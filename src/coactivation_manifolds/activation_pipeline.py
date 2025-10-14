"""End-to-end activation logging with Gemma and an SAE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from .activation_writer import ActivationRecord, ActivationWriter


@dataclass
class PipelineConfig:
    dataset: Dataset | Iterable[dict]
    text_field: str = "text"
    batch_size: int = 2
    max_length: int = 1024
    layer_index: int = 12
    device: str = "cuda"
    use_device_map: bool = False  # True when model uses device_map="auto"


class ActivationPipeline:
    """Runs Gemma + SAE and streams activations to disk."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        sae,
        writer: ActivationWriter,
        config: PipelineConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.writer = writer
        self.config = config

        self.device = torch.device(config.device)
        self.use_device_map = config.use_device_map

        # Only move model if not using device_map (it's already distributed)
        if not self.use_device_map:
            self.model.to(self.device)
        self.model.eval()

        if hasattr(self.sae, "to"):
            self.sae.to(self.device)
        if hasattr(self.sae, "eval"):
            self.sae.eval()

        self._sae_input_dtype = self._resolve_sae_dtype()

        self._global_token_index = 0
        self._doc_counter = 0

    def run(
        self,
        *,
        max_tokens: Optional[int] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        show_progress: bool = False,
    ) -> None:
        progress = None
        if show_progress and progress_callback is None:
            progress = tqdm(total=max_tokens, unit="tok", desc="Logging activations")
            progress_callback = progress.update
        for batch_samples in self._iter_batches():
            if max_tokens is not None and self._global_token_index >= max_tokens:
                break

            texts = [sample[self.config.text_field] for sample in batch_samples]
            doc_ids = [self._resolve_doc_id(sample) for sample in batch_samples]

            tokenized = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            # Only manually move to device if not using device_map
            if not self.use_device_map:
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states[self.config.layer_index]

                # Move hidden states to SAE device if needed
                if hidden_states.device != self.device:
                    hidden_states = hidden_states.to(self.device)

                sae_inputs = (
                    hidden_states
                    if hidden_states.dtype == self._sae_input_dtype
                    else hidden_states.to(self._sae_input_dtype)
                )
                sae_outputs = self.sae.encode(sae_inputs)
                sae_outputs = torch.relu(sae_outputs)

            stop = self._emit_records(
                batch_samples,
                doc_ids,
                tokenized,
                sae_outputs,
                max_tokens=max_tokens,
                progress_callback=progress_callback,
            )
            if stop:
                break

        self.writer.finalize()
        if progress is not None:
            progress.close()

    def _iter_batches(self) -> Iterator[List[dict]]:
        dataset = self.config.dataset
        if isinstance(dataset, Dataset):
            iterator = (dataset[i] for i in range(len(dataset)))
        else:
            iterator = iter(dataset)

        batch: List[dict] = []
        for example in iterator:
            batch.append(example)
            if len(batch) == self.config.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _resolve_doc_id(self, sample: dict) -> int:
        doc_id = self._doc_counter
        self._doc_counter += 1
        return doc_id

    def _emit_records(
        self,
        batch_samples: List[dict],
        doc_ids: List[int],
        tokenized,
        sae_outputs: torch.Tensor,
        *,
        max_tokens: Optional[int] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        input_ids = tokenized["input_ids"].detach().cpu()
        sae_cpu = sae_outputs.detach().cpu()
        attention_mask = tokenized.get("attention_mask")
        mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None

        records: List[ActivationRecord] = []
        stop = False

        for batch_idx, sample in enumerate(batch_samples):
            doc_id = doc_ids[batch_idx]
            seq_ids = input_ids[batch_idx]
            seq_acts = sae_cpu[batch_idx]
            seq_len = int(seq_ids.shape[0])
            mask = mask_cpu[batch_idx] if mask_cpu is not None else torch.ones(seq_len, dtype=torch.long)

            position_in_doc = 0
            for token_pos in range(seq_len):
                if max_tokens is not None and self._global_token_index >= max_tokens:
                    stop = True
                    break
                if mask[token_pos] == 0:
                    continue

                feature_vector = seq_acts[token_pos]
                nz = torch.nonzero(feature_vector, as_tuple=False).squeeze(-1)

                if nz.ndim == 0:
                    nz = nz.unsqueeze(0)
                if nz.numel() == 0:
                    feature_ids = []
                    activations = []
                else:
                    nz_np = nz.numpy().astype("uint16")
                    feature_ids = nz_np.tolist()
                    activations = feature_vector[nz].numpy().astype("float16").tolist()

                snippet = self._make_snippet(seq_ids.numpy(), token_pos)

                record = ActivationRecord(
                    doc_id=doc_id,
                    token_index=self._global_token_index,
                    position_in_doc=position_in_doc,
                    feature_ids=feature_ids,
                    activations=activations,
                    token_text=snippet,
                )
                records.append(record)

                self._global_token_index += 1
                position_in_doc += 1

            if stop:
                break

        if records:
            self.writer.add_records(records)
            if progress_callback is not None:
                progress_callback(len(records))
        return stop

    def _resolve_sae_dtype(self) -> torch.dtype:
        dtype_attr = getattr(self.sae, "dtype", None)
        if isinstance(dtype_attr, torch.dtype):
            return dtype_attr

        cfg = getattr(self.sae, "cfg", None)
        if cfg is not None:
            cfg_dtype = getattr(cfg, "dtype", None)
            if isinstance(cfg_dtype, torch.dtype):
                return cfg_dtype
            if isinstance(cfg_dtype, str) and hasattr(torch, cfg_dtype):
                return getattr(torch, cfg_dtype)

        return torch.float32

    def _make_snippet(self, seq_ids: np.ndarray, token_pos: int, window: int = 10) -> str:
        seq_len = int(seq_ids.shape[0])
        left_start = max(token_pos - window, 0)
        right_end = min(token_pos + window + 1, seq_len)

        left_ids = seq_ids[left_start:token_pos]
        center_ids = seq_ids[token_pos : token_pos + 1]
        right_ids = seq_ids[token_pos + 1 : right_end]

        decode = self.tokenizer.decode
        left_text = decode(left_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        center_text = decode(center_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        right_text = decode(right_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

        snippet = f"{left_text} «{center_text}» {right_text}".strip()
        snippet = " ".join(snippet.split())
        return snippet
