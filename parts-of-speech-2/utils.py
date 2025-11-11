from typing import List
from transformers import AutoTokenizer


TAG_NAMES = [
    'NOUN',
    'PUNCT',
    'ADP',
    'NUM',
    'SYM',
    'SCONJ',
    'ADJ',
    'PART',
    'DET',
    'CCONJ',
    'PROPN',
    'PRON',
    'X',
    '_',
    'ADV',
    'INTJ',
    'VERB',
    'AUX'
]

def tokenize_with_labels(text: str, ud_tokens: List[str], ud_labels: List[int], ud_heads: List[str], tokenizer: AutoTokenizer) -> tuple[any, List[int]]:
    ud_filter_mask = [ud_head != "None" for ud_head in ud_heads]
    filtered_ud_tokens = [token for token, keep in zip(ud_tokens, ud_filter_mask) if keep]
    filtered_ud_labels = [label for label, keep in zip(ud_labels, ud_filter_mask) if keep]
    tokenized = tokenizer(
        text,
        truncation=True,
        is_split_into_words=False,
        return_offsets_mapping=True,
    )
    
    ud_token_idx = 0
    ud_token_char_idx = 0
    text_char_idx = 0

    text_char_idx_to_label = {}  # one label per char (unless we end early)

    while True:
        if ud_token_idx >= len(filtered_ud_tokens):
            break  # we matched every ud token, hooray!
        if text_char_idx >= len(text):
            raise ValueError("ran out of text before matching every ud token")  # bummer
        current_ud_token = filtered_ud_tokens[ud_token_idx]
        if text[text_char_idx] == current_ud_token[ud_token_char_idx]:
            # match
            text_char_idx_to_label[text_char_idx] = filtered_ud_labels[ud_token_idx]
            ud_token_char_idx += 1
            if ud_token_char_idx >= len(current_ud_token):
                ud_token_idx += 1
                ud_token_char_idx = 0
            text_char_idx += 1
        else:
            # didn't match
            text_char_idx += 1  # skip whitespace or other fuckery
    
    # now we have every char matched to a ud token index. Now we can label each model token.
    # if a model token spans multiple labels, we'll take the first one that isn't None.
    token_labels = []
    for (start, end) in tokenized["offset_mapping"]:
        if start == end:
            token_labels.append(None)  # this is BOS
            continue
        labels = [text_char_idx_to_label.get(i, None) for i in range(start, end)]
        filtered_labels = [label for label in labels if label is not None]
        if not filtered_labels:
            token_labels.append(None)  # this seems bad?
        else:
            token_labels.append(filtered_labels[0])  # take the first non-None label (good)

    return tokenized, token_labels

def upos_to_tag_name(upos_label: int) -> str:
    if upos_label is None:
        return "_"
    return TAG_NAMES[upos_label]


def tag_name_to_upos(tag_name: str):
    try:
        return TAG_NAMES.index(tag_name)
    except ValueError as exc:
        raise ValueError(f"Unknown tag name: {tag_name}") from exc
