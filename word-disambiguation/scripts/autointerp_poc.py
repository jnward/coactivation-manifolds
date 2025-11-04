#!/usr/bin/env python3
"""
Proof of Concept: Fetch max activating examples from Neuronpedia and run autointerp with OpenRouter
"""

import os
import asyncio
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from neuron_explainer.activations.activation_records import ActivationRecord, calculate_max_activation
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat

def parse_sae_name_to_neuronpedia_format(sae_name: str, feature_idx: int) -> tuple[str, str, str]:
    """
    Convert SAE name to Neuronpedia URL format.
    """
    parts = sae_name.split('/')
    model_id = "gemma-2-2b"
    layer_part = parts[2]
    layer_num = layer_part.split('_')[1]
    width_part = parts[3]
    width = width_part.split('_')[1]
    sae_id = f"{layer_num}-gemmascope-res-{width}"
    
    return model_id, sae_id, f"{model_id}/{sae_id}/{feature_idx}"

def fetch_neuronpedia_data(model_id: str, sae_id: str, feature_idx: int) -> Dict[str, Any]:
    """
    Fetch feature data from Neuronpedia API.
    """
    api_url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_idx}"
    # print(f"Fetching from: {api_url}")
    
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def format_activation_records(neuronpedia_data: Dict[str, Any]) -> List[ActivationRecord]:
    """
    Convert Neuronpedia data to ActivationRecord format for automated-interpretability.
    """
    activation_records = []
    
    if 'activations' in neuronpedia_data:
        for act_example in neuronpedia_data['activations']:
            tokens = act_example.get('tokens', [])
            activations = act_example.get('values', [])
            
            record = ActivationRecord(
                tokens=tokens,
                activations=activations
            )
            activation_records.append(record)
    
    return activation_records

async def run_autointerp(
    activation_records: List[ActivationRecord],
    model_name: str,
    api_base_url: str,
    *,
    num_samples: int,
    max_tokens: int,
    reasoning_effort: str | None,
) -> List[str]:
    """
    Run TokenActivationPairExplainer to generate explanation.
    """
    print(f"\nRunning autointerp with model: {model_name}")
    print(f"Number of activation records: {len(activation_records)}")

    if not activation_records:
        raise ValueError("No activation records available to explain. "
                         "Check Neuronpedia response or feature index.")

    # Calculate max activation for context
    max_act = calculate_max_activation(activation_records)
    print(f"Max activation value: {max_act}")

    # Create explainer with activation records
    explainer = TokenActivationPairExplainer(
        model_name=model_name,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
        base_api_url=api_base_url,
    )

    generation_kwargs = dict(
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=1.0,
        all_activation_records=activation_records,
        max_activation=max_act,
    )
    if reasoning_effort:
        generation_kwargs["reasoning_effort"] = reasoning_effort

    # Generate explanations (async call, no positional args)
    explanations = await explainer.generate_explanations(**generation_kwargs)
    
    return explanations

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration
    SAE_NAME = os.getenv(
        "AUTO_INTERP_SAE_NAME", "google/gemma-scope-2b-pt-res/layer_12/width_65k/average_l0_72"
    )
    FEATURE_IDX = int(os.getenv("AUTO_INTERP_FEATURE_IDX", "43412"))
    MODEL_NAME = os.getenv("AUTO_INTERP_MODEL_NAME", "anthropic/claude-haiku-4.5")
    MAX_ACTIVATION_RECORDS = int(os.getenv("AUTO_INTERP_MAX_RECORDS", "45"))
    NUM_SAMPLES = int(os.getenv("AUTO_INTERP_NUM_SAMPLES", "1"))
    MAX_COMPLETION_TOKENS = int(os.getenv("AUTO_INTERP_MAX_TOKENS", "4096"))
    reasoning_effort_setting = os.getenv("AUTO_INTERP_REASONING_EFFORT", "none").strip().lower()
    REASONING_EFFORT = None if reasoning_effort_setting in {"", "none"} else reasoning_effort_setting
    
    # Configure OpenRouter
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    api_base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    os.environ["OPENAI_API_BASE"] = api_base_url
    
    print("=" * 80)
    print("Neuronpedia AutoInterp PoC")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  SAE: {SAE_NAME}")
    print(f"  Feature Index: {FEATURE_IDX}")
    print(f"  Model: {MODEL_NAME}")
    print()
    
    # Parse SAE name
    model_id, sae_id, full_path = parse_sae_name_to_neuronpedia_format(SAE_NAME, FEATURE_IDX)
    print(f"Parsed Neuronpedia path: {full_path}\n")
    
    # Fetch data from Neuronpedia
    print("Step 1: Fetching data from Neuronpedia...")
    neuronpedia_data = fetch_neuronpedia_data(model_id, sae_id, FEATURE_IDX)
    print(f"✓ Successfully fetched data")
    
    # Format activation records
    print("\nStep 2: Formatting activation records...")
    activation_records = format_activation_records(neuronpedia_data)
    if len(activation_records) > MAX_ACTIVATION_RECORDS:
        activation_records = activation_records[:MAX_ACTIVATION_RECORDS]
        print(
            f"✓ Created {len(activation_records)} activation records "
            f"(truncated to max {MAX_ACTIVATION_RECORDS})"
        )
    else:
        print(f"✓ Created {len(activation_records)} activation records")
    
    # Run autointerp (async)
    print("\nStep 3: Running autointerp...")
    explanations = asyncio.run(
        run_autointerp(
            activation_records,
            MODEL_NAME,
            api_base_url,
            num_samples=NUM_SAMPLES,
            max_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort=REASONING_EFFORT,
        )
    )
    
    # Output results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nGenerated {len(explanations)} explanation(s):\n")
    for i, explanation in enumerate(explanations, 1):
        print(f"Explanation {i}:")
        print(f"  {explanation}")
        print()
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()
