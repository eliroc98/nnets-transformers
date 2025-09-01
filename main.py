import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_lens
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from huggingface_hub import list_repo_refs
import einops
import functools
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import re
from torch import Tensor
from typing import Callable, List, Dict
import os
import warnings

# --- Local Imports ---
# Assumes get_sentences.py is in the same directory
from get_sentences import get_induction_data

# Suppress unnecessary warnings from matplotlib
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Configuration ---
CONFIG = {
    "NUM_SENTENCES": 5,          # Number of sentences to analyze
    "SEQUENCE_LENGTH": 300,       # Max sequence length for tokens
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "OUTPUT_DIR": "ablation_data", # Directory to save the raw data
}

# --- Ablation and Analysis Functions (largely from your script) ---

def get_component_dict(model: HookedTransformer, attn_only=False):
    # (Your function code here - no changes needed)
    input_component_dict = {
        ('hook_embed', None): {'input': 'tokens', 'transformation_matrix': 'W_E', 'layer':0},
        ('hook_pos_embed', None): {'input': 'tokens', 'transformation_matrix': 'W_pos', 'layer':0}
    }
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            input_component_dict[(f'blocks.{layer}.attn.hook_q', head)] = {'input': f'blocks.{layer}.ln1.hook_normalized', 'transformation_matrix': 'W_Q', 'layer':layer, 'head':head}
            input_component_dict[(f'blocks.{layer}.attn.hook_k', head)] = {'input': f'blocks.{layer}.ln1.hook_normalized', 'transformation_matrix': 'W_K', 'layer':layer, 'head':head}
            input_component_dict[(f'blocks.{layer}.attn.hook_v', head)] = {'input': f'blocks.{layer}.ln1.hook_normalized', 'transformation_matrix': 'W_V', 'layer':layer, 'head':head}
            input_component_dict[(f'blocks.{layer}.attn.hook_result', head)] = {'input': f'blocks.{layer}.attn.hook_z', 'transformation_matrix': 'W_O', 'layer':layer, 'head':head}
        if not attn_only:
            input_component_dict[(f'blocks.{layer}.mlp.hook_pre', None)] = {'input': f'blocks.{layer}.ln2.hook_normalized', 'transformation_matrix': 'W_in', 'layer':layer}
            input_component_dict[(f'blocks.{layer}.hook_mlp_out', None)] = {'input': f'blocks.{layer}.mlp.hook_post', 'transformation_matrix': 'W_out', 'layer':layer}
    return input_component_dict

def get_hook_list(model: HookedTransformer, attn=True, mlp=True, ln=True, embed=True, qkv=True):
    hooks=[]
    if not attn: qkv=False
    for h in model.hook_dict:
        if 'pattern' in h or '.hook_attn_' in h or 'attn.hook_result' in h or '_input' in h or 'resid' in h or 'hook_mlp' in h or 'mlp.hook_pre' in h:
            continue
        if not attn and 'attn' in h:
            continue
        if not mlp and 'mlp' in h:
            continue
        if not ln and 'ln' in h:
            continue
        if not embed and 'embed' in h:
            continue
        if not qkv and ('_q' in h or '_k' in h or '_v' in h):
            continue
        hooks.append(h)
    return hooks


def zero_ablation_hook(
    component_output: Tensor,
    hook: HookPoint,
    **kwargs
) -> None:
    head_idx_to_ablate = kwargs.get('head_idx_to_ablate')
    if head_idx_to_ablate is None:
        component_output[:] = 0.0
    else:
        component_output[:,:,head_idx_to_ablate,:] = 0.0

def random_weight_ablation_hook(component_output: Tensor, hook: HookPoint, component_input: Tensor, transformation_shape: tuple, **kwargs):
    # (Your function code here - simplified for clarity, assuming device is passed)
    random_w = torch.empty(transformation_shape, device=component_output.device)
    nn.init.kaiming_normal_(random_w, nonlinearity='relu')
    head_idx_to_ablate = kwargs.get('head_idx_to_ablate')

    if 'hook_q' in hook.name or 'hook_k' in hook.name or 'hook_v' in hook.name:
        ablated_output = einops.einsum(component_input, random_w, "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head")
        if head_idx_to_ablate is not None:
            component_output[:, :, head_idx_to_ablate, :] = ablated_output[:, :, head_idx_to_ablate, :]
        else:
            component_output[:] = ablated_output
    elif 'hook_result' in hook.name:
        ablated_output = einops.einsum(component_input, random_w, "batch seq n_heads d_head, n_heads d_head d_model -> batch seq n_heads d_model")
        if head_idx_to_ablate is not None:
            component_output[:, :, head_idx_to_ablate, :] = ablated_output[:, :, head_idx_to_ablate, :]
        else:
            component_output[:] = ablated_output
    elif 'hook_embed' in hook.name:
        component_output[:] = random_w[component_input]
    elif 'hook_pos_embed' in hook.name:
        component_output[:] = random_w[torch.arange(component_input.shape[1], device=component_output.device)]
    elif 'mlp.hook_pre' in hook.name:
        component_output[:] = einops.einsum(component_input, random_w, "batch seq d_model, n_layers d_model d_ff -> batch seq d_ff")
    elif 'hook_mlp_out' in hook.name:
        component_output[:] = einops.einsum(component_input, random_w, "batch seq d_ff, n_layers d_ff d_model -> batch seq d_model")

def compute_ablation_score(component_output: Tensor, hook: HookPoint, clean_cache, cache_diffs):
    similarity = F.cosine_similarity(component_output, clean_cache[hook.name], dim=-1)
    cache_diffs[hook.name] = similarity.mean().item()

def compute_head_ablation_score(component_output: Tensor, hook: HookPoint, clean_cache, cache_diffs, head_idx_to_ablate):
    similarity = F.cosine_similarity(component_output[:,:,head_idx_to_ablate,:], clean_cache[hook.name][:,:,head_idx_to_ablate,:], dim=-1)
    cache_diffs[f"{hook.name}.{head_idx_to_ablate}"] = similarity.mean().item()

def get_random_ablation_scores(model: HookedTransformer, tokens: Tensor, component_to_ablate: str, component_input: str, transformation_shape: tuple, head_idx_to_ablate: int=None):
    _, clean_cache = model.run_with_cache(tokens)
    cache_dict_diffs = {}
    
    caching_hooks = []
    for name in clean_cache.keys():
        if 'hook_attn_scores' in name or 'hook_pattern' in name: continue # Skip non-tensor hooks
        temp_hook_fn_partial = functools.partial(compute_ablation_score, clean_cache=clean_cache, cache_diffs=cache_dict_diffs)
        caching_hooks.append((name, temp_hook_fn_partial))
        if name.endswith(('hook_q', 'hook_k', 'hook_v', 'hook_z', 'hook_result')):
            for head in range(model.cfg.n_heads):
                temp_hook_head = functools.partial(compute_head_ablation_score, clean_cache=clean_cache, cache_diffs=cache_dict_diffs, head_idx_to_ablate=head)
                caching_hooks.append((name, temp_hook_head))
    
    temp_hook_fn = functools.partial(random_weight_ablation_hook, component_input=tokens if component_input == 'tokens' else clean_cache[component_input], transformation_shape=transformation_shape, head_idx_to_ablate=head_idx_to_ablate)
    ablation_hook = (component_to_ablate, temp_hook_fn)
    
    model.run_with_hooks(tokens, fwd_hooks=[ablation_hook] + caching_hooks, return_type=None)
    model.reset_hooks()
    return cache_dict_diffs

def get_zero_ablation_scores(model: HookedTransformer, tokens: Tensor, component_to_ablate: str, components_to_cache:list, head_idx_to_ablate: int=None):
    clean_logits, clean_cache = model.run_with_cache(tokens)
    cache_dict_diffs = {}
    
    caching_hooks = []
    for name in components_to_cache:
        temp_hook_fn_partial = functools.partial(compute_ablation_score, clean_cache=clean_cache, cache_diffs=cache_dict_diffs)
        caching_hooks.append((name, temp_hook_fn_partial))
        if name.endswith(('hook_q', 'hook_k', 'hook_v', 'hook_z', 'hook_result')):
            for head in range(model.cfg.n_heads):
                temp_hook_head = functools.partial(compute_head_ablation_score, clean_cache=clean_cache, cache_diffs=cache_dict_diffs, head_idx_to_ablate=head)
                caching_hooks.append((name, temp_hook_head))
    
    temp_hook_fn = functools.partial(zero_ablation_hook, head_idx_to_ablate=head_idx_to_ablate)
    ablation_hook = (component_to_ablate, temp_hook_fn)
    
    logits=model.run_with_hooks(tokens, fwd_hooks=[ablation_hook] + caching_hooks, return_type='logits')
    model.reset_hooks()
    cache_dict_diffs['logit']=F.cosine_similarity(logits, clean_logits, dim=-1).mean().item()
    return cache_dict_diffs

def plot_similarity_comparison(matrix_list: List[pd.DataFrame], save_path: str):
    """
    Compares a list of similarity matrices and plots the result as a heatmap.
    """
    num_matrices = len(matrix_list)
    if num_matrices < 2:
        print("Need at least two sentences to compare graphs. Skipping comparison plot.")
        return
        
    correlation_matrix = np.zeros((num_matrices, num_matrices))

    # Flatten each DataFrame to a 1D vector for comparison
    flat_vectors = [df.values.flatten() for df in matrix_list]

    for i in range(num_matrices):
        for j in range(i, num_matrices):
            # Calculate Pearson correlation coefficient
            corr = np.corrcoef(flat_vectors[i], flat_vectors[j])[0, 1]
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f',
                xticklabels=[f'Sentence {i}' for i in range(num_matrices)],
                yticklabels=[f'Sentence {i}' for i in range(num_matrices)])
    plt.title('Similarity of Ablation Impact Across Sentences\n(Pearson Correlation of Impact Matrices)')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comparison heatmap to {save_path}")

# --- Main Execution Logic ---

def main():
    """
    Main function to run the full analysis pipeline.
    """
    print(f"Using device: {CONFIG['DEVICE']}")
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    
    model_id = "EleutherAI/pythia-14m"
    available_checkpoints = []

    # Fetch all the revisions for the model
    try:
        refs = list_repo_refs(model_id)
        print(f"Revisions for model: {model_id}")

        print("\n--- Branches ---")
        for branch in refs.branches:
            print(branch.name)
            if branch.name.startswith("step"):
                available_checkpoints.append(branch.name)

    except Exception as e:
        print(f"Could not fetch refs for {model_id}. Error: {e}")
        print("The model might not exist or could be private.")
    
    #HookedTransformer.from_pretrained("gpt2", device=CONFIG['DEVICE'], default_padding_side='left')
    #model.set_use_attn_result(True)
    
    text = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
    that they were perfectly normal, thank you very much. They were the last
    people you'd expect to be involved in anything strange or mysterious,
    because they just didn't hold with such nonsense.
    Mr. Dursley was the director of a firm called Grunnings, which made
    drills. He was a big, beefy man with hardly any neck, although he did
    have a very large mustache. Mrs. Dursley was thin and blonde and had
    nearly twice the usual amount of neck, which came in very useful as she
    spent so much of her time craning over garden fences, spying on the
    neighbors. The Durs"""
    
    available_checkpoints = available_checkpoints[::-1]
    # 2. Process Each Sentence
    for i, checkpoint in enumerate(available_checkpoints[1:]):
        print(f"\n--- Processing Model {i+1}/{len(available_checkpoints)} ({checkpoint})---")

        model = HookedTransformer.from_pretrained("EleutherAI/pythia-14m", device = CONFIG['DEVICE'], revision=checkpoint)
        tokens = model.to_tokens(text, padding_side='left')

        # Run the full ablation experiment for this sentence
        #component_dict = get_component_dict(model)
        component_list = get_hook_list(model, ln=False)
        component_dict={}
        for c in component_list:
            if 'embed' in c: layer = -1
            elif 'final' in c: layer = model.cfg.n_layers
            else:
                layer = next(int(s) for s in re.findall(r'\b\d+\b', c))
            component_dict[(c,None)]={'layer':layer,'head':None}
            if 'attn' in c:
                for head in range(model.cfg.n_heads):
                    component_dict[(c,head)]={'layer':layer, 'head':head}
        with torch.no_grad():
            for (component_to_ablate, head_idx), info in component_dict.items():              
                info['diffs'] = get_zero_ablation_scores(
                    model=model,
                    tokens=tokens,
                    component_to_ablate=component_to_ablate,
                    components_to_cache=component_list,
                    head_idx_to_ablate=head_idx
                )
        
        # Create the similarity dataframe from the results
        heatmap_data = {
            f"{c}{'' if h is None else '.'+str(h)}": info['diffs']
            for (c, h), info in component_dict.items()
        }
        df_sim = pd.DataFrame(heatmap_data).T
        save_path = os.path.join(CONFIG['OUTPUT_DIR'], f'{checkpoint}.csv')
        df_sim.to_csv(save_path)
        print(f"Saved similarity matrix to {save_path}")
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()