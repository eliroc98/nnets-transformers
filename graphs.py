import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
import re
import warnings

# Suppress unnecessary warnings from matplotlib
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Helper Functions (Copied from your previous script) ---

def convert_name_to_component(component_name,max_layer):
    if 'pos_embed' in component_name: return 'pos_emb', -1, None
    if 'embed' in component_name: return 'emb', -1, None
    if 'final' in component_name: return 'ln_'+component_name.split('_')[-1][:5],max_layer, None
    if 'logit' in component_name: return 'logits', max_layer+1, None
    parts = component_name.replace("hook_","").replace('_in',"").replace("_out","").split(".")[1:]
    if 'ln' in parts[1]:
        return f"{parts[1]}.{parts[2]}.{parts[0]}", int(parts[0]), None
    if 'attn' in parts[1]:
        try:
            return f"{parts[1]}.{parts[2]}.{parts[0]}.{parts[-1]}", int(parts[0]), int(parts[-1])
        except ValueError:
            return f"{parts[1]}.{parts[2]}.{parts[0]}", int(parts[0]), None
    if 'mlp' in parts[1]: return f'{parts[1]}_{parts[0]}', int(parts[0]), None
    return None

def create_component_graph(adj_list, max_layer):
    G = nx.DiGraph()
    for (source, dest, weight) in adj_list:
        # Note: weight here is the similarity score. The graph edge weight will be 1-similarity
        source_name, source_layer, _ = convert_name_to_component(source, max_layer)
        target_name, target_layer, _ = convert_name_to_component(dest, max_layer)
        G.add_node(source_name, layer=source_layer)
        G.add_node(target_name, layer=target_layer)
        G.add_edge(source_name, target_name, weight=(1 - weight))
    return G

def plot_layered_graph(G, threshold: float, save_path: str):
    if G.number_of_nodes() == 0:
        print(f"No connections found with threshold {threshold}. Skipping plot.")
        return

    pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')
    
    for k, v in pos.items(): pos[k] = (v[0] * 20, v[1] * -20)
    
    edge_widths = [5 * G[u][v]['weight'] for u, v in G.edges()]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(24, 18))

    layers = sorted(list(set(nx.get_node_attributes(G, 'layer').values())))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    layer_to_color = {layer: colors[i] for i, layer in enumerate(layers)}
    node_colors = [layer_to_color.get(data['layer'], 'gray') for node, data in G.nodes(data=True)]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    nx.draw_networkx_edges(G, pos, alpha=0.7, width=edge_widths, arrows=True, arrowsize=20, edge_color='cyan', connectionstyle='arc3,rad=0.1', node_size=2500, ax=ax)
    
    legend_patches = [mpatches.Patch(color=color, label=f'Layer {layer}') for layer, color in layer_to_color.items()]
    ax.legend(handles=legend_patches, title="Component Layers", loc='upper left')

    ax.set_title(f"Component Similarity Graph (Connections where similarity < {threshold})", fontsize=20, color='white')
    plt.axis('off')
    fig.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='darkslategrey')
    plt.close(fig)
    print(f"Saved graph to {save_path}")
    
def get_max_layer_from_df(df: pd.DataFrame) -> int:
    """Infers the number of layers from the component names in the DataFrame index."""
    max_layer = 0
    for name in df.index:
        match = re.search(r"blocks\.(\d+)\.", name)
        if match:
            layer_num = int(match.group(1))
            if layer_num > max_layer:
                max_layer = layer_num
    return max_layer

# --- Main Execution Logic ---

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all the data files to process
    try:
        data_files = [f for f in os.listdir(args.input_dir) if f.startswith('step') and f.endswith('.csv')]
        if not data_files:
            print(f"Error: No data files found in '{args.input_dir}'.")
            print("Please run 'generate_ablation_data.py' first.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    # Process each data file
    for filename in data_files:
        print(f"\n--- Processing {filename} with threshold={args.threshold} ---")
        
        # 1. Load the pre-computed similarity matrix
        file_path = os.path.join(args.input_dir, filename)
        df_sim = pd.read_csv(file_path, index_col=0)
        
        # Infer model architecture details from the data itself
        max_layer = get_max_layer_from_df(df_sim)

        # 2. Filter for low similarity scores based on the threshold
        low_values = df_sim[df_sim<args.threshold].stack()
        adj_tuples=[]
        for (row_name, col_name), value in low_values.items():
            _, row_layer, _ = convert_name_to_component(row_name, max_layer)
            _, col_layer, _ = convert_name_to_component(col_name, max_layer)
            
            if row_layer >= col_layer: continue
            adj_tuples.append((row_name, col_name, 1-value))            
        

        # 3. Create and plot the graph
        component_graph = create_component_graph(adj_tuples, max_layer)

        # Construct a descriptive output filename
        base_name = os.path.splitext(filename)[0]
        output_filename = f"graph_{base_name}_thresh_{args.threshold}.png"
        save_path = os.path.join(args.output_dir, output_filename)
        
        plot_layered_graph(component_graph, args.threshold, save_path)
        
    print("\nPlot generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot component graphs from ablation data.")
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.7,
        help="Similarity threshold. Connections are drawn for scores BELOW this value."
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='ablation_data',
        help="Directory containing the saved df_sim CSV files."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='analysis_plots',
        help="Directory where the output graph PNGs will be saved."
    )
    args = parser.parse_args()
    main(args)