# """
# Signature Visualizer for CVG Decomposition - Using matplotlib hist2d

# Creates clean 2D histogram visualizations with frequency-based coloring.
# Uses matplotlib's native hist2d for proper histogram rendering.
# Features adaptive bin scaling for 2nd and 3rd order signatures.

# X-axis: Path Length (discrete)
# Y-axis: Signature Value (adaptive bins)
# Color: Frequency with signature direction (Blue=negative, Red=positive)
# Display: Only count, details on hover

# Usage:
#     python signature_visualizer.py --all
#     python signature_visualizer.py --node 5
#     python signature_visualizer.py --all --bin-scale 20
# """

# import json
# import argparse
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import glob

# def load_decomposition_data(logs_dir='logs', single_node=None):
#     """
#     Load decomposition data from JSON files.
    
#     Args:
#         logs_dir: Directory containing JSON files
#         single_node: If specified, load only this node; otherwise load all
        
#     Returns:
#         Dictionary with first_order, second_order, third_order data and node_count
#     """
#     # Determine which files to load
#     if single_node is not None:
#         filepath = os.path.join(logs_dir, f'node_{single_node}_decomposition.json')
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"JSON file not found: {filepath}")
#         json_files = [filepath]
#         print(f"Loading single node: {single_node}")
#     else:
#         pattern = os.path.join(logs_dir, 'node_*_decomposition.json')
#         json_files = sorted(glob.glob(pattern))
#         if not json_files:
#             raise FileNotFoundError(f"No node_*_decomposition.json files found in {logs_dir}")
#         print(f"\nFound {len(json_files)} node decomposition files")
    
#     # Storage for all data
#     all_first_order = []
#     all_second_order = []
#     all_third_order = []
    
#     # Load each file
#     for filepath in json_files:
#         try:
#             with open(filepath, 'r') as f:
#                 data = json.load(f)
            
#             node_index = data['node_info']['index']
            
#             # Extract paths from this node
#             for path_entry in data['decomposition_paths']:
#                 path_id = path_entry['id']
#                 path_length = path_entry['length']
#                 path_nodes = path_entry['path']
#                 path_type = path_entry['type']
                
#                 # Create abbreviated ID
#                 parts = path_id.split('_')
#                 abbreviated_id = f"N{parts[1]}_P{parts[3]}" if len(parts) >= 4 else path_id
                
#                 # Common path info
#                 path_info = {
#                     'node_index': node_index,
#                     'path_id': path_id,
#                     'abbreviated_id': abbreviated_id,
#                     'length': path_length,
#                     'nodes': path_nodes,
#                     'type': path_type
#                 }
                
#                 # Add to appropriate order list
#                 sig1 = path_entry['signatures'].get('first_order')
#                 if sig1 is not None:
#                     all_first_order.append({**path_info, 'signature': sig1})
                
#                 sig2 = path_entry['signatures'].get('second_order')
#                 if sig2 is not None:
#                     all_second_order.append({**path_info, 'signature': sig2})
                
#                 sig3 = path_entry['signatures'].get('third_order')
#                 if sig3 is not None:
#                     all_third_order.append({**path_info, 'signature': sig3})
            
#             print(f"  ✓ Loaded node {node_index}")
            
#         except Exception as e:
#             print(f"  ✗ Error loading {filepath}: {e}")
#             continue
    
#     print(f"\n{'='*60}")
#     print("STATISTICS")
#     print("="*60)
#     print(f"Nodes processed: {len(json_files)}")
#     print(f"First-order paths: {len(all_first_order)}")
#     print(f"Second-order paths: {len(all_second_order)}")
#     print(f"Third-order paths: {len(all_third_order)}")
    
#     return {
#         'first_order': all_first_order,
#         'second_order': all_second_order,
#         'third_order': all_third_order,
#         'node_count': len(json_files)
#     }

# def get_bin_size_for_order(order, custom_bin_scale=None):
#     """
#     Get adaptive bin size based on signature order.
    
#     Args:
#         order: Signature order (1, 2, or 3)
#         custom_bin_scale: Optional custom bin size
        
#     Returns:
#         Bin size for Y-axis (signature values)
#     """
#     if custom_bin_scale is not None:
#         return custom_bin_scale
    
#     # Adaptive bin scaling: higher orders have larger ranges
#     bin_sizes = {1: 10, 2: 20, 3: 50}
#     return bin_sizes.get(order, 10)

# def create_clean_2d_histogram(signature_data, order, node_count, bin_scale=None):
#     """
#     Create clean 2D histogram using matplotlib's hist2d.
    
#     X-axis: Path Length (discrete)
#     Y-axis: Signature Value (adaptive bins)
#     Color: Frequency with signature direction (Blue=negative, Red=positive)
#     Display: Only count, details on hover
#     """
#     if not signature_data:
#         print(f"No {order} order signature data to visualize")
#         return None
    
#     print(f"\nCreating {order} order 2D histogram...")
    
#     lengths = np.array([d['length'] for d in signature_data])
#     signatures = np.array([d['signature'] for d in signature_data])
    
#     print(f"  Length range: [{lengths.min()}, {lengths.max()}]")
#     print(f"  Signature range: [{signatures.min():.2f}, {signatures.max():.2f}]")
    
#     # Get adaptive bin size
#     y_bin_size = get_bin_size_for_order(order, bin_scale)
#     print(f"  Using Y-axis bin size: {y_bin_size}")
    
#     # X-axis bins: one per discrete length
#     unique_lengths = sorted(set(lengths))
#     x_bins = np.arange(min(unique_lengths) - 0.5, max(unique_lengths) + 1.5, 1)
    
#     # Y-axis bins: adaptive based on order
#     sig_min = np.floor(signatures.min() / y_bin_size) * y_bin_size
#     sig_max = np.ceil(signatures.max() / y_bin_size) * y_bin_size
#     y_bins = np.arange(sig_min, sig_max + y_bin_size, y_bin_size)
    
#     print(f"  Grid: {len(x_bins)-1} lengths × {len(y_bins)-1} signature bins")
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(20, 12))
#     fig.patch.set_facecolor('white')
    
#     # Use matplotlib's hist2d
#     h, xedges, yedges, img = ax.hist2d(lengths, signatures, 
#                                         bins=[x_bins, y_bins],
#                                         cmap='RdBu_r',  # Red=positive, Blue=negative
#                                         cmin=0.5)  # Don't show empty bins
    
#     # Store bin information for hover
#     bin_data = {}
#     for data_point in signature_data:
#         length = data_point['length']
#         signature = data_point['signature']
        
#         x_idx = np.digitize(length, xedges) - 1
#         y_idx = np.digitize(signature, yedges) - 1
        
#         x_idx = max(0, min(len(xedges) - 2, x_idx))
#         y_idx = max(0, min(len(yedges) - 2, y_idx))
        
#         key = (y_idx, x_idx)
#         if key not in bin_data:
#             bin_data[key] = []
#         bin_data[key].append(data_point)
    
#     # Add count text to non-empty bins (ONLY frequency number)
#     for y_idx in range(len(yedges) - 1):
#         for x_idx in range(len(xedges) - 1):
#             count_val = h[x_idx, y_idx]
#             # Skip NaN or zero values
#             if np.isnan(count_val) or count_val == 0:
#                 continue
#             count = int(count_val)
#             if count > 0:
#                 x_center = (xedges[x_idx] + xedges[x_idx + 1]) / 2
#                 y_center = (yedges[y_idx] + yedges[y_idx + 1]) / 2
#                 avg_sig = (yedges[y_idx] + yedges[y_idx + 1]) / 2
                
#                 # Text color for visibility
#                 text_color = 'white' if abs(avg_sig) > (sig_max - sig_min) * 0.3 else 'black'
                
#                 # Display ONLY the count
#                 ax.text(x_center, y_center, str(count),
#                        ha='center', va='center',
#                        fontsize=12, weight='bold',
#                        color=text_color,
#                        family='sans-serif')
    
#     # Hover annotation
#     hover_annotation = ax.annotate('', xy=(0, 0), xytext=(20, 20),
#                                    textcoords='offset points',
#                                    bbox=dict(boxstyle='round,pad=0.8',
#                                            fc='#fffacd',
#                                            ec='#333',
#                                            alpha=0.98,
#                                            linewidth=2),
#                                    arrowprops=dict(arrowstyle='->',
#                                                  connectionstyle='arc3,rad=0.2',
#                                                  color='#333',
#                                                  lw=2),
#                                    fontsize=9,
#                                    family='monospace',
#                                    visible=False,
#                                    zorder=1000)
    
#     def on_hover(event):
#         if event.inaxes != ax:
#             hover_annotation.set_visible(False)
#             fig.canvas.draw_idle()
#             return
        
#         if event.xdata is not None and event.ydata is not None:
#             x_idx = np.digitize(event.xdata, xedges) - 1
#             y_idx = np.digitize(event.ydata, yedges) - 1
            
#             x_idx = max(0, min(len(xedges) - 2, x_idx))
#             y_idx = max(0, min(len(yedges) - 2, y_idx))
            
#             key = (y_idx, x_idx)
            
#             if key in bin_data:
#                 paths = bin_data[key]
#                 count = len(paths)
                
#                 length_val = int(xedges[x_idx] + 0.5)
#                 sig_min_bin = yedges[y_idx]
#                 sig_max_bin = yedges[y_idx + 1]
                
#                 hover_text = f"BIN DETAILS\n"
#                 hover_text += f"{'='*45}\n"
#                 hover_text += f"Length: {length_val}\n"
#                 hover_text += f"Signature: [{sig_min_bin:.1f} to {sig_max_bin:.1f}]\n"
#                 hover_text += f"Frequency: {count} paths\n"
#                 hover_text += f"{'-'*45}\n\n"
                
#                 display_count = min(count, 10)
#                 for i, path in enumerate(paths[:display_count]):
#                     hover_text += f"{i+1}. {path['path_id']}\n"
#                     hover_text += f"   Sig: {path['signature']:.2f} | "
#                     hover_text += f"Nodes: {path['nodes']}\n"
#                     hover_text += f"   Type: {path['type']}\n"
                
#                 if count > 10:
#                     hover_text += f"\n... +{count - 10} more paths"
                
#                 hover_annotation.xy = (event.xdata, event.ydata)
#                 hover_annotation.set_text(hover_text)
#                 hover_annotation.set_visible(True)
#                 fig.canvas.draw_idle()
#                 return
        
#         hover_annotation.set_visible(False)
#         fig.canvas.draw_idle()
    
#     fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
#     # Axes labels
#     ax.set_xlabel('Path Length', fontsize=16, fontweight='bold', labelpad=12)
#     ax.set_ylabel('Signature Value', fontsize=16, fontweight='bold', labelpad=12)
    
#     # Colorbar
#     cbar = plt.colorbar(img, ax=ax, pad=0.02)
#     cbar.set_label('Frequency (Path Count)', fontsize=14, weight='bold', labelpad=15)
#     cbar.ax.tick_params(labelsize=11)
    
#     # Title
#     order_names = {1: 'First', 2: 'Second', 3: 'Third'}
#     title = f'{order_names[order]} Order Signatures - 2D Histogram\n'
#     title += f'Total Paths: {len(signature_data)} | Nodes: {node_count} | '
#     title += f'Bin Size: {y_bin_size}'
#     ax.set_title(title, fontsize=17, fontweight='bold', pad=20)
    
#     # Grid
#     ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#666')
#     ax.set_axisbelow(True)
    
#     # Instructions
#     fig.text(0.5, 0.01, '◆ Hover over bins to see detailed path information',
#             ha='center', fontsize=13, style='italic', weight='bold',
#             bbox=dict(boxstyle='round,pad=0.6',
#                      facecolor='#e8f4f8',
#                      edgecolor='#2c3e50',
#                      linewidth=2))
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
#     print(f"  ✓ Histogram created")
#     print(f"    Non-empty bins: {len(bin_data)}")
    
#     return fig

# def visualize_all_orders(data_dict, bin_scale=None):
#     """Create 3 plots (1st, 2nd, 3rd order) with adaptive binning."""
#     node_count = data_dict['node_count']
    
#     print(f"\n{'='*60}")
#     print(f"CREATING 2D HISTOGRAMS")
#     print(f"Dataset: {node_count} nodes")
#     if bin_scale:
#         print(f"Custom bin scale: {bin_scale}")
#     print("="*60)
    
#     figs = []
    
#     for order, data_key in [(1, 'first_order'), (2, 'second_order'), (3, 'third_order')]:
#         if data_dict[data_key]:
#             fig = create_clean_2d_histogram(data_dict[data_key], order, node_count, bin_scale)
#             if fig:
#                 figs.append(fig)
    
#     print(f"\n{'='*60}")
#     print(f"✓ COMPLETE: {len(figs)} histograms created")
#     print("="*60)
    
#     plt.show()

# def main():
#     """Main entry point."""
#     parser = argparse.ArgumentParser(
#         description='Visualize path signatures - 2D Histogram with hist2d',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python signature_visualizer.py --all                    # All nodes
#   python signature_visualizer.py --node 5                 # Single node
#   python signature_visualizer.py --all --bin-scale 20     # Custom bins
#         """
#     )
    
#     parser.add_argument('--all', action='store_true',
#                        help='Load ALL nodes from logs/ directory')
#     parser.add_argument('--node', type=int,
#                        help='Load single node')
#     parser.add_argument('--bin-scale', type=float,
#                        help='Custom Y-axis bin size (overrides adaptive scaling)')
    
#     args = parser.parse_args()
    
#     if not args.all and args.node is None:
#         parser.error('Must specify either --all or --node')
    
#     # Load data using unified function
#     data_dict = load_decomposition_data(
#         logs_dir='logs',
#         single_node=args.node if args.node is not None else None
#     )
    
#     print(f"\n{'='*60}")
#     print("CVG SIGNATURE VISUALIZER - MATPLOTLIB HIST2D")
#     print("="*60)
    
#     visualize_all_orders(data_dict, args.bin_scale)

# if __name__ == '__main__':
#     main()

"""
Signature Visualizer for CVG Decomposition - Using matplotlib hist2d

Creates clean 2D histogram visualizations with frequency-based coloring.
Uses matplotlib's native hist2d for proper histogram rendering.
Features adaptive bin scaling for 2nd and 3rd order signatures.

X-axis: Path Length (discrete, bins start at 1)
Y-axis: Signature Value (adaptive bins)
Color: Frequency (viridis colormap)
Display: Only count, details on hover

Usage:
    python signature_visualizer.py --all
    python signature_visualizer.py --node 5
    python signature_visualizer.py --all --bin-scale 20
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

def load_decomposition_data(logs_dir='logs', single_node=None):
    """
    Load decomposition data from JSON files.
    
    Args:
        logs_dir: Directory containing JSON files
        single_node: If specified, load only this node; otherwise load all
        
    Returns:
        Dictionary with first_order, second_order, third_order data and node_count
    """
    # Determine which files to load
    if single_node is not None:
        filepath = os.path.join(logs_dir, f'node_{single_node}_decomposition.json')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        json_files = [filepath]
        print(f"Loading single node: {single_node}")
    else:
        pattern = os.path.join(logs_dir, 'node_*_decomposition.json')
        json_files = sorted(glob.glob(pattern))
        if not json_files:
            raise FileNotFoundError(f"No node_*_decomposition.json files found in {logs_dir}")
        print(f"\nFound {len(json_files)} node decomposition files")
    
    # Storage for all data
    all_first_order = []
    all_second_order = []
    all_third_order = []
    
    # Load each file
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            node_index = data['node_info']['index']
            
            # Extract paths from this node
            for path_entry in data['decomposition_paths']:
                path_id = path_entry['id']
                path_length = path_entry['length']
                path_nodes = path_entry['path']
                path_type = path_entry['type']
                
                # Create abbreviated ID
                parts = path_id.split('_')
                abbreviated_id = f"N{parts[1]}_P{parts[3]}" if len(parts) >= 4 else path_id
                
                # Common path info
                path_info = {
                    'node_index': node_index,
                    'path_id': path_id,
                    'abbreviated_id': abbreviated_id,
                    'length': path_length,
                    'nodes': path_nodes,
                    'type': path_type
                }
                
                # Add to appropriate order list
                sig1 = path_entry['signatures'].get('first_order')
                if sig1 is not None:
                    all_first_order.append({**path_info, 'signature': sig1})
                
                sig2 = path_entry['signatures'].get('second_order')
                if sig2 is not None:
                    all_second_order.append({**path_info, 'signature': sig2})
                
                sig3 = path_entry['signatures'].get('third_order')
                if sig3 is not None:
                    all_third_order.append({**path_info, 'signature': sig3})
            
            print(f"  ✓ Loaded node {node_index}")
            
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("STATISTICS")
    print("="*60)
    print(f"Nodes processed: {len(json_files)}")
    print(f"First-order paths: {len(all_first_order)}")
    print(f"Second-order paths: {len(all_second_order)}")
    print(f"Third-order paths: {len(all_third_order)}")
    
    return {
        'first_order': all_first_order,
        'second_order': all_second_order,
        'third_order': all_third_order,
        'node_count': len(json_files)
    }

def get_bin_size_for_order(order, custom_bin_scale=None):
    """
    Get adaptive bin size based on signature order.
    
    Args:
        order: Signature order (1, 2, or 3)
        custom_bin_scale: Optional custom bin size
        
    Returns:
        Bin size for Y-axis (signature values)
    """
    if custom_bin_scale is not None:
        return custom_bin_scale
    
    # Adaptive bin scaling: higher orders have larger ranges
    bin_sizes = {1: 10, 2: 40, 3: 300}
    return bin_sizes.get(order, 10)

def create_clean_2d_histogram(signature_data, order, node_count, bin_scale=None):
    """
    Create clean 2D histogram using matplotlib's hist2d.
    
    X-axis: Path Length (bins start at 1: [1-2), [2-3), [3-4), ...)
    Y-axis: Signature Value (adaptive bins)
    Color: Frequency (viridis colormap)
    Display: Only count, details on hover
    """
    if not signature_data:
        print(f"No {order} order signature data to visualize")
        return None
    
    print(f"\nCreating {order} order 2D histogram...")
    
    lengths = np.array([d['length'] for d in signature_data])
    signatures = np.array([d['signature'] for d in signature_data])
    
    print(f"  Length range: [{lengths.min()}, {lengths.max()}]")
    print(f"  Signature range: [{signatures.min():.2f}, {signatures.max():.2f}]")
    
    # Get adaptive bin size
    y_bin_size = get_bin_size_for_order(order, bin_scale)
    print(f"  Using Y-axis bin size: {y_bin_size}")
    
    # X-axis bins: Start at 1, bins are [1-2), [2-3), [3-4), etc.
    min_length = int(lengths.min())
    max_length = int(lengths.max())
    x_bins = np.arange(min_length, max_length + 2, 1)  # +2 to include the last bin edge
    
    # Y-axis bins: adaptive based on order, aligned to bin_size
    sig_min = np.floor(signatures.min() / y_bin_size) * y_bin_size
    sig_max = np.ceil(signatures.max() / y_bin_size) * y_bin_size
    y_bins = np.arange(sig_min, sig_max + y_bin_size, y_bin_size)
    
    print(f"  Grid: {len(x_bins)-1} lengths × {len(y_bins)-1} signature bins")
    print(f"  X-bins: {x_bins[:5]}... to ...{x_bins[-5:]}")
    print(f"  Y-bins: {y_bins[:5]}... to ...{y_bins[-5:]}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    # Use matplotlib's hist2d with lighter Blues colormap (white to medium blue)
    # Use only the lighter portion of Blues to avoid dark colors
    import matplotlib.colors as mcolors
    blues_cmap = plt.cm.Blues
    colors = blues_cmap(np.linspace(0, 0.7, 256))  # Use only 0-70% of Blues range
    light_blues = mcolors.LinearSegmentedColormap.from_list('LightBlues', colors)
    
    h, xedges, yedges, img = ax.hist2d(lengths, signatures, 
                                        bins=[x_bins, y_bins],
                                        cmap=light_blues,  # White to light/medium blue
                                        cmin=0.5)  # Don't show empty bins
    
    # Set explicit tick marks at bin edges
    ax.set_xticks(x_bins)
    ax.set_yticks(y_bins)
    
    # Store bin information for hover
    bin_data = {}
    for data_point in signature_data:
        length = data_point['length']
        signature = data_point['signature']
        
        x_idx = np.digitize(length, xedges) - 1
        y_idx = np.digitize(signature, yedges) - 1
        
        x_idx = max(0, min(len(xedges) - 2, x_idx))
        y_idx = max(0, min(len(yedges) - 2, y_idx))
        
        key = (y_idx, x_idx)
        if key not in bin_data:
            bin_data[key] = []
        bin_data[key].append(data_point)
    
    # Add count text to non-empty bins (ONLY frequency number)
    for y_idx in range(len(yedges) - 1):
        for x_idx in range(len(xedges) - 1):
            count_val = h[x_idx, y_idx]
            # Skip NaN or zero values
            if np.isnan(count_val) or count_val == 0:
                continue
            count = int(count_val)
            if count > 0:
                x_center = (xedges[x_idx] + xedges[x_idx + 1]) / 2
                y_center = (yedges[y_idx] + yedges[y_idx + 1]) / 2
                
                # Text color for visibility
                # Light blues colormap: always use black text since we avoid dark colors
                text_color = 'black'
                
                # Display ONLY the count
                ax.text(x_center, y_center, str(count),
                       ha='center', va='center',
                       fontsize=12, weight='bold',
                       color=text_color,
                       family='sans-serif')
    
    # Hover annotation
    hover_annotation = ax.annotate('', xy=(0, 0), xytext=(20, 20),
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.8',
                                           fc='#fffacd',
                                           ec='#333',
                                           alpha=0.98,
                                           linewidth=2),
                                   arrowprops=dict(arrowstyle='->',
                                                 connectionstyle='arc3,rad=0.2',
                                                 color='#333',
                                                 lw=2),
                                   fontsize=9,
                                   family='monospace',
                                   visible=False,
                                   zorder=1000)
    
    def on_hover(event):
        if event.inaxes != ax:
            hover_annotation.set_visible(False)
            fig.canvas.draw_idle()
            return
        
        if event.xdata is not None and event.ydata is not None:
            x_idx = np.digitize(event.xdata, xedges) - 1
            y_idx = np.digitize(event.ydata, yedges) - 1
            
            x_idx = max(0, min(len(xedges) - 2, x_idx))
            y_idx = max(0, min(len(yedges) - 2, y_idx))
            
            key = (y_idx, x_idx)
            
            if key in bin_data:
                paths = bin_data[key]
                count = len(paths)
                
                length_start = int(xedges[x_idx])
                length_end = int(xedges[x_idx + 1])
                sig_min_bin = yedges[y_idx]
                sig_max_bin = yedges[y_idx + 1]
                
                hover_text = f"BIN DETAILS\n"
                hover_text += f"{'='*45}\n"
                hover_text += f"Length: [{length_start}-{length_end})\n"
                hover_text += f"Signature: [{sig_min_bin:.1f}, {sig_max_bin:.1f})\n"
                hover_text += f"Frequency: {count} paths\n"
                hover_text += f"{'-'*45}\n\n"
                
                display_count = min(count, 10)
                for i, path in enumerate(paths[:display_count]):
                    hover_text += f"{i+1}. {path['path_id']}\n"
                    hover_text += f"   Sig: {path['signature']:.2f} | "
                    hover_text += f"Nodes: {path['nodes']}\n"
                    hover_text += f"   Type: {path['type']}\n"
                
                if count > 10:
                    hover_text += f"\n... +{count - 10} more paths"
                
                hover_annotation.xy = (event.xdata, event.ydata)
                hover_annotation.set_text(hover_text)
                hover_annotation.set_visible(True)
                fig.canvas.draw_idle()
                return
        
        hover_annotation.set_visible(False)
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    # Axes labels
    ax.set_xlabel('Path Length', fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel('Signature Value', fontsize=16, fontweight='bold', labelpad=12)
    
    # Colorbar
    cbar = plt.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label('Frequency (Path Count)', fontsize=14, weight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=11)
    
    # Title
    order_names = {1: 'First', 2: 'Second', 3: 'Third'}
    title = f'{order_names[order]} Order Signatures - 2D Histogram\n'
    title += f'Total Paths: {len(signature_data)} | Nodes: {node_count} | '
    title += f'Bin Size: {y_bin_size}'
    ax.set_title(title, fontsize=17, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#666')
    ax.set_axisbelow(True)
    
    # Instructions
    fig.text(0.5, 0.01, '◆ Hover over bins to see detailed path information',
            ha='center', fontsize=13, style='italic', weight='bold',
            bbox=dict(boxstyle='round,pad=0.6',
                     facecolor='#e8f4f8',
                     edgecolor='#2c3e50',
                     linewidth=2))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    print(f"  ✓ Histogram created")
    print(f"    Non-empty bins: {len(bin_data)}")
    
    return fig

def visualize_all_orders(data_dict, bin_scale=None):
    """Create 3 plots (1st, 2nd, 3rd order) with adaptive binning."""
    node_count = data_dict['node_count']
    
    print(f"\n{'='*60}")
    print(f"CREATING 2D HISTOGRAMS")
    print(f"Dataset: {node_count} nodes")
    if bin_scale:
        print(f"Custom bin scale: {bin_scale}")
    print("="*60)
    
    figs = []
    
    for order, data_key in [(1, 'first_order'), (2, 'second_order'), (3, 'third_order')]:
        if data_dict[data_key]:
            fig = create_clean_2d_histogram(data_dict[data_key], order, node_count, bin_scale)
            if fig:
                figs.append(fig)
    
    print(f"\n{'='*60}")
    print(f"✓ COMPLETE: {len(figs)} histograms created")
    print("="*60)
    
    plt.show()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize path signatures - 2D Histogram with hist2d',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python signature_visualizer.py --all                    # All nodes
  python signature_visualizer.py --node 5                 # Single node
  python signature_visualizer.py --all --bin-scale 20     # Custom bins
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Load ALL nodes from logs/ directory')
    parser.add_argument('--node', type=int,
                       help='Load single node')
    parser.add_argument('--bin-scale', type=float,
                       help='Custom Y-axis bin size (overrides adaptive scaling)')
    
    args = parser.parse_args()
    
    if not args.all and args.node is None:
        parser.error('Must specify either --all or --node')
    
    # Load data using unified function
    data_dict = load_decomposition_data(
        logs_dir='logs',
        single_node=args.node if args.node is not None else None
    )
    
    print(f"\n{'='*60}")
    print("CVG SIGNATURE VISUALIZER - MATPLOTLIB HIST2D")
    print("="*60)
    
    visualize_all_orders(data_dict, args.bin_scale)

if __name__ == '__main__':
    main()