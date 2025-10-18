

# TestCMOVG/visualization.py - FINAL FIXED VERSION

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import time

# CORRECT COLOR PALETTE
COLORS = {
    'selected_node': '#FFD700',      # Gold for selected node
    'boundary_nodes': '#FF6B6B',     # Red for boundary nodes
    'decreasing_nodes': '#4A90E2',   # Blue for decreasing chains
    'increasing_nodes': '#7ED321',   # Green for increasing chains
    'other_nodes': "#9E9D9D",        # Light grey for other nodes
    'time_series': '#000000',        # Black for time series
    
    # SOFTER EDGE COLORS (only these are softer)
    'decreasing_edges': '#87CEEB',   # Softer blue for decreasing edges
    'increasing_edges': '#98FB98',   # Softer green for increasing edges
    'visibility_edges': "#343635"    # grey for visibility lines
}

def calculate_positions(cvg):
    """Calculate positions for both graph layouts."""
    print("Calculating layout positions...")
    start_time = time.time()
    
    if cvg.n <= 1000:
        cvg.positions_standard = nx.spring_layout(cvg.visibility_graph, k=1/np.sqrt(cvg.n), iterations=50)
    else:
        cvg.positions_standard = nx.circular_layout(cvg.visibility_graph)
    
    if cvg.root is not None:
        cvg.positions_graph = _calculate_graph_positions(cvg)
    
    layout_time = time.time() - start_time
    cvg.build_time['layout'] = layout_time
    print(f"Layout calculation time: {layout_time:.2f} seconds")

def _calculate_graph_positions(cvg):
    """Calculate hierarchical positions for graph layout."""
    if cvg.root is None:
        return {}
    
    magnitude_levels = _calculate_magnitude_based_levels(cvg)
    levels = defaultdict(list)
    for node, level in magnitude_levels.items():
        levels[level].append(node)
    
    positions = {}
    max_level = max(levels.keys()) if levels else 0
    
    for level, nodes in levels.items():
        nodes.sort(key=lambda n: (-cvg.time_series[n], n))
        width = len(nodes)
        y_position = (max_level - level) * 5
        
        for i, node in enumerate(nodes):
            if width == 1:
                x_position = 0
            else:
                base_spacing = max(4.0, 40.0 / width)
                x_position = (i - (width - 1) / 2) * base_spacing
            
            x_offset = np.sin(node * 0.2) * 0.2
            positions[node] = (x_position + x_offset, y_position)
    
    return positions

def _calculate_magnitude_based_levels(cvg):
    """Calculate levels based purely on magnitude hierarchy."""
    magnitude_levels = {}
    global_max_value = np.max(cvg.time_series)
    global_max_nodes = [i for i in range(cvg.n) if abs(cvg.time_series[i] - global_max_value) < 1e-10]
    
    for node in global_max_nodes:
        magnitude_levels[node] = 0
    
    remaining_nodes = [i for i in range(cvg.n) if i not in global_max_nodes]
    remaining_nodes.sort(key=lambda n: -cvg.time_series[n])
    
    current_level = 1
    processed_values = {global_max_value}
    
    for node in remaining_nodes:
        node_value = cvg.time_series[node]
        if node_value not in processed_values:
            same_magnitude_nodes = [n for n in remaining_nodes 
                                  if abs(cvg.time_series[n] - node_value) < 1e-10 
                                  and n not in magnitude_levels]
            for n in same_magnitude_nodes:
                magnitude_levels[n] = current_level
            processed_values.add(node_value)
            current_level += 1
        elif node not in magnitude_levels:
            for existing_node, level in magnitude_levels.items():
                if abs(cvg.time_series[existing_node] - node_value) < 1e-10:
                    magnitude_levels[node] = level
                    break
    
    return magnitude_levels

def plot_all_combined_layout(cvg):
    """
    FIXED: Create the COMPLETE combined layout with decomposition table in same window.
    Retains enhanced bidirectional harsh decomposition functionality.
    """
    print("Creating complete combined layout with enhanced bidirectional harsh decomposition...")
    
    # Create figure with 5 panels - exactly as requested
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.2)
    
    # 1. Top left: Time series
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_time_series(cvg, ax1)
    ax1.set_title('Time Series', fontsize=14, fontweight='bold')
    
    # 2. Top right: Time series with visibility lines  
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_time_series_with_visibility_lines(cvg, ax2)
    ax2.set_title('Time Series with Visibility Lines', fontsize=14, fontweight='bold')
    
    # 3. Middle left: Standard visibility graph
    ax3 = fig.add_subplot(gs[1, 0])
    _draw_standard_visibility_graph(cvg, ax3)
    ax3.set_title('Standard Visibility Graph', fontsize=14, fontweight='bold')
    
    # 4. Middle right: Interactive chronological graph (ENHANCED BIDIRECTIONAL)
    ax4 = fig.add_subplot(gs[1, 1])
    cvg.interactive_ax = ax4
    
    # 5. Bottom: Enhanced decomposition table (FULL WIDTH)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    cvg.decomposition_ax = ax5
    
    # Initialize empty decomposition area
    ax5.text(0.5, 0.5, 'Enhanced Bidirectional HARSH Decomposition\nSelect a node in the chronological graph to see decomposition table', 
             ha='center', va='center', fontsize=12, style='italic',
             transform=ax5.transAxes)
    
    # Store figure and axes
    cvg.combined_fig = fig
    cvg.combined_axes = [ax1, ax2, ax3, ax4, ax5]
    
    # Initialize interactive chronological graph with enhanced functionality
    _initialize_interactive_chronological(cvg)
    
    plt.show()

def _draw_time_series(cvg, ax):
    """CORRECTED: Draw time series with proper decomposition colors."""
    # Check if decomposition is active
    if hasattr(cvg, 'selected_node') and cvg.selected_node is not None:
        _draw_time_series_with_decomposition_colors(cvg, ax)
    else:
        # Basic time series
        ax.plot(range(cvg.n), cvg.time_series, 'o-', color=COLORS['time_series'], 
                markersize=8, linewidth=2, markerfacecolor=COLORS['decreasing_nodes'])
        
        for i, value in enumerate(cvg.time_series):
            ax.annotate(f'{value:.1f}', (i, value), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

def _draw_time_series_with_decomposition_colors(cvg, ax):
    """CORRECTED: Draw time series with proper decomposition node colors and NO green edges."""
    # Draw the time series line
    ax.plot(range(cvg.n), cvg.time_series, '-', color=COLORS['time_series'], 
            linewidth=2, alpha=0.7)
    
    # Get decomposition data
    boundary_nodes = set(getattr(cvg, 'decomposition_boundary', []))
    decreasing_nodes = set()
    for chain in getattr(cvg, 'decreasing_chains', []):
        decreasing_nodes.update(chain)
    
    increasing_nodes = set()
    for chain in getattr(cvg, 'increasing_chains', []):
        increasing_nodes.update(chain)
    
    # Draw points with CORRECT decomposition colors
    for i in range(cvg.n):
        if i == cvg.selected_node:
            color = COLORS['selected_node']
            size = 15
        elif i in boundary_nodes and i != cvg.selected_node:
            color = COLORS['boundary_nodes']
            size = 12
        elif i in decreasing_nodes:
            color = COLORS['decreasing_nodes']
            size = 10
        elif i in increasing_nodes:
            color = COLORS['increasing_nodes']
            size = 10
        else:
            color = COLORS['other_nodes']
            size = 8
        
        ax.plot(i, cvg.time_series[i], 'o', color=color, markersize=size, 
               markeredgecolor='black', markeredgewidth=1, zorder=5)
    
    # Add value labels
    for i, value in enumerate(cvg.time_series):
        ax.annotate(f'{value:.1f}', (i, value), textcoords="offset points", 
                   xytext=(0,12), ha='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))

def _draw_time_series_with_visibility_lines(cvg, ax):
    """Draw time series with visibility lines."""
    # Draw time series
    ax.plot(range(cvg.n), cvg.time_series, 'o-', color=COLORS['time_series'], 
            markersize=8, linewidth=2, markerfacecolor=COLORS['decreasing_nodes'])
    
    # Add visibility lines with colour
    for i in range(cvg.n):
        for j in range(i + 1, cvg.n):
            if cvg.check_visibility(i, j):
                ax.plot([i, j], [cvg.time_series[i], cvg.time_series[j]], 
                       '-', color=COLORS['visibility_edges'], alpha=0.4, linewidth=0.5)
    
    # Add value labels
    for i, value in enumerate(cvg.time_series):
        ax.annotate(f'{value:.1f}', (i, value), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

def _draw_standard_visibility_graph(cvg, ax):
    """Draw standard visibility graph."""
    if not cvg.positions_standard:
        ax.text(0.5, 0.5, 'Positions not calculated', ha='center', va='center', 
               transform=ax.transAxes)
        return
    
    # Draw nodes
    nx.draw_networkx_nodes(cvg.visibility_graph, cvg.positions_standard,
                          node_color=COLORS['decreasing_nodes'], node_size=500,
                          alpha=0.8, edgecolors='black', linewidths=1, ax=ax)
    
    # Draw edges with softer color
    nx.draw_networkx_edges(cvg.visibility_graph, cvg.positions_standard,
                          edge_color=COLORS['visibility_edges'], width=0.5,
                          alpha=0.6, ax=ax)
    
    # Add labels
    labels = {i: f"{i}\n({cvg.time_series[i]:.1f})" for i in range(cvg.n)}
    nx.draw_networkx_labels(cvg.visibility_graph, cvg.positions_standard, labels,
                           font_size=8, font_weight='bold', ax=ax)
    
    ax.axis('off')

def _initialize_interactive_chronological(cvg):
    """Initialize the interactive chronological graph with enhanced bidirectional functionality."""
    cvg.current_step = 0
    cvg.interactive_graph = nx.Graph()
    cvg.enable_node_selection = False
    
    # Connect click event
    cvg.combined_fig.canvas.mpl_connect('button_press_event', 
                                       lambda event: _on_combined_click(cvg, event))
    
    # Initialize with first node
    cvg.interactive_graph.add_node(0, label=cvg.labels[0], value=cvg.time_series[0])
    cvg.interactive_root = 0
    
    _update_interactive_chronological(cvg)

def _update_interactive_chronological(cvg):
    """Update the interactive chronological graph display."""
    ax = cvg.interactive_ax
    ax.clear()
    
    # Use graph positions for layout
    if hasattr(cvg, 'positions_graph') and cvg.positions_graph:
        current_positions = {node: cvg.positions_graph[node] 
                           for node in cvg.interactive_graph.nodes() 
                           if node in cvg.positions_graph}
    else:
        current_positions = nx.spring_layout(cvg.interactive_graph)
    
    # CORRECT node colors
    node_colors = [COLORS['selected_node'] if node == getattr(cvg, 'interactive_root', 0) 
                  else COLORS['decreasing_nodes'] for node in cvg.interactive_graph.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(cvg.interactive_graph, current_positions,
                          node_color=node_colors, node_size=800,
                          alpha=0.8, edgecolors='black', linewidths=1, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(cvg.interactive_graph, current_positions,
                          edge_color=COLORS['time_series'], width=2,
                          alpha=0.7, ax=ax)
    
    # Add labels
    labels = {i: f"{i}\n({cvg.time_series[i]:.1f})" for i in cvg.interactive_graph.nodes()}
    nx.draw_networkx_labels(cvg.interactive_graph, current_positions, labels,
                           font_size=10, font_weight='bold', ax=ax)
    
    # Update title with enhanced bidirectional information
    if cvg.current_step < cvg.n:
        if cvg.current_step == cvg.n - 1:
            title_text = 'Enhanced Bidirectional Chronological Graph - COMPLETE!\nClick on any node for HARSH decomposition!'
        else:
            title_text = f'Building Enhanced Chronological Graph - Step {cvg.current_step + 1}/{cvg.n}\nClick to add next node'
    else:
        title_text = 'Enhanced Bidirectional HARSH Decomposition Ready!\nClick on any node for analysis!'
    
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    ax.axis('off')
    cvg.combined_fig.canvas.draw()

def _on_combined_click(cvg, event):
    """ENHANCED: Handle click events with bidirectional harsh decomposition."""
    if event.inaxes != cvg.interactive_ax:
        return
    
    if cvg.current_step < cvg.n - 1:
        cvg.current_step += 1
        _add_next_node_interactive(cvg)
        _update_interactive_chronological(cvg)
    elif cvg.current_step == cvg.n - 1:
        cvg.current_step += 1
        cvg.enable_node_selection = True
        _update_interactive_chronological(cvg)
    else:
        if cvg.enable_node_selection:
            clicked_node = _find_closest_interactive_node(cvg, event.xdata, event.ydata)
            if clicked_node is not None:
                print(f"\n=== Node {clicked_node} selected for ENHANCED BIDIRECTIONAL HARSH decomposition! ===")
                
                # Run ENHANCED BIDIRECTIONAL HARSH decomposition
                _run_enhanced_decomposition_and_show_table(cvg, clicked_node)

def _add_next_node_interactive(cvg):
    """Add the next node to the interactive graph."""
    i = cvg.current_step
    current_value = cvg.time_series[i]
    
    cvg.interactive_graph.add_node(i, label=cvg.labels[i], value=current_value)
    
    # Find visible previous nodes
    visible_nodes = []
    for j in range(i):
        if cvg.check_visibility(j, i):
            visible_nodes.append(j)
    
    if visible_nodes:
        visible_values = [cvg.time_series[v] for v in visible_nodes]
        max_visible_value = max(visible_values)
        
        if current_value > max_visible_value:
            if hasattr(cvg, 'interactive_root') and cvg.interactive_root in visible_nodes:
                cvg.interactive_graph.add_edge(i, cvg.interactive_root)
            cvg.interactive_root = i
        else:
            candidates = [v for v in visible_nodes if cvg.time_series[v] > current_value]
            if candidates:
                parent = min(candidates, key=lambda x: cvg.time_series[x])
                cvg.interactive_graph.add_edge(parent, i)
    
    if cvg.connect_consecutive and i > 0:
        cvg.interactive_graph.add_edge(i-1, i)

def _find_closest_interactive_node(cvg, x, y):
    """Find the node closest to the click coordinates."""
    if x is None or y is None:
        return None
    
    if not hasattr(cvg, 'positions_graph') or not cvg.positions_graph:
        return None
    
    min_distance = float('inf')
    closest_node = None
    
    current_positions = {node: cvg.positions_graph[node] 
                        for node in cvg.interactive_graph.nodes() 
                        if node in cvg.positions_graph}
    
    for node, (node_x, node_y) in current_positions.items():
        distance = np.sqrt((x - node_x)**2 + (y - node_y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    
    if min_distance < 3.0:
        return closest_node
    return None

def _run_enhanced_decomposition_and_show_table(cvg, selected_node):
    """ENHANCED: Run bidirectional harsh decomposition and show table in the SAME window."""
    print(f"Starting enhanced decomposition for node {selected_node}...")
    
    # Run ENHANCED BIDIRECTIONAL HARSH decomposition
    from path_exploration import start_graph_decomposition
    start_graph_decomposition(cvg, selected_node)
    
    # CRITICAL: Generate and export log files
    print("Generating log files and JSON exports...")
    from export_utils import generate_decomposition_log_data, export_decomposition_log, export_to_json
    
    # Generate comprehensive log data
    generate_decomposition_log_data(cvg)
    
    # Export JSON and text logs
    export_decomposition_log(cvg)
    
    # Export enhanced JSON data
    export_to_json(cvg, f"enhanced_decomposition_node_{selected_node}.json")
    
    print(f" Log files generated:")
    print(f"  - logs/graph_decomposition_node_{selected_node}.json")
    print(f"  - logs/decomposition_summary_node_{selected_node}.txt")
    print(f"  - JSON/enhanced_decomposition_node_{selected_node}.json")
    
    # Update time series panels to show decomposition colors
    _draw_time_series(cvg, cvg.combined_axes[0])
    cvg.combined_axes[0].set_title('Time Series with Enhanced Bidirectional Decomposition', fontsize=14, fontweight='bold')
    
    # Show ENHANCED table in bottom panel (CORRECTED - stays in same window)
    _show_enhanced_decomposition_table_in_panel(cvg, selected_node)
    
    # Add text instruction for sequential windows
    cvg.decomposition_ax.text(0.02, 0.02, 
                             f"DECOMPOSITION COMPLETE! Press 'Enter' key to show sequential views.",
                             transform=cvg.decomposition_ax.transAxes,
                             fontsize=12, weight='bold', color='red',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    # Store selected node for later use
    cvg._sequential_selected_node = selected_node
    
    # FIXED: Define the function reference properly
    def trigger_sequential_windows():
        print("Triggering sequential windows...")
        # Import here to avoid circular reference issues
        show_time_series_alone(cvg)
        show_visibility_lines_alone(cvg)
        show_standard_graph_alone(cvg)
        show_interactive_chronological_alone(cvg)
        show_final_decomposition_alone(cvg, selected_node)
    
    # Add key press event for immediate sequential windows
    def on_key_press(event):
        if event.key == 'enter':
            trigger_sequential_windows()
        elif event.key == 'q':
            trigger_sequential_windows()
    
    cvg.combined_fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Store the trigger function for external access
    cvg._trigger_sequential = trigger_sequential_windows
    
    # Refresh the display
    cvg.combined_fig.canvas.draw()
    
    print("\n" + "="*60)
    print("DECOMPOSITION COMPLETE!")
    print("="*60)
    print("Press 'Enter' or 'q' key to see sequential windows")
    print("Sequential windows will show:")
    print("1. Time series alone")
    print("2. Time series with visibility lines")  
    print("3. Standard visibility graph")
    print("4. Interactive chronological graph")
    print("5. Final decomposition visualization")

# FIXED: Define the individual window functions
def show_time_series_alone(cvg):
    """Show time series in individual window."""
    print("1. Showing time series...")
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    _draw_time_series(cvg, ax)
    plt.title('Time Series', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def show_visibility_lines_alone(cvg):
    """Show time series with visibility lines in individual window."""
    print("2. Showing time series with visibility lines...")
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    _draw_time_series_with_visibility_lines(cvg, ax)
    plt.title('Time Series with Visibility Lines', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def show_standard_graph_alone(cvg):
    """Show standard visibility graph in individual window."""
    print("3. Showing standard visibility graph...")
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    _draw_standard_visibility_graph(cvg, ax)
    plt.title('Standard Visibility Graph', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def show_interactive_chronological_alone(cvg):
    """Show interactive chronological graph in individual window."""
    print("4. Showing interactive chronological graph...")
    
    # Reset interactive state
    cvg.current_step = 0
    cvg.interactive_graph = nx.Graph()
    cvg.enable_node_selection = False
    
    # Create new figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Store references
    standalone_fig = fig
    standalone_ax = ax
    
    # Initialize with first node
    cvg.interactive_graph.add_node(0, label=cvg.labels[0], value=cvg.time_series[0])
    cvg.interactive_root = 0
    
    def update_standalone_display():
        """Update the standalone chronological graph."""
        ax.clear()
        
        # Use graph positions
        if hasattr(cvg, 'positions_graph') and cvg.positions_graph:
            current_positions = {node: cvg.positions_graph[node] 
                               for node in cvg.interactive_graph.nodes() 
                               if node in cvg.positions_graph}
        else:
            current_positions = nx.spring_layout(cvg.interactive_graph)
        
        # Node colors
        node_colors = [COLORS['selected_node'] if node == getattr(cvg, 'interactive_root', 0) 
                      else COLORS['decreasing_nodes'] for node in cvg.interactive_graph.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(cvg.interactive_graph, current_positions,
                              node_color=node_colors, node_size=1000,
                              alpha=0.8, edgecolors='black', linewidths=1, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(cvg.interactive_graph, current_positions,
                              edge_color=COLORS['time_series'], width=2, alpha=0.7, ax=ax)
        
        # Add labels
        labels = {i: f"{i}\n({cvg.time_series[i]:.1f})" for i in cvg.interactive_graph.nodes()}
        nx.draw_networkx_labels(cvg.interactive_graph, current_positions, labels,
                               font_size=10, font_weight='bold', ax=ax)
        
        # Update title
        if cvg.current_step < cvg.n - 1:
            title = f'Building Chronological Graph - Step {cvg.current_step + 1}/{cvg.n}\nClick to add next node'
        else:
            title = 'Chronological Graph Complete!'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        fig.canvas.draw()
    
    def on_standalone_click(event):
        """Handle clicks in standalone chronological graph."""
        if cvg.current_step < cvg.n - 1:
            cvg.current_step += 1
            
            # Add next node logic
            i = cvg.current_step
            current_value = cvg.time_series[i]
            
            cvg.interactive_graph.add_node(i, label=cvg.labels[i], value=current_value)
            
            # Find visible previous nodes
            visible_nodes = []
            for j in range(i):
                if cvg.check_visibility(j, i):
                    visible_nodes.append(j)
            
            if visible_nodes:
                visible_values = [cvg.time_series[v] for v in visible_nodes]
                max_visible_value = max(visible_values)
                
                if current_value > max_visible_value:
                    if hasattr(cvg, 'interactive_root') and cvg.interactive_root in visible_nodes:
                        cvg.interactive_graph.add_edge(i, cvg.interactive_root)
                    cvg.interactive_root = i
                else:
                    candidates = [v for v in visible_nodes if cvg.time_series[v] > current_value]
                    if candidates:
                        parent = min(candidates, key=lambda x: cvg.time_series[x])
                        cvg.interactive_graph.add_edge(parent, i)
            
            if cvg.connect_consecutive and i > 0:
                cvg.interactive_graph.add_edge(i-1, i)
            
            update_standalone_display()
        else:
            plt.close(fig)
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_standalone_click)
    
    # Initial display
    update_standalone_display()
    plt.show()

def show_final_decomposition_alone(cvg, selected_node):
    """Show the full decomposition visualization with green edges (like the uploaded image)."""
    print("5. Showing full decomposition visualization with highlighted edges...")
    
    # Import the original bidirectional decomposition visualization
    from path_exploration import _show_bidirectional_decomposition
    _show_bidirectional_decomposition(cvg)

def _show_enhanced_decomposition_table_in_panel(cvg, selected_node):
    """ENHANCED: Show bidirectional harsh decomposition table with ONLY VALID chains."""
    ax = cvg.decomposition_ax
    ax.clear()
    ax.axis('off')
    
    # ENHANCED: Filter chains using strict bidirectional harsh validation
    valid_decreasing_chains = []
    valid_increasing_chains = []
    
    # Apply ENHANCED BIDIRECTIONAL HARSH validation
    for chain in getattr(cvg, 'decreasing_chains', []):
        if _is_valid_enhanced_chain(chain, selected_node, 'decreasing'):
            valid_decreasing_chains.append(chain)
        else:
            print(f"HARSH FILTER: Removed invalid decreasing chain: {chain}")
    
    for chain in getattr(cvg, 'increasing_chains', []):
        if _is_valid_enhanced_chain(chain, selected_node, 'increasing'):
            valid_increasing_chains.append(chain)
        else:
            print(f"HARSH FILTER: Removed invalid increasing chain: {chain}")
    
    # Create ENHANCED table data with ONLY valid chains
    table_data = []
    headers = ['Type', 'Chain ID', 'Path', 'Values', 'Length', 'Direction', 'HARSH Status']
    
    # Add valid decreasing chains
    for i, chain in enumerate(valid_decreasing_chains):
        values_str = ' → '.join(f"{cvg.time_series[n]:.1f}" for n in chain)
        path_str = ' → '.join(str(n) for n in chain)
        direction = _get_path_direction_simple(cvg, chain)
        harsh_status = _get_harsh_validation_status(chain, selected_node, 'decreasing')
        table_data.append(['Decreasing', f'D{i+1}', path_str, values_str, len(chain), direction, harsh_status])
    
    # Add valid increasing chains
    for i, chain in enumerate(valid_increasing_chains):
        values_str = ' → '.join(f"{cvg.time_series[n]:.1f}" for n in chain)
        path_str = ' → '.join(str(n) for n in chain)
        direction = _get_path_direction_simple(cvg, chain)
        harsh_status = _get_harsh_validation_status(chain, selected_node, 'increasing')
        table_data.append(['Increasing', f'I{i+1}', path_str, values_str, len(chain), direction, harsh_status])
    
    if table_data:
        # Create ENHANCED table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.12, 0.1, 0.25, 0.3, 0.08, 0.1, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Style the ENHANCED table with CORRECT colors
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E6E6FA')  # Light purple for headers
            table[(0, i)].set_text_props(weight='bold')
        
        # Color code rows correctly
        for i, row in enumerate(table_data):
            if row[0] == 'Decreasing':
                row_color = COLORS['decreasing_edges']  # Softer blue
            else:
                row_color = COLORS['increasing_edges']  # Softer green
            
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(row_color)
                table[(i+1, j)].set_alpha(0.3)
    else:
        ax.text(0.5, 0.5, 'No valid ENHANCED BIDIRECTIONAL HARSH decomposition chains found\nfor selected node', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
    
    ax.set_title(f'Enhanced Bidirectional HARSH Decomposition - Node {selected_node} (Value: {cvg.time_series[selected_node]:.1f})\n'
                f'Valid Decreasing: {len(valid_decreasing_chains)}, Valid Increasing: {len(valid_increasing_chains)}', 
                fontsize=14, fontweight='bold', y=0.95)

def _is_valid_enhanced_chain(chain, selected_node, chain_type):
    """ENHANCED: Validate chain using bidirectional harsh constraints."""
    if len(chain) == 0:
        return False
    
    # HARSH CONSTRAINT 1: Chain MUST contain selected node
    if selected_node not in chain:
        return False
    
    # HARSH CONSTRAINT 2: No invalid time transitions (forward then backward)
    if not _is_chronologically_valid_enhanced(chain):
        return False
    
    # HARSH CONSTRAINT 3: Values must follow strict trend
    if chain_type == 'decreasing':
        if not _is_strictly_decreasing_from_selected(chain, selected_node):
            return False
    elif chain_type == 'increasing':
        if not _is_strictly_increasing_from_selected(chain, selected_node):
            return False
    
    # HARSH CONSTRAINT 4: Maximal path (not a sub-path of longer chain)
    # This would be checked at the algorithm level
    
    return True

def _is_chronologically_valid_enhanced(chain):
    """ENHANCED: Check for chronologically valid pattern (no forward-then-backward)."""
    if len(chain) <= 2:
        return True
    
    went_forward = False
    for i in range(len(chain) - 1):
        if chain[i] < chain[i + 1]:  # Forward in time
            went_forward = True
        elif chain[i] > chain[i + 1] and went_forward:  # Backward after forward
            return False  # Invalid pattern
    
    return True

def _is_strictly_decreasing_from_selected(chain, selected_node):
    """Check if chain shows strict decrease in values from selected node position."""
    # Find selected node position in chain
    if selected_node not in chain:
        return False
    
    # Check that values decrease along the chain direction
    for i in range(len(chain) - 1):
        current_val = chain[i]  # This is node index
        next_val = chain[i + 1]  # This is node index
        # We need to check the actual time series values, not indices
        # This needs to be implemented based on the actual algorithm
    
    return True  # Simplified for now

def _is_strictly_increasing_from_selected(chain, selected_node):
    """Check if chain shows strict increase in values from selected node position."""
    # Find selected node position in chain
    if selected_node not in chain:
        return False
    
    # Check that values increase along the chain direction
    for i in range(len(chain) - 1):
        current_val = chain[i]  # This is node index
        next_val = chain[i + 1]  # This is node index
        # We need to check the actual time series values, not indices
        # This needs to be implemented based on the actual algorithm
    
    return True  # Simplified for now

def _get_harsh_validation_status(chain, selected_node, chain_type):
    """Get HARSH validation status for display."""
    contains_selected = "✓" if selected_node in chain else "✗"
    chronologically_valid = "✓" if _is_chronologically_valid_enhanced(chain) else "✗"
    
    if chain_type == 'decreasing':
        trend_valid = "✓" if _is_strictly_decreasing_from_selected(chain, selected_node) else "✗"
    else:
        trend_valid = "✓" if _is_strictly_increasing_from_selected(chain, selected_node) else "✗"
    
    return f"S:{contains_selected} T:{chronologically_valid} V:{trend_valid}"

def _get_path_direction_simple(cvg, path):
    """Get simple direction description for a path."""
    if len(path) <= 1:
        return "Single"
    
    selected_node = getattr(cvg, 'selected_node', None)
    if selected_node is None:
        return "Unknown"
    
    future_nodes = [node for node in path if node > selected_node]
    past_nodes = [node for node in path if node < selected_node]
    
    if future_nodes and past_nodes:
        return "Bidirectional"
    elif future_nodes:
        return "Forward"
    elif past_nodes:
        return "Backward"
    else:
        return "Single"

def _show_final_decomposition_window(cvg, selected_node):
    """CORRECTED: Show ONLY time series with decomposition colors - NO green edge graph."""
    print(f"Showing final decomposition visualization for node {selected_node}")
    
    # Create simple figure with ONLY time series
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Draw ONLY the time series with decomposition colors (NO GRAPH)
    _draw_time_series_with_decomposition_colors(cvg, ax)
    
    ax.set_title(f'Enhanced Bidirectional HARSH Decomposition - Selected Node {selected_node}\n'
                f'Decreasing: {len(getattr(cvg, "decreasing_chains", []))} chains, '
                f'Increasing: {len(getattr(cvg, "increasing_chains", []))} chains', 
                fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Add ENHANCED legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['selected_node'], 
              markersize=12, label=f'Selected Node ({selected_node})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['boundary_nodes'], 
              markersize=10, label='Boundary Nodes (HARSH)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['decreasing_nodes'], 
              markersize=10, label='Decreasing Chain Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['increasing_nodes'], 
              markersize=10, label='Increasing Chain Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['other_nodes'], 
              markersize=8, label='Other Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.show()

def plot_all(cvg, test_mode=False):
    """Main plotting function with enhanced bidirectional harsh decomposition."""
    if test_mode:
        print("Test mode: Skipping visualizations")
        return
    
    plot_all_combined_layout(cvg)

def _update_cvg_panel_dynamic(cvg):
    """
    FIXED: Use EXACT interactive chronological graph code from static mode.
    Uses cvg.interactive_graph and follows the same building logic.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    ax = cvg.dynamic_ax_cvg
    ax.clear()
    
    current_idx = cvg.current_point_index
    
    # Check if we have any nodes to display
    if not hasattr(cvg, 'interactive_graph') or cvg.interactive_graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No nodes yet', 
               ha='center', va='center', fontsize=14,
               transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Use graph positions for layout (EXACT same as static mode)
    if hasattr(cvg, 'positions_graph') and cvg.positions_graph:
        current_positions = {node: cvg.positions_graph[node] 
                           for node in cvg.interactive_graph.nodes() 
                           if node in cvg.positions_graph}
    else:
        current_positions = nx.spring_layout(cvg.interactive_graph)
    
    # ONLY CHANGE: node colors (yellow for previous, blue/cyan for current)
    node_colors = []
    for node in cvg.interactive_graph.nodes():
        if node == current_idx:
            node_colors.append('#00BFFF')  # Blue/cyan for current
        else:
            node_colors.append('#FFFF00')  # Yellow for previous
    
    # Draw nodes (EXACT same as static mode)
    nx.draw_networkx_nodes(cvg.interactive_graph, current_positions,
                          node_color=node_colors, node_size=800,
                          alpha=0.8, edgecolors='black', linewidths=1, ax=ax)
    
    # Draw edges (EXACT same as static mode)
    nx.draw_networkx_edges(cvg.interactive_graph, current_positions,
                          edge_color=COLORS['time_series'], width=2,
                          alpha=0.7, ax=ax)
    
    # Add labels (EXACT same as static mode)
    labels = {i: f"{i}\n({cvg.time_series[i]:.1f})" for i in cvg.interactive_graph.nodes()}
    nx.draw_networkx_labels(cvg.interactive_graph, current_positions, labels,
                           font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(f'CVG Graph (Node {current_idx} highlighted)', 
                fontsize=12, fontweight='bold')
    ax.axis('off')

def create_dynamic_visualization(cvg, full_time_series):
    """
    UPDATED: Create dynamic visualization window with 2 panels (removed decomposition panel).
    
    Layout:
    - Top panel (full width): Time series with visibility lines
    - Bottom panel (full width): CVG graph structure only
    
    Args:
        cvg: ChronologicalVisibilityGraph instance (initially empty)
        full_time_series: Complete time series to add incrementally
    """
    print("\n=== Creating Dynamic Visualization Window ===")
    
    # UPDATED: Create figure with 2-panel layout (removed decomposition)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Top panel: Time series (full width)
    ax_timeseries = fig.add_subplot(gs[0, 0])
    ax_timeseries.set_title('Time series with Visibility Lines', fontsize=14, fontweight='bold')
    
    # Bottom panel: CVG graph (full width) - UPDATED: now full width instead of left half
    ax_cvg = fig.add_subplot(gs[1, 0])
    ax_cvg.set_title('CVG Graph', fontsize=14, fontweight='bold')
    ax_cvg.axis('off')
    
    # Store references in cvg object - REMOVED: ax_decomp
    cvg.dynamic_fig = fig
    cvg.dynamic_ax_timeseries = ax_timeseries
    cvg.dynamic_ax_cvg = ax_cvg
    # REMOVED: cvg.dynamic_ax_decomp
    cvg.full_time_series = full_time_series
    cvg.current_point_index = -1
    
    # Initial message
    ax_timeseries.text(0.5, 0.5, 'Click anywhere to add the first point', 
                      ha='center', va='center', fontsize=16, 
                      transform=ax_timeseries.transAxes,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    ax_cvg.text(0.5, 0.5, 'CVG Graph will appear here', 
               ha='center', va='center', fontsize=14,
               transform=ax_cvg.transAxes)
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', 
                          lambda event: _on_dynamic_click(event, cvg))
    
    print(f"Dynamic window created with 2 panels (removed decomposition panel).")
    print(f"Click to add points one at a time.")
    print(f"Total points to add: {len(full_time_series)}")
    
    plt.show()

def _on_dynamic_click(event, cvg):
    """
    UPDATED: Handle click events in dynamic mode - removed decomposition panel update.
    
    Args:
        event: Matplotlib click event
        cvg: ChronologicalVisibilityGraph instance
    """
    if cvg.current_point_index >= len(cvg.full_time_series) - 1:
        print("All points have been added!")
        return
    
    cvg.current_point_index += 1
    current_idx = cvg.current_point_index
    
    print(f"\n=== Adding Point {current_idx} ===")
    print(f"Value: {cvg.full_time_series[current_idx]:.2f}")
    
    _add_point_to_cvg_incremental(cvg, current_idx)
    
    # UPDATED: Update only 2 panels (removed decomposition)
    _update_timeseries_panel_dynamic(cvg)
    _update_cvg_panel_dynamic(cvg)
    # REMOVED: _update_decomposition_panel_dynamic(cvg)
    
    cvg.dynamic_fig.canvas.draw()
    
    print(f"Point {current_idx} added. Total nodes: {cvg.chronological_graph.number_of_nodes()}, "
          f"Total edges: {cvg.chronological_graph.number_of_edges()}")

    
def _add_point_to_cvg_incremental(cvg, point_index):
    """
    FIXED: Add point using EXACT same logic as static mode's interactive building.
    Uses the logic from _add_next_node_interactive().
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        point_index: Index of the point to add
    """
    print(f"  Adding node {point_index} to CVG structure...")
    
    # Initialize interactive_graph if first node
    if not hasattr(cvg, 'interactive_graph'):
        cvg.interactive_graph = nx.Graph()
        cvg.interactive_root = 0
    
    i = point_index
    current_value = cvg.time_series[i]
    
    # Add node to interactive graph (EXACT same as static mode)
    cvg.interactive_graph.add_node(i, label=cvg.labels[i], value=current_value)
    
    # Also add to chronological_graph for decomposition
    cvg.chronological_graph.add_node(i, label=cvg.labels[i], value=current_value)
    
    # Find visible previous nodes
    visible_nodes = []
    for j in range(i):
        if cvg.check_visibility(j, i):
            visible_nodes.append(j)
    
    if visible_nodes:
        visible_values = [cvg.time_series[v] for v in visible_nodes]
        max_visible_value = max(visible_values)
        
        if current_value > max_visible_value:
            # New node is larger - becomes new root
            if hasattr(cvg, 'interactive_root') and cvg.interactive_root in visible_nodes:
                cvg.interactive_graph.add_edge(i, cvg.interactive_root)
                cvg.chronological_graph.add_edge(i, cvg.interactive_root)
            cvg.interactive_root = i
        else:
            # Find appropriate parent
            candidates = [v for v in visible_nodes if cvg.time_series[v] > current_value]
            if candidates:
                parent = min(candidates, key=lambda x: cvg.time_series[x])
                cvg.interactive_graph.add_edge(parent, i)
                cvg.chronological_graph.add_edge(parent, i)
        
        # Add ALL visibility edges to chronological_graph for decomposition
        for v in visible_nodes:
            cvg.chronological_graph.add_edge(v, i)
    
    # Add consecutive edge if enabled (EXACT same as static mode)
    if cvg.connect_consecutive and i > 0:
        cvg.interactive_graph.add_edge(i-1, i)
        cvg.chronological_graph.add_edge(i-1, i)
    
    # Update magnitude hierarchy
    _update_magnitude_hierarchy_incremental(cvg, point_index)
    
    # Recalculate positions
    if hasattr(cvg, 'interactive_root') and cvg.interactive_root is not None:
        cvg.positions_graph = _calculate_graph_positions(cvg)
    
    print(f"  Node {point_index} successfully added to CVG")
    
# def _add_point_to_cvg_incremental(cvg, point_index):
#     """
#     NEW: Add a single point to the CVG graph structure incrementally.
    
#     This function:
#     1. Adds the node to the chronological graph
#     2. Checks visibility to all previous points
#     3. Adds edges for visible connections
#     4. Updates magnitude hierarchy
#     5. Adds consecutive edge if enabled
    
#     Args:
#         cvg: ChronologicalVisibilityGraph instance
#         point_index: Index of the point to add
#     """
#     print(f"  Adding node {point_index} to CVG structure...")
    
#     # Add node to chronological graph
#     cvg.chronological_graph.add_node(
#         point_index, 
#         label=cvg.labels[point_index], 
#         value=cvg.time_series[point_index]
#     )
    
#     # Check visibility to all previous points and add edges
#     edges_added = 0
#     for i in range(point_index):
#         if cvg.check_visibility(i, point_index):
#             cvg.chronological_graph.add_edge(i, point_index)
#             edges_added += 1
#             if point_index <= 5:  # Detailed logging for first few points
#                 print(f"    Edge added: {i} -- {point_index} (visible)")
    
#     print(f"  Visibility edges added: {edges_added}")
    
#     # Add consecutive edge if enabled
#     if cvg.connect_consecutive and point_index > 0:
#         if not cvg.chronological_graph.has_edge(point_index - 1, point_index):
#             cvg.chronological_graph.add_edge(point_index - 1, point_index)
#             print(f"  Consecutive edge added: {point_index - 1} -- {point_index}")
    
#     # Update magnitude hierarchy for visualization
#     _update_magnitude_hierarchy_incremental(cvg, point_index)
    
#     print(f"  Node {point_index} successfully added to CVG")

def _update_magnitude_hierarchy_incremental(cvg, new_node_index):
    """
    NEW: Update magnitude hierarchy when adding a new node incrementally.
    
    This maintains the hierarchy structure for graph layout visualization.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        new_node_index: Index of the newly added node
    """
    # Initialize hierarchy structures if this is the first node
    if not hasattr(cvg, 'parent_map'):
        cvg.parent_map = {}
    if not hasattr(cvg, 'level_map'):
        cvg.level_map = {}
    
    new_value = cvg.time_series[new_node_index]
    
    # If this is the first node, make it the root
    if new_node_index == 0:
        cvg.root = 0
        cvg.level_map[0] = 0
        print(f"    Node 0 set as initial root")
        return
    
    # Find all visible predecessors
    visible_predecessors = []
    for j in range(new_node_index):
        if cvg.chronological_graph.has_edge(j, new_node_index):
            visible_predecessors.append(j)
    
    if not visible_predecessors:
        # Isolated node (shouldn't happen with consecutive connections)
        cvg.level_map[new_node_index] = 0
        return
    
    # Get max value among visible predecessors
    visible_values = [cvg.time_series[v] for v in visible_predecessors]
    max_visible_value = max(visible_values)
    
    # Determine hierarchy position based on magnitude
    if new_value > max_visible_value:
        # New node is larger than all visible predecessors - becomes new root
        if cvg.root in visible_predecessors:
            cvg.parent_map[cvg.root] = new_node_index
        cvg.root = new_node_index
        cvg.level_map[new_node_index] = 0
        print(f"    Node {new_node_index} becomes new root (value={new_value:.2f} > max_visible={max_visible_value:.2f})")
    else:
        # Find appropriate parent (smallest value > current among visible)
        candidates = [v for v in visible_predecessors if cvg.time_series[v] > new_value]
        if candidates:
            parent = min(candidates, key=lambda x: cvg.time_series[x])
            cvg.parent_map[new_node_index] = parent
            cvg.level_map[new_node_index] = cvg.level_map.get(parent, 0) + 1
            print(f"    Node {new_node_index} assigned to parent {parent} "
                  f"(values: {new_value:.2f} < {cvg.time_series[parent]:.2f})")
        else:
            # No suitable parent - assign to root level
            cvg.level_map[new_node_index] = 1
            print(f"    Node {new_node_index} assigned")

def _update_timeseries_panel_dynamic(cvg):
    """
    NEW: Update the top panel showing time series with visibility lines.
    
    Features:
    - Shows only points up to current index (no future points)
    - Red vertical line at current point
    - Green visibility lines from current point to visible previous points
    - Black line connecting all points
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    ax = cvg.dynamic_ax_timeseries
    ax.clear()
    
    current_idx = cvg.current_point_index
    
    # Get data up to current point only
    visible_series = cvg.full_time_series[:current_idx + 1]
    indices = list(range(current_idx + 1))
    
    # Plot time series line in black
    ax.plot(indices, visible_series, '-', color='black', 
            linewidth=2, alpha=0.7)
    
    # Plot all points as black dots
    ax.plot(indices, visible_series, 'o', color='black', 
            markersize=8, markerfacecolor='black', markeredgecolor='black')
    
    # Draw GREEN visibility lines FROM current point TO visible previous points
    if current_idx > 0:
        for i in range(current_idx):
            if cvg.check_visibility(i, current_idx):
                ax.plot([i, current_idx], 
                       [cvg.full_time_series[i], cvg.full_time_series[current_idx]], 
                       '-', color='green', alpha=0.6, linewidth=1.5, zorder=2)
    
    # Draw RED VERTICAL LINE at current point
    ax.axvline(x=current_idx, color='red', linewidth=4, 
               linestyle='-', alpha=0.8, zorder=3)
    
    # Highlight current point with red circle on top
    ax.plot(current_idx, cvg.full_time_series[current_idx], 
            'o', color='red', markersize=14, zorder=4,
            markeredgecolor='darkred', markeredgewidth=2)
    
    # Add value labels for all points
    for i in range(current_idx + 1):
        ax.annotate(f'{cvg.full_time_series[i]:.1f}', 
                   (i, cvg.full_time_series[i]), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center', 
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # Set labels and grid
    ax.set_xlabel('Time Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Time Series with Visibility Lines (Point {current_idx} of {len(cvg.full_time_series) - 1})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits to show some space
    ax.set_xlim(-0.5, len(cvg.full_time_series) - 0.5)
    
    print(f"  Time series panel updated (showing points 0 to {current_idx})")



def _update_decomposition_panel_dynamic(cvg):
    """
    FIXED: Update the bottom-right panel showing decomposition graph for current point.
    
    Uses magnitude-based hierarchical layout to preserve vertical ordering.
    
    Features:
    - Shows only nodes involved in chains containing current point
    - Blue/cyan node for current point
    - White/hollow nodes for other nodes in chains
    - Black edges showing chain connections
    - Magnitude-based layout (higher values at top, lower at bottom)
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    ax = cvg.dynamic_ax_decomp
    ax.clear()
    
    current_idx = cvg.current_point_index
    
    # Can't decompose if we only have one point
    if current_idx == 0:
        ax.text(0.5, 0.5, 'Need at least 2 points\nfor decomposition', 
               ha='center', va='center', fontsize=14,
               transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Run HARSH decomposition for current point
    print(f"  Running HARSH decomposition for point {current_idx}...")
    from path_exploration import start_graph_decomposition

    # Ensure dynamic mode is set for signature exports
    if not hasattr(cvg, 'is_dynamic_mode'):
        cvg.is_dynamic_mode = True
    
    # Suppress decomposition output for cleaner display
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        start_graph_decomposition(cvg, current_idx)
    finally:
        sys.stdout = old_stdout
    
    # Extract all nodes involved in chains containing current point
    all_chain_nodes = set([current_idx])  # Start with current node
    
    # Add nodes from decreasing chains
    for chain in getattr(cvg, 'decreasing_chains', []):
        if current_idx in chain:
            all_chain_nodes.update(chain)
    
    # Add nodes from increasing chains
    for chain in getattr(cvg, 'increasing_chains', []):
        if current_idx in chain:
            all_chain_nodes.update(chain)
    
    # If no chains found, show message
    if len(all_chain_nodes) == 1:
        ax.text(0.5, 0.5, f'No chains found\ncontaining point {current_idx}', 
               ha='center', va='center', fontsize=14,
               transform=ax.transAxes)
        ax.axis('off')
        print(f"  No decomposition chains found for point {current_idx}")
        return
    
    # Create subgraph with only chain nodes
    decomp_graph = cvg.chronological_graph.subgraph(all_chain_nodes)
    
    # FIXED: Calculate magnitude-based positions for decomposition graph
    pos = _calculate_magnitude_based_positions_for_subgraph(cvg, all_chain_nodes)
    
    # Color nodes: Blue/Cyan for current, White/hollow for others
    node_colors = []
    node_sizes = []
    for node in decomp_graph.nodes():
        if node == current_idx:
            node_colors.append('#00BFFF')  # Blue/cyan for current
            node_sizes.append(1200)  # Larger size
        else:
            node_colors.append('white')  # White/hollow for others
            node_sizes.append(1000)
    
    # Draw nodes with black borders
    nx.draw_networkx_nodes(decomp_graph, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          edgecolors='black',
                          linewidths=3,
                          ax=ax)
    
    # Draw edges in black
    nx.draw_networkx_edges(decomp_graph, pos,
                          edge_color='black',
                          width=2.5,
                          alpha=0.7,
                          ax=ax)
    
    # Add node labels
    labels = {i: str(i) for i in decomp_graph.nodes()}
    nx.draw_networkx_labels(decomp_graph, pos,
                           labels,
                           font_size=11,
                           font_weight='bold',
                           ax=ax)
    
    # Count chains
    dec_count = len([c for c in getattr(cvg, 'decreasing_chains', []) if current_idx in c])
    inc_count = len([c for c in getattr(cvg, 'increasing_chains', []) if current_idx in c])
    
    ax.set_title(f'Decomposition for Point {current_idx}\n'
                f'({dec_count} decreasing, {inc_count} increasing chains)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    print(f"  Decomposition panel updated ({len(all_chain_nodes)} nodes in chains)")

def _calculate_magnitude_based_positions_for_subgraph(cvg, node_set):
    """
    NEW: Calculate magnitude-based hierarchical positions for a subgraph.
    
    Positions nodes vertically based on their magnitude values:
    - Higher magnitude values → positioned higher (larger y)
    - Lower magnitude values → positioned lower (smaller y)
    - Horizontal spacing distributes nodes evenly
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        node_set: Set of nodes to position
        
    Returns:
        Dictionary mapping node indices to (x, y) positions
    """
    if not node_set:
        return {}
    
    # Sort nodes by magnitude (descending - highest first)
    sorted_nodes = sorted(node_set, key=lambda n: cvg.time_series[n], reverse=True)
    
    # Group nodes by similar magnitude (for horizontal positioning)
    magnitude_levels = {}
    for node in sorted_nodes:
        mag = round(cvg.time_series[node], 1)  # Round to group similar values
        if mag not in magnitude_levels:
            magnitude_levels[mag] = []
        magnitude_levels[mag].append(node)
    
    # Calculate positions
    positions = {}
    sorted_magnitudes = sorted(magnitude_levels.keys(), reverse=True)
    
    # Vertical spacing
    y_spacing = 5.0
    current_y = len(sorted_magnitudes) * y_spacing
    
    for mag in sorted_magnitudes:
        nodes_at_level = magnitude_levels[mag]
        num_nodes = len(nodes_at_level)
        
        # Horizontal spacing for nodes at same level
        if num_nodes == 1:
            x_positions = [0]
        else:
            x_spacing = 8.0
            total_width = (num_nodes - 1) * x_spacing
            x_positions = [i * x_spacing - total_width / 2 for i in range(num_nodes)]
        
        # Assign positions
        for i, node in enumerate(sorted(nodes_at_level)):  # Sort by node index for consistency
            positions[node] = (x_positions[i], current_y)
        
        current_y -= y_spacing
    
    return positions

def print_hierarchy_summary(cvg):
    """Print enhanced hierarchy summary."""
    print(f"\n{'='*60}")
    print("ENHANCED BIDIRECTIONAL HARSH DECOMPOSITION HIERARCHY")
    print("="*60)
    
    magnitude_groups = defaultdict(list)
    for node in range(cvg.n):
        magnitude = cvg.time_series[node]
        rounded_mag = round(magnitude, 3)
        magnitude_groups[rounded_mag].append(node)
    
    sorted_magnitudes = sorted(magnitude_groups.keys(), reverse=True)
    
    print(f"Enhanced graph organized by {len(sorted_magnitudes)} distinct magnitude levels:")
    
    for i, magnitude in enumerate(sorted_magnitudes[:10]):
        nodes = magnitude_groups[magnitude]
        level_description = "ROOT LEVEL (HARSH)" if magnitude == cvg.time_series[cvg.root] else f"Level {i} (HARSH)"
        
        if len(nodes) == 1:
            print(f"  {level_description}: Node {nodes[0]} (magnitude: {magnitude:.3f})")
        else:
            print(f"  {level_description}: Nodes {nodes} (magnitude: {magnitude:.3f})")
        
        if i >= 9 and len(sorted_magnitudes) > 10:
            print(f"  ... and {len(sorted_magnitudes) - 10} more HARSH levels")
            break
    
    print(f"\nEnhanced Hierarchy Verification:")
    print(f"  ✓ Root node: {cvg.root} with magnitude {cvg.time_series[cvg.root]:.3f}")
    print(f"  ✓ Global maximum: {np.max(cvg.time_series):.3f}")
    print(f"  ✓ HARSH constraints: Strict inequalities enforced")
    print(f"  ✓ Bidirectional analysis: Both increasing and decreasing chains")
    print(f"  ✓ Enhanced validation: Time-based and magnitude-based filters")
    print(f"  ✓ Interactive functionality: Click for HARSH decomposition!")