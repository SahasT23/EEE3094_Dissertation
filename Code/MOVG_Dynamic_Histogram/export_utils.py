# TestCMOVG/export_utils.py - UPDATED VERSION

import json
import os
import numpy as np
from collections import defaultdict
from visibility_utils import print_memory_status
from signature_utils import calculate_path_signatures

def generate_decomposition_log_data(cvg):
    """
    ENHANCED: Generate log data with corrected subpath enumeration.
    UPDATED: Path length = (node count) - 1
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    if not hasattr(cvg, 'selected_node') or cvg.selected_node is None:
        cvg.decomposition_log_data = {'selected_node': None, 'paths': []}
        print("No decomposition available for logging")
        return
    
    print(f"Generating decomposition log data for node {cvg.selected_node}")
    print_memory_status("before log data generation")
    
    selected_node = cvg.selected_node
    
    from path_exploration import is_chronologically_valid_path, has_direct_cvg_connectivity
    
    cvg.decomposition_log_data = {
        'selected_node': selected_node,
        'node_label': cvg.labels[selected_node],
        'node_value': float(cvg.time_series[selected_node]),
        'boundary_nodes': getattr(cvg, 'decomposition_boundary', []),
        'boundary_values': [float(cvg.time_series[node]) for node in getattr(cvg, 'decomposition_boundary', [])],
        'stats': {
            'decreasing_chains_count': len(getattr(cvg, 'decreasing_chains', [])),
            'increasing_chains_count': len(getattr(cvg, 'increasing_chains', [])),
            'total_subpaths_count': len(getattr(cvg, 'all_sub_paths', [])),
            'total_decomposed_nodes': len(set(
                [node for chain in getattr(cvg, 'decreasing_chains', []) for node in chain] +
                [node for chain in getattr(cvg, 'increasing_chains', []) for node in chain] +
                getattr(cvg, 'decomposition_boundary', [])
            ))
        },
        'chains': {
            'decreasing': [],
            'increasing': []
        },
        'subpaths_analysis': {
            'total_count': len(getattr(cvg, 'all_sub_paths', [])),
            'length_distribution': {},
            'sample_subpaths': []
        }
    }
    
    # Process decreasing chains - UPDATED: length = node_count - 1
    if hasattr(cvg, 'decreasing_chains'):
        for i, path in enumerate(cvg.decreasing_chains):
            if selected_node in path and len(path) >= 2:
                path_info = {
                    'id': i + 1,
                    'nodes': path,
                    'values': [float(cvg.time_series[node]) for node in path],
                    'length': len(path) - 1,  # UPDATED: length = node_count - 1
                    'direction': _get_path_direction(cvg, path),
                    'contains_selected': True,
                    'selected_position': path.index(selected_node),
                    'cvg_connected': has_direct_cvg_connectivity(cvg, path),
                    'validation_status': 'VALID'
                }
                cvg.decomposition_log_data['chains']['decreasing'].append(path_info)
    
    # Process increasing chains - UPDATED: length = node_count - 1
    if hasattr(cvg, 'increasing_chains'):
        for i, path in enumerate(cvg.increasing_chains):
            if selected_node in path and len(path) >= 2:
                path_info = {
                    'id': i + 1,
                    'nodes': path,
                    'values': [float(cvg.time_series[node]) for node in path],
                    'length': len(path) - 1,  # UPDATED: length = node_count - 1
                    'direction': _get_path_direction(cvg, path),
                    'contains_selected': True,
                    'selected_position': path.index(selected_node),
                    'cvg_connected': has_direct_cvg_connectivity(cvg, path),
                    'validation_status': 'VALID'
                }
                cvg.decomposition_log_data['chains']['increasing'].append(path_info)
    
    # Process subpaths - UPDATED: length = node_count - 1
    if hasattr(cvg, 'all_sub_paths'):
        length_counts = defaultdict(int)
        
        for subpath in cvg.all_sub_paths:
            if len(subpath) >= 2:
                path_length = len(subpath) - 1  # UPDATED
                length_counts[path_length] += 1
        
        cvg.decomposition_log_data['subpaths_analysis'] = {
            'total_count': len([p for p in cvg.all_sub_paths if len(p) >= 2]),
            'length_distribution': dict(length_counts),
            'sample_subpaths': [
                {
                    'nodes': subpath,
                    'values': [float(cvg.time_series[node]) for node in subpath],
                    'length': len(subpath) - 1,  # UPDATED
                    'cvg_connected': has_direct_cvg_connectivity(cvg, subpath),
                    'validation_status': 'VALID'
                }
                for subpath in cvg.all_sub_paths[:20] if len(subpath) >= 2
            ],
            'explanation': f"All contiguous subsequences (length ≥ 1) from chains containing node {selected_node}"
        }
    
    print_memory_status("after log data generation")
    print(f"Generated log data: {len(cvg.decomposition_log_data['chains']['decreasing'])} decreasing + {len(cvg.decomposition_log_data['chains']['increasing'])} increasing chains")
    print(f"Subpaths: {cvg.decomposition_log_data['subpaths_analysis']['total_count']} total contiguous subsequences")

def _get_path_direction(cvg, path):
    """Determine the direction of a path relative to selected node."""
    if len(path) == 1:
        return "Single Node"
    
    if not hasattr(cvg, 'selected_node') or cvg.selected_node is None:
        return "Unknown"
    
    selected_node = cvg.selected_node
    future_nodes = [node for node in path if node > selected_node]
    past_nodes = [node for node in path if node < selected_node]
    
    if future_nodes and past_nodes:
        return "Bidirectional"
    elif future_nodes:
        return "Forward"
    elif past_nodes:
        return "Backward"
    else:
        return "Single Node"

def export_decomposition_log(cvg):
    """
    ENHANCED: Export optimized decomposition data to text only.
    UPDATED: Path length = node_count - 1
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    if not hasattr(cvg, 'decomposition_log_data') or not cvg.decomposition_log_data:
        print("No decomposition log data to export")
        return
    
    selected_node = cvg.decomposition_log_data['selected_node']
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    try:
        print(f"Creating directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Export summary with UPDATED format
        summary_filename = os.path.join(log_dir, f"decomposition_summary_node_{selected_node}.txt")
        try:
            with open(summary_filename, 'w') as f:
                _write_decomposition_summary_new_format(cvg, f)
            print(f"Decomposition summary exported to {summary_filename}")
        except Exception as e:
            print(f"Error writing summary file {summary_filename}: {e}")
    
    except Exception as e:
        print(f"Error creating directory {log_dir}: {e}")

def _write_decomposition_summary_new_format(cvg, f):
    """
    Write detailed decomposition summary.
    UPDATED: Path length = node_count - 1
    """
    data = cvg.decomposition_log_data
    
    f.write(f"Graph Decomposition Summary for Node {data['selected_node']} ({data['node_label']})\n")
    f.write(f"Value: {data['node_value']:.2f}\n")
    f.write("="*60 + "\n\n")
    
    # Statistics
    f.write("DECOMPOSITION STATISTICS:\n")
    stats = data['stats']
    f.write(f"  Total decomposed nodes: {stats['total_decomposed_nodes']}\n")
    f.write(f"  Decreasing chains: {stats['decreasing_chains_count']}\n")
    f.write(f"  Increasing chains: {stats['increasing_chains_count']}\n")
    f.write(f"  Total subpaths: {stats['total_subpaths_count']} (all contiguous subsequences, length ≥ 1)\n\n")
    
    # Boundary information
    f.write("UPWARD BOUNDARY:\n")
    f.write(f"Nodes: {data['boundary_nodes']}\n")
    for i, node in enumerate(data['boundary_nodes']):
        f.write(f"  - Node {node} ({cvg.labels[node]}): {data['boundary_values'][i]:.2f}\n")
    f.write("\n")
    
    # Decreasing trends - UPDATED: use length from path_info
    decreasing_chains = data['chains']['decreasing']
    f.write(f"DECREASING TRENDS ({len(decreasing_chains)}):\n\n")
    for chain_info in decreasing_chains:
        f.write("DECREASING TREND\n")
        path = chain_info['nodes']
        for node in path:
            f.write(f"Index {node} — {cvg.time_series[node]:.1f}\n")
        f.write(f"  Length: {chain_info['length']}, Direction: {chain_info['direction']}\n")
        f.write("\n")
    
    # Increasing trends - UPDATED: use length from path_info
    increasing_chains = data['chains']['increasing']
    f.write(f"INCREASING TRENDS ({len(increasing_chains)}):\n\n")
    for chain_info in increasing_chains:
        f.write("INCREASING TREND\n")
        path = chain_info['nodes']
        for node in path:
            f.write(f"Index {node} — {cvg.time_series[node]:.1f}\n")
        f.write(f"  Length: {chain_info['length']}, Direction: {chain_info['direction']}\n")
        f.write("\n")
    
    # Subpaths analysis
    if 'subpaths_analysis' in data:
        subpaths_data = data['subpaths_analysis']
        f.write(f"SUBPATHS ANALYSIS:\n")
        f.write(f"  {subpaths_data['explanation']}\n")
        f.write(f"  Total subpaths: {subpaths_data['total_count']}\n")
        f.write(f"  Length distribution:\n")
        
        for length in sorted(subpaths_data['length_distribution'].keys()):
            count = subpaths_data['length_distribution'][length]
            f.write(f"    Length {length}: {count} subpaths\n")
        
        if subpaths_data.get('sample_subpaths'):
            f.write(f"\n  Sample subpaths (first 20):\n")
            for i, subpath_info in enumerate(subpaths_data['sample_subpaths']):
                nodes_str = " → ".join(str(n) for n in subpath_info['nodes'])
                values_str = " → ".join(f"{v:.1f}" for v in subpath_info['values'])
                f.write(f"    {i+1}: [{nodes_str}] = [{values_str}] (length: {subpath_info['length']})\n")

def _is_forward_in_time(path):
    """
    Check if path moves forward in time (monotonically increasing indices).
    
    Args:
        path: List of node indices
        
    Returns:
        bool: True if path only moves forward in time
    """
    if len(path) < 2:
        return True
    
    for i in range(len(path) - 1):
        if path[i] >= path[i + 1]:
            return False
    
    return True

def export_node_json_with_signatures(cvg, selected_node, dynamic_mode=False):
    """
    NEW: Export decomposition data with path signatures in new JSON format.
    UPDATED: Path length = node_count - 1
    UPDATED: Only exports for dynamic mode, file naming changed to node_X_decomposition.json
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        selected_node: The node being decomposed
        dynamic_mode: If True, use new format. If False, skip export.
    """
    if not dynamic_mode:
        print(f"Skipping signature JSON export for node {selected_node} (static mode)")
        return
    
    json_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    try:
        os.makedirs(json_dir, exist_ok=True)
        print_memory_status("before signature JSON export")
        
        # UPDATED: Filename format
        filename = os.path.join(json_dir, f"node_{selected_node}_decomposition.json")
        
        # Collect all valid paths
        all_paths = []
        
        def try_add_path_chronologically(chain, original_type, is_maximal):
            """Helper to reorder path chronologically and add if valid."""
            filtered_chain = [node for node in chain if node <= selected_node]
            
            if selected_node not in filtered_chain or len(filtered_chain) < 2:
                return
            
            chronological_path = sorted(filtered_chain)
            
            if _is_forward_in_time(chronological_path):
                path_type = _determine_path_type(cvg, chronological_path)
                all_paths.append({
                    'path': chronological_path,
                    'type': path_type,
                    'is_maximal_chain': is_maximal
                })
        
        # Add chains
        for chain in getattr(cvg, 'decreasing_chains', []):
            try_add_path_chronologically(chain, 'decreasing', True)
        
        for chain in getattr(cvg, 'increasing_chains', []):
            try_add_path_chronologically(chain, 'increasing', True)
        
        for subpath in getattr(cvg, 'all_sub_paths', []):
            filtered_subpath = [node for node in subpath if node <= selected_node]
            
            if selected_node not in filtered_subpath or len(filtered_subpath) < 2:
                continue
            
            chronological_subpath = sorted(filtered_subpath)
            
            if _is_forward_in_time(chronological_subpath):
                path_type = _determine_path_type(cvg, chronological_subpath)
                all_paths.append({
                    'path': chronological_subpath,
                    'type': path_type,
                    'is_maximal_chain': False
                })

        # Remove duplicates
        unique_paths = []
        seen = set()
        for path_dict in all_paths:
            path_tuple = tuple(path_dict['path'])
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path_dict)
        
        # Sort by length (longest first)
        unique_paths.sort(key=lambda x: (-len(x['path']), x['path'][0]))
        
        # Build JSON structure
        json_data = {
            "node_info": {
                "index": selected_node,
                "magnitude": float(cvg.time_series[selected_node])
            },
            "decomposition_paths": [],
            "signature_groups": {
                "first_order": [],
                "second_order": [],
                "third_order": [],
            },
            "signature_length_tuples": {
                "first_order": [],
                "second_order": [],
                "third_order": [],
            }
        }
        
        # Process each path - UPDATED: length = node_count - 1
        print(f"Calculating signatures for {len(unique_paths)} paths...")
        for idx, path_dict in enumerate(unique_paths):
            path = path_dict['path']
            path_id = f"Point_{selected_node}_Path_{idx + 1}"
            path_length = len(path) - 1  # UPDATED: length = node_count - 1
            
            # Calculate signatures
            signatures = calculate_path_signatures(cvg, path, max_order=3)
            
            path_entry = {
                "id": path_id,
                "path": path,
                "magnitudes": [float(cvg.time_series[node]) for node in path],
                "length": path_length,  # UPDATED
                "type": path_dict['type'],
                "is_maximal_chain": path_dict['is_maximal_chain'],
                "signatures": {
                    "first_order": signatures['first_order'],
                    "second_order": signatures['second_order'],
                    "third_order": signatures['third_order'],
                }
            }

            json_data["decomposition_paths"].append(path_entry)
            
            # Add to signature groups
            if signatures['first_order'] is not None:
                json_data["signature_groups"]["first_order"].append({
                    "path_id": path_id,
                    "signature": signatures['first_order']
                })
                
                json_data["signature_length_tuples"]["first_order"].append({
                    "path_id": path_id,
                    "tuple": [signatures['first_order'], path_length]  # UPDATED: use corrected length
                })
            
            if signatures['second_order'] is not None:
                json_data["signature_groups"]["second_order"].append({
                    "path_id": path_id,
                    "signature": signatures['second_order']
                })
                
                json_data["signature_length_tuples"]["second_order"].append({
                    "path_id": path_id,
                    "tuple": [signatures['second_order'], path_length]  # UPDATED
                })
            
            if signatures['third_order'] is not None:
                json_data["signature_groups"]["third_order"].append({
                    "path_id": path_id,
                    "signature": signatures['third_order']
                })
                
                json_data["signature_length_tuples"]["third_order"].append({
                    "path_id": path_id,
                    "tuple": [signatures['third_order'], path_length]  # UPDATED
                })
        
        # Write JSON file
        try:
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Node decomposition with signatures exported to {filename}")
            print(f"  Total paths: {len(json_data['decomposition_paths'])}")
            print(f"  First-order signatures: {len(json_data['signature_groups']['first_order'])}")
            print(f"  Second-order signatures: {len(json_data['signature_groups']['second_order'])}")
            print(f"  Third-order signatures: {len(json_data['signature_groups']['third_order'])}")
            print_memory_status("after signature JSON export")
        except Exception as e:
            print(f"Error writing JSON file {filename}: {e}")
    
    except Exception as e:
        print(f"Error creating directory {json_dir}: {e}")

def _determine_path_type(cvg, path):
    """
    Determine if a path is increasing or decreasing based on MAGNITUDE trend.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        path: List of node indices
        
    Returns:
        str: 'increasing_over_time', 'decreasing_over_time', or 'mixed'
    """
    if len(path) < 2:
        return 'single'
    
    values = [cvg.time_series[node] for node in path]
    
    is_mag_increasing = all(values[i+1] > values[i] for i in range(len(values)-1))
    is_mag_decreasing = all(values[i+1] < values[i] for i in range(len(values)-1))
    
    if is_mag_increasing:
        return 'increasing_over_time'
    elif is_mag_decreasing:
        return 'decreasing_over_time'
    else:
        return 'mixed'

# REMOVED: export_to_json() function - no longer needed

def print_analysis(cvg):
    """
    ENHANCED: Print comprehensive analysis.
    UPDATED: Path length = node_count - 1
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    print(f"\n{'='*60}")
    print("ENHANCED GRAPH ANALYSIS")
    print("="*60)
    
    current_memory = print_memory_status()
    print(f"Current Memory Usage: {current_memory:.1f} MB")
    
    print(f"Dataset Size: {cvg.n:,} points")
    print(f"Time Series Statistics:")
    print(f"  - Min Value: {np.min(cvg.time_series):.2f}")
    print(f"  - Max Value: {np.max(cvg.time_series):.2f}")
    print(f"  - Mean Value: {np.mean(cvg.time_series):.2f}")
    print(f"  - Std Dev: {np.std(cvg.time_series):.2f}")
    
    print(f"\nStandard Visibility Graph:")
    print(f"  - Nodes: {cvg.visibility_graph.number_of_nodes():,}")
    print(f"  - Edges: {cvg.visibility_graph.number_of_edges():,}")
    if cvg.visibility_graph.number_of_nodes() > 0:
        avg_degree = sum(dict(cvg.visibility_graph.degree()).values()) / cvg.visibility_graph.number_of_nodes()
        print(f"  - Average Degree: {avg_degree:.2f}")
    
    print(f"\nChronological Graph:")
    print(f"  - Nodes: {cvg.chronological_graph.number_of_nodes():,}")
    print(f"  - Edges: {cvg.chronological_graph.number_of_edges():,}")
    print(f"  - Root Node: {cvg.root} ({cvg.labels[cvg.root]})")
    print(f"  - Graph Height: {_calculate_graph_height_safe(cvg)}")
    
    print(f"\nPerformance Metrics:")
    total_time = sum(cvg.build_time.values())
    for stage, time_taken in cvg.build_time.items():
        percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
        print(f"  - {stage.replace('_', ' ').title()}: {time_taken:.2f}s ({percentage:.1f}%)")
    print(f"  - Total Build Time: {total_time:.2f}s")
    
    print(f"\nGraph Decomposition Analysis:")
    if hasattr(cvg, 'decomposition_log_data') and cvg.decomposition_log_data:
        data = cvg.decomposition_log_data
        print(f"  - Selected Node: {data['selected_node']} ({data['node_label']})")
        print(f"  - Node Value: {data['node_value']:.2f}")
        print(f"  - Boundary Nodes: {len(data['boundary_nodes'])}")
        
        stats = data['stats']
        print(f"  - Decreasing Chains: {stats['decreasing_chains_count']}")
        print(f"  - Increasing Chains: {stats['increasing_chains_count']}")
        print(f"  - Total Decomposed Nodes: {stats['total_decomposed_nodes']}")
        print(f"  - Total Subpaths: {stats['total_subpaths_count']} (length ≥ 1, CVG connected)")
        
        if 'subpaths_analysis' in data:
            subpaths_info = data['subpaths_analysis']
            print(f"  - Subpath Length Distribution:")
            for length in sorted(subpaths_info['length_distribution'].keys())[:5]:
                count = subpaths_info['length_distribution'][length]
                print(f"    * Length {length}: {count} subpaths")
            
            total_lengths = len(subpaths_info['length_distribution'])
            if total_lengths > 5:
                print(f"    * ... and {total_lengths - 5} more length categories")
        
        dec_chains = data['chains']['decreasing']
        inc_chains = data['chains']['increasing']
        
        if dec_chains:
            dec_lengths = [c['length'] for c in dec_chains]
            print(f"  - Decreasing Chain Lengths: min={min(dec_lengths)}, max={max(dec_lengths)}, avg={np.mean(dec_lengths):.1f}")
        
        if inc_chains:
            inc_lengths = [c['length'] for c in inc_chains]
            print(f"  - Increasing Chain Lengths: min={min(inc_lengths)}, max={max(inc_lengths)}, avg={np.mean(inc_lengths):.1f}")
    else:
        print("  - No decomposition data available")
    
    print(f"\n{'='*60}")

def _calculate_graph_height_safe(cvg):
    """Calculate the height of the graph using BFS."""
    if cvg.root is None:
        return 0
    
    from collections import deque
    visited = set()
    queue = deque([(cvg.root, 0)])
    max_height = 0
    
    while queue:
        node, height = queue.popleft()
        if node in visited:
            continue
            
        visited.add(node)
        max_height = max(max_height, height)
        
        for neighbor in cvg.chronological_graph.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, height + 1))
    
    return max_height