import time
import numpy as np

def build_chronological_graph(cvg):
    """
    Build the chronological visibility graph structure with ALL visible connections.
    
    FIXED ALGORITHM:
    - Creates edges between ALL mutually visible nodes (not just parent-child)
    - Maintains magnitude hierarchy information for visualization
    - Ensures comprehensive connectivity for decomposition algorithms
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    """
    Build the chronological visibility tree structure.
        
    Key Algorithm Rules (UNCHANGED):
    1. Nodes are added in chronological order (0, 1, 2, ...)
    2. When adding a new node, check visibility to all previous nodes
    3. Tree position is determined by magnitude comparison:
    - If new node is LARGER than ALL visible nodes → becomes new root
    - If new node EQUALS the largest visible node → placed at same level as that node
    - If new node is SMALLER → becomes child of smallest visible node that's larger
    4. This creates a hierarchical structure based on local maxima/minima
    5. Consecutive nodes are always connected if toggle is enabled (default: True)
    """
    print("\n=== BUILDING CHRONOLOGICAL VISIBILITY GRAPH ===")
    print(f"Consecutive node connections: {'ENABLED' if cvg.connect_consecutive else 'DISABLED'}")
    print("FIXED: Creating edges for ALL visible node pairs")
    start_time = time.time()
    
    # Add all nodes first
    for i in range(cvg.n):
        cvg.chronological_graph.add_node(i, label=cvg.labels[i], value=cvg.time_series[i])
    
    # Create ALL visibility edges (not just hierarchical ones)
    print("\nPhase 1: Adding ALL visibility edges...")
    edges_added = 0
    
    for i in range(cvg.n):
        for j in range(i + 1, cvg.n):
            if cvg.check_visibility(i, j):
                cvg.chronological_graph.add_edge(i, j)
                edges_added += 1
                if cvg.n <= 20:  # Detailed logging for small datasets
                    print(f"  Added edge: {i} ({cvg.time_series[i]:.1f}) -- {j} ({cvg.time_series[j]:.1f})")
        
        # Progress indicator for larger datasets
        if cvg.n > 20 and i % max(1, cvg.n // 10) == 0:
            print(f"  Processed {i+1}/{cvg.n} nodes, {edges_added} edges so far")
    
    print(f"Phase 1 complete: {edges_added} visibility edges added")
    
    # Add consecutive connections if enabled
    if cvg.connect_consecutive:
        print("\nPhase 2: Adding consecutive connections...")
        consecutive_added = 0
        for i in range(cvg.n - 1):
            if not cvg.chronological_graph.has_edge(i, i + 1):
                cvg.chronological_graph.add_edge(i, i + 1)
                consecutive_added += 1
                if cvg.n <= 20:
                    print(f"  Added consecutive edge: {i} -- {i + 1}")
        print(f"Phase 2 complete: {consecutive_added} consecutive edges added")
    
    # Build magnitude hierarchy for visualization (separate from connectivity)
    print("\nPhase 3: Building magnitude hierarchy for visualization...")
    _build_magnitude_hierarchy(cvg)
    
    build_time = time.time() - start_time
    cvg.build_time['chronological_graph'] = build_time
    
    total_edges = cvg.chronological_graph.number_of_edges()
    print(f"\nCHRONOLOGICAL GRAPH COMPLETE:")
    print(f"  Nodes: {cvg.n}")
    print(f"  Total edges: {total_edges}")
    print(f"  Visibility edges: {edges_added}")
    if cvg.connect_consecutive:
        print(f"  Consecutive edges: {consecutive_added}")
    print(f"  Build time: {build_time:.2f} seconds")
    
    # Verify connectivity for debugging
    if cvg.n <= 20:
        print(f"\nConnectivity verification:")
        for node in range(cvg.n):
            neighbors = list(cvg.chronological_graph.neighbors(node))
            neighbors.sort()
            neighbor_values = [f"{n}({cvg.time_series[n]:.1f})" for n in neighbors]
            print(f"  Node {node} ({cvg.time_series[node]:.1f}) -> {neighbor_values}")

def _build_magnitude_hierarchy(cvg):
    """
    Build magnitude hierarchy for visualization purposes.
    This creates parent-child relationships for tree layout while preserving full connectivity.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    print("  Building magnitude-based hierarchy for visualization...")
    
    # Initialize hierarchy tracking
    cvg.parent_map = {}
    cvg.level_map = {}
    
    # Find global maximum as root
    max_value = np.max(cvg.time_series)
    root_candidates = [i for i in range(cvg.n) if abs(cvg.time_series[i] - max_value) < 1e-10]
    cvg.root = min(root_candidates)  # Use earliest occurrence as root
    cvg.level_map[cvg.root] = 0
    
    print(f"    Root node: {cvg.root} (value={cvg.time_series[cvg.root]:.2f})")
    
    # Process nodes chronologically to build hierarchy
    for i in range(cvg.n):
        if i == cvg.root:
            continue
            
        current_value = cvg.time_series[i]
        
        # Find all visible previous nodes for hierarchy assignment
        visible_predecessors = []
        for j in range(i):
            if cvg.chronological_graph.has_edge(j, i):  # Use actual graph edges
                visible_predecessors.append(j)
        
        if not visible_predecessors:
            # Isolated node - shouldn't happen with consecutive connections
            cvg.level_map[i] = 0
            continue
        
        # Assign parent based on magnitude hierarchy (smallest value > current)
        visible_values = [cvg.time_series[v] for v in visible_predecessors]
        max_visible_value = max(visible_values)
        
        if current_value > max_visible_value:
            # New node is larger than ALL visible predecessors
            if cvg.root in visible_predecessors:
                cvg.parent_map[cvg.root] = i  # Old root becomes child
            cvg.root = i  # This becomes new root
            cvg.level_map[i] = 0
            if cvg.n <= 20:
                print(f"    Node {i} becomes new root (value={current_value:.2f})")
        else:
            # Find appropriate parent (smallest value > current among visible)
            candidates = [v for v in visible_predecessors if cvg.time_series[v] > current_value]
            if candidates:
                parent = min(candidates, key=lambda x: cvg.time_series[x])
                cvg.parent_map[i] = parent
                cvg.level_map[i] = cvg.level_map.get(parent, 0) + 1
                if cvg.n <= 20:
                    print(f"    Node {i} -> parent {parent} (values: {current_value:.2f} < {cvg.time_series[parent]:.2f})")
            else:
                # No suitable parent found - assign to root level
                cvg.level_map[i] = 1
                if cvg.n <= 20:
                    print(f"    Node {i} assigned to level 1 (no suitable parent)")
    
    print(f"    Hierarchy complete: Root={cvg.root}, {len(cvg.parent_map)} parent assignments")
