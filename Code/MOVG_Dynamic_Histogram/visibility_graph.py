# TestCMOVG/visibility_graph.py

import numpy as np
import networkx as nx
from visibility_utils import (
    check_visibility_batch_fast, 
    print_memory_status, 
    check_visibility_memory_efficient,
    find_visible_neighbors_fast
)
import time
import gc

def build_standard_visibility_graph(cvg):
    """
    ENHANCED: Build the standard visibility graph with memory optimization.
    Uses different strategies based on dataset size for optimal performance.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    print("Building standard visibility graph...")
    print_memory_status("start of visibility graph building")
    start_time = time.time()
    
    # Add all nodes first
    for i in range(cvg.n):
        cvg.visibility_graph.add_node(i, label=cvg.labels[i], value=cvg.time_series[i])
    
    edges_added = 0
    
    if cvg.use_fast_computation:
        if cvg.n > 2000:
            # Use memory-efficient chunked approach for very large datasets
            print(f"  Using memory-efficient chunked processing for {cvg.n} points...")
            visible_edges = check_visibility_memory_efficient(cvg, chunk_size=50000)
            
            for i, j in visible_edges:
                cvg.visibility_graph.add_edge(i, j)
                edges_added += 1
            
            print(f"  Chunked processing complete: {edges_added:,} edges added")
            
        elif cvg.n > 500:
            # Use optimized batch processing for large datasets
            print(f"  Using optimized batch processing for {cvg.n} points...")
            
            # Generate pairs in memory-efficient batches
            batch_size = min(100000, (cvg.n * (cvg.n - 1)) // 4)  # Reduced batch size
            total_pairs = (cvg.n * (cvg.n - 1)) // 2
            
            processed_pairs = 0
            
            # Process in batches to manage memory
            for i in range(cvg.n):
                # Calculate batch for this row
                row_pairs = []
                for j in range(i + 1, cvg.n):
                    row_pairs.append((i, j))
                    
                    if len(row_pairs) >= batch_size:
                        # Process batch
                        batch_pairs = np.array(row_pairs)
                        batch_results = check_visibility_batch_fast(cvg.time_series, batch_pairs)
                        
                        # Add visible edges
                        for k, is_visible in enumerate(batch_results):
                            if is_visible:
                                edge_i, edge_j = batch_pairs[k]
                                cvg.visibility_graph.add_edge(edge_i, edge_j)
                                edges_added += 1
                        
                        processed_pairs += len(row_pairs)
                        row_pairs = []
                        
                        # Memory cleanup
                        del batch_pairs, batch_results
                        
                        # Progress indicator with memory monitoring
                        if processed_pairs % (batch_size * 5) == 0:
                            progress = (processed_pairs / total_pairs) * 100
                            memory_mb = print_memory_status()
                            print(f"    Progress: {progress:.1f}% ({edges_added:,} edges, {memory_mb:.1f}MB)")
                
                # Process remaining pairs in this row
                if row_pairs:
                    batch_pairs = np.array(row_pairs)
                    batch_results = check_visibility_batch_fast(cvg.time_series, batch_pairs)
                    
                    for k, is_visible in enumerate(batch_results):
                        if is_visible:
                            edge_i, edge_j = batch_pairs[k]
                            cvg.visibility_graph.add_edge(edge_i, edge_j)
                            edges_added += 1
                    
                    processed_pairs += len(row_pairs)
                    del batch_pairs, batch_results
                
                # Periodic garbage collection for large datasets
                if i % 100 == 0:
                    gc.collect()
        
        else:
            # Use standard fast computation for medium datasets
            print(f"  Using standard fast computation for {cvg.n} points...")
            
            pairs = []
            for i in range(cvg.n):
                for j in range(i + 1, cvg.n):
                    pairs.append((i, j))
            
            pairs = np.array(pairs)
            batch_results = check_visibility_batch_fast(cvg.time_series, pairs)
            
            for k, is_visible in enumerate(batch_results):
                if is_visible:
                    i, j = pairs[k]
                    cvg.visibility_graph.add_edge(i, j)
                    edges_added += 1
    
    else:
        # Original implementation for smaller datasets
        print(f"  Using original implementation for {cvg.n} points...")
        for i in range(cvg.n):
            for j in range(i + 1, cvg.n):
                if cvg.check_visibility(i, j):
                    cvg.visibility_graph.add_edge(i, j)
                    edges_added += 1
            
            # Progress for larger datasets
            if cvg.n > 200 and i % max(1, cvg.n // 20) == 0:
                progress = (i / cvg.n) * 100
                memory_mb = print_memory_status()
                print(f"    Progress: {progress:.1f}% ({edges_added:,} edges, {memory_mb:.1f}MB)")
    
    # Final cleanup
    gc.collect()
    
    build_time = time.time() - start_time
    cvg.build_time['visibility_graph'] = build_time
    
    final_memory = print_memory_status("end of visibility graph building")
    print(f"Standard visibility graph: {cvg.n} nodes, {cvg.visibility_graph.number_of_edges():,} edges")
    print(f"Build time: {build_time:.2f} seconds")
    print(f"Final memory usage: {final_memory:.1f} MB")

def build_visibility_graph_neighbors_only(cvg, target_nodes=None):
    """
    NEW: Build visibility connections only for specific nodes.
    Useful for decomposition algorithms that only need local connectivity.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        target_nodes: List of node indices to find neighbors for (default: all nodes)
    """
    if target_nodes is None:
        target_nodes = list(range(cvg.n))
    
    print(f"Building visibility neighbors for {len(target_nodes)} target nodes...")
    start_time = time.time()
    
    neighbors_dict = {}
    
    for node in target_nodes:
        neighbors = find_visible_neighbors_fast(cvg.time_series, node, cvg.n)
        neighbors_dict[node] = neighbors.tolist()
    
    build_time = time.time() - start_time
    print(f"Neighbor finding complete in {build_time:.2f} seconds")
    
    return neighbors_dict