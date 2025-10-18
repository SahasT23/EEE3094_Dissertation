import numpy as np
import psutil
import os
from numba import jit, prange
from numba.typed import List
from numba.core import types

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_status(stage=""):
    """Print current memory usage"""
    memory_mb = get_memory_usage()
    print(f"Memory usage {stage}: {memory_mb:.1f} MB")
    return memory_mb

# ENHANCED: Optimized visibility checking using Numba JIT compilation with memory optimization
@jit(nopython=True, cache=True)
def check_single_visibility_fast(time_series, i, j):
    """
    ENHANCED: Optimized single visibility check with improved memory efficiency.
    
    IMPROVEMENTS:
    - Cached compilation for faster startup
    - Optimized floating point operations
    - Reduced memory allocations
    - Early termination conditions
    
    Args:
        time_series: NumPy array of time series values
        i, j: Node indices to check visibility between
        
    Returns:
        bool: True if nodes i and j are mutually visible
    """
    # Ensure chronological order
    if i > j:
        i, j = j, i
    
    # Adjacent points are always visible
    if j - i <= 1:
        return True
    
    y_a = time_series[i]
    y_b = time_series[j]
    
    # Pre-calculate constants for line equation
    delta_y = y_b - y_a
    delta_t = float(j - i)
    
    # Check all intermediate points with optimized calculations
    for k in range(i + 1, j):
        y_c = time_series[k]
        
        # Optimized line equation calculation
        t_offset = float(k - i)
        y_line = y_a + delta_y * t_offset / delta_t
        
        # Early termination with epsilon comparison
        if y_c > y_line + 1e-10:
            return False
    
    return True

@jit(nopython=True, parallel=True, cache=True)
def check_visibility_batch_fast(time_series, pairs):
    """
    ENHANCED: Optimized batch visibility checking with parallel processing.
    
    IMPROVEMENTS:
    - Cached compilation
    - Parallel processing with prange
    - Memory-efficient array operations
    - Optimized for large datasets
    
    Args:
        time_series: NumPy array of time series values
        pairs: NumPy array of (i,j) pairs to check
        
    Returns:
        NumPy array of boolean results for each pair
    """
    n_pairs = len(pairs)
    results = np.zeros(n_pairs, dtype=np.bool_)
    
    for idx in prange(n_pairs):
        i, j = pairs[idx]
        results[idx] = check_single_visibility_fast(time_series, i, j)
    
    return results

@jit(nopython=True, cache=True)
def check_visibility_matrix_chunk(time_series, start_i, end_i, start_j, end_j):
    """
    NEW: Check visibility for a chunk of the visibility matrix.
    Optimized for memory-efficient processing of large datasets.
    
    Args:
        time_series: NumPy array of time series values
        start_i, end_i: Row range for the chunk
        start_j, end_j: Column range for the chunk
        
    Returns:
        List of (i, j) tuples representing visible pairs in the chunk
    """
    visible_pairs = List.empty_list(types.UniTuple(types.int64, 2))
    
    for i in range(start_i, end_i):
        for j in range(max(i + 1, start_j), end_j):
            if check_single_visibility_fast(time_series, i, j):
                visible_pairs.append((i, j))
    
    return visible_pairs

@jit(nopython=True, cache=True)
def find_visible_neighbors_fast(time_series, node_idx, max_nodes):
    """
    NEW: Fast neighbor finding for a specific node.
    Optimized for decomposition algorithms.
    
    Args:
        time_series: NumPy array of time series values
        node_idx: Index of the node to find neighbors for
        max_nodes: Maximum number of nodes to check
        
    Returns:
        NumPy array of visible neighbor indices
    """
    neighbors = List.empty_list(types.int64)
    
    # Check all potential neighbors
    for i in range(max_nodes):
        if i != node_idx and check_single_visibility_fast(time_series, node_idx, i):
            neighbors.append(i)
    
    # Convert to numpy array
    if len(neighbors) == 0:
        return np.array([], dtype=np.int64)
    
    result = np.zeros(len(neighbors), dtype=np.int64)
    for i in range(len(neighbors)):
        result[i] = neighbors[i]
    
    return result

def check_visibility_memory_efficient(cvg, chunk_size=10000):
    """
    NEW: Memory-efficient visibility checking for very large datasets.
    Processes the visibility matrix in chunks to reduce memory usage.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        chunk_size: Size of chunks for processing
        
    Returns:
        List of (i, j) tuples representing all visible pairs
    """
    print_memory_status("before visibility checking")
    
    visible_edges = []
    total_chunks = (cvg.n * cvg.n) // chunk_size + 1
    processed_chunks = 0
    
    print(f"Processing visibility matrix in {total_chunks} chunks of size {chunk_size}")
    
    # Process matrix in chunks
    for i_start in range(0, cvg.n, int(np.sqrt(chunk_size))):
        i_end = min(i_start + int(np.sqrt(chunk_size)), cvg.n)
        
        for j_start in range(0, cvg.n, int(np.sqrt(chunk_size))):
            j_end = min(j_start + int(np.sqrt(chunk_size)), cvg.n)
            
            # Skip lower triangle and diagonal
            if j_start <= i_start:
                continue
            
            # Process chunk
            chunk_pairs = check_visibility_matrix_chunk(
                cvg.time_series, i_start, i_end, j_start, j_end
            )
            
            # Convert numba list to python list
            for pair in chunk_pairs:
                visible_edges.append(pair)
            
            processed_chunks += 1
            if processed_chunks % 100 == 0:
                progress = (processed_chunks / total_chunks) * 100
                memory_mb = print_memory_status(f"chunk {processed_chunks}/{total_chunks}")
                print(f"Progress: {progress:.1f}%, Found {len(visible_edges)} edges so far")
    
    print_memory_status("after visibility checking")
    return visible_edges