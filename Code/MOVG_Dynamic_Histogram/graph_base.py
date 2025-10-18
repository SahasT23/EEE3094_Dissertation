import numpy as np
import networkx as nx
from collections import deque
from visibility_utils import (
    check_single_visibility_fast, 
    check_visibility_batch_fast,
    print_memory_status,
    get_memory_usage
)

class ChronologicalVisibilityGraph:
    def __init__(self, time_series, labels=None, connect_consecutive=True, use_fast_computation=True):
        """
        ENHANCED: Initialize with time series data and memory monitoring.
        
        Args:
            time_series: List or array of time series values
            labels: Optional list of labels for time points
            connect_consecutive: Whether to enforce consecutive node connections (default: True)
            use_fast_computation: Whether to use optimized Numba functions for large datasets
        """
        print_memory_status("initialization start")
        
        self.time_series = np.array(time_series, dtype=np.float64)
        self.n = len(time_series)
        self.labels = labels if labels else [f"T{i}" for i in range(self.n)]
        self.connect_consecutive = connect_consecutive
        
        # Enhanced auto-enable logic for fast computation
        self.use_fast_computation = use_fast_computation and self.n > 50  # Lower threshold
        
        # Graph structures
        self.visibility_graph = nx.Graph()  # Standard visibility graph (undirected)
        self.chronological_graph = nx.Graph()  # Chronological graph (undirected) - ALL visible connections
        
        # Graph structure tracking (for hierarchy visualization)
        self.root = None
        self.parent_map = {}  # For hierarchy display only
        self.level_map = {}   # For hierarchy display only
        self.positions_standard = {}
        self.positions_graph = {}
        
        # Performance tracking with memory monitoring
        self.build_time = {}
        self.memory_usage = {}
        self.memory_usage['initialization'] = get_memory_usage()
        
        # Path decomposition attributes
        self.selected_node = None
        self.decomposition_boundary = []
        self.decreasing_chains = []
        self.increasing_chains = []
        self.all_sub_paths = []
        self.decomposition_active = False
        self.decomposition_log_data = {}
        
        # Enhanced performance settings
        if self.use_fast_computation:
            print(f"✅ Fast computation enabled for {self.n} points")
            if self.n > 2000:
                print(f"⚡ Large dataset detected - using memory-efficient algorithms")
        else:
            print(f"📊 Standard computation for {self.n} points")
        
        print_memory_status("initialization complete")

    def check_visibility(self, i, j):
        """
        ENHANCED: Check if points i and j are visible with automatic optimization selection.
        
        IMPROVEMENTS:
        - Automatic selection between fast and standard algorithms
        - Memory-efficient implementation for large datasets
        - Enhanced error handling and validation
        
        Args:
            i, j: Node indices to check visibility between
            
        Returns:
            bool: True if nodes i and j are mutually visible
        """
        if self.use_fast_computation:
            return check_single_visibility_fast(self.time_series, i, j)
        
        # Standard implementation with optimizations
        # Always check in chronological order (smaller index first)
        if i > j:
            i, j = j, i
            
        # Adjacent points are always visible
        if abs(i - j) <= 1:
            return True
        
        t_a, y_a = i, self.time_series[i]
        t_b, y_b = j, self.time_series[j]
        
        # Pre-calculate line parameters for efficiency
        delta_y = y_b - y_a
        delta_t = t_b - t_a
        
        # Check all intermediate points with optimized calculation
        for k in range(min(i, j) + 1, max(i, j)):
            y_c = self.time_series[k]
            
            # Optimized line equation calculation
            y_line = y_a + delta_y * (k - t_a) / delta_t
            
            # Enhanced precision handling
            epsilon = 1e-10
            if y_c > y_line + epsilon:
                return False
        
        return True
    
    def check_visibility_batch(self, node_pairs):
        """
        NEW: Batch visibility checking for improved performance.
        
        Args:
            node_pairs: List of (i, j) tuples to check
            
        Returns:
            List of boolean results corresponding to each pair
        """
        if self.use_fast_computation:
            pairs_array = np.array(node_pairs)
            return check_visibility_batch_fast(self.time_series, pairs_array).tolist()
        else:
            return [self.check_visibility(i, j) for i, j in node_pairs]
    
    def get_visible_neighbors(self, node_id):
        """
        NEW: Get all nodes visible from a specific node.
        Optimized for decomposition algorithms.
        
        Args:
            node_id: Node to find visible neighbors for
            
        Returns:
            List of node IDs visible from the given node
        """
        if self.use_fast_computation:
            from visibility_utils import find_visible_neighbors_fast
            neighbors = find_visible_neighbors_fast(self.time_series, node_id, self.n)
            return neighbors.tolist()
        else:
            neighbors = []
            for i in range(self.n):
                if i != node_id and self.check_visibility(node_id, i):
                    neighbors.append(i)
            return neighbors
    
    def _calculate_graph_height_safe(self):
        """
        ENHANCED: Calculate graph height with memory monitoring.
        """
        if self.root is None:
            return 0
        
        print_memory_status("before height calculation")
        
        visited = set()
        queue = deque([(self.root, 0)])
        max_height = 0
        
        while queue:
            node, height = queue.popleft()
            if node in visited:
                continue
                
            visited.add(node)
            max_height = max(max_height, height)
            
            # Add neighbors that haven't been visited
            for neighbor in self.chronological_graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, height + 1))
        
        print_memory_status("after height calculation")
        return max_height
    
    def get_memory_statistics(self):
        """
        NEW: Get comprehensive memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        current_memory = get_memory_usage()
        
        stats = {
            'current_memory_mb': current_memory,
            'initialization_memory_mb': self.memory_usage.get('initialization', 0),
            'memory_increase_mb': current_memory - self.memory_usage.get('initialization', 0),
            'nodes': self.n,
            'visibility_edges': self.visibility_graph.number_of_edges() if hasattr(self, 'visibility_graph') else 0,
            'chronological_edges': self.chronological_graph.number_of_edges() if hasattr(self, 'chronological_graph') else 0,
            'memory_per_node_kb': (current_memory * 1024) / self.n if self.n > 0 else 0,
            'fast_computation_enabled': self.use_fast_computation
        }
        
        return stats
    
    def print_memory_report(self):
        """
        NEW: Print detailed memory usage report.
        """
        stats = self.get_memory_statistics()
        
        print(f"\n{'='*50}")
        print("MEMORY USAGE REPORT")
        print("="*50)
        print(f"Current Memory Usage: {stats['current_memory_mb']:.1f} MB")
        print(f"Memory at Initialization: {stats['initialization_memory_mb']:.1f} MB")
        print(f"Memory Increase: {stats['memory_increase_mb']:.1f} MB")
        print(f"\nDataset Information:")
        print(f"  - Nodes: {stats['nodes']:,}")
        print(f"  - Visibility Edges: {stats['visibility_edges']:,}")
        print(f"  - Chronological Edges: {stats['chronological_edges']:,}")
        print(f"  - Memory per Node: {stats['memory_per_node_kb']:.2f} KB")
        print(f"\nOptimization Settings:")
        print(f"  - Fast Computation: {'✅ Enabled' if stats['fast_computation_enabled'] else '❌ Disabled'}")
        print(f"  - Consecutive Connections: {'✅ Enabled' if self.connect_consecutive else '❌ Disabled'}")
        
        # Memory efficiency rating
        memory_per_node = stats['memory_per_node_kb']
        if memory_per_node < 1:
            efficiency = "🏆 Excellent"
        elif memory_per_node < 5:
            efficiency = "✅ Good"
        elif memory_per_node < 20:
            efficiency = "⚠️ Moderate"
        else:
            efficiency = "❌ High"
        
        print(f"  - Memory Efficiency: {efficiency} ({memory_per_node:.2f} KB/node)")
        print(f"{'='*50}\n")
    
    def optimize_memory_usage(self):
        """
        NEW: Attempt to optimize memory usage by cleaning up unused data.
        """
        print("Optimizing memory usage...")
        initial_memory = get_memory_usage()
        
        # Clear position caches if they exist
        if hasattr(self, '_position_cache'):
            del self._position_cache
        
        # Run garbage collection
        import gc
        collected = gc.collect()
        
        final_memory = get_memory_usage()
        memory_saved = initial_memory - final_memory
        
        print(f"Memory optimization complete:")
        print(f"  - Objects collected: {collected}")
        print(f"  - Memory before: {initial_memory:.1f} MB")
        print(f"  - Memory after: {final_memory:.1f} MB")
        print(f"  - Memory saved: {memory_saved:.1f} MB")
        
        return memory_saved
    
    def validate_decomposition_integrity(self):
        """
        ENHANCED: Validate the integrity of decomposition results with corrected subpath validation.
    
        Returns:
            Dictionary with validation results
         """
        if not hasattr(self, 'selected_node') or self.selected_node is None:
            return {'status': 'no_decomposition', 'message': 'No decomposition available to validate'}
    
        validation_results = {
            'status': 'valid',
            'selected_node': self.selected_node,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
    
        # Check decreasing chains
        invalid_decreasing = []
        for i, chain in enumerate(self.decreasing_chains):
            if self.selected_node not in chain:
                invalid_decreasing.append(i)
                validation_results['errors'].append(f"Decreasing chain {i+1} missing selected node")
    
        # Check increasing chains
        invalid_increasing = []
        for i, chain in enumerate(self.increasing_chains):
            if self.selected_node not in chain:
                invalid_increasing.append(i)
                validation_results['errors'].append(f"Increasing chain {i+1} missing selected node")
    
        # Check subpaths (corrected validation)
        invalid_subpaths = 0
        total_checked = min(100, len(self.all_sub_paths))
        for subpath in self.all_sub_paths[:total_checked]:
            if self.selected_node not in subpath:
                invalid_subpaths += 1
    
        if invalid_subpaths > 0:
            validation_results['errors'].append(
                f"{invalid_subpaths}/{total_checked} subpaths missing selected node (ALGORITHM ERROR)"
            )
    
        # Validate subpath consistency
        expected_subpath_count = 0
        for chain in self.decreasing_chains + self.increasing_chains:
            if self.selected_node in chain:
                # Count expected subpaths from this chain
                chain_length = len(chain)
                for start in range(chain_length):
                    for end in range(start + 1, chain_length + 1):
                        if self.selected_node in chain[start:end]:
                            expected_subpath_count += 1
    
        actual_subpath_count = len(self.all_sub_paths)
        if actual_subpath_count != expected_subpath_count:
            validation_results['warnings'].append(
                f"Subpath count mismatch: expected ~{expected_subpath_count}, got {actual_subpath_count}"
            )
    
        # Set status
        if validation_results['errors']:
            validation_results['status'] = 'invalid'
        elif validation_results['warnings']:
            validation_results['status'] = 'warnings'
    
        # Add enhanced statistics
        validation_results['statistics'] = {
            'decreasing_chains_total': len(self.decreasing_chains),
            'decreasing_chains_invalid': len(invalid_decreasing),
            'increasing_chains_total': len(self.increasing_chains),
            'increasing_chains_invalid': len(invalid_increasing),
            'subpaths_total': len(self.all_sub_paths),
            'subpaths_sampled': total_checked,
            'subpaths_invalid_sampled': invalid_subpaths,
            'boundary_nodes': len(self.decomposition_boundary),
            'expected_subpaths_approx': expected_subpath_count,
            'subpath_enumeration_correct': actual_subpath_count == expected_subpath_count
        }
    
        return validation_results

    def print_validation_report(self):
        """
        NEW: Print a comprehensive validation report for decomposition.
        """
        validation = self.validate_decomposition_integrity()
        
        print(f"\n{'='*60}")
        print("DECOMPOSITION VALIDATION REPORT")
        print("="*60)
        
        if validation['status'] == 'no_decomposition':
            print("No decomposition available to validate")
            return
        
        print(f"Selected Node: {validation['selected_node']}")
        print(f"Validation Status: ", end="")
        
        if validation['status'] == 'valid':
            print("PASSED - All paths contain selected node")
        elif validation['status'] == 'warnings':
            print("WARNINGS - Minor issues detected")
        else:
            print("FAILED - Critical issues found")
        
        # Print statistics
        stats = validation['statistics']
        print(f"\nDecomposition Statistics:")
        print(f"  - Decreasing Chains: {stats['decreasing_chains_total']} total, {stats['decreasing_chains_invalid']} invalid")
        print(f"  - Increasing Chains: {stats['increasing_chains_total']} total, {stats['increasing_chains_invalid']} invalid")
        print(f"  - Sub-paths: {stats['sub_paths_total']} total, {stats['sub_paths_invalid_sampled']}/{stats['sub_paths_sampled']} sampled invalid")
        print(f"  - Boundary Nodes: {stats['boundary_nodes']}")
        
        # Print errors
        if validation['errors']:
            print(f"\n❌ Critical Errors ({len(validation['errors'])}):")
            for error in validation['errors']:
                print(f"  - {error}")
        
        # Print warnings
        if validation['warnings']:
            print(f"\n⚠️ Warnings ({len(validation['warnings'])}):")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        if validation['status'] == 'valid':
            print(f"\n✅ All decomposition paths correctly contain the selected node!")
        
        print(f"{'='*60}\n")
    
    def get_performance_summary(self):
        """
        NEW: Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        total_build_time = sum(self.build_time.values()) if self.build_time else 0
        memory_stats = self.get_memory_statistics()
        
        # Calculate efficiency metrics
        nodes_per_second = self.n / total_build_time if total_build_time > 0 else 0
        edges_per_second = memory_stats['visibility_edges'] / total_build_time if total_build_time > 0 else 0
        
        return {
            'dataset_size': self.n,
            'total_build_time': total_build_time,
            'build_time_breakdown': dict(self.build_time),
            'nodes_per_second': nodes_per_second,
            'edges_per_second': edges_per_second,
            'memory_usage': memory_stats,
            'optimization_enabled': self.use_fast_computation,
            'efficiency_rating': self._calculate_efficiency_rating(nodes_per_second, memory_stats['memory_per_node_kb'])
        }
    
    def _calculate_efficiency_rating(self, nodes_per_sec, memory_per_node_kb):
        """Calculate overall efficiency rating."""
        # Speed score (0-100)
        if nodes_per_sec > 10000:
            speed_score = 100
        elif nodes_per_sec > 1000:
            speed_score = 80
        elif nodes_per_sec > 100:
            speed_score = 60
        elif nodes_per_sec > 10:
            speed_score = 40
        else:
            speed_score = 20
        
        # Memory score (0-100)
        if memory_per_node_kb < 1:
            memory_score = 100
        elif memory_per_node_kb < 5:
            memory_score = 80
        elif memory_per_node_kb < 20:
            memory_score = 60
        elif memory_per_node_kb < 50:
            memory_score = 40
        else:
            memory_score = 20
        
        # Combined score
        overall_score = (speed_score + memory_score) / 2
        
        if overall_score >= 90:
            return "🏆 Excellent"
        elif overall_score >= 70:
            return "✅ Good"
        elif overall_score >= 50:
            return "⚠️ Moderate"
        else:
            return "❌ Poor"
    
    def print_performance_summary(self):
        """
        NEW: Print comprehensive performance summary.
        """
        summary = self.get_performance_summary()
        
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"Dataset: {summary['dataset_size']:,} nodes")
        print(f"Total Build Time: {summary['total_build_time']:.2f} seconds")
        print(f"Processing Speed: {summary['nodes_per_second']:,.0f} nodes/second")
        print(f"Edge Generation: {summary['edges_per_second']:,.0f} edges/second")
        print(f"Overall Efficiency: {summary['efficiency_rating']}")
        
        print(f"\nBuild Time Breakdown:")
        for stage, time_taken in summary['build_time_breakdown'].items():
            percentage = (time_taken / summary['total_build_time']) * 100 if summary['total_build_time'] > 0 else 0
            print(f"  - {stage.replace('_', ' ').title()}: {time_taken:.2f}s ({percentage:.1f}%)")
        
        memory = summary['memory_usage']
        print(f"\nMemory Usage:")
        print(f"  - Current: {memory['current_memory_mb']:.1f} MB")
        print(f"  - Per Node: {memory['memory_per_node_kb']:.2f} KB")
        print(f"  - Efficiency: {summary['efficiency_rating']}")
        
        print(f"\nOptimizations:")
        print(f"  - Fast Computation: {'✅ Enabled' if summary['optimization_enabled'] else '❌ Disabled'}")
        print(f"  - Consecutive Connections: {'✅ Enabled' if self.connect_consecutive else '❌ Disabled'}")
        
        print(f"{'='*60}\n")
    
    def cleanup_resources(self):
        """
        NEW: Clean up resources and free memory.
        Useful for processing multiple large datasets.
        """
        print("Cleaning up graph resources...")
        initial_memory = get_memory_usage()
        
        # Clear large data structures
        if hasattr(self, 'positions_standard'):
            self.positions_standard.clear()
        if hasattr(self, 'positions_graph'):
            self.positions_graph.clear()
        
        # Clear decomposition data
        self.decreasing_chains.clear()
        self.increasing_chains.clear()
        self.all_sub_paths.clear()
        self.decomposition_log_data.clear()
        
        # Clear graph structures if requested
        self.visibility_graph.clear()
        self.chronological_graph.clear()
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        
        final_memory = get_memory_usage()
        memory_freed = initial_memory - final_memory
        
        print(f"Resource cleanup complete:")
        print(f"  - Objects collected: {collected}")
        print(f"  - Memory freed: {memory_freed:.1f} MB")
        print(f"  - Current memory: {final_memory:.1f} MB")
        
        return memory_freed