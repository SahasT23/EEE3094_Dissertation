import numpy as np
import argparse
import gc
import os
from graph_base import ChronologicalVisibilityGraph
from visibility_graph import build_standard_visibility_graph
from chronological_graph import build_chronological_graph
from path_exploration import start_graph_decomposition
from visualization import calculate_positions, plot_all, print_hierarchy_summary, create_dynamic_visualization
from export_utils import export_decomposition_log, print_analysis, generate_decomposition_log_data
from visibility_utils import print_memory_status

def demonstrate_algorithm(test_mode=False):
    """
    UPDATED: Demonstrate the algorithm with 15-point time series.
    REMOVED: export_to_json() calls - only text logs now
    """
    print("CHRONOLOGICAL VISIBILITY GRAPH WITH NEW VISUALIZATION LAYOUT")
    print("="*80)
    print(" NEW VISUALIZATION FEATURES:")
    print("- Combined layout: Time series, visibility lines, graphs, and decomposition table")
    print("- Softer color palette (lighter blues, reds, and greens)")
    print("- Interactive chronological graph building (retained)")
    print("- Sequential window display after decomposition")
    print("- Decomposition table showing all chains")
    print("- Final visualization with highlighted edges")
    print("="*80)
    
    print_memory_status("demo start")
    
    print("\n15-Point Time Series with Enhanced Visualization:")
    np.random.seed(5)
    series_size = 15
    time_series = np.random.uniform(1, 50, series_size)
    labels = [f"T{i}" for i in range(len(time_series))]
    
    print(f"Creating graph with {series_size} nodes...")
    cvg = ChronologicalVisibilityGraph(time_series, labels, connect_consecutive=True)
    cvg.is_dynamic_mode = False

    print(f"Initial memory usage: {print_memory_status():.1f} MB")
    
    print("Building visibility structures...")
    build_standard_visibility_graph(cvg)
    build_chronological_graph(cvg)
    calculate_positions(cvg)
    
    if not test_mode:
        print("\nPerformance Summary:")
        total_time = sum(cvg.build_time.values())
        for stage, time_taken in cvg.build_time.items():
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            print(f"  - {stage.replace('_', ' ').title()}: {time_taken:.3f}s ({percentage:.1f}%)")
        print(f"  - Total Build Time: {total_time:.3f}s")
    
    if not test_mode:
        print("\n" + "="*60)
        print("STARTING NEW VISUALIZATION LAYOUT")
        print("="*60)
        print("Layout structure:")
        print("  Top left: Time series")
        print("  Top right: Time series with visibility lines")
        print("  Middle left: Standard visibility graph")
        print("  Middle right: Interactive chronological graph")
        print("  Bottom: Decomposition table (after node selection)")
        print("\nINTERACTIONS:")
        print("1. Click in the chronological graph area to build step-by-step")
        print("2. After completion, click on any node to start decomposition")
        print("3. View decomposition table in the same window")
        print("4. Press 'q' key OR close window to see sequential views")
        print("5. Sequential windows will show all individual views")
        print("="*60)
        
        plot_all(cvg, test_mode=test_mode)
        
        print("\n" + "="*50)
        print("MANUAL SEQUENTIAL WINDOWS OPTION")
        print("="*50)
        print("If the automatic sequential windows don't appear,")
        print("you can trigger them manually by running:")
        print(f"show_sequential_windows(cvg, selected_node)")
        print("Where selected_node is the node you chose for decomposition")
        
        global show_sequential_windows
        show_sequential_windows = lambda cvg, node: _trigger_sequential_windows(cvg, node)
    
    # Analysis - REMOVED: export_to_json() call
    print_analysis(cvg)
    print_hierarchy_summary(cvg)
    
    # Memory cleanup
    if not test_mode:
        memory_saved = cvg.optimize_memory_usage()
        print(f"Memory optimization saved: {memory_saved:.1f} MB")
    
    print_memory_status("demo end")
    
    print(f"\n{'='*80}")
    print("NEW VISUALIZATION DEMONSTRATION COMPLETE")
    print("="*80)
    print("If sequential windows didn't appear automatically:")
    print("1. Check console for instructions about pressing 'q' key")
    print("2. Try closing and reopening the visualization window")
    print("3. Use the manual trigger option mentioned above")
    
    return cvg

def dynamic_mode():
    """
    NEW: Dynamic mode for incremental CVG construction and real-time decomposition.
    """
    print("Dynamic mode - incremental point addition")
    print_memory_status("dynamic mode start")
    
    np.random.seed(5)
    series_size = 30
    time_series = np.random.uniform(1, 50, series_size)
    labels = [f"T{i}" for i in range(len(time_series))]
    
    print(f"Generated {series_size}-point time series")
    print(f"Values: {[f'{x:.1f}' for x in time_series]}")
    
    cvg = ChronologicalVisibilityGraph(time_series, labels, connect_consecutive=True)
    cvg.is_dynamic_mode = True
    
    print(f"Initial memory usage: {print_memory_status():.1f} MB")
    
    from visualization import create_dynamic_visualization
    
    create_dynamic_visualization(cvg, time_series)
    
    print_memory_status("dynamic mode end")
    print("\nDynamic mode demonstration complete")

    print(f"\n{'='*60}")
    print("GENERATING DECOMPOSITION FILES FOR ALL NODES (DYNAMIC MODE)")
    print("="*60)
    print("This will create signature JSON files for every node...")
    export_all_node_decompositions(cvg)

def _trigger_sequential_windows(cvg, selected_node):
    """Manual trigger for sequential windows."""
    if not hasattr(cvg, 'selected_node') or cvg.selected_node is None:
        print("No decomposition has been performed yet.")
        print("Please run the visualization and select a node first.")
        return
    
    print("Manually triggering sequential windows...")
    from visualization import _show_sequential_views_after_table
    _show_sequential_views_after_table(cvg, selected_node)

def run_quick_test():
    """
    Quick test function for development/debugging.
    """
    print("Running quick test with new visualization...")
    
    np.random.seed(42)
    test_series = np.random.uniform(1, 20, 10)
    test_labels = [f"T{i}" for i in range(len(test_series))]
    
    print(f"Test series: {[f'{x:.1f}' for x in test_series]}")
    
    cvg = ChronologicalVisibilityGraph(test_series, test_labels)
    build_standard_visibility_graph(cvg)
    build_chronological_graph(cvg)
    calculate_positions(cvg)
    
    print("Graph built successfully!")
    print(f"Visibility edges: {cvg.visibility_graph.number_of_edges()}")
    print(f"Chronological edges: {cvg.chronological_graph.number_of_edges()}")
    print(f"Root node: {cvg.root}")
    
    if not hasattr(cvg, 'selected_node'):
        from path_exploration import start_graph_decomposition
        start_graph_decomposition(cvg, cvg.root)
        
        print(f"Decomposition for node {cvg.root}:")
        print(f"  Decreasing chains: {len(getattr(cvg, 'decreasing_chains', []))}")
        print(f"  Increasing chains: {len(getattr(cvg, 'increasing_chains', []))}")
    
    print("Quick test completed successfully!")

def export_all_node_decompositions(cvg):
    """
    NEW: Export decomposition JSON with signatures for EVERY node in the time series.
    UPDATED: Only works in dynamic mode (checks cvg.is_dynamic_mode flag).
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
    """
    if not getattr(cvg, 'is_dynamic_mode', False):
        print("Skipping all-node decomposition export (only available in dynamic mode)")
        print("To enable: python demo.py --mode dynamic")
        return
    
    from path_exploration import start_graph_decomposition
    import sys
    import io
    
    try:
        from tqdm import tqdm
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        print("Note: Install tqdm for progress bar: pip install tqdm")
    
    print(f"\n{'='*60}")
    print(f"GENERATING DECOMPOSITION FILES FOR ALL {cvg.n} NODES")
    print("="*60)
    print("Dynamic Mode: Signature JSONs will be generated")
    
    successful_exports = 0
    failed_exports = 0
    
    if use_progress_bar:
        iterator = tqdm(range(cvg.n), desc="Processing nodes", unit="node")
    else:
        iterator = range(cvg.n)
    
    for node_idx in iterator:
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                start_graph_decomposition(cvg, node_idx)
            finally:
                sys.stdout = old_stdout
            
            successful_exports += 1
            
            if use_progress_bar:
                iterator.set_postfix({'Success': successful_exports, 'Failed': failed_exports})
            else:
                if (node_idx + 1) % max(1, cvg.n // 10) == 0:
                    progress = ((node_idx + 1) / cvg.n) * 100
                    print(f"Progress: {progress:.1f}% ({node_idx + 1}/{cvg.n} nodes processed)")
            
        except Exception as e:
            if not use_progress_bar:
                print(f"Error processing node {node_idx}: {e}")
            failed_exports += 1
            continue
    
    print(f"\n{'='*60}")
    print("ALL NODE DECOMPOSITION EXPORT COMPLETE (DYNAMIC MODE)")
    print("="*60)
    print(f"Successful: {successful_exports}/{cvg.n}")
    print(f"Failed: {failed_exports}/{cvg.n}")
    print(f"\nJSON files with signatures created in: logs/")
    print(f"Format: logs/node_X_decomposition.json (where X = 0 to {cvg.n - 1})")
    
    print(f"\nExample files generated:")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    for i in range(min(5, cvg.n)):
        file_path = os.path.join(log_dir, f"node_{i}_decomposition.json")
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024
            print(f"  ✓ logs/node_{i}_decomposition.json ({file_size:.1f} KB)")
    if cvg.n > 5:
        print(f"  ... and {cvg.n - 5} more files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronological Visibility Graph - Static or Dynamic Mode")
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static",
                       help="Choose visualization mode: static (original) or dynamic (incremental)")
    parser.add_argument("--test", action="store_true", 
                       help="Run in test mode (no visualizations)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test for development")
    args = parser.parse_args()
    
    try:
        if args.quick:
            run_quick_test()
        elif args.mode == "dynamic":
            print("\n" + "="*80)
            print("DYNAMIC MODE - Incremental CVG Construction")
            print("="*80)
            print("Click to add points one at a time and see real-time decomposition")
            print("="*80 + "\n")
            dynamic_mode()
        else:
            demonstrate_algorithm(test_mode=args.test)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_memory_status("final cleanup")
        gc.collect()
        print("Demo cleanup complete")