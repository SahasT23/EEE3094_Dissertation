# TestCMOVG/signature_utils.py - CORRECTED FOR 1D TIME SERIES

import numpy as np
try:
    import esig
except ImportError:
    print("WARNING: esig library not installed. Install with: pip install esig")
    esig = None

def calculate_path_signatures(cvg, path, max_order=3):
    """
    Calculate path signatures for 1D discrete time series.
    
    For 1D time series, we create a 2D path: (time_index, magnitude_value)
    This gives us meaningful signatures that capture the trajectory shape.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        path: List of node indices representing the path
        max_order: Maximum signature order (default: 3)
        
    Returns:
        Dictionary with first_order, second_order, third_order signatures (each a single scalar)
    """
    if esig is None:
        return {
            'first_order': None,
            'second_order': None,
            'third_order': None
        }
    
    if len(path) < 2:
        return {
            'first_order': None,
            'second_order': None,
            'third_order': None
        }
    
    # Create 2D path: [time_index, magnitude_value] for each point
    path_data = np.array([
        [float(node), float(cvg.time_series[node])] 
        for node in path
    ])
    
    try:
        # Calculate signature up to max_order
        sig = esig.stream2sig(path_data, max_order)
        
        # Extract signatures - CORRECTED INDICES for 2D path
        # Level 1: sig[2] is dy (total magnitude change)
        first_order = float(sig[2]) if len(sig) > 2 else None
        
        # Level 2: sig[4] is dx⊗dy (signed area under curve)
        second_order = float(sig[4]) if len(sig) > 4 else None
        
        # Level 3: sig[8] is dx⊗dx⊗dy (time²-weighted magnitude change)
        third_order = float(sig[8]) if len(sig) > 8 else None
        
        return {
            'first_order': first_order,
            'second_order': second_order,
            'third_order': third_order
        }
    
    except Exception as e:
        print(f"Error calculating signature for path {path}: {e}")
        return {
            'first_order': None,
            'second_order': None,
            'third_order': None
        }

def calculate_simple_signatures(cvg, path):
    """
    ALTERNATIVE: Calculate simple signature approximations without esig.
    Use this as fallback or for simpler analysis.
    
    Args:
        cvg: ChronologicalVisibilityGraph instance
        path: List of node indices
        
    Returns:
        Dictionary with first_order, second_order, third_order signatures
    """
    if len(path) < 2:
        return {
            'first_order': None,
            'second_order': None,
            'third_order': None
        }
    
    values = [cvg.time_series[node] for node in path]
    
    # First order: Total change in value
    first_order = float(values[-1] - values[0])
    
    # Second order: Sum of incremental changes (captures volatility)
    second_order = float(sum(abs(values[i+1] - values[i]) for i in range(len(values)-1)))
    
    # Third order: Sum of second-order differences (captures acceleration)
    if len(values) >= 3:
        third_order = float(sum(
            abs((values[i+2] - values[i+1]) - (values[i+1] - values[i]))
            for i in range(len(values)-2)
        ))
    else:
        third_order = 0.0
    
    return {
        'first_order': first_order,
        'second_order': second_order,
        'third_order': third_order
    }

def format_signature_for_display(signature, order):
    """
    Format signature value for readable display.
    
    Args:
        signature: Single signature value (float or None)
        order: Order of the signature (1, 2, or 3)
        
    Returns:
        Formatted string representation
    """
    if signature is None:
        return f"Order {order}: Not calculated"
    
    return f"Order {order}: {signature:.6f}"