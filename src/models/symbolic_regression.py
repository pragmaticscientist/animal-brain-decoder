from pysr import PySRRegressor


def get_symbolic_regression_model(config):
    
    model = PySRRegressor(
        denoise=config['denoising'],              
        batching=config['batching'],             

        # --- Strict Simplicity ---
        maxsize=config['maxsize'],                  # Keep it very small. Real laws in 2D are rarely > 18 nodes.
        maxdepth=config['maxdepth'],                  # Prevent deep nesting

        # --- Robust Loss ---
        elementwise_loss=config['elementwise_loss'],  # Use L1 (Mean Absolute Error) to ignore outliers
        
        
        # --- Constraints ---
        binary_operators=config['binary_operators'],
        unary_operators=config['unary_operators'],
 
        
        # --- Technical ---
        progress=config['progress'],
        turbo=config['turbo'],
    )

    return model