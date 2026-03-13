from pysr import PySRRegressor


def get_symbolic_regression_model(config):
    sym_regression_config = config['sym_regression']
    model = PySRRegressor(
        batching=sym_regression_config['batching'],
        maxdepth=sym_regression_config['maxdepth'],
        niterations=sym_regression_config['niterations'],
        elementwise_loss=sym_regression_config['elementwise_loss'],
        binary_operators=sym_regression_config['binary_operators'],
        unary_operators=sym_regression_config['unary_operators'],
        complexity_of_operators=sym_regression_config['complexity_of_operators'],
        progress=sym_regression_config['progress'],
        turbo=sym_regression_config['turbo'],
    )

    return model