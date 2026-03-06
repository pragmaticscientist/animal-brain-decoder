def get_model(config):
    model_type = config['type']
    if model_type == 'linear_regression':
        from src.models.linear_regression import get_linear_regression_model
        return get_linear_regression_model(config)
    elif model_type == 'logistic_regression':
        from src.models.logistic_regression import get_logistic_regression_model
        return get_logistic_regression_model(config)
    elif model_type == 'symbolic_regression':
        from src.models.symbolic_regression import get_symbolic_regression_model
        return get_symbolic_regression_model(config)
    elif model_type == 'pointnet_pp_cls':
        from src.models.pointnetpp.pointnet_pp import pointnet_pp_cls
        return pointnet_pp_cls(config)
    elif model_type == 'mlp':
        from src.models.mlp import MLP
        return MLP(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")