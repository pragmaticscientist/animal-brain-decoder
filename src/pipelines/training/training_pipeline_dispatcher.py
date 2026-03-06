def training_pipeline_dispatcher(config, model, train_dataset, test_dataset):
    model_type = config['model']['type']
    if model_type in ["symbolic_regression", "linear_regression", "logistic_regression"]:
        from src.pipelines.training.reg_training_pipeline import training_pipeline
    elif model_type in ["pointnet_pp_cls","mlp"]:
        from src.pipelines.training.nn_training_pipeline.nn_training_pipeline import training_pipeline
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return training_pipeline(config, model, train_dataset, test_dataset)