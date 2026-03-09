from src.data.data_module import create_dataset, tailor_dataset
from src.data import transformations as transformations_module
from src.data.data_module import load_dataset
from src.data.split import split_dataset


def data_preparation_pipeline(config):
    # Load and preprocess data
    n_copies=config.get('copies', 1)
    raw_data = load_dataset(n_copies)
    print("===================================================")
    print(f"Dataset length after loading: {len(raw_data)}")
    transformations = config.get('transformations')
    fn_names = [t['name'] for t in transformations]
    fn_params = {t['name']: t['parameters'] for t in transformations}
    transform_fn = transformations_module.get_transformation(fn_names, fn_params)
    tailored_data = tailor_dataset(transform_fn, raw_data)
    print(f"Dataset length after tailoring: {len(tailored_data)}")
    
    # Split dataset
    split_type = config['split'].get('type')
    seed = config['split'].get('seed', None)
    train_ratio = config['split'].get('train_ratio', 0.8)
    
    train_data, test_data = split_dataset(tailored_data, split_type, seed, train_ratio)

    # Create datasets
    # Remove unused features and rename input/output features
    input_feature = config['task'].get('input')
    output_feature = config['task'].get('output')
    train_dataset = create_dataset(train_data, input_feature, output_feature)
    test_dataset = create_dataset(test_data, input_feature, output_feature)
    
    return train_dataset, test_dataset