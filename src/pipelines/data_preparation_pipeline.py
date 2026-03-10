from src.data.data_module import create_dataset, tailor_dataset
from src.data import transformations as transformations_module
from src.data.data_module import load_dataset
from src.data.split import split_dataset


def data_preparation_pipeline(data_config, split_config, task_config):
    # Load and preprocess data
    raw_data = load_dataset()
    print("===================================================")
    print(f"Dataset length after loading: {len(raw_data)}")
    # Split dataset
    split_type = split_config.get('type')
    seed = split_config.get('seed', None)
    train_ratio = split_config.get('train_ratio', 0.8)
    
    train_data, test_data = split_dataset(raw_data, split_type, seed, train_ratio)
    n_copies=data_config.get('copies', 1)
    
    for data_point in train_data:
        copies = []
        for _ in range(1, n_copies):
            copies.append(data_point.clone())
    train_data.extend(copies)
    # apply transformations
    
    transformations = data_config.get('transformations')
    fn_names = [t['name'] for t in transformations]
    fn_params = {t['name']: t['parameters'] for t in transformations}
    transform_fn = transformations_module.get_transformation(fn_names, fn_params)
    train_data = tailor_dataset(transform_fn, train_data)
    test_data = tailor_dataset(transform_fn, test_data)
    print(f"(Train) Dataset length after tailoring: {len(train_data)}")
    print(f"(Test) Dataset length after tailoring: {len(test_data)}")
    
    # Create datasets
    # Remove unused features and rename input/output features
    input_feature = task_config.get('input')
    output_feature = task_config.get('output')
    train_dataset = create_dataset(train_data, input_feature, output_feature)
    test_dataset = create_dataset(test_data, input_feature, output_feature)
    
    return train_dataset, test_dataset