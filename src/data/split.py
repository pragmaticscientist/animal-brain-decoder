from sklearn.model_selection import train_test_split
import numpy as np

"""
Split the dataset into training and test sets.
Parameters:
    - filtered_data: list of data points
    - type: str, type of split ('random', 'species', etc.)
    - seed: int, random seed for reproducibility
    - train_ratio: float, ratio of training data
Returns:
    - train_set: list of training data points
    - test_set: list of testing data points
"""

def split_dataset(data, type, seed = None, train_ratio=0.8):
    if seed is None:
        seed = np.random.randint(0, 1_000_000)  # generate random seed
    
    test_size = 1 - train_ratio
    
    if type == 'random':
        train, test = train_test_split(
            data,
            test_size=test_size,
            shuffle=True,
            random_state=seed
        )
    
    if type == 'species':
        species_to_data = {}
        for data_point in data:
            species = data_point.species
            if species not in species_to_data:
                species_to_data[species] = []
            species_to_data[species].append(data_point)
        
        train = []
        test = []
        train_species, test_species = train_test_split(
            list(species_to_data.keys()),
            test_size=test_size,
            shuffle=True,
            random_state=seed
        )

        rng = np.random.default_rng(seed)
        for species in train_species:
            points = species_to_data[species]
            rng.shuffle(points)  # optional internal shuffle
            train.extend(points)
        for species in test_species:
            points = species_to_data[species]
            rng.shuffle(points)
            test.extend(points)
    
    return train, test    