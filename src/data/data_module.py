from attrs import fields
from torch_geometric.data import Data
import torch
import os 
import src.data.transformations as transformations_module
import src.data.data_loader as data_loader
import src.data.dataset as dataset_module

PATH_VOLUME = "data/raw/allVols.csv"
PATH_COMMON_TO_SPECIES = "data/raw/animal-species-label-map.txt"
PATH_DIURNALITY = "data/raw/behaviors/day_night_fixed.csv"
PATH_DIURNALITY_BINARY = "data/raw/behaviors/day_night_binary.csv"
PATH_EATING = "data/raw/behaviors/eating-behavior.csv"
PATH_HABITATS = "data/raw/behaviors/num_of_habitats.csv"
PATH_SOCIABILITY = "data/raw/behaviors/sociability.csv"
PATH_ORDERS = "data/raw/behaviors/species_with_orders.csv"
PATH_POINT_CLOUDS = "data/raw/unaligned_brains/"
PATH_ORIENTATION = "data/raw/orientation.xlsx"

def load_raw_data():
    """
    Loads all raw data and returns it.
    Returns:
    - id -> point cloud
    - id -> common name
    - id -> volume
    - common name -> species name
    - species -> behavior
    """
    # id -> volume
    volume_data = data_loader.load_volume(PATH_VOLUME)
    # common_name -> species_name
    common_to_species = data_loader.load_common_to_species(PATH_COMMON_TO_SPECIES)
    # species -> behavior
    diurnality_data = data_loader.load_behavior_data(PATH_DIURNALITY, separator=",")
    diurnality_binary_data = data_loader.load_behavior_data(PATH_DIURNALITY_BINARY, separator=",")
    eating_data = data_loader.load_behavior_data(PATH_EATING, separator=",")
    habitats_data = data_loader.load_behavior_data(PATH_HABITATS, separator=",")
    sociability_data = data_loader.load_behavior_data(PATH_SOCIABILITY, separator=",")
    orders_data = data_loader.load_behavior_data(PATH_ORDERS, separator=" ")
    # id -> point cloud, id -> common name
    id_to_pc, id_to_common_name = data_loader.load_raw_data(PATH_POINT_CLOUDS)
    # animal -> brain orientation
    orientation_data = data_loader.load_orientation_data(PATH_ORIENTATION)

    return id_to_pc, id_to_common_name, volume_data, common_to_species, diurnality_data, diurnality_binary_data, eating_data, habitats_data, sociability_data, orders_data, orientation_data

def load_dataset(save_path=None):
    id_to_pc, id_to_common_name, volume_data, common_to_species, diurnality_data, diurnality_binary_data, eating_data, habitats_data, sociability_data, orders_data, orientation_data = load_raw_data()
    data_list = []
    for id in id_to_pc.keys():
        #print(f"Processing {id}...")
        common_name = id_to_common_name[id]
        species_name = common_to_species.get(common_name)
        diurnality_one_hot, diurnality = diurnality_data.get(species_name, (None, None))
        diurnality_binary_one_hot, diurnality_binary = diurnality_binary_data.get(species_name, (None, None))
        eating_one_hot, eating = eating_data.get(species_name, (None, None))
        habitats_one_hot, habitats = habitats_data.get(species_name, (None, None))
        sociability_one_hot, sociability = sociability_data.get(species_name, (None, None))
        order_one_hot, order = orders_data.get(species_name, (None, None))
        ball_volume, normalized_ball_volume, brain_volume, normalized_brain_volume, mc_volume, normalized_mc_volume = volume_data.get(id)  
        pc = id_to_pc[id]
        pc_orientation = orientation_data.get(id, None)
    
        fields = [
            common_name, species_name, diurnality, diurnality_binary, eating, habitats,
            sociability, order, ball_volume, normalized_ball_volume,
            brain_volume, normalized_brain_volume, mc_volume, normalized_mc_volume, pc_orientation
            ]

        if not all(field is not None for field in fields):
            #print(f"Skipping {id} due to None values")
            #print(f"common_name: {common_name}, species_name: {species_name}, diurnality: {diurnality}, eating: {eating}, habitats: {habitats}, sociability: {sociability}, order: {order}, ball_volume: {ball_volume}, normalized_ball_volume: {normalized_ball_volume}, brain_volume: {brain_volume}, normalized_brain_volume: {normalized_brain_volume}, mc_volume: {mc_volume}, normalized_mc_volume: {normalized_mc_volume}, pc_orientation: {pc_orientation}")
            #print("===================================================")
            continue

        bundle = Data(x = pc, edge_index = torch.empty(2,0, dtype=torch.long), pc_orientation = pc_orientation, id = id, species = species_name, brain_volume = brain_volume, normalized_brain_volume = normalized_brain_volume, ball_volume = ball_volume, normalized_ball_volume = normalized_ball_volume, mc_volume = mc_volume, normalized_mc_volume = normalized_mc_volume, diurnality_one_hot = diurnality_one_hot, diurnality = diurnality, diurnality_binary = diurnality_binary, diurnality_binary_one_hot = diurnality_binary_one_hot, eating_one_hot = eating_one_hot, eating = eating, habitats_one_hot = habitats_one_hot, habitats = habitats, sociability_one_hot = sociability_one_hot, sociability = sociability, order_one_hot = order_one_hot, order = order)
        # save the bundle
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(bundle, os.path.join(save_path, id + ".pth"))
        
        data_list.append(bundle)
        """
        for _ in range(1,copies):
            data_list.append(bundle.clone())
        """
    return data_list

def tailor_dataset(transform_fn, dataset):
    dataset = [transform_fn(data) for data in dataset]
    return dataset

def data_preprocessing(transformations):
    raw_data = load_dataset()
    fn_names, fn_params = transformations
    transform_fn = transformations_module.get_transformation(fn_names, fn_params)
    tailored_data = tailor_dataset(transform_fn, raw_data)
    return tailored_data

def filter_dataset(full_dataset, input_features, output_feature):
    filtered_data = []
    for data in full_dataset:
        input_dict = {feature: getattr(data, feature) for feature in input_features}
        output_value = getattr(data, output_feature)
        filtered_data.append((input_dict, output_value))
    return filtered_data

def create_dataset(filtered_data, input_features, output_feature):
    return dataset_module.SimpleDataset(filtered_data, input_features, output_feature)


