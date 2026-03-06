import torch
import glob
import os
import re
import numpy as np
import csv
import pandas as pd

def load_raw_data(directory):
    """
    Loads all point clouds from a directory, converts them into tensors, and
    stacks them into a single tensor.

    Parameters:
    - directory: str, path to the directory containing point cloud files

    Returns:
    - A dictionary id -> point cloud.
    - A dictionary id -> common name.
    """
    id_to_pc = {}
    id_to_common_name = {} 
    for filepath in sorted(glob.glob(os.path.join(directory, '*.txt'))):
        #print(filepath)
        # Load point cloud (3 floats per line, space-separated)
        points = []
        #print(f"filepath {filepath}")
        with open(filepath, 'r') as f:
            animal_id = os.path.splitext(os.path.basename(f.name))[0].lower()
            match = re.match(r"([A-Za-z]+)", os.path.basename(f.name))
            common_name = (match.group(1)).lower()
            for line in f:
                if line.strip():  # skip empty lines
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])

        id_to_pc[animal_id] = torch.tensor(points, dtype=torch.float32)
        id_to_common_name[animal_id] = common_name
    return id_to_pc, id_to_common_name

def load_volume(filepath):
    volumes = {}

    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        # Read all rows into a list
        rows = list(reader)

        # Extract columns and convert to float32
        ball_volumes = [np.float32(row[1]) for row in rows]
        brain_volumes = [np.float32(row[2]) for row in rows]
        mc_volumes = [np.float32(row[3]) for row in rows]
        # Compute mean and std
        ball_volumes_mean = np.mean(ball_volumes)
        brain_volumes_mean = np.mean(brain_volumes)
        mc_volumes_mean = np.mean(mc_volumes)
        ball_volumes_std = np.std(ball_volumes)
        brain_volumes_std = np.std(brain_volumes)
        mc_volumes_std = np.std(mc_volumes)

        # Normalize and store
        for row in rows:
            key = row[0].lower()
            ball_volume = np.float32(row[1])
            brain_volume = np.float32(row[2])
            mc_volume = np.float32(row[3])
            normalized_ball_volume = (ball_volume - ball_volumes_mean) / ball_volumes_std
            normalized_brain_volume = (brain_volume - brain_volumes_mean) / brain_volumes_std
            normalized_mc_volume = (mc_volume - mc_volumes_mean) / mc_volumes_std
            volumes[key] = (ball_volume, normalized_ball_volume, brain_volume, normalized_brain_volume, mc_volume, normalized_mc_volume)

    return volumes

def load_common_to_species(txt_file):
    """
    Loads a whitespace-separated file that maps common names (file names) to species names.
    """
    common_to_species = {}
    with open(txt_file, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                common_name = parts[0].lower()
                species_name = parts[1].lower()
                common_to_species[common_name] = species_name
    return common_to_species

def load_behavior_data(file_path, separator = ','):
    rows = []
    class_set = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=separator)
        print("Fieldnames:", reader.fieldnames)  # Exact headers
        print("First row keys:", list(next(reader).keys()))  # If any
        for row in reader:
            species = row["Species"].strip().lower()
            label = row["Char"].strip()
            rows.append((species, label))
            class_set.add(label)

    class_list = sorted(class_set)
    class_list.reverse()  
    class_to_index = {label: idx for idx, label in enumerate(class_list)}
    print(f"Class to index = {class_to_index}")
    num_classes = len(class_list)

    result = {}
    for species, label in rows:
        index = class_to_index[label]
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[index] = 1.0
        result[species] = (one_hot, index)

    return result

def load_orientation_data(file_path):
    df = pd.read_excel(file_path)
    animal_brain_dict = dict(
        zip(
            df['AnimalName'].astype(str).str.lower().str.replace(" ", "", regex=False),
            df['Brain Orientation']
        )
    )
    return animal_brain_dict

if __name__ == "__main__":
    _, id_to_common_name = load_raw_data("data/raw/unaligned_brains")
    print(id_to_common_name)