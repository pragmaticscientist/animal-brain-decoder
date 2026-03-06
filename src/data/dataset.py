from torch_geometric.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data_list, input_features, output_feature):
        super().__init__()
        for data in data_list:
            data.input = getattr(data, input_features)
            #if input_features=="x":
            #    data.input = data.input.permute(1, 0)  # Transpose to [num_features, num_points]
            data.output = getattr(data, output_feature)

            # safely remove other attributes
            for feature in list(data.keys()):
                if feature not in ['id', 'input', 'output']:
                    delattr(data, feature)

        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        return data.id, data.input, data.output