import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import os

INPUT_LENGTH = 125766


class GenomicCNN(nn.Module):
    def __init__(self, x_len, use_dp=False):
        super(GenomicCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=10)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(10, 8, kernel_size=8)
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(8, 6, kernel_size=6)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        final_len = (x_len - 21) // 2
        if use_dp:
            self.norm1 = nn.LayerNorm([6, final_len])
            self.norm2 = nn.LayerNorm(4)
        else:
            self.norm1 = nn.BatchNorm1d(6)
            self.norm2 = nn.BatchNorm1d(4)

        self.fc1 = nn.Linear(6 * final_len, 24)
        self.fc2 = nn.Linear(24, 16)
        self.fc3 = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.norm2(x)
        return x


def get_data_loaders(privacy_type, partition_id=0, is_server=False, num_clients=4, seed=42):
    mapping = {
        "no-dp": "cnn",
        "low-dp": "dpcnn200",
        "high-dp": "dpcnn10",
        "cnn": "cnn",
        "dpcnn10": "dpcnn10",
        "dpcnn200": "dpcnn200"
    }
    actual_folder = mapping.get(privacy_type, privacy_type)
    file_prefix = "cnn" if actual_folder == "cnn" else "dpcnn"
    
    base_path = "attack_targets"
    if not os.path.exists(base_path):
        base_path = os.path.join("problem1", "attack_targets")

    all_data = []

    path = f"{base_path}/{actual_folder}/{actual_folder}_challenge_records.dat"
    with open(path, "rb") as f:
        all_data.append(pickle.load(f))

    for i in range(1, 5):
        path = f"{base_path}/{actual_folder}/client_{i}/"
        with open(f"{path}{file_prefix}_{i}_relevant_records.dat", "rb") as f:
            all_data.append(pickle.load(f))
        with open(f"{path}{file_prefix}_{i}_external_records.dat", "rb") as f:
            all_data.append(pickle.load(f))

    full_data = np.concatenate(all_data, axis=0)

    rng = np.random.RandomState(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.8)
    train_data = full_data[:split_idx]
    test_data = full_data[split_idx:]

    def preprocess(dataset):
        x = dataset[:, :-1].astype(np.float32)
        y = dataset[:, -1].astype(np.int64)
        return torch.tensor(x).unsqueeze(1), torch.tensor(y)

    if is_server:
        test_x, test_y = preprocess(test_data)
        return DataLoader(TensorDataset(test_x, test_y), batch_size=64)
    else:
        chunk_size = len(train_data) // num_clients
        start_idx = partition_id * chunk_size
        end_idx = (partition_id + 1) * chunk_size if partition_id < num_clients - 1 else len(train_data)
        
        client_data = train_data[start_idx:end_idx]
        train_x, train_y = preprocess(client_data)
        return DataLoader(TensorDataset(train_x, train_y), batch_size=15)
