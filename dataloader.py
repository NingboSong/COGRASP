import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_index_mapping(edge_index):
    unique_nodes = torch.unique(edge_index).numpy()
    node_mapping = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    return node_mapping

def convert_edge_index(edge_index, node_mapping):
    for i in range(edge_index.size(1)):
        edge_index[0, i] = node_mapping[edge_index[0, i].item()]
        edge_index[1, i] = node_mapping[edge_index[1, i].item()]
    return edge_index


def load_graph_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    edge_index = []
    edge_weight = []
    for src in df.columns:
        for tgt in df.index:
            weight = df.at[tgt, src]
            if weight > 0:
                edge_index.append([int(src), int(tgt)])
                edge_weight.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    node_mapping = build_index_mapping(edge_index)
    edge_index = convert_edge_index(edge_index, node_mapping)

    max_weight = edge_weight.max()
    if max_weight >= 0:
        edge_weight = edge_weight / max_weight

    return edge_index, edge_weight

def preprocess_data(windows):
    features = ['open',	'close', 'high', 'low', 'amount', 'volume', 'amplitude', 'momentum', 'momentum_volume', 'turnover']
    train_df = pd.read_csv('dataset_2025/train.csv')
    test_df = pd.read_csv('dataset_2025/test.csv')
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    def generate_batches(data_df):
        grouped = data_df.groupby('date')
        valid_dates = [name for name, group in grouped if len(group) == 300]
        X, y = [], []

        for i in range(windows, len(valid_dates)):
            temp_df = data_df[data_df['date'].isin(valid_dates[i-windows:i])]
            if temp_df.shape[0] == 300 * windows:
                X.append(temp_df[features].values.reshape(300, windows, -1))
                temp_target_df = data_df[data_df['date'] == valid_dates[i]]
                if temp_target_df['momentum'].values is not None:
                    y.append(temp_target_df['momentum'].values)
                else:
                    print('No target value for date:', valid_dates[i])
                    y.append(0.0)

        return np.array(X), np.array(y)

    X_train, y_train = generate_batches(train_df)
    X_test, y_test = generate_batches(test_df)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data(windows, graph_file):

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(windows)

    edge_index, edge_weight = load_graph_data(graph_file)

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    node_features = 10
    node_tensor = torch.ones((300, node_features)) 
    graph_data = (node_tensor, edge_index, edge_weight)
    return train_loader, val_loader, test_loader, graph_data 


