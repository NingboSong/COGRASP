import torch
from torch import nn
from torch_geometric.nn import ChebConv
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        return context_vector, attn_weights
    

class ALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, sequence_length=5,dropout_rate=0.5):
        super(ALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, num_stocks, sequence_length, feature_dim = x.size()
        if x.size(1) < self.sequence_length:
            raise ValueError("out of sequence length")
        x = x[:, :, -self.sequence_length:, :].reshape(-1, self.sequence_length, self.input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_output, _ = self.lstm(x, (h0, c0))
        context_vector, attn_weights = self.attention(lstm_output)
        out = self.fc(context_vector)
        return out.view(batch_size, num_stocks)
    
class ChebNet(nn.Module):
    def __init__(self, node_features, hidden_dim, num_layers=2, K=2):
        super(ChebNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(node_features, hidden_dim, K=K))
        for _ in range(num_layers - 1):
            self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        for conv in self.layers:
            x = conv(x, edge_index, edge_weight)
            x = self.relu(x)
        return x
    
class M_ALSTM_S(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout_rate=0):
        super(M_ALSTM_S, self).__init__()
        self.alstm_1 = ALSTM(input_dim, hidden_dim, num_layers, output_dim, sequence_length=5, dropout_rate=dropout_rate)
        self.alstm_2 = ALSTM(input_dim, hidden_dim, num_layers, output_dim, sequence_length=10, dropout_rate=dropout_rate)
        self.alstm_3 = ALSTM(input_dim, hidden_dim, num_layers, output_dim, sequence_length=15, dropout_rate=dropout_rate)
        self.weights = nn.Parameter(torch.ones(3, 1), requires_grad=True)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        out1 = self.alstm_1(x)
        out2 = self.alstm_2(x)
        out3 = self.alstm_3(x)
        outputs = torch.stack([out1, out2, out3], dim=1)
        
        weights_normalized = F.softmax(self.weights, dim=0)
        out = torch.sum(outputs * weights_normalized, dim=1)

        return out
    

class COGRASP(nn.Module):
    def __init__(self, features, gnn_hidden_dim, lstm_hidden_dim, gnn_num_layers, lstm_num_layers, output_dim=1):
        lstm_input_dim = features + gnn_hidden_dim
        super(COGRASP, self).__init__()
        self.gnn = ChebNet(features, gnn_hidden_dim, gnn_num_layers)
        self.M_ALSTM_S = M_ALSTM_S(lstm_input_dim, lstm_hidden_dim, lstm_num_layers, output_dim)

    def forward(self, graph_data, sequence_data):
      
        x, edge_index, edge_weight = graph_data 
        
        graph_features = self.gnn(x, edge_index, edge_weight)
        graph_features = graph_features.unsqueeze(0).unsqueeze(2)
        graph_features = graph_features.repeat(1, 1, sequence_data.size(2), 1)

        combined_input = torch.cat((sequence_data, graph_features), dim=-1)

        output = self.M_ALSTM_S(combined_input)

        return output
