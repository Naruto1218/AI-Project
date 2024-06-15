import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data.iloc[idx][['open', 'close', 'low', 'high', 'volume', 'money', 'change']].values
        labels = self.data.iloc[idx][['open', 'close', 'low', 'high', 'volume', 'money', 'change']].values
        return features, labels

class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=input_dim, num_encoder_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)
    return predictions, true_labels

def calculate_metrics(predictions, true_labels):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
    import numpy as np
    
    rmse = mean_squared_error(true_labels, predictions, squared=False)
    mae = mean_absolute_error(true_labels, predictions)
    mape = np.mean(np.abs((true_labels - predictions) / true_labels)) * 100
    direction_accuracy = np.mean(np.sign(predictions) == np.sign(true_labels))
    explained_var = explained_variance_score(true_labels, predictions)
    return rmse, mae, mape, direction_accuracy, explained_var
