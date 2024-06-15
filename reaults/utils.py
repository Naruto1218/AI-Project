import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx].values[:-1], self.data.iloc[idx].values[-1]

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
