import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
class SimpleMinMaxScaler:
    def __init__(self):
        self.min_ = 0
        self.max_ = 1
        
    def fit_transform(self, data):
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        denom = (self.max_ - self.min_)
        denom[denom == 0] = 1.0 # Avoid division by zero
        return 2 * ((data - self.min_) / denom) - 1
        
    def transform(self, data):
        denom = (self.max_ - self.min_)
        # Handle cases where min/max are same or not yet fit
        if isinstance(denom, np.ndarray):
            denom[denom == 0] = 1.0
        elif denom == 0:
            denom = 1.0
        return 2 * ((data - self.min_) / denom) - 1
        
    def inverse_transform(self, data):
        data_arr = np.array(data)
        return (data_arr + 1) / 2 * (self.max_ - self.min_) + self.min_


class CrowdLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=128, num_layers=2):
        super(CrowdLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class CrowdLSTMModel:
    """LSTM wrapper for Time Series density prediction."""
    def __init__(self, sequence_length=20, epochs=50, lr=0.01):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.model = CrowdLSTM()
        self.scaler = SimpleMinMaxScaler()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.is_trained = False
        
    def create_inout_sequences(self, input_data):
        inout_seq = []
        L = len(input_data)
        for i in range(L - self.sequence_length):
            train_seq = input_data[i:i+self.sequence_length]
            train_label = input_data[i+self.sequence_length:i+self.sequence_length+1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def train(self, df):
        if len(df) <= self.sequence_length:
            return False

        # Use count, hour, and day as features
        features = ['y', 'hour', 'day']
        
        # Check for data variation to avoid scaling issues
        for feat in features:
            if df[feat].nunique() <= 1:
                # Add tiny noise to avoid constant value scaling issues if necessary, 
                # but better to just warn or skip if it's too degenerate.
                pass 

        train_data = df[features].values
        train_data_normalized = self.scaler.fit_transform(train_data)
        train_data_tensor = torch.FloatTensor(train_data_normalized)

        # Re-initialize model and optimizer to avoid inplace operation errors with persistent graphs
        self.model = CrowdLSTM()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_inout_seq = self.create_inout_sequences(train_data_tensor)
        if not train_inout_seq:
            return False

        # Labels are only the 'count' (index 0)
        dataset = TensorDataset(
            torch.stack([s for s, _ in train_inout_seq]), 
            torch.stack([l[:, 0] for _, l in train_inout_seq])
        )
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        self.model.train()
        for i in range(self.epochs):
            for seq_batch, label_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(seq_batch)
                single_loss = self.loss_function(y_pred, label_batch.view(-1, 1))
                single_loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
        self.is_trained = True
        return True
        
    def save(self, paths=('models/lstm_weights.pth', 'models/scaler.pkl')):
        import pickle
        import os
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
        torch.save(self.model.state_dict(), paths[0])
        with open(paths[1], 'wb') as f:
            pickle.dump(self.scaler, f)
        return True
        
    def load(self, paths=('models/lstm_weights.pth', 'models/scaler.pkl')):
        import pickle
        import os
        if not os.path.exists(paths[0]) or not os.path.exists(paths[1]):
            return False
            
        self.model.load_state_dict(torch.load(paths[0]))
        with open(paths[1], 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        return True

    def predict(self, recent_data, periods=30):
        if not self.is_trained or len(recent_data) < self.sequence_length:
            return None
            
        self.model.eval()
        features = ['y', 'hour', 'day']
        
        # If we don't have future hour/day, we'll have to synthesize them or assume constant for eval
        # In a real scenario, we'd pass the known future hour/day.
        test_inputs = recent_data[features].values[-self.sequence_length:]
        test_inputs_normalized = self.scaler.transform(test_inputs).tolist()
        
        predictions_normalized = []
        
        with torch.no_grad():
            for i in range(periods):
                seq = torch.FloatTensor(test_inputs_normalized[-self.sequence_length:]).unsqueeze(0)
                pred = self.model(seq).item()
                
                # To continue predicting, we need hour/day for the next step.
                # We'll synthesize them by assuming 5-min increments from the last timestamp
                last_ts = pd.to_datetime(recent_data['ds'].iloc[-1]) + pd.to_timedelta((i+1)*5, unit='min')
                next_hour = last_ts.hour
                next_day = last_ts.dayofweek
                
                # Prepare next input row: [predicted_count_normalized, next_hour_normalized, next_day_normalized]
                next_feat_raw = np.array([[0, next_hour, next_day]]) # placeholder for count
                next_feat_norm = self.scaler.transform(next_feat_raw)[0]
                next_feat_norm[0] = pred # Use the predicted count
                
                predictions_normalized.append([pred])
                test_inputs_normalized.append(next_feat_norm.tolist())
                
        # Inverse transform only the count column
        # We need to pad with zeros to match scaler shape
        preds_padded = np.zeros((len(predictions_normalized), 3))
        preds_padded[:, 0] = [p[0] for p in predictions_normalized]
        predictions = self.scaler.inverse_transform(preds_padded)[:, 0]
        return predictions
