import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import optuna
from Lasso_selection import Lasso
from data_factory import create_datasets, load_macro_data, load_news_data, load_his_data, load_report_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
import os, sys


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]  # batch size
        seq_length = query.shape[1]

        # batch_size, num_heads, seq_length, head_dim
        Q = self.query(query).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # mask is applied in sometimes
        if mask is not None:
            energy += energy.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(energy, dim=-1)

        #  attention-weighted values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(N, seq_length, -1)  # Reshape to original shape

        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention block
        attention = self.attention(src, src, src)
        attention = self.dropout(attention)
        src = self.layernorm1(src + attention)

        # Feedforward block
        ff_output = self.feed_forward(src)
        ff_output = self.dropout(ff_output)
        out = self.layernorm2(src + ff_output)

        return out


# Time Series Model
class TSTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim):
        super(TSTransformer, self).__init__()

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dim_feedforward)
                                     for _ in range(num_layers)])

        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # Input embedding and positional encoding
        src = self.input_embedding(src)
        src = self.positional_encoding(src)

        # Apply Transformer layers
        for layer in self.layers:
            src = layer(src)

        output = src[:, -1, :]  # Take the last time step

        # Output layer for final prediction
        return self.output_layer(output)


def generate_synthetic_data(num_samples=1000, num_features=14):
    # Generate synthetic data
    data = np.random.rand(num_samples, num_features)  # Random values
    return data


def smape(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    denominator = (np.abs(true) + np.abs(pred))
    # Handle division by zero (replace 0/0 with 0, keep other divisions)
    smape_values = np.where(denominator == 0, 0, 200 * np.abs(pred - true) / denominator)
    return np.mean(smape_values)


def objective(trial):
    # Hyperparameters to tune

    # d_model % num_heads == 0 d_model must be divisible by num_heads, using in QKV calculation
    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256, 512, 1024])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])

    # the number of Transformer encoder layers stacked
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 4, 8])

    # Feed-Forward Network (FFN) block parameter
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512, 1024, 2048, 4096])

    # Training parameter
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4, 5e-5, 1e-3, 5e-3, 1e-2])
    # learning_rate = trial.suggest_categorical("learning_rate", [1e-3])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # batch_size = trial.suggest_categorical("batch_size", [2])
    num_epochs = trial.suggest_categorical("num_epochs", [10, 20, 50, 100])
    # num_epochs = trial.suggest_categorical("num_epochs", [5])

    # Initialize model with hyperparameters
    model = TSTransformer(data_X.shape[1], d_model, num_heads, num_layers, dim_feedforward, output_dim).to(device)

    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            # output = scaler.inverse_transform(output.cpu().detach().numpy())
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print('training epoch:{}, loss:{}'.format(epoch + 1, loss))

        # Valid
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                val_loss += criterion(output, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Test
    model.eval()
    with torch.no_grad():
        true_list = []
        pred_list = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            pred_list.append(output.cpu().numpy())
            true_list.append(batch_y.cpu().numpy())

        true_values = np.concatenate([true.flatten() for true in true_list])
        pred_values = np.concatenate([pred.flatten() for pred in pred_list])

        # evaluations
        mae = mean_absolute_error(true_values, pred_values)
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        mape = mean_absolute_percentage_error(true_values, pred_values)
        smape_score = smape(true_values, pred_values)

        study_results.append({
            "Trial": trial.number,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "True Value": true_list,
            "Prediction": pred_list,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "SMAPE": smape_score

        })

        true = np.vstack(true_list)
        pred = np.vstack(pred_list)
        epsilon = 1e-8
        bias_percentage = ((pred - true) / (true + epsilon)) * 100

        true_df = pd.DataFrame(true, columns=[f"stock_{i + 1}" for i in range(true.shape[1])])
        pred_df = pd.DataFrame(pred, columns=[f"stock_{i + 1}" for i in range(pred.shape[1])])
        bias_df = pd.DataFrame(bias_percentage, columns=[f"stock_{i + 1}" for i in range(bias_percentage.shape[1])])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"trial_{trial.number}_results_{timestamp}.xlsx"
        output_file_path = os.path.join(folder_name, output_file)

        with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
            true_df.to_excel(writer, sheet_name="true", index=False)
            pred_df.to_excel(writer, sheet_name="pred", index=False)
            bias_df.to_excel(writer, sheet_name="bias", index=False)

    return avg_val_loss


if __name__ == '__main__':

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, message):
            for file in self.files:
                file.write(message)

        def flush(self):
            for file in self.files:
                file.flush()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'output_log_{timestamp}.txt'
    log_file = open(log_filename, 'w')
    sys.stdout = Tee(sys.stdout, log_file)

    start_date = '2019-02-04'
    end_date = '2024-5-22'
    # "H/D/W/Q"
    macro_granularity = 'D'
    use_MS = False
    use_lasso = False
    use_news = False
    use_report = False
    seq_len = 20  # look back window
    forecast_horizon = 1  # look forward window
    optimize_test_num = 100  # The optuna search running itrs

    # START
    experiment = 1
    for use_MS in [True, False]:
        for use_lasso in [True, False]:
            for use_news in [True, False]:
                for use_report in [True, False]:

                    print('experiment:{}'.format(experiment))
                    # save config
                    lasso_status = "with_L" if use_lasso else "no_L"
                    macro_scale_status = "with_M" if use_MS else "no_M"
                    news_status = "with_N" if use_news else "no_N"
                    report_status = "with_R" if use_report else "no_R"
                    output_file = (
                        f"results_LB{seq_len}_LF{forecast_horizon}_{lasso_status}_{macro_scale_status}_{news_status}_{report_status}.xlsx"
                    )
                    folder_name = os.path.splitext(output_file)[0]
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)

                    macro_df = load_macro_data(macro_granularity)
                    news_df = load_news_data()
                    data_y, his_df = load_his_data()
                    report_df = load_report_data()

                    output_dim = data_y.shape[1]  # Predicting the next time point (1 output)

                    if use_MS:
                        # with scaling for macro data change
                        macro_df.interpolate(method='linear', inplace=True)
                        macro_df = (macro_df.diff(periods=1) / macro_df.shift(periods=1)) * 100
                        macro_df.replace([np.inf, -np.inf], 0, inplace=True)
                        macro_df.fillna(0, inplace=True)
                        drop_change_df = macro_df.filter(like='Change').columns
                        macro_df.drop(columns=drop_change_df, inplace=True)

                    data_X = pd.merge(his_df, macro_df, left_index=True, right_index=True, how="left")
                    if use_news:
                        data_X = pd.merge(data_X, news_df, left_index=True, right_index=True, how="left")
                    if use_report:
                        data_X = pd.merge(data_X, report_df, left_index=True, right_index=True, how="left")

                    data_y = data_y.loc[start_date:end_date]
                    data_X = data_X.loc[start_date:end_date]

                    if sorted(data_y.index) != sorted(data_X.index):
                        print("warning, data are not matching!")

                    # apply missing values using -1 or mean or ...
                    data_X.fillna(-1, inplace=True)

                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # data_x = generate_synthetic_data(num_samples=num_samples, num_features=num_features)
                    # data_x = pd.concat([y, X], axis=1).to_numpy(dtype='float64')

                    if use_lasso:
                        data_X = Lasso(data_X, data_y, use_corr=False)
                    train_dataset, val_dataset, test_dataset, scaler = create_datasets(data_X, data_y, seq_len,
                                                                                       forecast_horizon)

                    study_results = []

                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=optimize_test_num)

                    results_df = pd.DataFrame(study_results)
                    results_df.to_excel(os.path.join(folder_name, os.path.basename(output_file)), index=False, engine='openpyxl')

                    experiment = experiment+1
                    # Print the best parameters and validation loss
                    print("Best hyperparameters: ", study.best_params)
                    print("Best validation loss: ", study.best_value)
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    sys.stdout = sys.__stdout__
    # Log that the output has been saved
    print(f"Logging complete. Output saved as {log_filename}")