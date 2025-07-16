import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob, os
import talib



def load_macro_data(data_granularity):
    # load_macro_data
    marco_bond_data = ['UK1YBOND', 'UK10YBOND', 'US1YBOND', 'US1YBOND']
    marco_index_data = ['UKCPI', 'UKGDP', 'USCPI', 'USGDP']
    clean_df = []
    # load_multi_cols
    for each_marco in marco_bond_data:
        root = 'macro_data_root' + each_marco + '.csv'
        data = pd.read_csv(root)
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data.set_index('Date', inplace=True)
        filtered_df = data.apply(lambda col: col.str.replace(',', '').str.replace('%', '').astype(float) / 100 if col.dtypes == 'object' else col)
        resampled_df = filtered_df.astype(float).resample(data_granularity).mean()
        resampled_df = resampled_df.rename(columns={col: each_marco + '_' + col for col in resampled_df.columns})
        clean_df.append(resampled_df)
    # load_simple_col
    for each_marco in marco_index_data:
        root = 'macro_data_root' + each_marco + '.csv'
        data = pd.read_csv(root)
        data['DATE'] = pd.to_datetime(data['DATE'])
        data.set_index('DATE', inplace=True)
        resampled_df = data.astype(float).resample(data_granularity).mean()
        resampled_df = resampled_df.rename(columns={col: each_marco for col in resampled_df.columns})
        clean_df.append(resampled_df)
    macro_df = pd.concat(clean_df, axis=1)
    return macro_df


def load_news_data():
    news_data_feature_use = ['Date', '1_day_close_change label_012', 'sentiment_score',
       'Next Day Close Direction Prediction']
    root = "news_prediction_data_root"
    csv_files = glob.glob(os.path.join(root, "*.csv"))

    dfs = [pd.read_csv(file) for file in csv_files]

    df_list = []
    for each_df in dfs:
        company_name = each_df['Name'].unique()
        renamed_df = each_df[news_data_feature_use]
        renamed_df = renamed_df.rename(columns={col: f"{company_name}{col}" for col in renamed_df.columns if col != "Date"})
        df_list.append(renamed_df)

    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined['DATE'] = pd.to_datetime(df_combined['Date'])
    df_combined.drop(columns=["Date"], inplace=True)
    df_combined = df_combined.set_index('DATE')

    # use mean for multi news in a day
    df_combined = df_combined.groupby(df_combined.index).mean()

    date_range = pd.date_range(start=df_combined.index.min(), end=df_combined.index.max(), freq='D')
    df_daily = df_combined.reindex(date_range, fill_value=None)
    df_daily = apply_wma_to_fill_missing(df_daily)

    df_daily = df_daily.interpolate(method='linear', limit_direction='both')

    return df_daily


# Function to apply WMA and fill missing values
def apply_wma_to_fill_missing(df, timeperiod=30):
    df_filled = df.copy()

    # Apply WMA to each column (ignore NaNs during calculation)
    for column in df.columns:
        wma_values = talib.WMA(df[column].fillna(0), timeperiod)  # Fill NaNs with 0 for calculation
        # Replace missing values (NaNs) with the corresponding WMA values
        df_filled[column] = df[column].fillna(pd.Series(wma_values, index=df.index))

    return df_filled


def load_his_data():
    target_stocks = ['75fd75fa-66a1-43b4-a2c4-3f4fd4d15b74', '5288ad95-72bf-4ea1-880a-b1a39c547855', 'd842b0c9-4255-4b3b-8fcd-8e3b5c779a5c', 'b6eafb78-c0fa-42ff-a9d4-55f92c9cad27', 'c8b6e318-be37-44bc-b2c5-9be36cbc7d90']
    his_data = []
    # ['open_price', 'close_price', 'high_price', 'low_price', 'volatility']
    # historical_feature_use = ['open_price', 'close_price', 'high_price', 'low_price']
    historical_feature_use = ['close', 'open', 'high', 'low']
    for historical_feature in historical_feature_use:
        # data_path = f'D:/my/my/school/school/phd/2023/subjects/parameter-optimisation/nlp/data/historical_data/eod_'+historical_feature+'_all_202405230947_staging.csv'
        data_path = f'historical_data_root' + historical_feature + '_09012025.csv'
        data = pd.read_csv(data_path)
        data.interpolate(method='linear', inplace=True)
        data['quote_date'] = pd.to_datetime(data['quote_date'])
        data.set_index('quote_date', inplace=True)
        data = data[target_stocks]
        data = data.add_prefix(historical_feature)
        his_data.append(data)

    data_y = his_data[0]
    data_y.interpolate(method='linear', inplace=True)
    data_y.fillna(data_y.mean(), inplace=True)
    his_df = pd.concat(his_data, axis=1, join='inner')

    return data_y, his_df


def load_report_data():
    data_path = "report_data_root"
    raw = pd.read_excel(data_path, engine='openpyxl')
    df = raw.pivot(index='YEAR', columns='COMPANY', values=raw.columns.tolist()[2:])
    df.columns = [f'{col[0]}_{col[1]}_score' for col in df.columns]
    df.reset_index(inplace=True)
    df['YEAR'] = df['YEAR'].apply(convert_quarter_to_datetime)
    df = df.set_index('YEAR').sort_index()

    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df_daily = df.reindex(date_range, fill_value=None)
    df_daily = apply_wma_to_fill_missing(df_daily)

    df_daily = df_daily.interpolate(method='linear', limit_direction='both')
    return df_daily


def convert_quarter_to_datetime(quarter_str):
    year, quarter = quarter_str.split('_')
    quarter_to_month = {'Q1': '04', 'Q2': '07', 'Q3': '10', 'Q4': '01'}
    # Default to '01' for invalid quarter
    month = quarter_to_month.get(quarter, '01')

    # If it's Q4, increment the year by 1 since Q4 maps to January of the next year
    if quarter == 'Q4':
        year = str(int(year) + 1)  # Increment the year by 1

    return pd.to_datetime(f'{year}-{month}-01')


class TimeSeriesDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

def create_datasets(X, y, seq_len, forecast_horizon=1, train_split=0.6, val_split=0.2):
    num_days, num_features = X.shape
    num_stocks = y.shape[1]

    y = y.loc[X.index]

    # Prepare input and output sequences
    data_input = []
    data_output = []

    for i in range(num_days - seq_len - forecast_horizon + 1):
        data_input.append(X.iloc[i:i + seq_len].values)  # Sequence of length `seq_len`
        data_output.append(y.iloc[i + seq_len + forecast_horizon - 1].values)  # Target `forecast_horizon` days ahead

    # Convert lists to arrays
    data_input = np.array(data_input)
    data_output = np.array(data_output)

    # Split into training, validation, and test sets
    train_size = int(len(data_input) * train_split)
    val_size = int(len(data_input) * val_split)

    X_train, X_val, X_test = (
        data_input[:train_size],
        data_input[train_size:train_size + val_size],
        data_input[train_size + val_size:],
    )
    y_train, y_val, y_test = (
        data_output[:train_size],
        data_output[train_size:train_size + val_size],
        data_output[train_size + val_size:],
    )

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])  # Flatten for fitting scaler
    scaler.fit(X_train_flat)

    # Apply the scaler to the data (training, validation, and test sets)
    X_train_normalized = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_normalized = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_normalized = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_normalized, y_train)
    val_dataset = TimeSeriesDataset(X_val_normalized, y_val)
    test_dataset = TimeSeriesDataset(X_test_normalized, y_test)

    return train_dataset, val_dataset, test_dataset, scaler