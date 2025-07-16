import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLassoCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from sklearn.feature_selection import VarianceThreshold
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel



def create_lag_features(X, y, lag=1):
    # Create lagged features for each feature column
    X_lagged = pd.concat([X.shift(i) for i in range(1, lag + 1)], axis=1)
    X_lagged.columns = [f'{col}_lag{i}' for i in range(1, lag + 1) for col in X.columns]

    # Drop rows with NaN values created due to shifting
    X_lagged.dropna(inplace=True)
    y_lagged = y.iloc[lag:]

    return X_lagged, y_lagged

# # Generate synthetic data
# np.random.seed(42)
# time_points = 100
# num_features = 5
#
# # Generate random data
# trend = np.linspace(50, 150, time_points)  # Linearly increasing trend (stock price rising)
# seasonality = 10 * np.sin(np.linspace(0, 3 * np.pi, time_points))  # 3 cycles of seasonality
# noise = np.random.normal(0, 5, size=time_points)  # Gaussian noise with mean 0 and standard deviation 5
# target_stock_price = trend + seasonality + noise
# X = np.zeros((time_points, num_features))
#
# # Features
# # X[:, 0] = trend * 0.5 + np.random.normal(0, 5, size=time_points)
# X[:, 0] = np.random.normal(2, 5, size=time_points)
# X[:, 1] = seasonality * 2 + np.random.normal(0, 5, size=time_points)
# # X[:, 2] = np.cumsum(np.random.normal(0, 1, size=time_points))
# X[:, 2] = np.random.normal(-1, 5, size=time_points)
# X[:, 3] = noise * 2 + np.random.normal(0, 5, size=time_points)
# X[:, 4] = np.linspace(0, 50, time_points) + np.random.normal(0, 5, size=time_points)
#
#
# # to DataFrame
# X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, num_features + 1)])
# y_df = pd.Series(target_stock_price, name='target')

# def load_data(company_name):
#     start_date = '2021-1-1'
#     end_date = '2023-12-31'
#     data_granularity = 'Q'
#
#     # load_macro_data
#     marco_bond_data = ['UK1YBOND', 'UK10YBOND', 'US1YBOND', 'US1YBOND']
#     marco_index_data = ['UKCPI', 'UKGDP', 'USCPI', 'USGDP']
#     clean_df = []
#     # load_multi_cols
#     for each_marco in marco_bond_data:
#         root = 'data/macro_data/data/' + each_marco + '.csv'
#         data = pd.read_csv(root)
#         data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
#         filtered_df = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
#         filtered_df.set_index('Date', inplace=True)
#         filtered_df = filtered_df.apply(lambda col: col.str.replace(',', '').str.replace('%', '').astype(float) / 100 if col.dtypes == 'object' else col)
#         resampled_df = filtered_df.astype(float).resample(data_granularity).mean()
#         resampled_df = resampled_df[resampled_df.index <= pd.to_datetime(end_date)]
#         resampled_df = resampled_df.rename(columns={col: each_marco + '_' + col for col in resampled_df.columns})
#         clean_df.append(resampled_df)
#     # load_simple_col
#     for each_marco in marco_index_data:
#         root = 'data/macro_data/data/' + each_marco + '.csv'
#         data = pd.read_csv(root)
#         data['DATE'] = pd.to_datetime(data['DATE'])
#         filtered_df = data.loc[(data['DATE'] >= start_date) & (data['DATE'] <= end_date)]
#         filtered_df.set_index('DATE', inplace=True)
#         resampled_df = filtered_df.astype(float).resample(data_granularity).mean()
#         resampled_df = resampled_df[resampled_df.index <= pd.to_datetime(end_date)]
#         resampled_df = resampled_df.rename(columns={col: each_marco for col in resampled_df.columns})
#         clean_df.append(resampled_df)
#     macro_df = pd.concat(clean_df, axis=1)
#
#     # load_other_data
#     # data = pd.read_excel(r'D:\my\my\school\school\phd\2023\subjects\parameter-optimisation\nlp\data\test_data.xlsx')
#     # valid_data = data.iloc[:-3, 2:]
#     # valid_data = valid_data.astype('float64')
#     data = pd.read_csv('data/new_test_data/'+ company_name + ' (1).csv')
#     company_df = data.iloc[:12, 8:-1]
#     company_df.index = macro_df.index
#     valid_data = pd.concat([macro_df, company_df], axis=1)
#
#
#     y_target = -3
#     y = valid_data.iloc[:, y_target]
#     X = valid_data.iloc[:, [i for i in range(valid_data.shape[1]) if i != y_target]]
#     print('target is :{}'.format(valid_data.columns[y_target]))
#     print('raw data -- time point:{}, features:{}'.format(X.shape[0], X.shape[1]))
#     return X, y

def Lasso(X_df, y_df, use_corr):
    if use_corr:
        print('using corr')
        # test with corr() using the mean abs as the threshold
        correlation_matrix = X_df.corr().abs()
        average_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].mean()
        mean_correlations = correlation_matrix.mean(axis=1)
        selected_features = mean_correlations[mean_correlations < average_correlation].index.tolist()
        X_df = X_df[selected_features]
        print('after corr selection-- features:{}'.format(X_df.shape[1]))
    X_raw = X_df
    y_raw = y_df

    # last one season (90 days) to be count
    X_df = X_df[-90:]
    y_df = y_df[-90:]
    # Create lagged features with a lag of 1
    X_lagged, y_lagged = create_lag_features(X_df, y_df, lag=1)

    # # Reduce dimensionality (variance threshold)
    # selector = VarianceThreshold(threshold=0.01)
    # X_lagged = selector.fit_transform(X_lagged)
    #
    # rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    # rf.fit(X_lagged, y_lagged.mean(axis=1))  # Use mean of targets for simplicity
    # selector = SelectFromModel(rf, threshold="mean", prefit=True)
    # X_lagged = selector.transform(X_lagged)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_lagged)

    # 5-fold time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # LassoCV for cross-validation and automatic alpha selection
    # lasso_cv = MultiTaskLassoCV(alphas=np.logspace(-4, 0, 50), cv=tscv, max_iter=10000)
    logging.basicConfig(level=logging.INFO)
    lasso_cv = MultiTaskLassoCV(alphas=np.logspace(-3, 0, 20), cv=tscv, max_iter=10000, n_jobs=-1, verbose=True)

    lasso_cv.fit(X_scaled, y_lagged)

    # Get the coefficients from the Lasso model
    lasso_coef = lasso_cv.coef_
    # abs_mean = np.mean(np.abs(lasso_coef))
    # sig_features = lasso_coef[np.abs(lasso_coef) > abs_mean]
    # important_features = X_df.columns[np.where(np.abs(lasso_coef) > abs_mean)[0]]
    # X_df = X_df[important_features]
    # print('after lasso selection-- features:{}'.format(X_df.shape[1]))
    # X = X_df.corr().round(2)
    # corr_matrix = X.mask(np.triu(np.ones(X.shape), k=0).astype(bool))
    # print('the features corr :{}'.format(tabulate(corr_matrix, headers="keys", tablefmt="pretty", floatfmt=".2f")))

    important_features = X_raw.columns[np.sum(lasso_coef, axis=0) != 0]
    X_df = X_raw[important_features]

    # abs_mean_coef = np.mean(np.abs(lasso_coef), axis=0)
    # sig_features = np.where(np.abs(lasso_coef).mean(axis=0) > abs_mean_coef)[0]
    # important_features = X_df.columns[sig_features]
    # X_df = X_df[important_features]
    print(f"after lasso selection-- features: {X_df.shape[1]}")
    X_corr = X_df.corr().round(2)
    corr_matrix = X_corr.mask(np.triu(np.ones(X_corr.shape), k=0).astype(bool))
    print(f"the features corr: {tabulate(corr_matrix, headers='keys', tablefmt='pretty', floatfmt='.2f')}")

    return X_df


# # Find the indices of important features (non-zero coefficients)
# non_zero_coefficients = lasso_coef[lasso_coef != 0]
# mean_threshold = np.mean(np.abs(non_zero_coefficients))
#
# important_feature_indices = np.where(np.abs(lasso_coef) >= mean_threshold)[0]
#
# # Get the names of important features
# important_features = X_lagged.columns[important_feature_indices]
#
# # Print the important features
# print("Important Features Selected by LassoCV:")
# for feature in important_features:
#     print(feature)
#
# y_pred = lasso_cv.predict(X_scaled)
# mse = mean_squared_error(y_lagged, y_pred)
# print("Mean Squared Error:", mse)
