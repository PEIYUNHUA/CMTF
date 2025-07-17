# CMTF
Official code for "Cross-Modal Temporal Fusion for Financial Market Forecasting", accepted by ECAI-PAIS 2025.
https://arxiv.org/abs/2504.13522

Before run transf_main.py

environment settings: 
please use
pip install -r requirements.txt

(1)adjust roots in data_factory.py [line 16, 26, 40, 70] (load_macro_data, load_news_data, load_his_data)
(2)adjust running config for the training in transf_main.py [line 273] (main)
(3)adjust the hyperparameter searching space for the model in transf_main.py [line 150] (objective)

model input:
(timepoints*features (historical + macro_index + news_prediction) )
model output
(batch_size, the prediction results of the close price of the given stocks)

hyperparameters including:
    for data:
        start_date
        end_date
        macro_granularity
        use_MS  # using Macro_scaling
        use_lasso   #   using Lasso_selection
        use_news    # using news_prediction as the features

    for prediction:
        seq_len  # look back window
        forecast_horizon  # look forward window T+n
        optimize_test_num  # The optuna search running itrs

    for training & model:
        d_model
        num_heads
        num_layers
        dim_feedforward
        learning_rate
        batch_size
        num_epochs

using optuna to search the best combination of the whole stock market

The 8 macro index data are from:
https://fred.stlouisfed.org/series/GDP
https://fred.stlouisfed.org/series/CPIAUCSL
https://uk.investing.com/rates-bonds/u.s.-1-year-bond-yield-historical-data
https://uk.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data
https://fred.stlouisfed.org/series/UKNGDP
https://fred.stlouisfed.org/series/CPALTT01GBM659N
https://uk.investing.com/rates-bonds/uk-1-year-bond-yield-historical-data
https://uk.investing.com/rates-bonds/uk-10-year-bond-yield-historical-data

To request other data (historical prices, company news, and financial reports), please contact us at https://www.stratiphy.io/about


