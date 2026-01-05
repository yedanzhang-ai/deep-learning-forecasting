import pandas as pd
import numpy as np

import torch
from pytorch_lightning.callbacks import EarlyStopping
from darts import concatenate
from darts.metrics import smape, mae, mape, rmse, mase, rmsse

from models import build_model
from model_tuning import create_model_x_y_fn, create_fit_pred_kwargs_fn

#### Define function to calculate wape
def wape(actual_series, pred_series):
    """Weighted Absolute Percentage Error - not available in Darts."""
    actual_vals = actual_series.values().flatten()
    pred_vals = pred_series.values().flatten()
    return np.sum(np.abs(actual_vals - pred_vals)) / np.sum(np.abs(actual_vals)) * 100

#### Implement DL & ML Models
# when is_backtest = True, retrain tuned model on train data and check model performance on valid data
# when is_backtest = False, retrain tuned model on train + valid data and check model_performance on test data
def implement_dl_ml_model_fn(model_name, model_cfg, hp, dataset, forecast_horizon, is_backtest, calendar_features, static_encoders):
    """Train DL/ML model and evaluate on validation or test set."""
    
    # Retrieve model meta data
    model_type = model_cfg['model_type']
    cov_support_type = model_cfg['covariate_support']
    supports_static_covariates = model_cfg['supports_static_covariates']
    SEED = model_cfg['seed']

    if model_type == 'dl':
        use_gpu=torch.cuda.is_available()
        n_epochs=model_cfg['n_epochs_train']       
        early_stop=EarlyStopping(
            monitor="train_loss",
            mode="min",
            patience=model_cfg['early_stop_patience'], 
            min_delta=1e-4, 
            verbose=False,
            )
    else:
        use_gpu=False
        n_epochs=None
        early_stop=None

    # Get data fed to model
    y_valid  = dataset['y_valid']
    y_scaler = dataset['y_scaler']
    y_test   = dataset['y_test']
    y_train, y_train_valid, X_train, X_train_valid, X_full = create_model_x_y_fn(model_type, dataset, supports_static_covariates)

    # Build model
    model=build_model(
        model_name, 
        hp, 
        n_epochs, 
        cov_support_type, 
        early_stop, 
        forecast_horizon, 
        SEED, 
        use_gpu, 
        calendar_features
        )
        
    # Run backtest
    if is_backtest:
        fit_kwargs, pred_kwargs = create_fit_pred_kwargs_fn(
            model_type=model_type,
            cov_support_type=cov_support_type,
            fit_series= y_train,
            fit_covariates= X_train,
            pred_series = y_train_valid,
            pred_covariates =X_train_valid,
            series_start = y_valid[0].start_time(),
            forecast_horizon = forecast_horizon,
            is_backtest = is_backtest
        )
        model.fit(**fit_kwargs)
        forecast = model.historical_forecasts(**pred_kwargs)
        y_pred = y_scaler.inverse_transform(forecast) if model_type=='dl' else forecast       
        y_actual = y_valid

    # Retrain model to forecast
    else:
        fit_kwargs, pred_kwargs = create_fit_pred_kwargs_fn(
            model_type=model_type,
            cov_support_type=cov_support_type,
            fit_series= y_train_valid,
            fit_covariates= X_train_valid,
            pred_series = y_train_valid,
            pred_covariates =X_full,
            series_start = len(y_test[0]),
            forecast_horizon = forecast_horizon,
            is_backtest = is_backtest
            )
        model.fit(**fit_kwargs)
        forecast = model.predict(**pred_kwargs)
        y_pred = y_scaler.inverse_transform(forecast) if model_type=='dl' else forecast            
        y_actual = y_test

    # Collect model implementation result
    results = []
    scores_smape = []
    scores_mape = []
    scores_mase = []
    scores_wape = []
    scores_rmsse = []
    scores_mae = []   
    scores_rmse = []
  

    # Get unscaled training series for MASE/RMSSE calculation   
    y_insample = dataset['y_train'] if is_backtest else dataset['y_train_valid']

    for actual, pred_list, train_series in zip(y_actual, y_pred, y_insample):
        pred = concatenate(pred_list) if isinstance(pred_list, list) else pred_list
        actual_slice = actual.slice_intersect(pred)
        pred_slice = pred.slice_intersect(actual)
        pred_slice = pred_slice.map(lambda x: np.maximum(x, 0))

        dept_id_encoded = int(actual.static_covariates.iloc[0]['dept_id'])
        store_id_encoded = int(actual.static_covariates.iloc[0]['store_id'])
        dept_id = static_encoders['dept_id'].inverse_transform([dept_id_encoded])[0]
        store_id = static_encoders['store_id'].inverse_transform([store_id_encoded])[0]

        # Calculate all metrics
        scores_smape.append(smape(actual_slice, pred_slice))
        scores_mape.append(mape(actual_slice, pred_slice))
        scores_mase.append(mase(actual_slice, pred_slice, insample=train_series, m=52))
        scores_wape.append(wape(actual_slice, pred_slice))
        scores_rmsse.append(rmsse(actual_slice, pred_slice, insample=train_series, m=52))        
        scores_mae.append(mae(actual_slice, pred_slice))       
        scores_rmse.append(rmse(actual_slice, pred_slice))
        

        # Create row for each timestep
        for date, pred_val, actual_val in zip(
            pred_slice.time_index,
            pred_slice.values().flatten(),
            actual_slice.values().flatten()
            ):
            results.append({
                'dept_id': dept_id,
                'store_id': store_id,
                'date': date,
                'forecast': pred_val,
                'actual': actual_val
                })

    results_df = pd.DataFrame(results)
    results_df['forecast']=results_df['forecast'].round(0).astype(int)   

    return {       
        'model': None if is_backtest else model,
        'model_type': model_type,
        'model_name': model_name,
        'model_score_type': 'backtest' if is_backtest else 'forecast',
        'smape': round(np.mean(scores_smape), 4),
        'mape': round(np.mean(scores_mape), 4),
        'mase': round(np.mean(scores_mase), 4),
        'wape': round(np.mean(scores_wape), 4),
        'rmsse': round(np.mean(scores_rmsse), 4),
        'mae': round(np.mean(scores_mae), 4),        
        'rmse': round(np.mean(scores_rmse), 4),        
        'model_forecast': results_df
    }

