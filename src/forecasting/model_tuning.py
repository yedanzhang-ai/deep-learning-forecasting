import pandas as pd
import numpy as np

import torch
from pytorch_lightning.callbacks import EarlyStopping

from darts import concatenate
from darts.metrics import smape

import optuna

from models import build_model

import warnings
warnings.filterwarnings("ignore")

#### Standardize Unscaled and Scaled Datasets
def create_model_x_y_fn(model_type, dataset, supports_static_covariates):
    """Extract scaled or unscaled X/y data based on model type and static covariate support."""
    
    # Select X prefix based on static covariate support
    x_prefix = 'X' if supports_static_covariates else 'X_with_static'

    if model_type == 'dl':       
        y_train = dataset['y_train_scaled']
        X_train = dataset[f'{x_prefix}_train_scaled']
        y_train_valid = dataset['y_train_valid_scaled']
        X_train_valid = dataset[f'{x_prefix}_train_valid_scaled']
        X_full = dataset[f'{x_prefix}_full_scaled']

    else:
        y_train = dataset['y_train']
        X_train = dataset[f'{x_prefix}_train']
        y_train_valid = dataset['y_train_valid']
        X_train_valid = dataset[f'{x_prefix}_train_valid']
        X_full = dataset[f'{x_prefix}_full']

    return y_train, y_train_valid, X_train, X_train_valid, X_full

#### Create fit_kwargs and pred_kwargs
def create_fit_pred_kwargs_fn(
    model_type, 
    cov_support_type, 
    fit_series, 
    fit_covariates, 
    pred_series, 
    pred_covariates, 
    series_start, 
    forecast_horizon, 
    is_backtest
    ):
    """Build fit and predict kwargs based on model type and covariate support."""
    
    if model_type in ['dl', 'ml']:
        fit_kwargs = {'series': fit_series}
        if is_backtest:
            pred_kwargs = {
                'series': pred_series,
                'start': series_start,
                'forecast_horizon': forecast_horizon,
                'stride': forecast_horizon,
                'retrain': False,
                'last_points_only': False,
                'verbose': False
            }
        else:
            pred_kwargs = {
                'n': series_start,
                'series': pred_series,
                'verbose': False
            }

        if cov_support_type == 'future':
            fit_kwargs['future_covariates'] = fit_covariates
            pred_kwargs['future_covariates'] = pred_covariates
        elif cov_support_type == 'past':
            fit_kwargs['past_covariates'] = fit_covariates
            pred_kwargs['past_covariates'] = pred_covariates
       
        return fit_kwargs, pred_kwargs
    
#### Get Hyperparameters
def sample_hp_from_space(trial, hp_space):
    """Sample hyperparameters from hp_space using Optuna trial."""
    
    hp = {}

    for param_name, param_def in hp_space.items():
        if isinstance(param_def, list):          
            hp[param_name] = trial.suggest_categorical(param_name, param_def)
        elif isinstance(param_def, dict):
            param_type = param_def.get("type", "float")
            low = param_def["low"]
            high = param_def["high"]

            if param_type == "float":
                hp[param_name] = trial.suggest_float(param_name, low, high)
            elif param_type == "log_float":
                hp[param_name] = trial.suggest_float(param_name, low, high, log=True)
            else:
                hp[param_name] = trial.suggest_int(param_name, low, high)
        else:            
            hp[param_name] = param_def

    return hp

### Tune DL & ML model
def tune_dl_ml_fn(model_name, model_cfg, dataset, forecast_horizon, calendar_features):
    """Run Optuna hyperparameter tuning for DL or ML models."""
    
    # Get model meta data
    model_type = model_cfg['model_type']
    cov_support_type = model_cfg['covariate_support']
    supports_static_covariates = model_cfg['supports_static_covariates']
    hp_space = model_cfg['hp_space']
    n_trials = model_cfg['n_trials']
    SEED = model_cfg['seed']

    if model_type == 'dl':
        use_gpu = torch.cuda.is_available()
        n_epochs = model_cfg['n_epochs_tune']
        early_stop_patience = model_cfg['early_stop_patience']
        early_stop = EarlyStopping(
            monitor="train_loss",
            mode="min",
            patience=early_stop_patience, 
            min_delta=1e-4, 
            verbose=False,
            )
    else:
        use_gpu = False
        n_epochs = None
        early_stop = None

    # Get data fed to models for tuning
    y_valid = dataset['y_valid']
    y_scaler = dataset['y_scaler']
    y_train, y_train_valid, X_train, X_train_valid, _ = create_model_x_y_fn(model_type, dataset, supports_static_covariates)

    fit_kwargs, pred_kwargs = create_fit_pred_kwargs_fn(
        model_type=model_type,
        cov_support_type=cov_support_type,
        fit_series= y_train,
        fit_covariates= X_train,
        pred_series = y_train_valid,
        pred_covariates =X_train_valid,
        series_start = y_valid[0].start_time(),
        forecast_horizon = forecast_horizon,
        is_backtest = True
    )
    
    # Set up Optuna Objective
    def objective(trial):
        hp = sample_hp_from_space(trial, hp_space)
        model = build_model(
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

        # Train model and predict
        model.fit(**fit_kwargs)
        forecast = model.historical_forecasts(**pred_kwargs)
        y_pred = y_scaler.inverse_transform(forecast) if model_type=='dl' else forecast
        
        # Get smape for tuning
        scores = []
        for actual, pred_list in zip(y_valid, y_pred):
            pred = concatenate(pred_list) if isinstance(pred_list, list) else pred_list
            actual_slice = actual.slice_intersect(pred)
            pred_slice = pred.slice_intersect(actual)
            pred_slice = pred_slice.map(lambda x: np.maximum(x, 0))
            score = smape(actual_slice, pred_slice)
            scores.append(score)

        mean_score = np.mean(scores)
        
        # Clean cache 
        if model_type == 'dl':
            torch.cuda.empty_cache()
            
        return mean_score
        
    # Tune model
    study = optuna.create_study(study_name=f'{model_name}_tuning', direction='minimize')
    study.optimize(objective, n_trials = n_trials, show_progress_bar = True)

    return {
        'model_type': model_type,
        'model_name': model_name,
        'cov_support_type': cov_support_type,
        'best_score': study.best_value,
        'best_params': study.best_params
        }

