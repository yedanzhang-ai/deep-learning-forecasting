# Silence noisy warnings
import os
import logging
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Silence Lightning + Torch
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["PL_LOGGING"] = "0"
os.environ["LIGHTNING_LOG_LEVEL"] = "ERROR"

# Silence all known noisy loggers
NOISY_LOGGERS = [
    "pytorch_lightning",
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "torch",
    "optuna",
    "darts",
]

for name in NOISY_LOGGERS:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    logger.handlers.clear()

import pytorch_lightning.utilities.rank_zero as rz
rz.rank_zero_info = lambda *args, **kwargs: None
rz.rank_zero_warn = lambda *args, **kwargs: None

# Add needed tools
import pandas as pd
import numpy as np 
from pathlib import Path
import yaml

import json
import plotly.express as px
import kaleido 

from raw_data_processing import process_raw_data_fn
from model_data import build_dataset_from_config_fn
from models import build_model
from model_tuning import tune_dl_ml_fn
from model_implementation import implement_dl_ml_model_fn
from experiment_tracking import get_tracker


#### Load models.yaml
def load_config_file(file_path):
    """Load and parse a YAML configuration file."""
    
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

#### Load Model Configurations
def load_model_config(cfg, model_name):
    """Get model configuration by name from config dict."""
    
    model_name_lower = model_name.lower()
    
    for section in ["dl", "ml"]:
        if section in list(cfg["models"].keys()) and model_name_lower in cfg["models"][section]:
            return cfg["models"][section][model_name_lower]

    raise ValueError(f"Model '{model_name}' not found in config")
    
#### Tune Models Across DL and ML Models
def model_tuning_fn(models_config, models_list, dataset, forecast_horizon, calendar_features):
    """Tune all models and return the best one based on SMAPE."""
    
    model_tuning_results = {}

    for model_name in models_list:
        model_cfg = load_model_config(models_config, model_name)          
        model_tuning_results[model_name] = tune_dl_ml_fn(model_name, model_cfg, dataset, forecast_horizon, calendar_features)         
  
    return model_tuning_results
    
#### Select the Best Model through Backtesting Tuned Models Across DL and ML Models
def model_selection_fn(
    models_config, 
    model_tuning_results, 
    dataset, 
    forecast_horizon, 
    calendar_features
    ):
    """Backtest tuned models on validation data"""
    
    optimal_models_all_forecasts = pd.DataFrame()
    backtest_result = {}
    
    for model_name, opt_model_tuning_res in model_tuning_results.items():   
        
        model_cfg = load_model_config(models_config, model_name)  
        hp = opt_model_tuning_res['best_params']        
        
        backtest_result[model_name] = implement_dl_ml_model_fn(
            model_name,
            model_cfg,
            hp,
            dataset,
            forecast_horizon,
            is_backtest=True,
            calendar_features=calendar_features,
            static_encoders=dataset['static_encoders']
        )
      
    optimal_models_score = pd.DataFrame([
        {
            'model_type': v['model_type'],
            'model_name': v['model_name'],
            'model_score_type': v['model_score_type'],
            'smape': v['smape'],
            'mape': v['mape'],
            'mase': v['mase'],
            'wape': v['wape'],
            'rmsse': v['rmsse'],
            'mae': v['mae'],            
            'rmse': v['rmse'],          
        }
        for v in backtest_result.values()
    ])

    optimal_models_all_forecasts = pd.concat([
        v['model_forecast'].assign(model_name=v['model_name'])
        for v in backtest_result.values()
    ], ignore_index=True)
    
    best_model_name = optimal_models_score.loc[optimal_models_score['smape'].idxmin(), 'model_name']
    
    return best_model_name, optimal_models_score, optimal_models_all_forecasts    

#### Retrain the Selected Model and Test Its Performance on Test Dataset
def retrain_selected_model_with_test_performance_fn(
    best_model_name, 
    model_tuning_results, 
    models_config, 
    dataset, 
    forecast_horizon, 
    calendar_features
    ):
    """Retrain the selected model on train + validation dataset and check performance on test dataset"""
    
    best_model_type = model_tuning_results[best_model_name]['model_type']
    best_model_cfg  = load_model_config(models_config, best_model_name)
    best_model_hp   = model_tuning_results[best_model_name]['best_params']
        
    best_model_result = implement_dl_ml_model_fn(
        best_model_name,
        best_model_cfg,
        best_model_hp,
        dataset,
        forecast_horizon,
        is_backtest=False,
        calendar_features=calendar_features,
        static_encoders=dataset['static_encoders']
    )  

    best_model_test_score = pd.DataFrame([
        {
            'model_type': best_model_type,
            'model_name': best_model_name,
            'model_score_type': best_model_result['model_score_type'],
            'smape': best_model_result['smape'],
            'mape': best_model_result['mape'],
            'mase': best_model_result['mase'],
            'wape': best_model_result['wape'],
            'rmsse': best_model_result['rmsse'],
            'mae': best_model_result['mae'],          
            'rmse': best_model_result['rmse'],           
        }
    ])
    best_model = best_model_result['model']
    best_model_test_forecast = best_model_result['model_forecast']
    best_model_test_forecast['model_name'] = best_model_name
    
    return best_model, best_model_test_score, best_model_test_forecast
    
#### Plot Forecasts vs Actuals through Interactive Menu
def plot_with_interactive_menu_fn(df, output_dir):     
    """Plot tuned models' forecasts versus validation dataset given the dept and store chosen by end users"""

    df_long = df.melt(
        id_vars=['dept_id', 'store_id', 'date', 'model_name'],
        value_vars=['actual', 'forecast'],
        var_name='type', value_name='value'
    )

    # Get unique values
    depts = sorted(df['dept_id'].unique())
    stores = sorted(df['store_id'].unique())
    models = df['model_name'].unique().tolist() 

    # Create figure with first dept/store as default
    default_data = df_long[(df_long['dept_id'] == depts[0]) & (df_long['store_id'] == stores[0])]

    fig = px.line(
        default_data,
        x='date', y='value', color='type',
        facet_col='model_name', facet_col_wrap=3, markers=True,
        color_discrete_map={'actual': 'black', 'forecast': 'red'}
    )

    # Build dropdown
    def get_update_args(dept, store):
        filtered = df_long[(df_long['dept_id'] == dept) & (df_long['store_id'] == store)]
        
        x_vals, y_vals = [], []
        
        for t in ['actual', 'forecast']:
            for model in models:
                data = filtered[(filtered['type'] == t) & (filtered['model_name'] == model)].sort_values('date')
                x_vals.append(data['date'].tolist())
                y_vals.append(data['value'].tolist())
        
        return [{'x': x_vals, 'y': y_vals}, {'title': f'Forecasts: {dept} - {store}'}]

    buttons = [
        dict(label=f"{dept} - {store}", method="update", args=get_update_args(dept, store))
        for dept in depts for store in stores
    ]

    fig.update_layout(
        height=1200,
        width=2000,
        title=f'Forecasts: {depts[0]} - {stores[0]}',
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.08, yanchor="top",
            direction="down"
        )]
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(line=dict(dash='solid'), selector=dict(name='forecast'))

    fig.write_html(f"{output_dir}/backtest_results_interactive.html")
    fig.show() 
        
#### Plot Forecasts of One Model
def plot_single_model_forecasts_fn(model_name, df, output_dir):
    """Plot the selected model's forecasts versus test dataset"""
    
    test_res_df_long = df.melt(
        id_vars=['dept_id', 'store_id', 'date', 'model_name'],
        value_vars=['actual', 'forecast'],
        var_name='type', value_name='value'
    )

    test_res_df_long['id'] = test_res_df_long['dept_id'] + '_' + test_res_df_long['store_id']
    new_fig = px.line(
        test_res_df_long, 
        x='date', 
        y='value', 
        color='type',
        facet_col='id', 
        facet_col_wrap=5, 
        markers=True,
        facet_row_spacing=0.04,  
        facet_col_spacing=0.02,
        color_discrete_map={'actual': 'black', 'forecast': 'red'}
    )
    new_fig.update_layout(height=3500, width=2000, autosize=True, title=f'{model_name} Test Results')
    new_fig.update_yaxes(matches=None, showticklabels=True)
    new_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    new_fig.show(renderer="iframe")
    new_fig.write_html(f"{output_dir}/selected_model_test_results.html")
    new_fig.write_image(f"{output_dir}/selected_model_test_results.png")
    
#### Build Pipeline Function
def pipeline_fn(experiment_name, enable_tracking):
    """Execute full forecasting pipeline: tune, backtest, and predict."""
    
    # Get files paths
    project_path = str(Path.cwd()).replace('\\', '/')
    data_config_path = project_path + '/config/data.yaml'
    models_config_path = project_path + '/config/models.yaml'
    output_dir = project_path + '/output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize experiment tracking
    tracker = None
    if enable_tracking:
        tracker = get_tracker(experiment_name)
        tracker.start_run(run_name="pipeline_run")
        tracker.set_tags({"data_config": data_config_path, "models_config": models_config_path})
        
    # Load data configurations
    data_configs = load_config_file(data_config_path)
    
    # Process raw data
    raw_data_cfg = data_configs['raw_data']
    processed_data_file_path = process_raw_data_fn(raw_data_cfg, project_path)
    
    print("Finished processing raw data...")

    # Generate data for models    
    calendar_features = data_configs['data']['calendar_features']
    dataset, forecast_horizon = build_dataset_from_config_fn(data_configs['data'], processed_data_file_path)
    
    print("Finished building datasets for models...")

    # Log dataset parameters
    if tracker:
        tracker.log_params({"forecast_horizon": forecast_horizon, 
            "n_series": len(dataset['y_train'])})

    # Get model list
    models_list = []
    models_config = load_config_file(models_config_path)
    
    # Tune models  
    for model_type in ['dl', 'ml']:
        models_list += list(models_config['models'][model_type].keys())          
          
    # Tune models
    model_tuning_results = model_tuning_fn(models_config, models_list, dataset, forecast_horizon, calendar_features)
    
    print("Finished tuning models ...")

    # Log tuning results to MLflow
    if tracker:
        for model_name, results in model_tuning_results.items():
            tracker.log_model_results(
                model_name=model_name,
                model_type=results['model_type'],
                best_params=results['best_params'],
                score=results['best_score'] if results['best_score'] is not None else float('inf')
            )

    # Backtest tuned models  
    best_model_name, optimal_models_score, optimal_models_all_forecasts = model_selection_fn(
        models_config, 
        model_tuning_results, 
        dataset, 
        forecast_horizon, 
        calendar_features
        )
        
    print("Finished backtesting tuned models and model_selection")
        
    # Save backtesting results
    optimal_models_score.to_csv(f"{output_dir}/tuned_models_backtest_performance.csv", index=False)
    optimal_models_all_forecasts.to_csv(f"{output_dir}/tuned_models_backtest_forecasts.csv", index=False)
    
    print("Finished storing tuned models'results...")

    # Log backtest metrics
    if tracker:
        for _, row in optimal_models_score.iterrows():
            tracker.log_metric(f"backtest_smape_{row['model_name']}", row['smape'])
            tracker.log_metric(f"backtest_mae_{row['model_name']}", row['mae'])
            tracker.log_metric(f"backtest_rmse_{row['model_name']}", row['rmse'])          

    # Train selected model and Assess Performance on test dataset     
    best_model, best_model_test_score, best_model_test_forecast = retrain_selected_model_with_test_performance_fn(
        best_model_name, 
        model_tuning_results, 
        models_config, 
        dataset, 
        forecast_horizon, 
        calendar_features
        )
        
    print("Finished retraining best models and get the model's performance...")

    # Store the selected model across all experimented DL, ML, and STAT models
    os.makedirs(f"{output_dir}/model/", exist_ok=True)
    model_path = os.path.join(f"{output_dir}/model/", f"selected_model_{best_model_name}")
    if best_model is not None:
        best_model.save(model_path)
    
    # Save forecasting results
    best_model_test_score.to_csv(f"{output_dir}/best_model_test_performance.csv", index=False)
    best_model_test_forecast.to_csv(f"{output_dir}/best_model_test_forecasts.csv", index=False)
    
    print("Finished storing the best model's results...")

    # Log test metrics
    if tracker:        
        tracker.log_dataframe(best_model_test_score, "test_performance")

    # Save hyperparameters
    hp_path = os.path.join(f"{output_dir}/model/", f"{best_model_name}_best_params.json")
    best_model_hp = model_tuning_results[best_model_name]['best_params']
    with open(hp_path, 'w') as f:
        json.dump(best_model_hp, f, indent=2)

    # Log model artifacts to MLflow
    if tracker:
        import mlflow
        mlflow.log_artifacts(f"{output_dir}/model/", artifact_path="model")

    # End tracking run
    if tracker:
        tracker.end_run()

    # Plot backtesting results
    plot_with_interactive_menu_fn(optimal_models_all_forecasts, output_dir)  
    
    
    print("Finished interactive plot...")
    
    # Plot test result
    plot_single_model_forecasts_fn(best_model_name, best_model_test_forecast, output_dir)   
    print("Finished plotting best model's results")
    print("Finished pipeline")

    
 ####  Run Pipeline
def run():    
    """Entry point to run the forecasting pipeline with auto-detected config paths."""   
    pipeline_fn(experiment_name='ts_forecasting', enable_tracking=True)