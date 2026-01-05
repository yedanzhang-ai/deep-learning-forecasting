import pandas as pd
import numpy as np

from category_encoders import TargetEncoder

from darts import TimeSeries, concatenate

from sklearn.preprocessing import RobustScaler, LabelEncoder
from darts.dataprocessing.transformers import Scaler

import warnings
warnings.filterwarnings("ignore")

#### Load Processed Data
def load_processed_data_fn(
    file_path,
    id_var,
    target_var,
    static_cov_vars,
    dt_var,
    event_cov_vars,
    num_cov_var,
    bool_cov_var,
    ): 
    """Load CSV and cast columns to appropriate dtypes."""
    
    df = pd.read_csv(file_path)
    
    df[dt_var] = pd.to_datetime(df[dt_var])
    df[bool_cov_var] = df[bool_cov_var].astype('bool')
    
    for col in [id_var] + static_cov_vars + event_cov_vars:
        df[col] = df[col].astype('category')

    for col in [target_var] + [num_cov_var]:
        df[col] = df[col].astype('float')

    df = df.sort_values([id_var, dt_var])

    return df  


#### Split Data
def split_data_fn(df, dt_var, forecast_horizon):
    """Split data into train/valid/test sets based on time."""
    
    n = df[dt_var].nunique()
    train_end = int(max(1, min(n * 0.85, n - forecast_horizon * 15)))
    valid_end = int(max(1, max(n * 0.95, n - forecast_horizon * 10)))
    train_end_week = df[dt_var].min() + pd.Timedelta(weeks=train_end)
    valid_end_week = df[dt_var].min() + pd.Timedelta(weeks=valid_end)
    
    df_train = df[df[dt_var]<=train_end_week].copy()
    df_valid   = df[(df[dt_var]>train_end_week) & (df[dt_var]<=valid_end_week)].copy()
    df_test  = df[df[dt_var]>valid_end_week].copy()

    return df_train, df_valid, df_test

#### Encode Events Covariates
def encode_event_covs_fn(df_train, df_valid, df_test, target_var, event_cov_vars):
    """Target-encode categorical event covariates."""
    
    encoder = TargetEncoder(
        cols = event_cov_vars,
        smoothing=10,
        handle_unknown='value',
        handle_missing='value'
    )

    encoder.fit(df_train, df_train[target_var])

    # Transform (replaces original columns)
    df_train = encoder.transform(df_train)
    df_valid = encoder.transform(df_valid)
    df_test = encoder.transform(df_test)

    return encoder, df_train, df_valid, df_test

#### Encode Static Covariates
def encode_static_covs_fn(df_train, df_valid, df_test, static_cov_vars):
    """Label-encode categorical static covariates to integers for model compatibility."""

    static_encoders = {}

    for col in static_cov_vars:
        le = LabelEncoder()
        # Fit on all unique values across all splits to ensure consistent encoding
        all_values = pd.concat([df_train[col], df_valid[col], df_test[col]]).unique()
        le.fit(all_values)

        df_train[col] = le.transform(df_train[col])
        df_valid[col] = le.transform(df_valid[col])
        df_test[col] = le.transform(df_test[col])

        static_encoders[col] = le

    return static_encoders, df_train, df_valid, df_test

#### Create Darts Time Series
def make_darts_ts_list_fn(df, id_var, dt_var, target_var, freq, dynamic_cov_vars, static_cov_vars): 
    """Convert DataFrame to lists of Darts TimeSeries for targets and covariates."""
    
    target_list = []
    cov_list    = []
    cov_with_static_list = []

    for group_id, group in df.groupby(id_var):
        group = group.sort_values(dt_var)

        target_ts = TimeSeries.from_dataframe(
            group,
            time_col=dt_var,
            value_cols=target_var,
            fill_missing_dates=True,
            freq = freq            
        ).astype(np.float32)

        static_values = group[static_cov_vars].iloc[0:1].reset_index(drop=True)
        target_ts = target_ts.with_static_covariates(static_values)
        target_list.append(target_ts)

        cov_ts = TimeSeries.from_dataframe(
            group,
            time_col = dt_var,
            value_cols = dynamic_cov_vars,
            fill_missing_dates=True,
            freq = freq
        ).astype(np.float32)
        cov_list.append(cov_ts)
      
        cov_vars = dynamic_cov_vars + static_cov_vars
        cov_ts_with_static = TimeSeries.from_dataframe(
            group,
            time_col=dt_var,
            value_cols=cov_vars,
            fill_missing_dates=True,
            freq=freq
        ).astype(np.float32)
        cov_with_static_list.append(cov_ts_with_static)
        
    return target_list, cov_list, cov_with_static_list

#### Concatenate Train and Valid Dataset with Static Covariates
def combine_train_val_with_static_fn(train_list, valid_list):
    """Concatenate train and validation series while preserving static covariates."""
    
    combined_list = []
    for train_s, val_s in zip(train_list, valid_list):
        merged = concatenate([train_s, val_s], axis=0)
        if train_s.static_covariates is not None:
            merged = merged.with_static_covariates(train_s.static_covariates)
        combined_list.append(merged)
        
    return combined_list

#### Convert Time Series Data to Float32
def _transform_and_convert_fn(scaler, ts_list):
    """Transform and convert list of TimeSeries to float32"""
    return [ts.astype(np.float32) for ts in scaler.transform(ts_list)]

#### Scale Data for DL and STAT Models
def scale_data_fn(dataset, scale_obj):
    """Apply RobustScaler to train/valid/test TimeSeries lists."""
 
    scaler = Scaler(scaler=RobustScaler())
    scaler.fit(dataset[f'{scale_obj}_train'])
    
    dataset[f'{scale_obj}_train_scaled'] = _transform_and_convert_fn(scaler, dataset[f'{scale_obj}_train'])
    dataset[f'{scale_obj}_valid_scaled'] = _transform_and_convert_fn(scaler, dataset[f'{scale_obj}_valid'])
    dataset[f'{scale_obj}_train_valid_scaled'] = _transform_and_convert_fn(scaler, dataset[f'{scale_obj}_train_valid'])

    if scale_obj.startswith('X'):
        dataset[f'{scale_obj}_test_scaled'] = _transform_and_convert_fn(scaler, dataset[f'{scale_obj}_test'])
    else:
        dataset[f'{scale_obj}_test_scaled'] = None
        
    return scaler, dataset[f'{scale_obj}_train_scaled'], dataset[f'{scale_obj}_valid_scaled'], dataset[f'{scale_obj}_train_valid_scaled'], dataset[f'{scale_obj}_test_scaled']

# Create Datasets for Models
def create_model_datasets_fn(
    df_train,
    df_valid,
    df_test,
    id_var,
    dt_var,
    target_var,
    freq,
    event_cov_vars,
    num_cov_var,
    bool_cov_var,
    static_cov_vars
    ):
    """Build dataset dict with scaled/unscaled TimeSeries for all splits."""
    
    dynamic_cov_vars = event_cov_vars + [num_cov_var] + [bool_cov_var]
    
    dataset = {}
    
    # Create Darts time series for train dataset
    dataset['y_train'], dataset['X_train'], dataset['X_with_static_train'] = make_darts_ts_list_fn(
        df_train, 
        id_var, 
        dt_var, 
        target_var, 
        freq, 
        dynamic_cov_vars,
        static_cov_vars
    ) 
    
    # Create Darts time series for valid dataset
    dataset['y_valid'], dataset['X_valid'], dataset['X_with_static_valid'] = make_darts_ts_list_fn(
        df_valid, 
        id_var, 
        dt_var, 
        target_var, 
        freq, 
        dynamic_cov_vars,
        static_cov_vars
    ) 
    
    # Create Darts time series for test dataset
    dataset['y_test'], dataset['X_test'], dataset['X_with_static_test'] = make_darts_ts_list_fn(
        df_test, 
        id_var, 
        dt_var, 
        target_var, 
        freq, 
        dynamic_cov_vars,
        static_cov_vars
    ) 
    
    # Create Darts time series for train + valid dataset and full dataset for covariates
    dataset['y_train_valid'] = combine_train_val_with_static_fn(dataset['y_train'], dataset['y_valid'])
    dataset['X_train_valid'] = [concatenate([t, v], axis=0) for t, v in zip(dataset['X_train'], dataset['X_valid'])]
    dataset['X_full'] = [concatenate([t, v], axis=0) for t, v in zip(dataset['X_train_valid'], dataset['X_test'])]
    
    dataset['X_with_static_train_valid'] = [concatenate([t, v], axis=0) for t, v in zip(dataset['X_with_static_train'], dataset['X_with_static_valid'])]
    dataset['X_with_static_full'] = [concatenate([t, v], axis=0) for t, v in zip(dataset['X_with_static_train_valid'], dataset['X_with_static_test'])]
    
    

    # Scale datasets for DL and STAT models
    y_scaler, dataset['y_train_scaled'], dataset['y_valid_scaled'], dataset['y_train_valid_scaled'], _ = scale_data_fn(dataset, scale_obj='y')
    _, dataset['X_train_scaled'], dataset['X_valid_scaled'], dataset['X_train_valid_scaled'], dataset['X_test_scaled'] = scale_data_fn(dataset, scale_obj='X')
    _, dataset['X_with_static_train_scaled'], dataset['X_with_static_valid_scaled'], dataset['X_with_static_train_valid_scaled'], dataset['X_with_static_test_scaled'] = scale_data_fn(dataset, scale_obj='X_with_static')
    
    dataset['X_full_scaled'] = [concatenate([t, v], axis=0) for t, v in zip(dataset['X_train_valid_scaled'], dataset['X_test_scaled'])]
    dataset['X_with_static_full_scaled'] = [concatenate([t, v], axis=0) for t, v in zip(dataset['X_with_static_train_valid_scaled'], dataset['X_with_static_test_scaled'])] 
    dataset['y_scaler'] = y_scaler

    return dataset
    
 # Orchestration Function
def build_dataset_from_config_fn(data_cfg, processed_data_file_path):
    """Load config, process data, and return dataset dict with forecast horizon."""   
    
    id_var = data_cfg['id_var']
    target_var = data_cfg['target_var']
    static_cov_vars = data_cfg['static_cov_vars']
    dt_var = data_cfg['dt_var']
    event_cov_vars = data_cfg['event_cov_vars']
    num_cov_var = data_cfg['num_cov_var']
    bool_cov_var = data_cfg['bool_cov_var']
    forecast_horizon = data_cfg['forecast_horizon']
    freq = data_cfg['freq']
    
    data_proc = load_processed_data_fn(
        processed_data_file_path, 
        id_var, 
        target_var,
        static_cov_vars, 
        dt_var, 
        event_cov_vars, 
        num_cov_var, 
        bool_cov_var
        )
    
    df_train, df_valid, df_test = split_data_fn(data_proc, dt_var, forecast_horizon)    
    
    encoder, df_train, df_valid, df_test = encode_event_covs_fn(
        df_train,
        df_valid,
        df_test,
        target_var,
        event_cov_vars
        )

    static_encoders, df_train, df_valid, df_test = encode_static_covs_fn(
        df_train,
        df_valid,
        df_test,
        static_cov_vars
        )

    dataset = create_model_datasets_fn(
        df_train, 
        df_valid, 
        df_test, 
        id_var, 
        dt_var, 
        target_var, 
        freq, 
        event_cov_vars,
        num_cov_var,
        bool_cov_var, 
        static_cov_vars
        ) 
        
    # Store encoders for later decoding if needed
    dataset['static_encoders'] = static_encoders

    print("Datasets for models are ready")

    return dataset, forecast_horizon


