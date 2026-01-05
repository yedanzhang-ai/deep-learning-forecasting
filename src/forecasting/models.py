
from darts.models import (
    # Deep Learning models
    RNNModel,
    TCNModel,
    NBEATSModel,
    TFTModel,
    TiDEModel,
    # Machine Learning models
    CatBoostModel,
    XGBModel,
    LightGBMModel,  
)

#### Add Model Encoders
def _add_model_encoders(model_type, calendar_features, cov_support_type):
    """Build encoder dict for calendar features based on covariate type."""
    
    if model_type == 'dl':
        add_model_encoders={"datetime_attribute": {cov_support_type:calendar_features["date_features"]},
                            "cyclic": {cov_support_type: calendar_features["cyclic_features"]},
                            "position": {cov_support_type: "relative"}
                           }
    else:        
        add_model_encoders={"cyclic": {cov_support_type: calendar_features["cyclic_features"]}}

    return add_model_encoders

#### Set Up Trainer for DL Models
def _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu):
    """Build PyTorch Lightning trainer kwargs for DL models."""
    
    kwargs = {
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": 1,
        "precision": "32-true",
        "enable_progress_bar": False,
        "logger": False,
        "enable_model_summary": False,
        "enable_checkpointing": False,
        "max_epochs": n_epochs,

    }

    # Only add callbacks if early_stop is provided
    if early_stop is not None:
        kwargs["callbacks"] = [early_stop]

    return kwargs

#### Build Specific DL Models
def build_gru_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts RNNModel with GRU architecture."""
    
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    add_encoders = _add_model_encoders('dl', calendar_features, cov_support_type)

    kwargs = dict(
        model="GRU",
        input_chunk_length=hp["lookback"],
        output_chunk_length=1,
        training_length=hp["lookback"] + forecast_horizon * 2,
        n_rnn_layers=hp['layers'],
        hidden_dim=hp["hidden"],
        dropout=hp["dropout"],
        batch_size=hp["batch"],
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return RNNModel(n_epochs=n_epochs, add_encoders=add_encoders, **kwargs)


def build_lstm_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts RNNModel with LSTM architecture."""
    
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    add_encoders = _add_model_encoders('dl', calendar_features, cov_support_type)

    kwargs = dict(
        model="LSTM",
        input_chunk_length=hp["lookback"],
        output_chunk_length=1,
        training_length=hp["lookback"] + forecast_horizon * 2,
        n_rnn_layers=hp['layers'],
        hidden_dim=hp["hidden"],
        dropout=hp["dropout"],
        batch_size=hp["batch"],
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return RNNModel(n_epochs=n_epochs, add_encoders=add_encoders, **kwargs)


def build_nbeats_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts N-BEATS model (univariate - no covariates)."""

    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    
    kwargs = dict(
        input_chunk_length=hp["lookback"],
        output_chunk_length=forecast_horizon,
        generic_architecture=True,
        num_stacks=hp["stacks"],
        num_blocks=hp["blocks"],
        layer_widths=hp["width"],
        dropout=hp["dropout"],
        batch_size=hp["batch"],
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return NBEATSModel(n_epochs=n_epochs, **kwargs)

def build_tcn_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts TCN (Temporal Convolutional Network) model."""
    
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    add_encoders = _add_model_encoders('dl', calendar_features, cov_support_type)

    kwargs = dict(
        input_chunk_length=hp["lookback"],
        output_chunk_length=forecast_horizon,
        num_filters=hp["filters"],
        kernel_size=hp["kernel"],
        dilation_base=hp["dilation_base"],
        dropout=hp["dropout"],
        weight_norm=False,
        batch_size=hp["batch"],
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },

        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return TCNModel(n_epochs=n_epochs, add_encoders=add_encoders, **kwargs)

def build_tft_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts TFT (Temporal Fusion Transformer) model."""
    
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    add_encoders = _add_model_encoders('dl', calendar_features, cov_support_type)

    kwargs = dict(
        input_chunk_length=hp["lookback"],
        output_chunk_length=forecast_horizon,
        hidden_size=hp['hidden'],
        lstm_layers=hp["lstm_layers"],
        num_attention_heads=hp["heads"],
        add_relative_index=True,
        dropout=hp["dropout"],
        batch_size=hp["batch"],
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },
        use_static_covariates=True,
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return TFTModel(n_epochs=n_epochs, add_encoders=add_encoders, **kwargs)


def build_tide_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts TiDE model."""
    
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    add_encoders = _add_model_encoders('dl', calendar_features, cov_support_type)

    kwargs = dict(
        input_chunk_length=hp["lookback"],
        output_chunk_length=forecast_horizon,
        num_encoder_layers=hp["encoder_layers"],
        num_decoder_layers=hp["decoder_layers"],
        decoder_output_dim=hp["decoder_dim"],
        hidden_size=hp["hidden"],
        temporal_width_past=hp["temporal_width"],
        temporal_width_future=hp["temporal_width"],
        dropout=hp["dropout"],
        batch_size=hp["batch"],
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },
        use_static_covariates=True,
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return TiDEModel(n_epochs=n_epochs, add_encoders=add_encoders, **kwargs)

#### Convert Lag Parameters to List for ML Models
def _to_lag_list(lag_value, include_zero=False):
    """Convert lag integer to list of lag indices for Darts ML models."""
    
    # Convert int to list: e.g., 4 -> [-4, -3, -2, -1] or [-4, -3, -2, -1, 0]
    if isinstance(lag_value, (list, tuple)):
        return [-x for x in lag_value]
   
    end = 1 if include_zero else 0
    
    return list(range(-lag_value, end))
    
#### Build Specific ML Models
def build_xgboost_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts XGBoost model."""
  
    lags = _to_lag_list(hp.get("lags", 12), include_zero=False)
    lags_future = _to_lag_list(hp.get("lags_future_covariates", 12), include_zero=True)

    add_encoders = _add_model_encoders('ml', calendar_features, cov_support_type)

    kwargs = dict(
        lags=lags,
        lags_future_covariates=lags_future,
        output_chunk_length=forecast_horizon,
        n_estimators=hp.get("n_estimators", 100),
        max_depth=hp.get("max_depth", 6),
        learning_rate=hp.get("learning_rate", 0.1),
        reg_alpha=hp.get("reg_alpha", 0.0),
        reg_lambda=hp.get("reg_lambda", 0.0),
        use_static_covariates=True,
        random_state=SEED      
    )

    return XGBModel(add_encoders=add_encoders, **kwargs)

def build_lightgbm_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts LightGBM model."""
   
    lags = _to_lag_list(hp.get("lags", 12), include_zero=False)
    lags_future = _to_lag_list(hp.get("lags_future_covariates", 12), include_zero=True)

    add_encoders = _add_model_encoders('ml', calendar_features, cov_support_type)

    kwargs = dict(
        lags=lags,
        lags_future_covariates=lags_future,
        output_chunk_length=forecast_horizon,
        n_estimators=hp.get("n_estimators", 100),
        max_depth=hp.get("max_depth", -1),
        learning_rate=hp.get("learning_rate", 0.1),
        num_leaves=hp.get("num_leaves", 31),
        num_boost_round=500,
        reg_alpha=hp.get("reg_alpha", 0.0),
        reg_lambda=hp.get("reg_lambda", 0.0),
        use_static_covariates=True,
        random_state=SEED,     
        verbosity=-1
    )

    return LightGBMModel(add_encoders=add_encoders, **kwargs)

def build_catboost_model_fn(hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Build a Darts CatBoost model."""
   
    lags = _to_lag_list(hp.get("lags", 12), include_zero=False)
    lags_future = _to_lag_list(hp.get("lags_future_covariates", 12), include_zero=True)

    add_encoders = _add_model_encoders('ml', calendar_features, cov_support_type)

    kwargs = dict(
        lags=lags,
        lags_future_covariates=lags_future,
        output_chunk_length=forecast_horizon,
        iterations=hp.get("iterations", 100),
        depth=hp.get("depth", 6),
        learning_rate=hp.get("learning_rate", 0.1),
        l2_leaf_reg=hp.get("l2_leaf_reg", 3.0),
        use_static_covariates=True, 
        random_state=SEED,     
        verbose=False
    )

    return CatBoostModel(add_encoders=add_encoders, **kwargs)

#### Create Generic Model Builder
model_builders = {
    # Deep Learning models
    "gru": build_gru_model_fn,
    "lstm": build_lstm_model_fn,
    "nbeats": build_nbeats_model_fn,
    "tcn": build_tcn_model_fn,
    "tide": build_tide_model_fn,
    "tft": build_tft_model_fn,
    # Machine Learning models
    "catboost": build_catboost_model_fn,
    "xgboost": build_xgboost_model_fn,
    "lightgbm": build_lightgbm_model_fn,   
}

def build_model(model_name, hp, n_epochs, cov_support_type, early_stop, forecast_horizon, SEED, use_gpu, calendar_features):
    """Factory function to build a Darts model by name."""
    
    name = model_name.lower()
    if name not in model_builders:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Available: {list(model_builders.keys())}")

    builder = model_builders[name]
    return builder(
        hp=hp,
        n_epochs=n_epochs,
        cov_support_type = cov_support_type,
        early_stop=early_stop,
        forecast_horizon=forecast_horizon,
        SEED=SEED,
        use_gpu=use_gpu,
        calendar_features=calendar_features,

    )

