from DSIPTS.dsipts import RNN, Autoformer
# Configure RNN model
def define_RNN(ts, past_steps, future_steps, use_quantiles=False):
    config = dict(
    model_configs=dict(
        cat_emb_dim=16,
        kind='gru', 
        hidden_RNN=128,  # Hidden size of the GRU
        num_layers_RNN=2,  # Number of GRU layers
        sum_emb=True,
        kernel_size=15, 
        past_steps=past_steps, 
        future_steps=future_steps, 
        past_channels=len(ts.num_var),  # Number of numerical variables
        future_channels=len(ts.future_variables),  # Number of future variables
        embs=[ts.dataset[c].nunique() for c in ts.cat_var],  # Unique categories for embeddings
        quantiles=[0.1, 0.5, 0.9] if use_quantiles else [],  # Removed quantiles for binary classification
        dropout_rate=0.2,  # Dropout for regularization
        loss_type='BCEWithLogitsLoss',  # Binary classification loss
        remove_last=True, 
        use_bn=False,  # Batch normalization (off)
        optim='torch.optim.AdamW',  # Optimizer
        activation='torch.nn.LeakyReLU',  # Activation
        out_channels=1  # Single output channel for binary classification
    ),
    scheduler_config=dict(
        gamma=0.05,  # Learning rate decay factor
        step_size=20000000  # Decay step size
    ),
    optim_config=dict(
        lr=0.0001,  # Learning rate
        weight_decay=0.0001  # Regularization (weight decay)
    )
)


    ts.set_model(
        RNN(**config['model_configs'], optim_config=config['optim_config'], scheduler_config=config['scheduler_config'], verbose=True),
        config=config)
    return ts

# Configure Autoformer model
def define_Autoformer(ts, past_steps, future_steps, use_quantiles=True):
    config = dict(
        model_configs=dict(
            past_steps=past_steps,  # Number of past data points
            future_steps=future_steps,  # Number of steps to predict into the future
            label_len=0,#past_steps // 2,  # Overlap length, typically half of past_steps
            past_channels=len(ts.num_var)-1,  # Number of past numerical variables
            future_channels=len(ts.future_variables),  # Number of future variables
            out_channels=1,  # Single output channel for regression
            d_model=128,  # Dimension of the attention model
            embs=[ts.dataset[c].nunique() for c in ts.cat_var],  # Embeddings for categorical variables
            kernel_size=15,  # Kernel size for convolution
            activation='torch.nn.ReLU',  # Activation function
            factor=5,  # Factor for TSA stage (paper detail)
            n_head=4,  # Number of attention heads
            n_layer_encoder=3,  # Number of encoder layers
            n_layer_decoder=3,  # Number of decoder layers
            hidden_size=512,  # Hidden size for feed-forward layers
            persistence_weight=0.0,  # Persistence weight (0 = no divergence)
            loss_type='mse',  # Use MSE loss for regression
            quantiles=[0.1, 0.5, 0.9] if use_quantiles else [],  # No quantiles for regression; add for probabilistic outputs
            dropout_rate=0.1,  # Dropout rate for regularization
            optim='torch.optim.AdamW',  # Optimizer
            ),
            scheduler_config=dict(
                gamma=0.1,  # Learning rate decay factor
                step_size=10000  # Decay step size
            ),
            optim_config=dict(
                lr=0.0001,  # Learning rate
                weight_decay=0.0001  # Regularization (weight decay)
            )
        )


    ts.set_model(
        Autoformer(**config['model_configs'], optim_config=config['optim_config'], scheduler_config=config['scheduler_config'], verbose=True),
        config=config)
    return ts
