# Effect of model size on synthetic data experiments

import numpy as np
from utils.train_synthetic import run_synthetic_experiments

# Our imports
import torch
import chronos
import itertools

device = torch.device(
    'cuda' if torch.cuda.is_available() else 
    'mps' if torch.backends.mps.is_available() else 
    'cpu'
)
print(device)

# Got results with
for size in ['tiny', 'small', 'base', 'large']:
    
    pipeline = chronos.ChronosPipeline.from_pretrained(
        f'amazon/chronos-t5-{size}',
        device_map = device,
        torch_dtype = torch.bfloat16,
    )

    for experiment, seed in itertools.product(['static', 'time_dependent', 'sample_complexity', 'static_long'], range(5)):

        extra_path_info = f'size={size}'
        print(extra_path_info, seed, experiment)
        
        if experiment == 'static_long':
            experiment = 'static'
            horizon = 100
            save_model = True
        else:
            horizon = None
            save_model = False

        run_synthetic_experiments(
            experiment = experiment, 
            baseline = 'CHRONOS_CONFORMAL',
            n_train = 2000,
            retrain_auxiliary = False,
            recompute_dataset = True,
            save_model = False, # True
            save_results =True,
            rnn_mode='LSTM',
            horizon=horizon,
            chronos_kwargs={
                'pipeline': pipeline,
                'pred_kwargs':{
                    # 'num_samples': num_samples,
                    # 'temperature': temperature,
                    'limit_prediction_length': False
                },
            },
            extra_path_info = extra_path_info,
            seed = seed
        )

print('done')