# Synthetic data experiments with chronos
import numpy as np
from utils.train_synthetic import run_synthetic_experiments

# Our imports
import torch
import chronos
import itertools, argparse, os

device = torch.device(
    'cuda' if torch.cuda.is_available() else 
    'mps' if torch.backends.mps.is_available() else 
    'cpu'
)
print(device)

def main():

    pipeline = chronos.ChronosPipeline.from_pretrained(
        f'amazon/chronos-t5-base',
        device_map = device,
        torch_dtype = torch.bfloat16,
    )

    for baseline, rag, experiment, seed in itertools.product(
        ['CHRONOS_CONFORMAL', 'CHRONOS_CONFORMAL_ALL'], [True, False],
        ['static', 'time_dependent', 'sample_complexity', 'static_long'], range(5)):

        extra_path_info = f'main_rag={rag}'
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
            baseline = baseline,
            n_train = 2000,
            retrain_auxiliary = False,
            recompute_dataset = True,
            save_model = save_model,
            save_results = True,
            rnn_mode ='LSTM',
            horizon = horizon,
            chronos_kwargs = {
                'pipeline': pipeline,
                'pred_kwargs':{
                    # 'num_samples': num_samples,
                    # 'temperature': temperature,
                    'rag': rag,
                    'limit_prediction_length': False,
                },
            },
            extra_path_info = extra_path_info,
            seed = seed
        )

    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='static')
    args = parser.parse_args()

    experiments = {f.__name__:f for f in [main]}
    assert args.experiment in experiments

    print(f'Running {args.experiment} {experiments[args.experiment]}')
    experiments[args.experiment]()