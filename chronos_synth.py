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

def size_effect():
    for size in ['tiny', 'small', 'base', 'large']:
        
        pipeline = chronos.ChronosPipeline.from_pretrained(
            f'amazon/chronos-t5-{size}',
            device_map = device,
            torch_dtype = torch.bfloat16,
        )

        for experiment, seed in itertools.product(['static', 'time_dependent', 'sample_complexity', 'static_long'], range(5)):

            extra_path_info = f'size_effect_size={size}'
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
                save_model = save_model,
                save_results = True,
                rnn_mode ='LSTM',
                horizon = horizon,
                chronos_kwargs = {
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


def explore_hyper():

    pipeline = chronos.ChronosPipeline.from_pretrained(
        f'amazon/chronos-t5-base',
        device_map = device,
        torch_dtype = torch.bfloat16,
    )

    for num_samples, temperature, n_beams, rag, experiment, seed in itertools.product(
        [1, 5, 20], [None, 0.1, 0.5], [1, 5, 10], [True, False],
        ['static', 'time_dependent'], range(5)):

        extra_path_info = f'explore_hyper_num-samples={num_samples}_temperature={temperature}_n-beams={n_beams}_rag={rag}'
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
            save_model = save_model,
            save_results = True,
            rnn_mode ='LSTM',
            horizon = horizon,
            chronos_kwargs = {
                'pipeline': pipeline,
                'pred_kwargs':{
                    'num_samples': num_samples,
                    'temperature': temperature,
                    'n_beams': n_beams,
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

    experiments = {f.__name__:f for f in [size_effect, explore_hyper]}
    assert args.experiment in experiments

    print(f'Running {args.experiment} {experiments[args.experiment]}')
    experiments[args.experiment]()