
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str)
parser.add_argument('--dry-run', action='store_true')
pargs = parser.parse_args()


df = pd.read_csv(pargs.csv)
metrics = ['gain', 'ugbw', 'cost', 'tset', 'ibias', 'offset_sys', 'psrr']
models = ['mlp', 'gat']
datasizes = ['full', 'half', 'RI', '0p25', '0p1', '0p01']

for _, row in tqdm(df.iterrows()):
    exp_name = row['config/exp_name']
    exp_id = row['ID']

    metric_str = None
    for metric in metrics:
        if exp_name.startswith(metric):
            metric_str = metric
            break
    
    model_str = None
    for model in models:
        if model.upper() in exp_name:
            model_str = model
            break
    
    ds_str = None
    for ds in datasizes:
        if f'FT_{ds}' in exp_name:
            ds_str = ds
            break

    cf_path = f'configs/downstream/regression/{metric_str}/finetune/{model_str}/{ds_str}/config.py'

    base_ckpt_path = Path(f'/store/nosnap/results/cgl/cgl/{exp_id}/checkpoints')
    glob_res = list(base_ckpt_path.glob('cgl*valid_loss_total_ema*.ckpt'))
    if glob_res and (len(glob_res) == 1) and glob_res[0].is_file():
        ckpt_path = glob_res[0]
    else:
        raise ValueError(f'Could not find ckpt path at {str(base_ckpt_path)}')

    command = [
        'python', 
        'scripts/train_downstream.py', 
        '--path', cf_path,
        '--gpus', '1',
        '--train', '0',
        '--ckpt', str(ckpt_path),
        '--wandb_id', exp_id
    ]

    print(' '.join(command))
    if not pargs.dry_run:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()