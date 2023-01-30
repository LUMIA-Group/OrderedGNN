import time
import wandb
import yaml
import os
from run import count_sweep
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='agent sweep id and gpu index')
    parser.add_argument('--sweep_file', type=str)
    args = parser.parse_args()

    params_config = yaml.load(open('%s'%(str(args.sweep_file))).read(), Loader=yaml.Loader)
    search_space = {}
    hash_keys = []
    ignore_keys = []
    for k,v in params_config.items():
        if len(v)>1:
            search_space[k] = {"values":v}
            hash_keys.append(k)
        else:
            search_space[k] = {"value":v[0]}
    ignore_keys.append('index_split')
    ignore_keys.append('index_runs')
    ignore_keys.append('seed')
    search_space['hash_keys'] = {"value":hash_keys}
    search_space['ignore_keys'] = {"value":ignore_keys}

    entity,project = os.getenv('WANDB_entity'),os.getenv('WANDB_project')
    sweep_config =  {}
    sweep_config['method'] = 'grid'
    sweep_config['metric'] = {'name':'metric/final', 'goal':'maximize'}

    sweep_config['parameters'] = search_space
    sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
    time.sleep(3)
    print('Start new sweep [%s].'%(sweep_id))
    print('Sweep search space size: [%s]'%(count_sweep(mode='space', id=sweep_id)))