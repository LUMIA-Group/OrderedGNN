import os
import wandb
from multiprocessing import Process, Queue
from run import agent, count_sweep, runner
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='agent sweep id and gpu index')
    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--gpu_allocate', type=str)
    args = parser.parse_args()
    sweep_id = args.sweep_id
    list_gpu_id = sum([[int(item.split(':')[0]) for i in range(int(item.split(':')[1]))] for item in args.gpu_allocate.split('-')], [])
    print('GPU allocate: [%s]'%(list_gpu_id))

    api = wandb.Api()
    entity,project = os.getenv('WANDB_entity'),os.getenv('WANDB_project')
    sweep = api.sweep('%s/%s/%s'%(entity, project, sweep_id))
    params_default = {}
    params_default['wandb_log_code_path'] = os.getcwd()
    
    params_config = sweep.config['parameters']

    wandb_config = {}
    wandb_config['entity'] = entity
    wandb_config['project'] = project
    wandb_config['reinit'] = True
    wandb_config['group'] = sweep_id
    agent_package = params_config['agent_package']['value']

    os.environ["WANDB_START_METHOD"] = "thread"
    q = Queue()
    for gpu_id in list_gpu_id:
        q.put(gpu_id)
    while True:
        num_space = count_sweep(mode='space', id=sweep_id)
        num_now = count_sweep(mode='now', id=sweep_id)
        if num_now<num_space:
            gpu_id = q.get()
            p = Process(target=agent, args=(q, gpu_id, sweep_id, wandb_config, params_default, agent_package, ))
            p.start()
        else:
            print('Sweep is done, do not need more agent')
            break

    # for debug
    # params_default['gpu_index'] = list_gpu_id[0]
    # for key in params_config.keys():
    #     if 'value' in params_config[key].keys():
    #         params_default[key] = params_config[key]['value']
    #     else:
    #         params_default[key] = params_config[key]['values'][0]
    # runner(wandb_config, params_default)
    
    print('Search finished')