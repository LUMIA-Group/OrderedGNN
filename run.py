from torch.utils.tensorboard import SummaryWriter
import torch
import importlib
import os
import random
import numpy as np
import wandb
import time
import json
import hashlib
import tempfile
import sys
import traceback
import shutil

def count_sweep(mode, id):
    # mode: space, now
    api = wandb.Api()

    entity,project = os.getenv('WANDB_entity'),os.getenv('WANDB_project')
    sweep = api.sweep('%s/%s/%s'%(entity, project, id))
    
    if mode=='space':
        cnt = 1
        params= sweep.config['parameters']
        for key in params.keys():
            if 'values' in params[key].keys():
                cnt *= len(params[key]['values'])
    else:
        cnt = len(sweep.runs)
    return cnt

def get_hash(dict_in, hash_keys, ignore_keys):
    dict_in = {k:v for k,v in dict_in.items() if k in hash_keys}
    dict_in = {k:v for k,v in dict_in.items() if k not in ignore_keys}
    hash_out = hashlib.blake2b(json.dumps(dict_in, sort_keys=True).encode(), digest_size=4).hexdigest()
    return str(hash_out)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def runner(wandb_config, params_default):
    # folder_temp = tempfile.TemporaryDirectory(dir="remote")
    # tmpdirname = folder_temp.name
    # os.chmod(tmpdirname, 0o777)

    tmpdirname = "remote"

    wandb_config['dir'] = tmpdirname
    wandb.init(**wandb_config, config=params_default)

    try:
        wandb_run_id = str(wandb.run.id)
        wandb.config.update({'params_hash':get_hash(wandb.config, wandb.config['hash_keys'], wandb.config['ignore_keys'])}, allow_val_change=True)
        
        params = dict(wandb.config)
        print("This trial's parameters: %s"%(params))

        if params['save_model']==True:
            checkpoint_path = os.path.join(params['wandb_log_code_path'], 'outputs', params['task'], wandb.run.id, 'checkpoint')
            os.makedirs(checkpoint_path)
        tensorboard_path = os.path.join(wandb.run.dir, 'tensorboard')
        os.mkdir(tensorboard_path)

        get_trainer = importlib.import_module('task_node').get_trainer
        get_metric = importlib.import_module('task_node').get_metric

        seed = params['seed']
        if seed!="None":
            setup_seed(seed)

        trainer = get_trainer(params)
        writer = SummaryWriter(log_dir=tensorboard_path)

        bad_cnt = 0
        best_test_metric = 0
        best_val_metric = 0
        
        time_all = []

        for epoch in range(params['epochs']):
            
            start_time = time.time()
            metrics = get_metric(trainer=trainer, stage='train')
            end_time = time.time()
            time_consumed = end_time-start_time
            time_all.append(time_consumed)

            train_metric, train_loss, train_encode_values = metrics['metric'], metrics['loss'], metrics['encode_values']
            metrics = get_metric(trainer=trainer, stage='val')
            val_metric, val_loss, val_encode_values = metrics['metric'], metrics['loss'], metrics['encode_values']
            metrics = get_metric(trainer=trainer, stage='test')
            test_metric, test_loss, test_encode_values = metrics['metric'], metrics['loss'], metrics['encode_values']

            if epoch%params['log_freq']==0:
                writer.add_scalar('time/train', time_consumed, epoch)

                writer.add_scalar('metric/train', train_metric, epoch)
                writer.add_scalar('metric/val', val_metric, epoch)
                writer.add_scalar('metric/test', test_metric, epoch)
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar('loss/test', test_loss, epoch)
                wandb.log({'metric/train':train_metric, 'metric/val':val_metric, 'metric/test':test_metric, 'loss/train':train_loss, 'loss/val':val_loss, 'loss/test':test_loss, 'metric/best':best_test_metric})

            if val_metric>best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
                bad_cnt = 0
                if params['save_model']==True:
                    torch.save(trainer['model'].state_dict(), os.path.join(checkpoint_path,'model.pt'))
                    json.dump(trainer['params'], open(os.path.join(checkpoint_path,'model_config.json'), 'w'))
            else:
                bad_cnt += 1

            if bad_cnt==params['patience']:
                break
        
        print('Final metric is [%s]'%(best_test_metric))
        writer.close()
        wandb.run.summary["metric/final"] = best_test_metric
        wandb.run.summary["time/avg"] = sum(time_all)/len(time_all) if len(time_all)!=0 else 0

    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
    
    wandb.finish()

    # shutil.rmtree(tmpdirname)

    return wandb_run_id

def agent(q, gpu_id, sweep_id, wandb_config, params_default, agent_package):
    params_default['gpu_index'] = gpu_id
    print('Agent started with GPU [%s]'%(gpu_id))
    try:
        entity,project = os.getenv('WANDB_entity'),os.getenv('WANDB_project')
        wandb.agent(sweep_id, function=lambda:runner(wandb_config, params_default), entity=entity, project=project, count=agent_package)
    except:
        time.sleep(10)
        print('Some error with this agent, skip')
    q.put(gpu_id)
    print('Agent finished and release GPU [%s]'%(gpu_id))