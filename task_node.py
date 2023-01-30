import os
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Actor, WebKB, WikipediaNetwork

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from datasets.datasets_linkx.dataset import load_nc_dataset
from torch_geometric.data import Data
from torch_sparse import SparseTensor

def get_trainer(params):
    dataset_name = params['task']
    split = params['index_split']

    if dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='datasets/datasets_pyg/', geom_gcn_preprocess=True, name=dataset_name, transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        data.x = torch.eye(data.x.shape[0])
        data.adj_t = data.adj_t.t()
        params['in_channel']=data.x.shape[0]
        params['out_channel']=dataset.num_classes

    if dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='datasets/datasets_pyg/', name=dataset_name, transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes
    
    if dataset_name in ['Actor']:
        dataset = Actor(root='datasets/datasets_pyg/Actor', transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes
    
    if dataset_name in ['Cora_full','CiteSeer_full','PubMed_full']:
        dataset = Planetoid(root='datasets/datasets_pyg/', name='%s'%(dataset_name.split('_')[0]), split=dataset_name.split('_')[-1], transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes

    if dataset_name in ['Cora_geom','CiteSeer_geom','PubMed_geom']:
        dataset = Planetoid(root='datasets/datasets_pyg/', name='%s'%(dataset_name.split('_')[0]), split='public', transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        split_str = "%s_split_0.6_0.2_%s.npz"%(dataset_name.split('_')[0].lower(), str(split))
        split_file = np.load(os.path.join('datasets/datasets_geomgcn/', split_str))
        data = dataset[0]
        data.train_mask = torch.Tensor(split_file['train_mask'])==1
        data.val_mask = torch.Tensor(split_file['val_mask'])==1
        data.test_mask = torch.Tensor(split_file['test_mask'])==1
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes
    
    if dataset_name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(root='datasets/datasets_pyg/', name='ogbn-arxiv', transform=T.ToSparseTensor())
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name='ogbn-arxiv')
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes
    
    if dataset_name in ['arxiv-year']:
        dataset = load_nc_dataset(dataset_name, sub_dataname='')
        edge_index = dataset[0][0]['edge_index']
        num_nodes = dataset[0][0]['num_nodes']
        data = Data(x=dataset[0][0]['node_feat'], adj_t=SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)), y=dataset[0][1].view(-1), num_nodes=dataset[0][0]['num_nodes'])
        splits = np.load('datasets/datasets_linkx/data/arxiv-year-splits.npy', allow_pickle=True)
        sizes = (data.num_nodes, len(splits))
        data.train_mask = torch.zeros(sizes, dtype=torch.bool)
        data.val_mask = torch.zeros(sizes, dtype=torch.bool)
        data.test_mask = torch.zeros(sizes, dtype=torch.bool)
        for i, split_temp in enumerate(splits):
            data.train_mask[:, i][torch.tensor(split_temp['train'])] = True
            data.val_mask[:, i][torch.tensor(split_temp['valid'])] = True
            data.test_mask[:, i][torch.tensor(split_temp['test'])] = True
        data.train_mask = data.train_mask[:,int(split)]
        data.val_mask = data.val_mask[:,int(split)]
        data.test_mask = data.test_mask[:,int(split)]
        params['in_channel']=data.num_features
        params['out_channel']=5

    device = torch.device('cuda:%s'%(params['gpu_index']) if torch.cuda.is_available() else 'cpu')
    print("GPU device: [%s]"%(device))
    
    if params['model'] in ['ONGNN']:
        from model import GONN as Encoder
        model = Encoder(params).to(device)
    elif params['model']=='GAT':
        from baseline import GAT as Encoder
        model = Encoder(params).to(device)

    criterion = torch.nn.NLLLoss()
    
    if params['weight_decay2']=="None":
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    else:
        optimizer = torch.optim.Adam([dict(params=model.params_conv, weight_decay=params['weight_decay']),
                                dict(params=model.params_others, weight_decay=params['weight_decay2'])],
                                lr=params['learning_rate'])

    if dataset_name in ['ogbn-arxiv']:
        trainer = dict(zip(['data', 'device', 'model', 'criterion', 'optimizer', 'split_idx', 'evaluator', 'params'], [data, device, model, criterion, optimizer, split_idx, evaluator, params]))
    else:
        trainer = dict(zip(['data', 'device', 'model', 'criterion', 'optimizer', 'params'], [data, device, model, criterion, optimizer, params]))

    return trainer

def get_metric(trainer, stage):
    if trainer['params']['task'] in ['ogbn-arxiv']:
        data, device, model, criterion, optimizer, split_idx, evaluator, params= trainer.values()
    else:
        data, device, model, criterion, optimizer, params = trainer.values()

    if stage=='train':
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()
    
    data = data.to(device)

    if params['task']=='ogbn-arxiv':
        mask = split_idx['valid'] if stage=='val' else split_idx[stage]
        mask = mask.to(device)
        encode_values = model(data.x, data.adj_t)
        vec = encode_values['x']
        pred = F.log_softmax(vec, dim=-1)
        loss = criterion(pred[mask], data.y.squeeze(1)[mask])
    else:
        for _, mask_tensor in data(stage+'_mask'):
            mask = mask_tensor
        encode_values = model(data.x, data.adj_t)
        vec = encode_values['x']
        pred = F.log_softmax(vec, dim=-1)
        loss = criterion(pred[mask], data.y[mask])
    
    if stage=='train':
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if params['task']=='ogbn-arxiv':
        y_pred = vec.argmax(dim=-1, keepdim=True)
        acc = evaluator.eval({
            'y_true': data.y[mask],
            'y_pred': y_pred[mask],
        })['acc']
    else:
        acc = float((pred[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())

    metrics = dict(zip(['metric', 'loss', 'encode_values'], [acc, loss.item(), encode_values]))

    return metrics