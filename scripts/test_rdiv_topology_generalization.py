

# load the config file and the model
# load weights on to the model
# run validation error checker
# run test on new topology
#   create new topology graphs (locally first and then in another repo)

# Note: methodologically generate diverse topologies for rdiv


from collections import defaultdict
from pathlib import Path
import argparse
import imp
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from tqdm import tqdm 

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx

import pytorch_lightning as pl

# This should be imported since the checkpoint loader needs it to reconstruct the state_dict
from scripts.pretrain import ModelCheckpointNoOverride
from cgl.utils.pdb import register_pdb_hook
from cgl.utils.params import ParamDict

from rdiv.gen_test_data import get_dataset
from cgl.eval.evaluator import NodeEvaluator


register_pdb_hook()

class Tester:

    def __init__(self, pargs) -> None:
        
        pl.seed_everything(10)
        self.pargs = pargs
        conf = self.get_conf(pargs)
        model = conf.model

        ckpt_path = pargs.ckpt

        # test_dset = get_dataset(1000)
        # tdloader = DataLoader(test_dset, batch_size=100, drop_last=False, num_workers=0)
        tdloader = None

        evaluator = NodeEvaluator(bins=conf.mdl_conf.bins)

        if not ckpt_path:
            raise ValueError('Checkpoint path is not given.')
        else:
            print(f'Loading the checkpoint {Path(ckpt_path).absolute()} ...')
            model = model.load_from_checkpoint(ckpt_path).to(device=model.device, dtype=model.dtype)
            print('Checkpoint Loaded.')
            # HACK
            model.config = conf.mdl_conf

            # loaders = dict(valid=conf.valid_dloader, test=tdloader)
            loaders = dict(train=conf.train_dloader, valid=conf.valid_dloader, test=conf.test_dloader)#, valid=conf.valid_dloader, test=tdloader)
            acc_dict = defaultdict(lambda: [])
            results_loader = {}
            for mode, loader in loaders.items():
                if loader is None:
                    continue
                results_batch = ParamDict({'vdc_pred': [], 'vdc_target': [], 'graph_id': [], 'graph_list': []})
                for batch in tqdm(loader):
                    device = torch.device(pargs.device)
                    model = model.to(device)
                    if isinstance(batch, dict):
                        batch = ParamDict({k: v.to(device) for k, v in batch.items()})
                    else:
                        batch = batch.to(device)
                    results = model.predict(batch, compute_loss=True)
                    
                    mask = results.input.data.output_node_mask
                    batch_id = results.input.data.batch[mask]

                    results_batch['graph_id'].append(batch_id.cpu().numpy() + len(results_batch['graph_list']))
                    results_batch['vdc_pred'].append(results.output.vdc.cpu().numpy())
                    results_batch['vdc_target'].append(results.input.data.vdc.cpu().numpy())

                    results_batch['graph_list'] += results.input.data.to_data_list()
                    acc_dict[mode].append(results['eval']['vdc_acc'])
                
                results_loader[mode] = ParamDict()
                for k, v in results_batch.items():
                    if k not in  ['graph_list']:
                        results_loader[mode][k] = np.concatenate(v, axis=0)
                    else:
                        results_loader[mode][k] = v
            

            train_dict = {'y_true': results_loader['train'].vdc_target, 'y_pred': results_loader['train'].vdc_pred}
            labels = evaluator.eval(train_dict, return_cond=True).flatten()
            print(f'Train Label accuracy: {labels.sum() / labels.flatten().shape[0]}')
            
            for k in acc_dict:
                acc_dict[k] = np.mean(acc_dict[k])
            acc_dict = dict(acc_dict)

            # pprint(acc_dict)
            print(f'{acc_dict["train"]:.4f}/{acc_dict["valid"]:.4f}/{acc_dict["test"]:.4f}')


            #################################### Plotting pre-training insights

            # ## Plot histogram of target and predicted values and their distribution over wrong data points
            # min_val, max_val = min(results_loader['train'].vdc_target), max(results_loader['train'].vdc_target)
            # min_val, max_val = float(min_val), float(max_val)
            
            # plt.close()
            # _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            # ax1.hist(results_loader['train'].vdc_target.flatten(), range=[min_val, max_val], bins=conf.mdl_conf.bins)
            # ax1.set_title('Histogram of the ground truth for all data points')

            # ax2.hist(results_loader['train'].vdc_target[~labels].flatten(), range=[min_val, max_val], bins=conf.mdl_conf.bins)
            # ax2.hist(results_loader['train'].vdc_pred[~labels].flatten(), range=[min_val, max_val], bins=conf.mdl_conf.bins, color='orange', alpha=0.2)
            # ax2.set_title('Histogram of the ground truth for False data points')

            # plt.savefig('rladder_assets/train_hist_neg.png')
            # # plt.savefig('opamp_train_hist_neg.png') # testing the opamp results

            # ## Plot histogram of distance from ground trough for wrong data points
            # plt.close()

            # _, ax = plt.subplots(1, 1, figsize=(8, 6))
            # threshold = (max_val - min_val) / conf.mdl_conf.bins
            # norm_dist = (results_loader['train'].vdc_pred - results_loader['train'].vdc_target) / threshold

            # bins = conf.mdl_conf.bins
            # min_range, max_range = -bins/2, bins/2
            # ax.hist(norm_dist[~labels].flatten(), range=[min_range, max_range], bins=bins)
            # ax.set_title('Histogram of normalized distance for False data points')

            # plt.savefig('rladder_assets/train_hist_dist_neg.png')

            # ## For those graphs that have False predictions, what are the False nodes, what is the # nodes, and the percentage of False nodes?
            # graph_to_false_node_map = defaultdict(lambda: [])
            # for node_idx, graph_idx in enumerate(results_loader['train']['graph_id'][~labels]):
            #     graph_to_false_node_map[graph_idx].append(node_idx)
            
            # n_branch_map = {x: 0 for x in range(11, 12)}
            # for graph_idx in graph_to_false_node_map:
            #     graph_data = results_loader['train']['graph_list'][graph_idx]
            #     # this relationship is true only for rladder
            #     n_branch = (len(graph_data.vdc) - 1) // 2
            #     n_branch_map[n_branch] += 1

            # plt.close()
            # plt.pie(list(n_branch_map.values()), labels=n_branch_map.keys())
            # plt.savefig('rladder_assets/train_false_graph_pie.png')


            #################################### Plotting Validation set insights

            # result_key = 'train'
            # for node_id in [0, 1, 2]:
            #     vdc_target = results_loader[result_key].vdc_target[node_id::3]
            #     vdc_pred = results_loader[result_key].vdc_pred[node_id::3]
            #     valid_dict = {'y_true': vdc_target, 'y_pred': vdc_pred}
            #     labels = evaluator.eval(valid_dict, return_cond=True).flatten()
            #     acc = labels.sum() / labels.flatten().shape[0]
            #     if node_id == 1:
            #         print(f'[{node_id}] {result_key} Label accuracy: {acc}')
            #         min_val, max_val = min(vdc_target), max(vdc_target)
            #         min_val, max_val = float(min_val), float(max_val)

            #         plt.close()
            #         ax = plt.gca()
            #         ax.scatter(vdc_target.flatten(), vdc_pred.flatten(), s=5)
            #         ax.set_ylabel('Prediciton')
            #         ax.set_xlabel('Target')
            #         ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), linestyle='--')
            #         plt.savefig(f'rladder_assets/{result_key}_pred_vs_target.png')

                    
                    
            #         plt.close()
            #         _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            #         ax1.hist(vdc_target.flatten(), range=[min_val, max_val], bins=conf.mdl_conf.bins)
            #         ax1.set_title('Histogram of the ground truth for all data points')

            #         ax2.hist(vdc_target[~labels].flatten(), range=[min_val, max_val], bins=conf.mdl_conf.bins)
            #         ax2.hist(vdc_pred[~labels].flatten(), range=[min_val, max_val], bins=conf.mdl_conf.bins, color='orange', alpha=0.2)
            #         ax2.set_title('Histogram of the ground truth for False data points')

            #         plt.savefig(f'rladder_assets/{result_key}_hist_neg_node_{node_id}.png')

            #         ## Plot histogram of distance from ground trough for wrong data points
            #         plt.close()

            #         _, ax = plt.subplots(1, 1, figsize=(8, 6))
            #         threshold = (max_val - min_val) / conf.mdl_conf.bins
            #         norm_dist = (vdc_pred - vdc_target) / threshold

            #         bins = conf.mdl_conf.bins
            #         min_range, max_range = -bins/2, bins/2
            #         ax.hist(norm_dist[~labels].flatten(), range=[min_range, max_range], bins=bins)
            #         ax.set_title('Histogram of normalized distance for False data points')

            #         plt.savefig(f'rladder_assets/{result_key}_hist_dist_neg_node_{node_id}.png')


        
    def get_conf(self, pargs):
        conf_path = pargs.path
        print(f'Loading from the config file {conf_path}')
        conf_module = imp.load_source('conf', conf_path)

        conf = ParamDict(
            exp_name=conf_module.exp_name, 
            model=conf_module.model,
            train_dloader=conf_module.train_dloader,
            valid_dloader=conf_module.valid_dloader,
            test_dloader=conf_module.test_dloader,
            mdl_conf=conf_module.mdl_config,
        )

        return conf

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='config file path')
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--output', type=str, default='/store/nosnap/results/cgl', 
    #                     help='The output directory')
    # parser.add_argument('--max_epochs', type=int, default=None, 
    #                     help='The maximum number of training epochs (if earlier than max_steps)')
    # parser.add_argument('--max_steps', type=int, default=None, 
    #                     help='The maximum number of training steps (if earlier than max_epochs)')
    parser.add_argument('--ckpt', type=str, help='Resume from this checkpoint if valid.')
    # parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # parser.add_argument('--project', type=str, default='cgl-PT', help='project name for wandb')
    # parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    # parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')
    # parser.add_argument('--profile', type=int, default=0, help='Set to 1 to profile individual steps during training')
    # parser.add_argument('--test', type=int, default=1, help='Set to 0 to skip running the test scripts')
    # parser.add_argument('--log_freq', type=int, default=10, help='Wandb log every n steps')
    # parser.add_argument('--ckpt_intervel_steps', type=int, default=100, 
    #                     help='The frequency in terms of steps, at which checkpoint callback run at')
    # parser.add_argument('--val_intervel_steps', type=int, default=100, 
    #                     help='The frequency in terms of steps, at which validaiton loop run at')
    
    return parser.parse_args()


if __name__ == '__main__':
    Tester(parse_args())