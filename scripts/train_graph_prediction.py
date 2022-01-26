from pathlib import Path
import argparse
import imp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm
import torch
import os
import wandb
import itertools


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from cgl.utils.pdb import register_pdb_hook
from cgl.utils.params import ParamDict
from cgl.models.gnn import GraphRegression
# HACK: load_ckpt needs all the classes to be included (this is only a problem for opamp checkpoint which has used ModelcheckpointNoOverride)
from scripts.pretrain import ModelCheckpointNoOverride

import torch_scatter
from sklearn.manifold import TSNE


register_pdb_hook()

class Trainer:
    def __init__(self, pargs) -> None:

        if pargs.train and pargs.finetune:
            raise ValueError('')

        pl.seed_everything(pargs.seed)
        
        conf = self.get_conf(pargs)

        gnn = self.get_backbone_model(conf.bb_config)
        if pargs.gnn_ckpt:
            # ckpt_dict = torch.load(pargs.gnn_ckpt)
            # breakpoint()
            # gnn.load_state_dict(ckpt_dict['state_dict'])
            gnn = gnn.load_from_checkpoint(pargs.gnn_ckpt).to(device=gnn.device, dtype=gnn.dtype)
            print('GCN backbone pre-loaded.')

        if conf.mdl_conf.get('freeze_backbone', False):
            print('Backbone is frozen.')
            gnn.freeze()
        model = GraphRegression(conf.mdl_conf, gnn)
        model.hparams['seed'] = pargs.seed

        """
        ################## plotting the tsne based on class
    
        if pargs.ckpt:
            ckpt_dict = torch.load(pargs.ckpt, map_location=model.device)
            model.load_state_dict(ckpt_dict['state_dict'])

        xs, ys = [], []
        gnn.freeze()
        for batch in conf.test_dloader:
            input_struct = gnn.get_input_struct(batch)
            node_embs = gnn.get_node_features(input_struct)
            # counts = torch_scatter.scatter_sum(batch.output_node_mask.int(), batch.batch, dim=0)
            # group_ids = torch.tensor([[id]*count for id, count in enumerate(counts)]).view(-1).contiguous()
            # graph_embs = torch.stack([node_embs[group_ids == i].view(-1) for i in range(batch.num_graphs)], 0)
            graph_embs = node_embs[75::84]
            xs.append(graph_embs.detach().cpu().numpy())
            ys.append(batch.rout_cls.detach().cpu().numpy())

        x = np.concatenate(xs, 0)
        y = np.concatenate(ys, 0)
        print('Doing TSNE ....')
        x2d = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x2d[:, 0], x2d[:, 1], s=5, c=y)
        plt.savefig('tsne_pretrained_frozen_75.png')
        # plt.savefig('tsne_pretrained_frozen.png')
        breakpoint()
        ###################################################
        # """
        

        Path(pargs.output).mkdir(parents=True, exist_ok=True)
        callbacks = [LearningRateMonitor()]
        if conf.valid_dloader:
            # uncomment this line if you don't want to override your checkpoint
            # ckpt_callback = ModelCheckpointNoOverride(
            ckpt_callback = ModelCheckpoint(
                monitor='valid_loss_epoch',
                filename='cgl-{step}-{valid_loss_epoch:.4f}-{epoch:02d}',
                save_last=True,
                # save_top_k=10,
                # monitor 10 checkpoints overall
                every_n_train_steps=pargs.ckpt_intervel_steps,
            )
            # ckpt_callback = CheckpointEveryNSteps(save_step_frequency=1000)
            callbacks.append(ckpt_callback)
        else:
            ckpt_callback = None

        wandb_run = wandb.init(
            project=pargs.project,
            name=pargs.run_name,
            dir=pargs.output,
            id=pargs.wandb_id,
            resume='allow',
            config=dict(seed=pargs.seed),
            group=conf.exp_name,
        )
        wandb_run.log_code(root=Path(pargs.path).parent)
        wandb_logger = pl_loggers.WandbLogger(experiment=wandb_run, save_dir=pargs.output) #, settings=wandb.Settings(start_method="fork"))
        trainer = pl.Trainer(
                        max_epochs=pargs.max_epochs,
                        max_steps=pargs.max_steps,
                        gpus=pargs.gpus,
                        logger=wandb_logger,
                        callbacks=callbacks,
                        profiler='simple' if pargs.profile else None,
                        terminate_on_nan=True,
                        log_every_n_steps=pargs.log_freq,
                        num_sanity_val_steps=-1,
                        # check_val_every_n_epoch=max(int(conf.train_dloader.batch_size * pargs.val_intervel_steps / len(conf.train_dloader.dataset)), 1),
                        deterministic=True,
                    )   
        if pargs.train:
            trainer.fit(model, train_dataloader=conf.train_dloader, val_dataloaders=conf.valid_dloader)
            if ckpt_callback:
                model_path = ckpt_callback.best_model_path
                # ckpt_path = ckpt_callback.last_model_path
                print(f'Last model is stored in {model_path}.')
        else:
            model_path = pargs.ckpt

        # Test with the model stored on ckpt_path based on validation set
        if pargs.test:
            if not model_path:
                raise ValueError('Checkpoint path is not given.')
            else:
                print(f'Loading the checkpoint {Path(model_path).absolute()} ...')
                ckpt_dict = torch.load(model_path, map_location=model.device)
                model.load_state_dict(ckpt_dict['state_dict'])
                # model = model.load_from_checkpoint(model_path).to(device=model.device, dtype=model.dtype)
                print('Checkpoint Loaded.')
                # # HACK
                # model.config = conf.mdl_conf

            exp = None
            loaders = dict(test=conf.test_dloader,  valid=conf.valid_dloader, train=conf.train_dloader)
            for mode, loader in loaders.items():
                if loader is None:
                    continue
                results = trainer.test(model, test_dataloaders=loader, ckpt_path=model_path, verbose=False)[0]
                exp = model.logger.experiment
                for metric_name, metric_value in results.items():
                    exp.summary[f'{mode}_{metric_name}'] = metric_value

    def get_backbone_model(self, bb_path):
        print(f'Loading from the backbone config file {bb_path}')
        conf_module = imp.load_source('conf', bb_path)
        return conf_module.model

    def get_conf(self, pargs):
        conf_path = pargs.path
        print(f'Loading from the config file {conf_path}')
        conf_module = imp.load_source('conf', conf_path)

        conf = ParamDict(
            exp_name=conf_module.exp_name, 
            train_dloader=conf_module.train_dloader,
            valid_dloader=conf_module.valid_dloader,
            test_dloader=conf_module.test_dloader,
            mdl_conf=conf_module.mdl_config,
            bb_config=conf_module.backbone_config,
        )

        return conf

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='config file path')
    parser.add_argument('--train', type=int, default=1, help='set to zero to skip training')
    parser.add_argument('--finetune', type=int, default=0, help='set to zero to finetune')
    parser.add_argument('--gpus', type=int, default=0, help='set the training on gpu')
    parser.add_argument('--output', type=str, default='/store/nosnap/results/cgl', 
                        help='The output directory')
    parser.add_argument('--max_epochs', type=int, default=None, 
                        help='The maximum number of training epochs (if earlier than max_steps)')
    parser.add_argument('--max_steps', type=int, default=None, 
                        help='The maximum number of training steps (if earlier than max_epochs)')
    parser.add_argument('--ckpt', type=str, help='Resume from this checkpoint if valid.')
    parser.add_argument('--gnn_ckpt', type=str, help='GNN backbone checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--project', type=str, default='cgl-Rout', help='project name for wandb')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')
    parser.add_argument('--profile', type=int, default=0, help='Set to 1 to profile individual steps during training')
    parser.add_argument('--test', type=int, default=1, help='Set to 0 to skip running the test scripts')
    parser.add_argument('--log_freq', type=int, default=10, help='Wandb log every n steps')
    parser.add_argument('--ckpt_intervel_steps', type=int, default=100, 
                        help='The frequency in terms of steps, at which checkpoint callback run at')
    parser.add_argument('--val_intervel_steps', type=int, default=100, 
                        help='The frequency in terms of steps, at which validaiton loop run at')
    
    return parser.parse_args()


if __name__ == '__main__':
    Trainer(parse_args())