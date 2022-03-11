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


register_pdb_hook()

### watch the minimum of the validation loss and save the checkpoint based on that
### in that checkpoint save train / valid / test scores (log_metrics)
### plot the scatter waveforms for validation through out time every 10 epochs
### plot the final scatter for test
### take the avg/std perf over seeds
### enable loading the model ckpt and resuming (ckpts are save locally and stats are on wandb)
# TODO: checkpoint resuming works but with a new logger

class ModelCheckpointNoOverride(ModelCheckpoint):
    """This checkpoint call back does not delete the already saved checkpoint"""
    def _del_model(self, filepath: str) -> None:
        pass

class Trainer:
    def __init__(self, pargs) -> None:

        if pargs.train and pargs.finetune:
            raise ValueError('')

        pl.seed_everything(pargs.seed)
        
        conf = self.get_conf(pargs)
        model = conf.model

        # make sure despite loading config random state is still the same
        pl.seed_everything(pargs.seed)
        model.hparams['seed'] = pargs.seed
        model.reset_parameters()

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
        # debug tool for logging all the parameters and their gradients during training
        # wandb_run.watch(model, log='all', log_freq=pargs.log_freq)

        ckpt_path = pargs.ckpt

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

        wandb_logger = pl_loggers.WandbLogger(experiment=wandb_run, save_dir=pargs.output) #, settings=wandb.Settings(start_method="fork"))
        trainer = pl.Trainer(
            max_epochs=pargs.max_epochs,
            max_steps=pargs.max_steps,
            gpus=pargs.gpus,
            logger=wandb_logger,
            callbacks=callbacks,
            resume_from_checkpoint=ckpt_path if not pargs.finetune else None,
            profiler='simple' if pargs.profile else None,
            detect_anomaly=True,
            log_every_n_steps=pargs.log_freq,
            num_sanity_val_steps=-1,
            # check_val_every_n_epoch=max(int(conf.train_dloader.batch_size * pargs.val_intervel_steps / len(conf.train_dloader.dataset)), 1),
        )

        if pargs.train:
            pl.seed_everything(pargs.seed)
            trainer.fit(model, train_dataloaders=conf.train_dloader, val_dataloaders=conf.valid_dloader)
            if ckpt_callback:
                ckpt_path = ckpt_callback.best_model_path
                # print(f'Best model is stored in {ckpt_path}.')
                # ckpt_path = ckpt_callback.last_model_path
                print(f'Last model is stored in {ckpt_path}.')

        if pargs.finetune:
            pl.seed_everything(pargs.seed)
            model = model.load_from_checkpoint(ckpt_path).to(device=model.device, dtype=model.dtype)
            # tmp_model = model.load_from_checkpoint(ckpt_path).to(device=model.device, dtype=model.dtype)
            # model.layers.load_state_dict(tmp_model.layers.state_dict())
            # model.lin.load_state_dict(tmp_model.lin.state_dict())
            # model.proj.load_state_dict(tmp_model.proj.state_dict())
            # freezing the backbone
            # for name, param in model.named_parameters():
            #     if not name.startswith('proj.nets.8'):
            #         param.requires_grad = False
            #     else:
            #         print(name)
            # for param in itertools.chain(model.layers.parameters(), model.lin.parameters()):
            #     param.requires_grad = False
            model.config = conf.mdl_conf
            print('GCN backbone pre-loaded.')
            trainer.fit(model, train_dataloaders=conf.train_dloader, val_dataloaders=conf.valid_dloader)
            if ckpt_callback:
                ckpt_path = ckpt_callback.last_model_path
                print(f'Last model is stored in {ckpt_path}.')


        # Test with the model stored on ckpt_path based on validation set
        if pargs.test:
            if not ckpt_path:
                raise ValueError('Checkpoint path is not given.')
            else:
                print(f'Loading the checkpoint {Path(ckpt_path).absolute()} ...')
                model = model.load_from_checkpoint(ckpt_path).to(device=model.device, dtype=model.dtype)
                print('Checkpoint Loaded.')
                # HACK
                model.config = conf.mdl_conf

            exp = None
            loaders = dict(test=conf.test_dloader,  valid=conf.valid_dloader, train=conf.train_dloader)
            for mode, loader in loaders.items():
                if loader is None:
                    continue
                results = trainer.test(model, test_dataloaders=loader, ckpt_path=ckpt_path, verbose=False)[0]
                exp = model.logger.experiment
                for metric_name, metric_value in results.items():
                    exp.summary[f'{mode}_{metric_name}'] = metric_value

            # labels = model.output_labels

            # for mode, loader in zip(['test', 'train'], [conf.test_dloader, conf.train_dloader]):
            #     if loader is None:
            #         continue
            #     pred_values = {k: [] for k in labels}
            #     true_values = {k: [] for k in labels}
            #     n_samples = 0
            #     print(f'Running Prediction loop for {mode} dataset ...')
            #     for batch in tqdm(loader):
            #         device = torch.device('cuda') if pargs.gpus != 0 else torch.device('cpu')
            #         model = model.to(device)
            #         if isinstance(batch, dict):
            #             batch = ParamDict({k: v.to(device) for k, v in batch.items()})
            #         else:
            #             batch = batch.to(device)
            #         results = model.predict(batch, compute_loss=False)
            #         for label in labels:
            #             true_value = batch[label].reshape(-1)
            #             pred_value = results.output[label].reshape(-1)
            #             pred_values[label].append(pred_value.detach().cpu().numpy())
            #             true_values[label].append(true_value.detach().cpu().numpy())

            #         # to cap the size of the generated graph, limit the number of plotted pts to 500.
            #         n_samples += batch.x.shape[0]
            #         if mode == 'train' and n_samples > 500:
            #             break

                # output_fig_dir = Path(f'fig_{conf.mdl_conf.exp_name}')
                # output_fig_dir.mkdir(exist_ok=True)

                # for label in labels:
                #     pred_value = np.concatenate(pred_values[label], -1)
                #     true_value = np.concatenate(true_values[label], -1)

                #     # plot pred vs true value onto wandb
                #     plt.close()
                #     _, ax = plt.subplots(1, 1)
                #     ax.scatter(true_value, pred_value, s=1)
                #     min_true, max_true = true_value.min(), true_value.max()
                #     x = np.linspace(min_true, max_true, 1000)
                #     ax.plot(x, x, linestyle='dashed', color='k', linewidth=1)
                #     ax.set_xlabel('True Value')
                #     ax.set_ylabel('Pred Value')
                #     ax.set_title(label)
                #     exp.log({f'{mode}_chart_{label}': wandb.Image(ax)})
                #     print(f'Plotted {mode}_chart_{label}')
                    
                #     # # plot train and test figures for qualitative comparison
                #     # cur_output_dir = output_fig_dir / mode / label
                #     # cur_output_dir.mkdir(parents=True, exist_ok=True)

                #     # pred_value = pred_value.reshape(-1, 101)
                #     # true_value = true_value.reshape(-1, 101)

                #     # non_nan_inds = np.where(~np.isnan(true_value[:, 0]))[0]
                #     # # look at first 200 nodes
                #     # pred_value = pred_value[non_nan_inds][:200]
                #     # true_value = true_value[non_nan_inds][:200]

                #     # for i, (pred_vec, true_vec) in enumerate(zip(pred_value, true_value)):
                #     #     plt.close()
                #     #     plt.plot(true_vec, linestyle='solid', color='b', label='true')
                #     #     plt.plot(pred_vec, linestyle='dashed', color='b', label='pred')
                #     #     plt.legend()

                #     #     plt.savefig(cur_output_dir / f'{i}.png')
                        
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
    parser.add_argument('--train', type=int, default=1, help='set to zero to skip training')
    parser.add_argument('--finetune', type=int, default=0, help='set to zero to finetune')
    parser.add_argument('--gpus', type=int, default=0, help='set the training on gpu')
    parser.add_argument('--output', type=str, default=os.environ.get('OUTPUT_PATH', '.') + '/results/cgl', 
                        help='The output directory')
    parser.add_argument('--max_epochs', type=int, default=None, 
                        help='The maximum number of training epochs (if earlier than max_steps)')
    parser.add_argument('--max_steps', type=int, default=-1, 
                        help='The maximum number of training steps (if earlier than max_epochs)')
    parser.add_argument('--ckpt', type=str, help='Resume from this checkpoint if valid.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--project', type=str, default='cgl-PT', help='project name for wandb')
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