from cgl.eval.evaluator import GraphEvaluator
from pathlib import Path
import argparse
import imp
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from cgl.utils.params import ParamDict
from cgl.utils.general import listdict2dictlist

class Trainer:
    def __init__(self, pargs) -> None:
        pl.seed_everything(pargs.seed)
        
        conf = self.get_conf(pargs)
        model = conf.model

        wandb_run = wandb.init(
            project='cgl',
            name=pargs.run_name,
            dir=pargs.output,
            id=pargs.wandb_id,
            resume='allow',
            config=dict(seed=pargs.seed)
        )
        wandb_run.log_code(root=Path(pargs.path).parent)

        ckpt_path = pargs.ckpt

        Path(pargs.output).mkdir(parents=True, exist_ok=True)
        callbacks = [LearningRateMonitor()]
        if conf.valid_dloader:
            ckpt_callback = ModelCheckpoint(
                monitor='valid_loss_total',
                filename='cgl-{epoch:02d}-{valid_loss_total:.4f}',
                save_last=True,
            )
            callbacks.append(ckpt_callback)
        else:
            ckpt_callback = None

        wandb_logger = pl_loggers.WandbLogger(experiment=wandb_run, save_dir=pargs.output)
        trainer = pl.Trainer(
            max_epochs=pargs.max_epochs, 
            gpus=pargs.gpus,
            logger=wandb_logger,
            callbacks=callbacks,
            resume_from_checkpoint=ckpt_path,
            profiler='simple' if pargs.profile else None,
            terminate_on_nan=True,
            # track_grad_norm=1,
            log_every_n_steps=pargs.log_freq
        )

        if pargs.train:
            # pre-load the node_emb_ckpt if given
            if pargs.node_emb_ckpt:
                print(f'Loading weights for the Node Embedding module from checkpoint {pargs.node_emb_ckpt} ...')
                state_dict = torch.load(pargs.node_emb_ckpt, map_location=model.device)['state_dict']
                model.node_emb_module.load_state_dict(state_dict)
                print('Checkpoint Loaded.')

            trainer.fit(model, train_dataloader=conf.train_dloader, val_dataloaders=conf.valid_dloader)
            if ckpt_callback:
                ckpt_path = ckpt_callback.best_model_path
                print(f'Best model is stored in {ckpt_path}.')


        # Test with the model stored on ckpt_path based on validation set
        if pargs.test:
            if not ckpt_path:
                raise ValueError('Checkpoint path is not given.')
            else:
                print(f'Loading the checkpoint {str(Path(ckpt_path).absolute())} ...')
                state_dict = torch.load(ckpt_path, map_location=model.device)['state_dict']
                model.load_state_dict(state_dict)
                print('Checkpoint Loaded.')

            mse_evaluator = GraphEvaluator('mse')
            bc_evaluator = GraphEvaluator('rocauc')
            # loaders = dict(train=conf.train_dloader, valid=conf.valid_dloader, test=conf.test_dloader)
            loaders = dict(test=conf.test_dloader)
            for mode, loader in loaders.items():
                prediction_list = []
                true_list = []
                for batch in tqdm(loader):
                    device = torch.device('cuda') if pargs.gpus != 0 else torch.device('cpu')
                    model = model.to(device)
                    if isinstance(batch, dict):
                        batch = ParamDict({k: v.to(device) for k, v in batch.items()})
                    else:
                        batch = batch.to(device)
                    batch_results = model.predict(batch, compute_loss=True)
                    print(f'{mode} gain MSE loss {batch_results.loss.gain.item()}')
                    breakpoint()
                    prediction_list.append({k:batch_results.output[k] for k in model.output_labels_dict})
                    true_list.append({k:batch[k] for k in model.output_labels_dict})
                    

                prediction_dict = listdict2dictlist(prediction_list)
                prediction_dict = {k: torch.cat(v, 0) for k, v in prediction_dict.items()}
                true_dict = listdict2dictlist(true_list)
                true_dict = {k: torch.cat(v, 0) for k, v in true_dict.items()}

                # Regression tasks
                y_pred = torch.stack([prediction_dict[k] for k in model.reg_keys], -1)
                # in mlp dataset the true output has shape of (batch, 1)
                y_true = torch.stack([true_dict[k].squeeze(-1) for k in model.reg_keys], -1)
                mse_dict = mse_evaluator.eval(dict(y_pred=y_pred, y_true=y_true))

                if mode == 'test':
                    for label in model.reg_keys:
                        y_pred = prediction_dict[label].detach().cpu().numpy()
                        y_true = true_dict[label].squeeze(-1).detach().cpu().numpy()
                        plt.close()
                        _, ax = plt.subplots(1, 1)
                        ax.scatter(y_true, y_pred, s=1)
                        min_true, max_true = y_true.min(), y_true.max()
                        x = np.linspace(min_true, max_true, 1000)
                        ax.plot(x, x, linestyle='dashed', color='k', linewidth=1)
                        ax.set_xlabel('True Value')
                        ax.set_ylabel('Pred Value')
                        ax.set_title(label)
                        wandb_run.log({f'test_chart_{label}': wandb.Image(ax)})

                # classification tasks
                y_pred = torch.stack([prediction_dict[k] for k in model.class_keys], -1)
                y_true = torch.stack([true_dict[k].squeeze(-1) for k in model.class_keys], -1)
                bc_dict = bc_evaluator.eval(dict(y_pred=y_pred, y_true=y_true))

                wandb_run.summary[f'{mode}_rocauc'] = bc_dict['rocauc']
                wandb_run.summary[f'{mode}_mse'] = mse_dict['mse']


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
            output_labels_dict=conf_module.output_labels_dict,
        )

        return conf


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='config file path')
    parser.add_argument('--train', type=int, default=1, help='set to zero to skip training')
    parser.add_argument('--gpus', type=int, default=0, help='set the training on gpu')
    parser.add_argument('--output', type=str, default='/store/nosnap/results/cgl', 
                        help='The output directory')
    parser.add_argument('--max_epochs', type=int, default=100, 
                        help='The maximum number of training epochs')
    parser.add_argument('--ckpt', type=str, help='Resume from this checkpoint if valid.')
    parser.add_argument('--node_emb_ckpt', type=str, help='Load the node embedding module from this checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')
    parser.add_argument('--profile', type=int, default=0, help='Set to 1 to profile individual steps during training')
    parser.add_argument('--test', type=int, default=1, help='Set to 0 to skip running the test scripts')
    parser.add_argument('--log_freq', type=int, default=50, help='Wandb log every n steps')
    
    return parser.parse_args()


if __name__ == '__main__':
    Trainer(parse_args())