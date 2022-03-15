
import argparse
from pathlib import Path
from posixpath import split

import wandb


from torch import optim
from cgl.bagnet_gnn.data import BagNetDatasetTrain, BagNetDatasetTest
from cgl.bagnet_gnn.trainer import BagNetLightning, BagNetDataModule
from torch.utils.data import DataLoader

from torch.utils.data import random_split


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--lr', '-lr', default=3e-4, type=float)
    parser.add_argument('--max_steps', default=1000, type=int) # max_steps per round
    
    # dataloader
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--use_gnn_backbone', action='store_true')

    # checkpoint resuming and testing
    parser.add_argument('--gnn_ckpt', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')

    # wandb
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')
    return parser.parse_args()


def main(pargs):
    pl.seed_everything(pargs.seed)

    dataset = BagNetDatasetTrain('datasets/bagnet_gnn', is_graph=pargs.use_gnn_backbone)
    

    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=pargs.batch_size)
    # batch = next(iter(test_loader))

    conf = dict(
        lr=pargs.lr,
        test_every_n_epoch=50,
    )
    pl_model = BagNetLightning(conf) 

    exp_name = 'bagnet'
    run_name = exp_name if not pargs.run_name else f'{exp_name}_{pargs.run_name}'
    wandb_run = wandb.init(
        project='bagnet_gnn',
        name=run_name,
        dir='./wandb_logs',
        id=pargs.wandb_id,
        resume='allow',
        config=dict(seed=pargs.seed),
    )
    logger = WandbLogger(experiment=wandb_run, save_dir='./wandb_logs')

    
    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)

    trainer = pl.Trainer(
        # max_steps=pargs.max_steps,
        max_epochs=50 * (dataset.round_max + 1),
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
        callbacks=[ModelSummary(max_depth=-1)],
        reload_dataloaders_every_n_epochs=50,
    )
    datamodule = BagNetDataModule(batch_size=pargs.batch_size, use_gnn_backbone=pargs.use_gnn_backbone)

    trainer.fit(pl_model, datamodule=datamodule)
    trainer.test(pl_model, datamodule=datamodule)



if __name__ == '__main__':

    main(_parse_args())