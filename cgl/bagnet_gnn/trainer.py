
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from cgl.bagnet_gnn.model import BagNetComparisonModel
from cgl.bagnet_gnn.data import MAX_SPECS, MIN_SPECS, BagNetDatasetTrain, BagNetDatasetTest


COMPARISON_KWRDS = MAX_SPECS + MIN_SPECS

class LightningBase(pl.LightningModule):

    def log_ret_dict(self, dictionary, step: str = 'train', is_batch: bool = True, **kwargs) -> None:
        suf = 'batch' if is_batch else 'epoch'
        for key in dictionary:
            if key.startswith(('loss', 'acc')):
                prog_bar = not is_batch and key == 'loss'
                self.log(f'{step}_{key}_{suf}', dictionary[key], prog_bar=prog_bar ,**kwargs)

    def log_ret_epoch(self, outputs, step: str = 'train', **kwargs):
        metric_vals = defaultdict(list)
        for output in outputs:
            for key in output:
                if key.startswith(('loss', 'acc')):
                    metric_vals[key].append(output[key])
        metric_vals = {k: torch.stack(v, 0).mean() for k, v in metric_vals.items()}
        self.log_ret_dict(metric_vals, step, is_batch=False, **kwargs)


class BagNetLightning(LightningBase):

    def __init__(self, conf) -> None:
        super().__init__()
        self.save_hyperparameters(conf)
        self.conf = conf

        self.optim_round = 0

        feature_ext_config = dict(
            input_features=8,
            output_features=20,
            hidden_dim=20,
            n_layers=2,
            drop_out=0.2,
        )

        comparison_config = dict(
            hidden_dim=20,
            n_layers=1,
            drop_out=0.2,
        )

        self.model = BagNetComparisonModel(
            comparison_kwrds=COMPARISON_KWRDS,
            feature_exractor_config=feature_ext_config,
            comparison_model_config=comparison_config, 
            is_gnn=False
        )


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.conf['lr'])
        return optimizer

    def ff(self, batch):
        model_output = self.model(batch)

        loss = torch.tensor(0., device=self.device)
        total_heads = len(self.model.comparison_heads)
        for key in self.model.comparison_heads:
            loss += model_output['losses'][key] / total_heads

        acc = {}
        all_correct = None
        for key in self.model.comparison_heads:
            pred = model_output['outputs'][key]['prob'].argmax(-1)
            target = batch[key]

            cond = pred == target
            acc[f'acc_{key}'] = cond.float().mean(0).detach()
            if all_correct is None:
                all_correct = cond
            else:
                all_correct = all_correct & cond

        acc['acc_all'] = all_correct.float().mean(0).detach()

        return {'loss': loss, **acc}

    def training_step(self, batch):
        ret = self.ff(batch[0])
        self.log_ret_dict(ret, step='train', is_batch=True)
        return ret
        
    def training_epoch_end(self, outputs) -> None:
        self.log_ret_epoch(outputs, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0 or (dataloader_idx == 1 and self.current_epoch % self.conf['test_every_n_epoch'] == 0):
            # run test evaluation if idx=1 and the current epoch divides the fequency
            ret = self.ff(batch)
            return ret

    def validation_epoch_end(self, outputs) -> None:
        for dataloader_idx in range(len(outputs)):
            if dataloader_idx == 0:
                self.log_ret_epoch(outputs[dataloader_idx], 'valid')
            else:
                if outputs[dataloader_idx]:
                    self.log_ret_epoch(outputs[dataloader_idx], 'test')

    def test_step(self, batch, batch_idx):
        ret = self.ff(batch)
        return ret

    def test_epoch_end(self, outputs) -> None:
        self.log_ret_epoch(outputs, 'test')


class BagNetDataModule(pl.LightningDataModule):

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None, batch_size=1, use_gnn_backbone=False):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)

        self.save_hyperparameters('batch_size', 'use_gnn_backbone')
        self.batch_size = batch_size
        self.use_gnn_backbone = use_gnn_backbone

        self.optim_round = 0

    def setup(self, stage) -> None:
        self.test_dataset = BagNetDatasetTest('datasets/bagnet_gnn')

        if stage in ('fit', 'validate'):
            self.train_dataset, self.valid_dataset = self._get_cur_train_valid_dataset()

    def _get_cur_train_valid_dataset(self):
        dataset = BagNetDatasetTrain('datasets/bagnet_gnn', optim_round=self.optim_round, is_graph=self.use_gnn_backbone)
        
        split_lens = [int(0.8 * len(dataset))]
        split_lens.append(len(dataset) - split_lens[-1])

        train_dataset, valid_dataset = random_split(dataset, lengths=split_lens)
        return train_dataset, valid_dataset 

    def _update_datasets(self):
        self.train_dataset, self.valid_dataset = self._get_cur_train_valid_dataset()
        self.optim_round += 1
        print('updated, next_optim_round: ',  self.optim_round)

    def train_dataloader(self):
        self._update_datasets()

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        return [train_loader]

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        return [valid_loader, test_loader]

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        return test_loader
        