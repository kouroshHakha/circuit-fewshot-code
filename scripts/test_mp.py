from cgl.data.collect_test_json import main
from typing import Optional

import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.pool import sag_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.data.dataloader import DataLoader

import pytorch_lightning as pl



class CustomSAGEConv(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = nn.Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t, x) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return nn.matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Net(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, output_dim, nlayers):
        super().__init__()
        self.save_hyperparameters()

        self.gnn = nn.ModuleList([CustomSAGEConv(input_dim if i == 0 else hidden_dim,
                                         hidden_dim, aggr='add') for i in range(nlayers)])
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.crit = nn.CrossEntropyLoss()

    def forward(self, data):
        hidden_x = data.x
        for layer in self.gnn:
            hidden_x = layer(hidden_x, data.edge_index)
        out = self.output_head(hidden_x)
        return out


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


    def training_step(self, batch, batch_idx):
        mask = batch.train_mask
        batch_output = self(batch)[mask]
        loss = self.crit(batch_output, batch.y[mask])
        self.log('train_loss', loss, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_output = self(batch)
        val_acc = self._get_acc(batch_output[batch.val_mask], batch.y[batch.val_mask])
        test_acc = self._get_acc(batch_output[batch.test_mask], batch.y[batch.test_mask])
        self.log('val_acc', val_acc, logger=True, prog_bar=True)
        self.log('test_acc', test_acc, logger=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        mask = batch.test_mask
        batch_output = self(batch)
        acc = self._get_acc(batch_output[mask], batch.y[mask])
        self.log('Final Test accuracy', acc, logger=True, prog_bar=False)

    def _get_acc(self, output_logits, ground_truth):
        pred = output_logits.argmax(-1)
        acc = (pred == ground_truth).sum() / pred.shape[0]
        return acc

if __name__ == '__main__':
    dataset = Planetoid(root="/store/nosnap/datasets/gnn_tut", name= "Cora")
    data = dataset[0]
    dloader = DataLoader(dataset)

    pl.seed_everything(seed=0)
    trainer = pl.Trainer(
        max_epochs=300,
        default_root_dir="/store/nosnap/results/gnn_tut/sage",
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        gpus=1,
    )
    net = Net(
        input_dim=data.num_features,
        hidden_dim=512,
        output_dim=len(set(data.y.numpy())),
        nlayers=5,
    )
    trainer.fit(net, dloader, val_dataloaders=[dloader])
    trainer.test(net, dloader, verbose=True)