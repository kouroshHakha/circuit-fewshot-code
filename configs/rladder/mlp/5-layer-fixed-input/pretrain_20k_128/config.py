import time

from torch_geometric.data import DataLoader

from cgl.utils.params import ParamDict

from rdiv.data import RLadderDatasetMLP
from cgl.models.mlp import MLPFixedInput
from torch.utils.data import Subset
import numpy as np

s = time.time()
print('Loading the dataset ...')
dset = RLadderDatasetMLP(root='rdiv', train_fname='rladder_r128_train_20k_mlp.pt', test_fname='rladder_r128_test_mlp.pt')
print(f'Dataset was loaded in {time.time() - s:.6f} seconds.')

fract = 1.0
split = 0.9
train_idx = int(split * dset.train_idx)
train_dset = Subset(dset, np.arange(int(fract * train_idx)))
valid_dset = Subset(dset, np.arange(train_idx, dset.train_idx))
test_dset = Subset(dset, np.arange(dset.train_idx, len(dset)))

sample_x, sample_y = train_dset[0]

lr = 1e-3
in_channels = sample_x.shape[-1]
hidden_channels = 2048
num_layers = 5
train_batch_size = min(256, len(train_dset))
valid_batch_size = min(256, len(valid_dset)) 
test_batch_size = min(256, len(test_dset)) 

exp_name = f'RLadder_MLPFixedInput_scratch_r128_h{hidden_channels}_nls{num_layers}_bs{train_batch_size}_lr{lr}'

mdl_config = ParamDict(
    exp_name=exp_name,
    num_nodes=sample_y.shape[-1],
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    num_layers=num_layers,
    dropout=0,
    output_labels={'vdc': 1},
    output_sigmoid=['vdc'],
    lr=lr,
    bins=100,
    with_bn=True,
)

train_dloader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True, num_workers=4)
valid_dloader = DataLoader(valid_dset, batch_size=valid_batch_size, num_workers=4)
test_dloader = DataLoader(test_dset, batch_size=test_batch_size, num_workers=0)

# .to converts the weight dtype to match input
model = MLPFixedInput(mdl_config).to(sample_x.dtype)
