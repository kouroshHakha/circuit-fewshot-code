import time

from torch_geometric.data import DataLoader

from cgl.utils.params import ParamDict

from rdiv.data import RLadderDataset
from cgl.models.mlp import MLPPyramid

s = time.time()
print('Loading the dataset ...')
dset = RLadderDataset(root='rdiv', train_fname='rladder_r32_train_20k.pt', test_fname='rladder_r32_test.pt')
print(f'Dataset was loaded in {time.time() - s:.6f} seconds.')

fract = 1.0
split = 0.9
train_idx = int(split * dset.train_idx)
train_dset = dset[:int(fract * train_idx)]
valid_dset = dset[train_idx:dset.train_idx]
test_dset = dset[dset.train_idx:]
train_dset = train_dset.shuffle()
sample_data = train_dset[0]


lr = 1e-4
in_node_dim = sample_data.x.shape[-1] + sample_data.type_tens.shape[-1]
in_channels = in_node_dim * len(sample_data.x)
hidden_channels = 512
num_layers = 5
train_batch_size = min(256, len(train_dset))
valid_batch_size = min(256, len(valid_dset)) 
test_batch_size = min(256, len(test_dset)) 

exp_name = f'RLadder_MLP_scratch_r32_h{hidden_channels}_nls{num_layers}_bs{train_batch_size}_lr{lr}'

mdl_config = ParamDict(
    exp_name=exp_name,
    num_nodes=sample_data.vdc.shape[0],
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
model = MLPPyramid(mdl_config).to(sample_data.x.dtype)