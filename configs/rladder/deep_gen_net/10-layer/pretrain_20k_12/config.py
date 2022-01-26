import time

from torch_geometric.data import DataLoader

from cgl.utils.params import ParamDict

from cgl.models.gnn import DeepGENNet
from rdiv.data import RLadderDataset

s = time.time()
print('Loading the dataset ...')
dset = RLadderDataset(root='rdiv', train_fname='rladder_r12_train_20k.pt', test_fname='rladder_r12_test.pt')
print(f'Dataset was loaded in {time.time() - s:.6f} seconds.')


fract = 1.0
split = 0.9
train_idx = int(split * dset.train_idx)
train_dset = dset[:int(fract * train_idx)]
valid_dset = dset[train_idx:dset.train_idx]
test_dset = dset[dset.train_idx:]
train_dset = train_dset.shuffle()
sample_data = train_dset[0]


lr = 1e-3
activation = 'relu'
hidden_channels = 128
num_layers = 10
train_batch_size = min(256, len(train_dset))
valid_batch_size = min(256, len(valid_dset)) 
test_batch_size = min(256, len(test_dset)) 


exp_name = f'RLadder_DeepGEN_scratch_r12_h{hidden_channels}_nl{num_layers}_bs{train_batch_size}_lr{lr:.0e}_{activation}'

mdl_config = ParamDict(
    exp_name=exp_name,
    num_nodes=sample_data.vdc.shape[0],
    in_channels=sample_data.x.shape[-1] + sample_data.type_tens.shape[-1],
    hidden_channels=hidden_channels,
    num_layers=num_layers,
    dropout=0,
    lr=lr,
    activation=activation,
    bins=100,
    # lr_warmup={'warmup': 0, 'max_iters': 100},
    lr_warmup={'peak_lr': lr, 'weight_decay': 0, 
               'warmup_updates': 1000, 'tot_updates': 30000, 'end_lr': 1e-4},
    output_labels={'vdc': 1},
    proj_n_layers=3,
    output_sigmoid=['vdc'],
)

train_dloader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True, num_workers=8)
valid_dloader = DataLoader(valid_dset, batch_size=valid_batch_size, num_workers=0)
test_dloader = DataLoader(test_dset, batch_size=test_batch_size, num_workers=0)

# .to converts the weight dtype to match input
model = DeepGENNet(mdl_config).to(sample_data.x.dtype)

