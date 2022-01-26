import time

from cgl.utils.params import ParamDict

from torch_geometric.data import DataLoader

from rdiv.data import RLadderDataset
from cgl.models.gnn import GCNNet

s = time.time()
print('Loading the dataset ...')
dset = RLadderDataset(root='rdiv')
print(f'Dataset was loaded in {time.time() - s:.6f} seconds.')

train_dset = dset[:dset.train_idx]
valid_dset = dset[dset.train_idx:]
train_dset = train_dset.shuffle()
sample_data = train_dset[0]


lr = 1e-3
activation = 'relu'
hidden_channels = 128
num_layers = 10
train_batch_size = min(256, len(train_dset))
valid_batch_size = min(256, len(valid_dset)) 

exp_name = f'RLadder_GCNNet_h{hidden_channels}_nl{num_layers}_bs{train_batch_size}_lr{lr:.0e}_{activation}'

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
    # lr_warmup={'warmup': 0, 'max_iters': 200},
    lr_warmup={'peak_lr': lr, 'weight_decay': 0, 
               'warmup_updates': 1000, 'tot_updates': 30000, 'end_lr': 1e-4},
    output_labels={'vdc': 1},
    proj_n_layers=3,
    output_sigmoid=['vdc'],
)

train_dloader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True, num_workers=4)
valid_dloader = DataLoader(valid_dset, batch_size=valid_batch_size, drop_last=False, num_workers=0)
test_dloader = None

# .to converts the weight dtype to match input
model = GCNNet(mdl_config).to(sample_data.x.dtype)


