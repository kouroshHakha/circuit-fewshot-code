import time

import torch
from torch_geometric.data import DataLoader

from cgl.utils.params import ParamDict
from cgl.data.graph_data import CircuitInMemDataset, CircuitGraphDataset

from cgl.models.gnn import DeepGENNet

s = time.time()
print('Loading the dataset ...')
root = '/store/nosnap/results/ngspice_biased_pmos_gain/two_stage_biased_pmos'
cir_dset = CircuitGraphDataset(root=root, mode='train')
node_output_idx = next(iter(cir_dset.graph_nodes.values()))['V_net6']
vout_idx = torch.where((torch.where(cir_dset[0].output_node_mask)[0] == node_output_idx))[0].item()

def transform_fn(data):
    data.gain = data.vac_mag[vout_idx, 0]
    return data

dset = CircuitInMemDataset(root=root, mode='train', transform=transform_fn)
print(f'Dataset was loaded in {time.time() - s:.6f} seconds.')

# gains = []
# for idx in range(len(dset)):
#     dset[idx].gain = dset[idx].vac_mag[vout_idx, 0]

sample_data = dset[0]

fract = 0.5
splits = dset.splits
train_idx = int(fract * len(splits['train']))
train_dset = dset[splits['train'][:train_idx]]
valid_dset = dset[splits['valid']]
test_dset = dset[splits['test']]

lr = 3e-4
activation = 'relu'
hidden_channels = 128
num_layers = 15
train_batch_size = min(256, len(train_dset))
valid_batch_size = min(256, len(valid_dset)) 
test_batch_size = min(256, len(test_dset)) 

exp_name = f'BIASED_PMOS_FT_5_DeepGEN_h{hidden_channels}_nl{num_layers}_bs{train_batch_size}_lr{lr:.0e}_{activation}'

mdl_config = ParamDict(
    exp_name=exp_name,
    num_nodes=sample_data.vdc.shape[0],
    in_channels=sample_data.x.shape[-1] + sample_data.type_tens.shape[-1],
    hidden_channels=hidden_channels,
    num_layers=num_layers,
    dropout=0,
    lr=lr,
    activation=activation,
    bins=200,
    freeze_backbone=False,
    # lr_warmup={'peak_lr': lr, 'weight_decay': 0, 
    #            'warmup_updates': 100, 'tot_updates': 600, 'end_lr': 1e-4},
    lr_warmup={'warmup': 0, 'max_iters': 1000},
    output_labels={'vdc': 1},
    proj_n_layers=3,
)

train_dloader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True, num_workers=0)
valid_dloader = DataLoader(valid_dset, batch_size=valid_batch_size, num_workers=0)
test_dloader = DataLoader(test_dset, batch_size=test_batch_size, num_workers=0)

# .to converts the weight dtype to match input
model = DeepGENNet(mdl_config).to(sample_data.x.dtype)

