import hashlib
import time
import hashlib

from torch_geometric.data import DataLoader

from cgl.utils.params import ParamDict

# from cgl.models.gnn import DeepGENNet
from rdiv.data import RLadderDataset

s = time.time()
print('Loading the dataset ...')
##### preparing classification data
def transform(data):
    data.rout_cls = (data.rout > 0.25).long()
    return data
dset = RLadderDataset(root='rdiv', transform=transform, train_fname='rladder_r10_rout_train_1k.pt', test_fname='rladder_r10_rout_test_1k.pt')
print(f'Dataset was loaded in {time.time() - s:.6f} seconds.')

fract = 0.75
split = 0.9
train_idx = int(split * dset.train_idx)
train_dset = dset[:int(fract * train_idx)]
valid_dset = dset[train_idx:dset.train_idx]
test_dset = dset[dset.train_idx:]
train_dset = train_dset.shuffle()
sample_data = train_dset[0]

# # debugging the graph conversion ...
# from torch_geometric.utils import to_networkx
# import networkx as nx
# import matplotlib.pyplot as plt
# g = to_networkx(sample_data, to_undirected=True)

# nx.draw_circular(g, with_labels=True)
# plt.savefig('graph.png', dpi=250)
# breakpoint()

backbone_config = 'configs/rladder/deep_gen_net/15-layer/pretrain_20k_2_10/config.py'
bb_id = hashlib.sha256(backbone_config.encode('utf-8')).hexdigest()[:6]

lr = 1e-3
activation = 'relu'
hidden_channels = 128
num_layers = 3
train_batch_size = min(256, len(train_dset))
valid_batch_size = min(256, len(valid_dset)) 
test_batch_size = min(256, len(test_dset)) 

exp_name = f'ROUT_prediction_pt_AVGPooling_{bb_id}_bs{train_batch_size}_lr{lr:.0e}'

mdl_config = ParamDict(
    exp_name=exp_name,
    num_nodes=sample_data.vdc.shape[0],
    in_channels=sample_data.x.shape[-1] + sample_data.type_tens.shape[-1],
    hidden_channels=hidden_channels,
    num_layers=num_layers,
    dropout=0,
    activation=activation,
    bins=100,
    lr=lr,
    freeze_backbone=False,
    use_pooling=True,
    output_label='rout',
    output_sigmoid=True,
    lr_warmup={'peak_lr': lr, 'weight_decay': 0, 
               'warmup_updates': 50, 'tot_updates': 600, 'end_lr': 1e-5},
)

train_dloader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True, num_workers=4)
valid_dloader = DataLoader(valid_dset, batch_size=valid_batch_size, num_workers=0)
test_dloader = DataLoader(test_dset, batch_size=test_batch_size, num_workers=0)

# # .to converts the weight dtype to match input
# model = DeepGENNet(mdl_config).to(sample_data.x.dtype)

