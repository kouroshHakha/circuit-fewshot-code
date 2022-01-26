import time

from torch_geometric.data import DataLoader

from cgl.utils.params import ParamDict
from cgl.data.graph_data import CircuitInMemDataset

from cgl.models.transformer import InterleavedPointTransformer

s = time.time()
print('Loading the dataset ...')
dset = CircuitInMemDataset(root='/store/nosnap/datasets/two_stage_graph', mode='train')
print(f'Dataset was loaded in {time.time() - s:.6f} seconds.')

sample_data = dset[0]
splits = dset.splits
train_dset = dset[splits['train']]
valid_dset = dset[splits['valid']]
test_dset = dset[splits['test']]


lr = 1e-3
activation = 'relu'
hidden_channels = 128

# (1 + 1)x5 + 1
pre_gnn_depth=1
depth=5
num_layers = 1
transformer_depth = 1

train_batch_size = min(256, len(train_dset))
valid_batch_size = min(256, len(valid_dset)) 
test_batch_size = min(256, len(test_dset)) 

# exp_name = f'Opamp_ACM_DeepGEN_h{hidden_channels}_nl{num_layers}_bs{train_batch_size}_lr{lr:.0e}_{activation}'
exp_name = f'Opamp_DC_InterleavePtTrsfGCNNet_{pre_gnn_depth}xgcn+({num_layers}xgcn+{transformer_depth}xtf)x{depth}'

mdl_config = ParamDict(
    exp_name=exp_name,
    num_nodes=sample_data.vdc.shape[0],
    in_channels=sample_data.x.shape[-1] + sample_data.type_tens.shape[-1],
    hidden_channels=hidden_channels,
    pre_gnn_depth=pre_gnn_depth,  # prepipeline gnns
    depth=depth,  # in units of modules
    num_layers=num_layers, # number of gcn layers per module
    transformer_depth=transformer_depth,  # depth of transformer per module
    nhead=4,
    dropout=0,
    lr=lr,
    activation=activation,
    bins=200,
    lr_warmup={'warmup': 0, 'max_iters': 100},
    output_labels={'vdc': 1},
    proj_n_layers=3,
)

train_dloader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True, num_workers=64)
valid_dloader = DataLoader(valid_dset, batch_size=valid_batch_size, num_workers=16)
test_dloader = DataLoader(test_dset, batch_size=test_batch_size, num_workers=16)

# .to converts the weight dtype to match input
model = InterleavedPointTransformer(mdl_config).to(sample_data.x.dtype)
