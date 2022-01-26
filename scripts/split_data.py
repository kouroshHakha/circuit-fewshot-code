"""Script for splitting the data into train/valid/test subsets based on stimulous configuration"""

from os import write
from cgl.data.graph_data import CircuitInMemDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

dataset = CircuitInMemDataset(root='/store/nosnap/datasets/two_stage_graph', mode='train')
params_list = []
for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    params_list.append(data.circuit_params)
params_np = np.stack(params_list, 0)

vin1_mag_1 = (params_np[:, 17] == 1)
vin1_phase_180 = (params_np[:, 18] == 180)
vin2_mag_1 = (params_np[:, 20] == 1)
vin2_phase_180 = (params_np[:, 21] == 180)
# vin1=1, vin2=-1
valid_cond = vin1_mag_1 & vin1_phase_180 & vin2_mag_1 & ~vin2_phase_180
valid = torch.from_numpy(np.argwhere(valid_cond)[:, 0])
# vin1=0, vin2=1
test_cond = ~vin1_mag_1 & vin2_mag_1 & ~vin2_phase_180
test = torch.from_numpy(np.argwhere(test_cond)[:, 0])
train = torch.from_numpy(np.argwhere(~valid_cond & ~test_cond)[:, 0])

save_dir = Path('/store/nosnap/datasets/two_stage_graph/train/processed')
torch.save(dict(train=train, valid=valid, test=test), save_dir / 'splits.pt')