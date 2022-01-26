
## Installation
Clone the repo, install the requirements and the package using `pip`:

```bash
cd acnet
pip install -r requirements.txt
```

### Data Loaders

We prepare Pytorch Geometric data loaders that can be used right out of the box.

**Rladder**. You can download the files and put them in `rdiv` folder (size ~).

```python
from rdiv.data import RLadderDataset
from torch_geometric.data import DataLoader

dset = RLadderDataset(root='rdiv', train_fname='rladder_r32_train.pt', test_fname='rladder_r32_test.pt')
train_dset = dset[:dset.train_idx]
valid_dset = dset[dset.train_idx:]
sample_data = train_dset[0]

train_dloader = DataLoader(train_dset, batch_size=16)
```

**OpAmp pretraining**. The file size is ~2.37GB.
```python
from acnet.data.graph_data import CircuitInMemDataset


dset = CircuitInMemDataset(root='/tmp/dataset/opamp', mode='train')

sample_data = dset[0]
splits = dset.splits
train_dset = dset[splits['train']]
valid_dset = dset[splits['valid']]
test_dset = dset[splits['test']]

train_dloader = DataLoader(train_dset, batch_size=16)
```

**OpAmp biased pmos**. The file size is ~2.GB
```python
from acnet.data.graph_data import CircuitInMemDataset


dset = CircuitInMemDataset(root='/tmp/dataset/opamp', mode='train')

sample_data = dset[0]
splits = dset.splits
train_dset = dset[splits['train']]
valid_dset = dset[splits['valid']]
test_dset = dset[splits['test']]

train_dloader = DataLoader(train_dset, batch_size=16)
```

# Running the experiments
All of our experiments with different GNN architectures can be reproduced by running the `pretrain.py` script. This script works with `wandb` for logging. So you have to define env variable [`WANDB_API_KEY`](https://docs.wandb.ai/guides/track/advanced/environment-variables) for the code to recognize your wandb workspace. 

You can now launch pre-training the GNNs by running:
```bash
python scripts/pretrain.py --path <path_to_config> --gpus 1
```
All the config files are provided in the `configs` folder for reproducibility. 

For more information on the options see `python scripts/pretrain.py -h`.

For fine-tuning with a pretrained checkpoint to do voltage prediction on a new circuit: 
```bash
scripts/pretrain.py --project=<wandb_project_name> --gpus=1 --max_steps=50000 --ckpt=<path_to_pretrained_network.ckpt> --train=0 --finetune=1 --path= <path_to_downstream_config>
```

For fine-tuning on a graph property prediction run:
```bash
python scripts/train_graph_prediction.py --project=<wandb_project_name> --gpus=1 --max_steps=5000 --gnn_ckpt=<gnn_backbone_ckpt> --path=<config_path>
```

### Evaluator
The evaluator used in this work is implemented in as `NodeEvaluator` class in `acnet/eval/evaluator.py`. For example usage see `acnet/models/base.py`.

