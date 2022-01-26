
from pathlib import Path
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import Dataset

class RDivDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None, pre_filter=None, valid_idx=0):
        super().__init__(root, transform, pre_transform, pre_filter)

        if data_list is None:
            dataset_list = torch.load(Path(root) / 'data.pt')
            assert len(dataset_list) == 15
            reordered_list = []
            for i in range(15):
                if i != valid_idx:
                    reordered_list += dataset_list[i]

            self.train_idx = len(reordered_list)
            reordered_list += dataset_list[valid_idx]
        else:
            reordered_list = data_list
        self.data, self.slices = self.collate(reordered_list)

    @classmethod
    def from_datalist(cls,  root, data_list):
        return cls(root, data_list)


    
class RLadderDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, train_fname='', test_fname=''):
        super().__init__(root, transform, pre_transform, pre_filter)

        if not train_fname:
            train_fname = Path(root) / 'rladder_pretrain.pt'
        else:
            train_fname = Path(root) / train_fname

        if not test_fname:
            test_fname = Path(root) / 'rladder_r11_train.pt'
        else:
            test_fname = Path(root) / test_fname

        train_list = torch.load(train_fname)
        test_list = torch.load(test_fname)
        self.train_idx = len(train_list)
        dataset_list = train_list + test_list
        self.data, self.slices = self.collate(dataset_list)


class RLadderDatasetMLP(TorchDataset):
    def __init__(self, root, train_fname='', test_fname=''):
        super().__init__()

        train_fname = Path(root) / train_fname
        test_fname = Path(root) / test_fname

        train_data = torch.load(train_fname)
        test_data = torch.load(test_fname)
        self.train_idx = len(train_data['x'])

        self.x = torch.cat([train_data['x'], test_data['x']], 0)
        self.y = torch.cat([train_data['y'], test_data['y']], 0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)