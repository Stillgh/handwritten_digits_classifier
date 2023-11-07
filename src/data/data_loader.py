from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class DataPreparator():

    def __init__(self):
        self.train_data = datasets.MNIST(
            root='data',
            train=True,
            transform=ToTensor(),
            download=True
        )
        self.test_data = datasets.MNIST(
            root='data',
            train=False,
            transform=ToTensor(),
            download=True
        )
        self.loaders = {
            'train': DataLoader(
                self.train_data,
                batch_size=100,
                shuffle=True,
                num_workers=1),

            'test': DataLoader(
                self.test_data,
                batch_size=100,
                shuffle=True,
                num_workers=1)
        }

    def get_data(self):
        return self.train_data, self.test_data, self.loaders