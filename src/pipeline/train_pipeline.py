import os

import torch.nn as nn
import torch.optim as optim
import torch
from jinja2 import loaders
from torch.utils.data import DataLoader

from src.model.models import CNN
from src.utils import save_object


class Trainer():
    def __init__(self,  model: nn.Module = CNN(), optimizer: optim = None,
                 loss: nn.modules.loss._Loss = nn.CrossEntropyLoss()):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) if not optimizer else optimizer
        self.loss_fn = loss

    def train(self, epoch, loaders: DataLoader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                print(f'Train epoch: {epoch} loss: {loss.item()}')

    def test(self, loaders: DataLoader):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in loaders['test']:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / len(loaders['test'].dataset)

            test_loss /= len(loaders['test'].dataset)
            print(f'Test avg loss: {test_loss:.4f}, Accuracy: {accuracy}')

    def save(self, path):
        save_object(
            file_path=path,
            obj=self.model
        )
