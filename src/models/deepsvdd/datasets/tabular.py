import torch
from torch.utils.data import DataLoader

from models.deepsvdd.base.base_dataset import BaseADDataset


class ADBenchDataset(BaseADDataset):
    def __init__(self, X_train, y_train, weights):
        super(ADBenchDataset, self).__init__(root="")
        X_train = torch.from_numpy(X_train).float()
        y_train = (
            torch.from_numpy(y_train).float()
            if y_train is not None
            else torch.zeros(len(X_train), dtype=torch.float32)
        )
        index = torch.arange(0, len(X_train))

        if weights is not None:
            self.train_data = list(zip(X_train, y_train, index, weights))
        else:
            self.train_data = list(zip(X_train, y_train, index))

    def loaders(
        self, batch_size: int = 64, num_workers: int = 0
    ) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return train_loader, None
