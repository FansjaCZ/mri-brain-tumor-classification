from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from matplotlib import pyplot as plt

from model import Model
from data import Nickparvar
from train import Trainer
import config

from pathlib import Path
import time
import sys

def main(path=None):
    dataset = Nickparvar(str(Path(__file__).parent.parent / config.dataset_dir), config.im_size)
    test_dataloader = DataLoader(dataset.get_test(), config.batch_size, True)
    train_dataloader = DataLoader(dataset.get_train(), config.batch_size, True)

    model = Model(config.im_size)
    model.to(config.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config.learning_rate)

    trainer = Trainer(model, optimizer, loss_fn, train_dataloader, test_dataloader)

    if path and Path(path).exists():
        trainer.load(path)
        print(f"Loaded training from {path}")

    epoch_start = trainer.epoch
    time_start = time.time()

    trainer.train_until(config.epochs)

    print(f"Completed {config.epochs-epoch_start} epochs in {round(time.time()-time_start, config.time_decimals)}s")

    if path:
        trainer.save(path)
        print(f"Saved training to {path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()