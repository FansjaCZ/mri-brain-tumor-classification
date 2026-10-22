import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import utils
import config

from model import Model

def train_batch(model, data_loader, loss_fn, optimizer):
    model.train()
    device = utils.get_model_device(model)

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        loss = loss_fn(out, y)

        dense_weights = model.block_dense1.weight
        regulated_loss = loss + 1e-5 * dense_weights.abs().sum() + 1e-4 * dense_weights.square().sum()

        regulated_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_batch(model, data_loader, loss_fn, scheduler):
    model.eval()
    device = utils.get_model_device(model)

    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            total_loss += loss_fn(out, y).item()
            total_correct += (out.argmax(1)==y).type(torch.float).sum().item()

    avg_loss = total_loss/len(data_loader)
    scheduler.step(avg_loss)
    accuracy = 100*(total_correct/len(data_loader.dataset))

    return avg_loss, accuracy, scheduler.get_last_lr()[0]

def get_sate_dict(training_save_path=str):
    save = torch.load(training_save_path)
    return save["model_state_dict"]

def get_accuracy_data(training_save_path=str):
    save = torch.load(training_save_path)
    return save["accuracy_history"]

def get_loss_data(training_save_path=str):
    save = torch.load(training_save_path)
    return save["avg_loss_history"]

def get_learning_rate_data(training_save_path=str):
    save = torch.load(training_save_path)
    return save["learning_rate_history"]

class Trainer():
    def __init__(self, model=Model, optimizer=Optimizer, scheduler=LRScheduler, loss_fn=CrossEntropyLoss, train_dataloader=DataLoader, test_dataloader=DataLoader):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.avg_loss_history = []
        self.accuracy_history = []
        self.learning_rate_history = []
        self.epoch = 0

    def load(self, path=str):
        save = torch.load(path)
        self.model.load_state_dict(save["model_state_dict"])
        self.optimizer.load_state_dict(save["optimizer_state_dict"])
        self.scheduler.load_state_dict(save["scheduler_state_dict"])
        self.avg_loss_history = save["avg_loss_history"]
        self.accuracy_history = save["accuracy_history"]
        self.learning_rate_history = save["learning_rate_history"]
        self.epoch = save["epoch"]

    def save(self, path=str):
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "scheduler_state_dict" : self.scheduler.state_dict(),
            "avg_loss_history" : self.avg_loss_history,
            "accuracy_history" : self.accuracy_history,
            "learning_rate_history" : self.learning_rate_history,
            "epoch" : self.epoch,
        }, path)

    def train(self, epochs=1):
        for i in range(epochs):
            train_batch(self.model, self.train_dataloader, self.loss_fn, self.optimizer)

            print(f"Epoch {str(self.epoch)} trained")

            avg_loss, accuracy, learning_rate = test_batch(self.model, self.test_dataloader, self.loss_fn, self.scheduler)

            print(f"Epoch {str(self.epoch)} tested. Loss: {round(avg_loss, config.loss_decimals)}, accuracy: {round(accuracy, config.percentage_decimals)}%, learning rate: {self.optimizer.param_groups[0]["lr"]}")

            self.avg_loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)
            self.learning_rate_history.append(learning_rate)

            self.epoch += 1

    def train_until(self, epoch):
        epochs = epoch - self.epoch
        if epochs > 0:
            self.train(epochs)