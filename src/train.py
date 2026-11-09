import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sklearn import metrics

import utils
import config

from model import Model

def train_batch(model, data_loader, loss_fn, optimizer):
    model.train()
    device = utils.get_model_device(model)

    total_loss, total_correct, total_samples = 0, 0, 0

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(X)
        loss = loss_fn(out, y)

        #l1_reg, l2_reg = 0, 0

        #for fc in model.apply_L1_and_L2_on:
            #l1_reg += fc.weight.abs().sum()
            #l2_reg += fc.weight.square().sum()

        #regulated_loss = loss + 1e-5*l1_reg + 1e-4*l2_reg

        l2_reg = 0
        
        for fc in model.apply_regularisation:
            l2_reg += fc.weight.square().sum()
        
        regulated_loss = loss + 1e-5*l2_reg

        regulated_loss.backward()
        optimizer.step()

        cur_batch_size = y.size(0)

        total_loss += loss.item() * cur_batch_size
        total_correct += (out.argmax(1)==y).sum().item()
        total_samples += cur_batch_size

    avg_loss = total_loss/total_samples
    accuracy = 100*(total_correct/total_samples)

    return avg_loss, accuracy

def test_batch(model, data_loader, loss_fn, scheduler):
    model.eval()
    device = utils.get_model_device(model)

    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)
            loss = loss_fn(out, y)

            cur_batch_size = y.size(0)

            total_loss += loss.item() * cur_batch_size
            total_correct += (out.argmax(1)==y).sum().item()
            total_samples += cur_batch_size

    avg_loss = total_loss/total_samples
    accuracy = 100*(total_correct/total_samples)

    scheduler.step(avg_loss)

    return avg_loss, accuracy, scheduler.get_last_lr()[0]

def get_confusion_matrix(model, data_loader):
    model.eval()
    device = utils.get_model_device(model)
    
    preds, labels = [], []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
    
            out = model(X)

            preds.extend(out.argmax(1).detach().contiguous().cpu().numpy())
            labels.extend(y.detach().contiguous().numpy())

    return torch.from_numpy(metrics.confusion_matrix(labels, preds, normalize="true"))

def get_sate_dict(training_save_path=str):
    save = torch.load(training_save_path)
    return save["model_state_dict"]

def get_training_data(training_save_path=str):
    save = torch.load(training_save_path)
    data = {
        "train_loss" : save["train_loss_history"],
        "test_loss" : save["test_loss_history"],
        "train_accuracy" : save["train_accuracy_history"],
        "test_accuracy" : save["test_accuracy_history"],
        "learning_rate" : save["learning_rate_history"],
        "confusion_matrix" : save["confusion_matrix"],
    }
    return data

class Trainer():
    def __init__(self, model=Model, optimizer=Optimizer, scheduler=LRScheduler, loss_fn=CrossEntropyLoss, train_dataloader=DataLoader, test_dataloader=DataLoader):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.loss_fn = loss_fn

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.train_loss_history = []
        self.test_loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.learning_rate_history = []

        self.epoch = 0

    def load(self, path=str):
        save = torch.load(path)
        self.model.load_state_dict(save["model_state_dict"])
        self.optimizer.load_state_dict(save["optimizer_state_dict"])
        self.scheduler.load_state_dict(save["scheduler_state_dict"])

        self.train_loss_history = save["train_loss_history"]
        self.test_loss_history = save["test_loss_history"]
        self.train_accuracy_history = save["train_accuracy_history"]
        self.test_accuracy_history = save["test_accuracy_history"]
        self.learning_rate_history = save["learning_rate_history"]

        self.epoch = save["epoch"]

    def save(self, path=str):
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "scheduler_state_dict" : self.scheduler.state_dict(),

            "train_loss_history" : self.train_loss_history,
            "test_loss_history" : self.test_loss_history,
            "train_accuracy_history" : self.train_accuracy_history,
            "test_accuracy_history" : self.test_accuracy_history,
            "learning_rate_history" : self.learning_rate_history,
            "confusion_matrix" : get_confusion_matrix(self.model, self.test_dataloader),

            "epoch" : self.epoch,
        }, path)

    def train(self, epochs=1):
        for i in range(epochs):
            train_loss, train_accuracy = train_batch(self.model, self.train_dataloader, self.loss_fn, self.optimizer)

            print(f"Epoch {str(self.epoch)} trained")

            test_loss, test_accuracy, learning_rate = test_batch(self.model, self.test_dataloader, self.loss_fn, self.scheduler)

            print(f"Epoch {str(self.epoch)} tested: loss: {round(test_loss, config.loss_decimals)}, accuracy: {round(test_accuracy, config.percentage_decimals)}%, learning rate: {self.optimizer.param_groups[0]["lr"]}")

            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.test_accuracy_history.append(test_accuracy)
            self.learning_rate_history.append(learning_rate)

            self.epoch += 1

    def train_until(self, epoch):
        epochs = epoch - self.epoch
        if epochs > 0:
            self.train(epochs)