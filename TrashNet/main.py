from model import LeNet, Resnet34
from dataset import Trash
import torch as t 
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable as V
import torchnet.meter as meter 
import numpy as numpy
from torchvision import models 
import time 
import os
# net = LeNet()
# net.save('pytorch_code')

def train(root, use_gpu = False):
    # data
    train_data = Trash(root, train = True)
    val_data = Trash(root, train = False)
    train_loader = DataLoader(
        train_data,
        batch_size = 16,
        shuffle = True,
        num_workers = 0
    )
    val_loader = DataLoader(
        val_data,
        batch_size = 4,
        shuffle = False,
        num_workers = 0
    )

    # model
    # model = Resnet34()
    # model.load('checkpoints/Resnet34_20191203_13:09:42.pth')
    model = models.resnet34(pretrained = True, num_classes = 1000) # use pretrained model
    model.fc = nn.Linear(512, 6)
    # model = LeNet()
    model.train()
    if use_gpu:
        model.cuda()

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = t.optim.Adam(
        model.parameters(),
        lr = lr,
        weight_decay = 1e-4
    )

    # training prepare
    avg_loss = meter.AverageValueMeter()
    cm = meter.ConfusionMeter(6)
    previous_loss = 1e100

    # train
    for epoch in range(300):
        avg_loss.reset()
        cm.reset()

        for batch_id, (data, label) in enumerate(train_loader):
            data, label = V(data), V(label)
            if use_gpu:
                data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()
            predict = model(data)
            loss = criterion(predict, label)
            loss.backward()

            optimizer.step()

            # record
            avg_loss.add(loss.item())
            cm.add(predict.data, label.data.long())

            if batch_id % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}, Train_acc: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                    100. * batch_id / len(train_loader), avg_loss.value()[0],
                    cm.value().trace()/cm.value().sum()
                    ))

        # model.save()

        # validation
        val_cm, val_accuracy = val(model, val_loader, use_gpu = use_gpu)
        print('Epoch:{epoch}, lr:{lr}, val_cm:\n{val_cm}, val_acc:{val_acc}'.format(
                    epoch = epoch,
                    lr = lr,
                    val_cm = val_cm,
                    val_acc = val_accuracy
                ))

        # 调整学习率
        if avg_loss.value()[0] > previous_loss:
            lr = lr * 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = avg_loss.value()[0]

    # model.save('')
    now = time.strftime('%Y%m%d_%H:%M:%S')
    check_path = os.path.join('checkpoints', str(model.__class__.__name__) + '_'+ now + '.pth')
    t.save(model.state_dict(), check_path)

def val(model, dataloader, use_gpu = False):
    model.eval()

    # confusion matrix, 6 classes
    cm = meter.ConfusionMeter(6)
    with t.no_grad():
        for data, label in dataloader:
            data, label = V(data), V(label)
            if use_gpu:
                data, label = data.cuda(), label.cuda()
            
            predict = model(data)
            cm.add(predict.data, label.data.long())

    model.train()
    accuracy = cm.value().trace() / cm.value().sum()
    return cm.value(), accuracy

if __name__ == '__main__':
    train('../datasets/dataset-resized', use_gpu = True)
