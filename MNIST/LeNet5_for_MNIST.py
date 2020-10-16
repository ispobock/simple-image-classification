import torch as t 
from torchvision import datasets
from torchvision import transforms as T 
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable as V
import torchnet.meter as meter 
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style("darkgrid")

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

# 训练集, 60000, (1, 28, 28)
trainset = datasets.MNIST(
    root = './',
    train = True,
    download = True,
    transform = transform
)

train_loader = DataLoader(
    trainset,
    batch_size = 32,
    shuffle = True,
    num_workers = 0
)

# 测试集, 10000
testset = datasets.MNIST(
    root = './',
    train = False,
    download = True,
    transform = transform
)

test_loader = DataLoader(
    testset,
    batch_size = 32,
    shuffle = False,
    num_workers = 0
)


# 定义LenNet-5网络结构
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

loss_list = []
test_acc_list = []

def train(use_gpu = False):
    model = LeNet5()
    model.train()
    if use_gpu:
        model.cuda()
    
    # loss & optimizer
    L = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = t.optim.Adam(
        model.parameters(),
        lr = lr,
        weight_decay = 1e-4
    )

    # training prepare
    avg_loss = meter.AverageValueMeter()
    cm = meter.ConfusionMeter(10)
    previous_loss = 1e100

    # train
    for epoch in range(100):
        avg_loss.reset()
        cm.reset()

        for batch_id, (data, label) in enumerate(train_loader):
            data, label = V(data), V(label)
            if use_gpu:
                data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()
            predict = model(data)
            loss = L(predict, label)
            loss.backward()

            optimizer.step()

            # record
            avg_loss.add(loss.item())
            cm.add(predict.data, label.data.long())

            if batch_id % 400 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}, Train_accuracy: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                    100. * batch_id / len(train_loader), avg_loss.value()[0],
                    cm.value().trace()/cm.value().sum()
                    ))

        # validation
        test_cm, test_accuracy = test(model, use_gpu = use_gpu)
        print('Epoch:{epoch}, lr:{lr}, test_cm:\n{test_cm}, test_accuracy:{test_acc}'.format(
                    epoch = epoch,
                    lr = lr,
                    test_cm = test_cm,
                    test_acc = test_accuracy
                ))
                
        loss_list.append(avg_loss.value()[0])
        test_acc_list.append(test_accuracy)

        # 调整学习率
        if avg_loss.value()[0] > previous_loss:
            lr = lr * 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = avg_loss.value()[0]

    # save model
    now = time.strftime('%Y%m%d_%H:%M:%S')
    check_path = os.path.join('checkpoints', str(model.__class__.__name__) + '_'+ now + '.pth')
    t.save(model.state_dict(), check_path)

def test(model, use_gpu = False):
    model.eval()
    # confusion matrix, 10 classes
    cm = meter.ConfusionMeter(10)
    with t.no_grad():
        for data, label in test_loader:
            data, label = V(data), V(label)
            if use_gpu:
                data, label = data.cuda(), label.cuda()
            
            predict = model(data)
            cm.add(predict.data, label.data.long())

    model.train()
    accuracy = cm.value().trace() / cm.value().sum()
    return cm.value(), accuracy

if __name__ == '__main__':
    train(use_gpu=True)

    # plot result
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 8))
    plt.subplot(1, 2, 1)
    sns.lineplot(range(1, 101), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss of LeNet5 for MNIST')
    plt.subplot(1, 2, 2)
    sns.lineplot(range(1, 101), test_acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy of LeNet5 for MNIST')
    plt.savefig('figs/LeNet5_mnist.jpg')

