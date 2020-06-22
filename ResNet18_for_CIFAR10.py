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

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# 训练集, 50000, (3, 32, 32)
trainset = datasets.CIFAR10(
    root = './',
    train = True,
    download = True,
    transform = transform_train
)

train_loader = DataLoader(
    trainset,
    batch_size = 128,
    shuffle = True,
    num_workers = 0
)

# 测试集, 10000
testset = datasets.CIFAR10(
    root = './',
    train = False,
    download = True,
    transform = transform_test
)

test_loader = DataLoader(
    testset,
    batch_size = 128,
    shuffle = False,
    num_workers = 0
)


# 定义ResNet-34网络结构
class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride = 1, shortcut = None):
        super(ResidualBlock, self).__init__()
        
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace = True),
            nn.Conv2d(c_out, c_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(c_out)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual 
        return F.relu(out)


class Resnet18(nn.Module):
    def __init__(self, num_classes = 10):
        super(Resnet18, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )

        self.layer1 = self.make_layer(64, 128, num_blocks = 2, stride = 1)
        self.layer2 = self.make_layer(128, 256, num_blocks = 2, stride = 2)
        self.layer3 = self.make_layer(256, 512, num_blocks = 2, stride = 2)
        self.layer4 = self.make_layer(512, 512, num_blocks = 2, stride = 2)

        self.fc = nn.Linear(512, num_classes)


    def make_layer(self, c_in, c_out, num_blocks, stride = 1):
        shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = 1, stride = stride, padding = 0, bias = False),
            nn.BatchNorm2d(c_out)
        )

        blocks = []
        blocks.append(ResidualBlock(c_in, c_out, stride = stride, shortcut = shortcut))
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(c_out, c_out, stride = 1))

        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, kernel_size = 4)
        x = x.view(x.size()[0], -1)
        return self.fc(x)

loss_list = []
test_acc_list = []

def train(use_gpu = False):
    model = Resnet18()
    model.load_state_dict(t.load('checkpoints/Resnet18_20191231_01:41:58.pth'))
    model.train()
    if use_gpu:
        model.cuda()
    
    # loss & optimizer
    L = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = t.optim.Adam(
        model.parameters(),
        lr = lr,
        weight_decay = 1e-3
    )

    # training prepare
    avg_loss = meter.AverageValueMeter()
    cm = meter.ConfusionMeter(10)
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
            loss = L(predict, label)
            loss.backward()

            optimizer.step()

            # record
            avg_loss.add(loss.item())
            cm.add(predict.data, label.data.long())

            if batch_id % 100 == 0:
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
        if epoch > 100 and epoch < 200:
            lr = 0.0001
        elif epoch > 200:
            if avg_loss.value()[0] > previous_loss:
                lr = lr * 0.90
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
    sns.lineplot(range(1, 301), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss of ResNet18 for CIFAR10')
    plt.subplot(1, 2, 2)
    sns.lineplot(range(1, 301), test_acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy of ResNet18 for CIFAR10')
    plt.savefig('figs/resnet18_cifar10.jpg')