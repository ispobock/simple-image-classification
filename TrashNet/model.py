import torch as t 
import torch.nn as nn
import time 
import os
import torch.nn.functional as F

# 加入load和save方法
class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, path):
        now = time.strftime('%Y%m%d_%H:%M:%S')
        check_path = os.path.join(path, str(self.__class__.__name__) + '_'+ now + '.pth')
        t.save(self.state_dict(), check_path)

# model = BasicModule()
# model.save('pytorch_code')

class LeNet(BasicModule):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # ((224 - 4) / 2 - 4) / 2
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

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


class Resnet34(BasicModule):
    def __init__(self, num_classes = 6):
        super(Resnet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.layer1 = self.make_layer(64, 128, num_blocks = 3, stride = 1)
        self.layer2 = self.make_layer(128, 256, num_blocks = 4, stride = 2)
        self.layer3 = self.make_layer(256, 512, num_blocks = 6, stride = 2)
        self.layer4 = self.make_layer(512, 512, num_blocks = 3, stride = 2)

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
        x = F.avg_pool2d(x, kernel_size = 7)
        x = x.view(x.size()[0], -1)
        return self.fc(x)
