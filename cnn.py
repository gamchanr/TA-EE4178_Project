import torch
import torch.nn as nn
import torch.nn.functional as F


#class ConvNet(nn.Module):
#    def __init__(self, num_classes):
#        super(ConvNet, self).__init__()
#        self.layer1 = nn.Sequential(
#                nn.Conv2d(3, 16, 5, stride=1, padding=2),
#                nn.BatchNorm2d(16),
#                nn.ReLU(),
#                nn.MaxPool2d(2))
#        self.layer2 = nn.Sequential(
#                nn.Conv2d(16, 32, 5, stride=1, padding=2),
#                nn.BatchNorm2d(32),
#                nn.ReLU(),
#                nn.MaxPool2d(2))
#        self.fc1 = nn.Linear(8*8*32, 120)
#        self.fc2 = nn.Linear(120, num_classes)
#
#    def forward(self, x):
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = x.reshape(x.size(0), -1)
#        x = F.relu(self.fc1(x))
#        x = F.softmax(self.fc2(x))
#        return x

class ConvNet(nn.Module):
    def __init__(self, num_classes=50):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, out_channels=6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Linear(in_features=8*8*16 ,out_features=120)
        self.layer4 = nn.Linear(in_features=120, out_features=84)
        self.out_layer = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.out_layer(x)
        return x
