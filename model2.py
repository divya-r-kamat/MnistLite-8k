import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 8, 3)   # 28 -> 26
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 10, 3)  # 26 -> 24
        self.bn2 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(2, 2)   # 24 -> 12
        # Transition
        self.conv3 = nn.Conv2d(10, 8, 1)
        self.bn3 = nn.BatchNorm2d(8)

        # Block 2
        self.conv4 = nn.Conv2d(8, 16, 3)  # 12 -> 10
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3)  # 10 -> 8
        self.bn5 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)    # 8 -> 4

        # Transition
        self.conv6 = nn.Conv2d(16, 10, 1)
        self.bn6 = nn.BatchNorm2d(10)
        self.conv7 = nn.Conv2d(10, 10, 3)  # 4 - > 2

        # GAP instead of conv7
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)

        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)

        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
