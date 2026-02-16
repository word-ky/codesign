import torch.nn as nn
class SEBlock(nn.Module):
    def __init__(self, channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, internal_neurons, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(internal_neurons, channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.avgpool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out