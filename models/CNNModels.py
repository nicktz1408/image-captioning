import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class CNN3Layer(nn.Module):
    def __init__(self, input_channels=3, output_dim=256):
        super(CNN3Layer, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4) # (B, 32, 56, 56)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=4) # (B, 64, 14, 14)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(64) # (B, 64, 14, 14)

        self.output_dim = 64 * 14 * 14
    
    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.pool1(x)

        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.pool2(x)

        x = self.bn3(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten the output for the RNN

        return x

'''
ResNet18 architecture:
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
'''

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)
        self.resnet18.fc = self.resnet18.avgpool = nn.Identity()  # Remove the final fully connected layer

        self.output_dim = 512 * 49  # I guess the final layer (layer4) feature output has 512 channels and is 7*7

    def forward(self, x):
        x = self.resnet18(x)

        #print(x.shape)
        return x

class ResNet18FeatureExtractorAttention(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18FeatureExtractorAttention, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)
        self.resnet18.fc = self.resnet18.avgpool = nn.Identity()  # Remove the final fully connected layer

        self.output_dim = 512 * 49  # I guess the final layer (layer4) feature output has 512 channels and is 7*7
        self.features_dim = 512 # 512 output channels on layer4

    def forward(self, x):
        x = self.resnet18(x) # flattened (B, 512 * 49)
        x = x.view(-1, 512, 49) # (B, 512, 49)

        x = x.permute(0, 2, 1)  # (B, 49, 512) focus on regional features

        return x
    
class ResNet18MultipleFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18MultipleFeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)

        self.conv1 = self.resnet18.conv1
        self.bn1 = self.resnet18.bn1
        self.relu = self.resnet18.relu
        self.maxpool = self.resnet18.maxpool
        self.layer1 = self.resnet18.layer1
        self.layer2 = self.resnet18.layer2
        self.layer3 = self.resnet18.layer3
        self.layer4 = self.resnet18.layer4

        self.output_dim = 128 * 14 * 14 + 512 * 49  # I guess the final layer (layer4) feature output has 512 channels and is 7*7
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)

        l3 = self.layer3(x)
        l4 = self.layer4(l3)

        l3 = l3[:, :128] # get only the first 128 channels of layer3 output

        l3 = l3.view(l3.size(0), -1)
        l4 = l4.view(l4.size(0), -1)

        x = torch.cat([l3, l4], dim=1)
        
        return x

class ResNet34FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34FeatureExtractor, self).__init__()
        self.resnet34 = resnet34(pretrained=pretrained)
        self.resnet34.fc = self.resnet34.avgpool = nn.Identity()  # Remove the final fully connected layer

        self.output_dim = 512 * 49  # same as ResNet18

    def forward(self, x):
        x = self.resnet34(x)
        
        return x

if __name__ == "__main__":
    resnet18 = ResNet18FeatureExtractor(pretrained=False)
    resnet18.eval()
    x = torch.randn(1, 3, 224, 224)  # Example input
    x = resnet18(x)  # Forward pass
    #print(resnet18)
    #print(resnet18.layer4[1].bn2.)

    print('ResNet34')
    print(resnet34(pretrained=False))
    print(resnet34(pretrained=False).layer4)
    #resnet34 = ResNet34FeatureExtractor(pretrained=False)

    

