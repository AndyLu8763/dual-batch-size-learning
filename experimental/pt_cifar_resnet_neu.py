import torch

class BasicBlock(torch.nn.Module):
    def __init__(self, inplanes: int, planes: int, down: bool = False) -> None:
        super().__init__()
        self.down = down
        self.conv1 = (
            torch.nn.Conv2d(inplanes, planes, 3, stride=2, padding=1, bias=False) if down
            else torch.nn.Conv2d(inplanes, planes, 3, padding='same', bias=False)
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, padding='same', bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        if self.down:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, 1, stride=2, bias=False),
                torch.nn.BatchNorm2d(planes)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BottleNeck(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # not complete yet, cause it does not used in cifar resnet 

class CIFAR_ResNet(torch.nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.inplanes = 16
        self.planes = self.inplanes
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.inplanes, kernel_size=3, padding='same', bias=False),
            torch.nn.BatchNorm2d(self.inplanes),
            torch.nn.ReLU(inplace=True)
        )
        self.layer1 = torch.nn.Sequential(
            BasicBlock(self.inplanes, self.planes),
            BasicBlock(self.planes, self.planes),
            BasicBlock(self.planes, self.planes),
        )
        self.inplanes = self.planes
        self.planes *= 2
        self.layer2 = torch.nn.Sequential(
            BasicBlock(self.inplanes, self.planes, down=True),
            BasicBlock(self.planes, self.planes),
            BasicBlock(self.planes, self.planes),
        )
        self.inplanes = self.planes
        self.planes *= 2
        self.layer3 = torch.nn.Sequential(
            BasicBlock(self.inplanes, self.planes, down=True),
            BasicBlock(self.planes, self.planes),
            BasicBlock(self.planes, self.planes),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(64, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
