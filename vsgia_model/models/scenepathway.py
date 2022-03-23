import torch.nn as nn

from vsgia_model.models.utils.resnet import resnet50

class SceneNet(nn.Module):


    def __init__(self,pretrained=False):

        super(SceneNet,self).__init__()

        org_resnet=resnet50(pretrained)

        self.conv1=nn.Conv2d(5,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=org_resnet.bn1
        self.relu=org_resnet.relu
        self.maxpool=org_resnet.maxpool
        self.layer1=org_resnet.layer1
        self.layer2=org_resnet.layer2
        self.layer3=org_resnet.layer3
        self.layer4=org_resnet.layer4

        # add
        # self.layer5=self._make_layer(Bottleneck,org_resnet.inplanes,256,2,stride=1)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample =  None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)


        return x