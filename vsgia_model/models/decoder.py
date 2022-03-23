import torch.nn as nn
from vsgia_model.models.utils.resnet import resnet50,Bottleneck

class HeatmapDecoder(nn.Module):

    def __init__(self):
        super(HeatmapDecoder,self).__init__()


        self.relu=nn.ReLU(inplace=True)


        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)



    def forward(self,scene_face_feat):

        x = self.deconv1(scene_face_feat)


        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x

class InoutDecoder(nn.Module):

    def __init__(self):
        super(InoutDecoder,self).__init__()

        self.relu=nn.ReLU(inplace=True)

        self.compress_conv1=nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=False)
        self.compress_bn1=nn.BatchNorm2d(256)
        self.compress_conv2=nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=False)
        self.compress_bn2=nn.BatchNorm2d(128)
        self.compress_conv3=nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.compress_bn3=nn.BatchNorm2d(1)

        self.fc_inout=nn.Linear(49,1)

    def forward(self,global_memory):

        x=self.compress_conv1(global_memory)
        x=self.compress_bn1(x)
        x=self.relu(x)

        x=self.compress_conv2(x)
        x=self.compress_bn2(x)
        x=self.relu(x)

        x=self.compress_conv3(x)
        x=self.compress_bn3(x)
        x=self.relu(x)

        x=x.view(-1,49)
        inout=self.fc_inout(x)

        return inout
