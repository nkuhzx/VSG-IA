import torch.nn as nn
from collections import OrderedDict

def get_activation(name):
    if name=='ReLU':
        return nn.ReLU(inplace=True)
    elif name=='Tanh':
        return nn.Tanh()
    # elif name=='Identity':
    #     return Identity()
    elif name=='Sigmoid':
        return nn.Sigmoid()
    elif name=='LeakyReLU':
        return nn.LeakyReLU(0.2,inplace=True)
    else:
        assert(False), 'Not Implemented'

class mlp(nn.Module):

    def __init__(self,layer_sizes,activation,bias=True,drop_prob=None):

        super(mlp,self).__init__()

        self.layers=nn.ModuleList()

        for i in range(len(layer_sizes)-1):

            layer=nn.Linear(layer_sizes[i],layer_sizes[i+1],bias=bias)

            activate_fuc=get_activation(activation[i])

            # add linear
            block=nn.Sequential(OrderedDict([
                ('linear{}'.format(i),layer)
            ]))
            # add activation function
            block.add_module("activate{}".format(i),activate_fuc)

            # add drop out
            if drop_prob:
                block.add_module("dropout".format(i),nn.Dropout(drop_prob))

            self.layers.append(block)


    def forward(self,x):

        for layer in self.layers:
            x=layer(x)
        return x