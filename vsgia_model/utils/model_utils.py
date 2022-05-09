import torch
import torch.nn as nn
import torch.optim as optim
import os
import math

import vsgia_model.models.gazenet as gazenet
from vsgia_model.models.utils.transformer import build_transformer
from vsgia_model.models.vsgat import GraphDealModule

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1 :

        nn.init.kaiming_normal_(m.weight.data)

    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.zero_()

def init_model(opt):

    transformer=build_transformer(opt.TRFM)
    vsgat=GraphDealModule(opt,opt.OTHER.device)

    model=gazenet.VSGIANet(transformer,vsgat,opt.TRFM.hidden_dim,opt.TRFM.num_queries,
                              pretrained=True,inout_branch=opt.MODEL.inout_branch)

    # model.head_backbone.load_state_dict(torch.load(opt.TRAIN.headpretrain))
    # model.head_backbone.layer5.apply(weights_init)
    # loss the transform
    # model.scene_backbone.conv1.apply(weights_init)

    # model.heatmap_decoder.apply(weights_init)

    if opt.MODEL.inout_branch:
        model.inout_decoder.apply(weights_init)


    model=model.to(opt.OTHER.device)
    return model


def setup_model(model,opt):

    if opt.TRAIN.criterion=="crossentropy":
        criterion=nn.NLLLoss()
    elif opt.TRAIN.criterion=="mse":
        criterion = nn.MSELoss(reduction='none')
    elif opt.TRAIN.criterion=="mixed":
        criterion=[nn.MSELoss(reduction='none'),nn.BCEWithLogitsLoss()]

    else:
        raise NotImplemented

    if opt.TRAIN.optimizer=="sgd":

        optimizer=optim.SGD(model.parameters(),
                            lr=opt.TRAIN.maxlr,
                            momentum=opt.TRAIN.momentum,
                            weight_decay=opt.TRAIN.weightDecay,
                            nesterov=opt.TRAIN.nesterov)

    elif opt.TRAIN.optimizer=="adam":

        optimizer=optim.Adam([{'params': model.parameters(), 'initial_lr': opt.TRAIN.maxlr}],
                             lr=opt.TRAIN.maxlr,)
                             #weight_decay=opt.TRAIN.weightDecay)

    else:
        raise NotImplemented

    return criterion,optimizer


def save_checkpoint(model,optimizer,best_error,epoch,opt):

    cur_state={
        'epoch':epoch+1,
        'state_dict':model.state_dict(),
        'best_err':best_error,
        'optimizer':optimizer.state_dict()
    }

    epochnum=str(epoch)

    filename='vsgia'+'_'+epochnum+'epoch.pth.tar'


    torch.save(cur_state,os.path.join(opt.TRAIN.store,filename))


def resume_checkpoint(model,optimizer,opt):

    checkpoint=torch.load(opt.TRAIN.resume_add)
    opt.TRAIN.start_epoch=checkpoint['epoch']
    best_error=checkpoint['best_err']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> Loading checkpoint '{}' (epoch {})".format(opt.TRAIN.resume,opt.TRAIN.start_epoch))

    return model, optimizer, best_error, opt

def init_checkpoint(model,opt):


    checkpoint=torch.load(opt.TRAIN.initmodel_add)

    model.load_state_dict(checkpoint['state_dict'])

    print("=> Loading init checkpoint ".format(opt.TRAIN.initmodel))

    return model