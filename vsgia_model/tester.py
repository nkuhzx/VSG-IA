import torch
import torch.nn as nn
import numpy as np
from vsgia_model.utils.utils import AverageMeter,euclid_dist,auc
from tqdm import tqdm

class Tester(object):

    def __init__(self,model,criterion,testloader,opt,writer=None):

        self.model=model
        self.criterion=criterion

        self.testloader=testloader

        self.dist=AverageMeter()
        self.mindist=AverageMeter()
        self.auc=AverageMeter()

        self.device=torch.device(opt.OTHER.device)

        self.opt=opt
        self.writer=writer

    @torch.no_grad()
    def test(self,opt):

        self.model.eval()

        self.dist.reset()
        self.mindist.reset()

        loader_capacity=len(self.testloader)
        pbar=tqdm(total=loader_capacity)

        for i,data in enumerate(self.testloader,0):

            x_img, x_face, x_hc = data["img"], data["face"], data["headloc"]

            x_maskimg, x_node_num, x_vis_feat, x_spa_feat = data["maskimg"], data["node_num"], data['vis_feat'], data[
                'spa_feat']

            gaze_value = data["gaze_label"]

            img_size=data["img_size"]

            x_img = x_img.to(self.device)
            x_face = x_face.to(self.device)
            x_hc = x_hc.to(self.device)

            x_maskimg = x_maskimg.to(self.device)
            x_vis_feat=x_vis_feat.to(self.device)
            x_spa_feat=x_spa_feat.to(self.device)

            inputs_size=x_img.tensors.size(0)

            outs=self.model(x_img, x_face, x_hc,x_maskimg,x_node_num,x_vis_feat,x_spa_feat)

            pred_heatmap=outs['heatmap']

            pred_heatmap=pred_heatmap.squeeze(1)
            pred_heatmap=pred_heatmap.data.cpu().numpy()

            # AUC
            auc_score=auc(gaze_value.numpy(),pred_heatmap,img_size.numpy())

            # mindist and avgdist
            disval=euclid_dist(pred_heatmap,gaze_value,type='avg')

            mindisval = euclid_dist(pred_heatmap, gaze_value, type='min')

            self.dist.update(disval,inputs_size)
            self.mindist.update(mindisval,inputs_size)
            self.auc.update(auc_score,inputs_size)

            pbar.set_postfix(dist=self.dist.avg,
                             mindist=self.mindist.avg,
                             auc=self.auc.avg)
            pbar.update(1)

        pbar.close()

        if self.writer is not None:

            self.writer.add_scalar("Val_avg_dist", self.dist.avg, global_step=opt.OTHER.global_step)
            self.writer.add_scalar("Val_min_dist", self.mindist.avg, global_step=opt.OTHER.global_step)
            self.writer.add_scalar("Val_auc", self.auc.avg, global_step=opt.OTHER.global_step)

        return self.dist.avg,self.mindist.avg,self.auc.avg