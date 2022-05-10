import torch
import torch.nn as nn
import numpy as np
from vsgia_model.utils.utils import AverageMeter,MovingAverageMeter,euclid_dist_videoatt,auc_videoatt,ap
from tqdm import tqdm

class Tester(object):

    def __init__(self,model,criterion,testloader,opt,writer=None):

        self.model=model
        self.criterion=criterion

        self.testloader=testloader

        self.dist=AverageMeter()
        self.ap=AverageMeter()
        self.auc=AverageMeter()

        self.device=torch.device(opt.OTHER.device)

        self.opt=opt
        self.writer=writer

    @torch.no_grad()
    def test(self,epoch,opt):

        self.model.eval()

        self.dist.reset()

        loader_capacity=len(self.testloader)
        pbar=tqdm(total=loader_capacity)

        label_inoutlist=[]
        input_inout_list=[]
        for i,data in enumerate(self.testloader,0):

            # data read
            x_img, x_face, x_hc = data["img"], data["face"], data["headloc"]

            x_maskimg, x_node_num, x_vis_feat, x_spa_feat = data["maskimg"], data["node_num"], data['vis_feat'], data[
                'spa_feat']

            heatmap = data["gaze_heatmap"]
            in_out = data["gaze_inside"]
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

            pred_inout=outs['inout']
            pred_inout=pred_inout.squeeze()
            pred_inout=pred_inout.data.cpu().numpy()
            in_out=in_out.squeeze().numpy()

            # for visualized
            # visualized(pred_heatmap, pred_gheatmap, heatmap, gaze_dv)

            # mindist and avgdist
            disval,disval_num=euclid_dist_videoatt(pred_heatmap,gaze_value,type='avg')


            label_inoutlist.extend(in_out)
            input_inout_list.extend(pred_inout)

            #AUC
            auc_score,aucval_num=auc_videoatt(gaze_value.numpy(),pred_heatmap,img_size.numpy())

            self.dist.update(disval,disval_num)
            self.auc.update(auc_score,aucval_num)

            pbar.set_postfix(dist=self.dist.avg,
                             ap=self.ap.avg,
                             auc=self.auc.avg)
            pbar.update(1)

        apval=ap(label_inoutlist,input_inout_list)
        self.ap.update(apval)
        pbar.set_postfix(dist=self.dist.avg,
                         ap=self.ap.avg,
                         auc=self.auc.avg)


        pbar.close()

        if self.writer is not None:

            self.writer.add_scalar("Val avg dist", self.dist.avg, global_step=opt.OTHER.global_step)
            self.writer.add_scalar("Val ap", self.ap.avg, global_step=opt.OTHER.global_step)


        return self.dist.avg,self.auc.avg,self.ap.avg

