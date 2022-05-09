import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from vsgia_model.models.scenepathway import SceneNet
from vsgia_model.models.headpathway import HeadNet
from vsgia_model.models.vsgat import GraphDealModule
from vsgia_model.models.decoder import HeatmapDecoder,InoutDecoder

from vsgia_model.models.utils.transformer import build_transformer
from vsgia_model.models.utils.position_encoding import PositionEmbeddingLearned

from vsgia_model.models.utils.misc import NestedTensor


class VSGIANet(nn.Module):

    def __init__(self, transformer,vsgraph, hidden_dim=256, num_queries=16,output_size=64, pretrained=False,inout_branch=False):

        super(VSGIANet, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.inout_branch=inout_branch

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.vsgraph=vsgraph

        self.scene_backbone=SceneNet(pretrained)
        self.head_backbone=HeadNet()

        self.input_proj = nn.Conv2d(3072, hidden_dim, kernel_size=1)
        self.encoder = transformer
        self.posembedding = PositionEmbeddingLearned(hidden_dim // 2)


        self.heatmap_decoder=HeatmapDecoder()

        if self.inout_branch:
            self.inout_decoder=InoutDecoder()


    def forward(self, simg: NestedTensor, face: NestedTensor, head_loc:NestedTensor, mask,nodenum,v_feat,s_feat):

        # batch_size
        bs = simg.tensors.size(0)
        wei, hei = simg.tensors.size(2), simg.tensors.size(3)

        # infer the interactive probability
        inter_prob=self.vsgraph(nodenum,v_feat,s_feat)
        inter_prob=inter_prob.unsqueeze(1)

        # infer the interactive attention map according to interactive
        mask=mask.unsqueeze(3)
        mask = mask.reshape(-1, 224 * 224, 1)
        max_value=torch.max(mask.detach(), dim=1)[0].reshape(-1,1,1)
        max_value[max_value==0]=1
        mask=mask/max_value

        interatt = torch.matmul(mask, inter_prob)
        interatt = interatt.reshape(-1, 224, 224)

        split_index = (np.array(nodenum) - 1).tolist()
        interatt_tuple = interatt.split(split_index, dim=0)

        batch_interatt = torch.stack([torch.mean(i, dim=0) for i in interatt_tuple], dim=0)
        batch_interatt=batch_interatt.unsqueeze(1)

        # obtain the saliency with interactive attention feat
        scene_feats = self.scene_backbone(torch.cat((simg.tensors, head_loc.tensors,batch_interatt), dim=1))
        scene_mask = F.interpolate(simg.mask[None].float(), size=scene_feats.shape[-2:]).to(torch.bool)[0]

        head_feats = self.head_backbone(face.tensors)
        head_mask=F.interpolate(face.mask[None].float(),size=head_feats.shape[-2:]).to(torch.bool)[0]

        # fed into the encoder builted by transformer
        input_src=torch.cat((scene_feats,head_feats),dim=1)
        input_mask=torch.bitwise_or(scene_mask,head_mask)

        input_features = NestedTensor(input_src, input_mask)
        input_pos = self.posembedding(input_features)

        query_embed= self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        global_memory = self.encoder(self.input_proj(input_src), input_mask, query_embed, input_pos)[0]

        # predicte the heatmap and inout
        heatmap = self.heatmap_decoder(global_memory)

        if self.inout_branch:

            inout=self.inout_decoder(global_memory)
        else:
            inout=None

        outs = {
            'heatmap': heatmap,
            'inout':inout,
            'inter_prob':inter_prob,
            "inter_att":batch_interatt
        }

        return outs


from vsgia_model.config import cfg

from torch.utils.data import DataLoader, DistributedSampler


import matplotlib.pyplot as plt

if __name__ == '__main__':

    # init the model
    check = False

    # get the data from the dataset
    cfg.merge_from_file("/home/nku120/Desktop/VSG-IA/vsgia_model/config/gazefollow_cfg.yaml")
    print(cfg.DATASET.train_anno)
    # train_dataset = GazeDataset(cfg.DATASET.test, type='test', opt=cfg, show=False)

    # construct the transform
    transform_component = build_transformer(cfg.TRFM)
    vsgat_component=GraphDealModule(cfg,device="cpu")


    # construct the gaze netowork
    testnet = VSGIANet(transform_component,vsgat_component ,cfg.TRFM.hidden_dim, pretrained=True)

    print(testnet)
    testnet.eval()

    raise NotImplemented
    # gazedirpara_path="/home/hzx/project/gaze_transformer/gazefollowmodel/model_para/gazedirmodule.pt"
    # testnet.head_backbone.load_state_dict(torch.load(gazedirpara_path))

    # loader
    # model_para_path = "/home/extend/hzx/gazetr_savemodels/gazetrnet_2epoch.pth.tar"
    # testnet.load_state_dict(torch.load(model_para_path)['state_dict'])

    # data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train = DataLoader(train_dataset, shuffle=True, batch_size=1,
                                   collate_fn=collate_fn)

    if check:
        check_point = torch.load(cfg.TRAIN.resume_add)

        testnet.load_state_dict(check_point['state_dict'])

    for idx, data in enumerate(data_loader_train, 0):
        # img, real_depthimg, face, head_loc, gaze_field, gaze_heatmap, gaze_inside, gaze_vector2d, gaze_value = data

        img,face,headloc=data["img"],data["face"],data["headloc"]

        maskimg,node_num,vis_feat,spa_feat=data["maskimg"],data["node_num"],data['vis_feat'],data['spa_feat']

        gaze_heatmap=data["gaze_heatmap"]
        gaze_inside=data["gaze_inside"]
        gaze_value=data["gaze_label"]

        outs = testnet(img, face, headloc, maskimg,node_num,vis_feat,spa_feat)

        pred_heatmap = outs["heatmap"]

        # pred_gazedir = outs["gazedir"]


        criterion = nn.CosineSimilarity(dim=1,eps=1e-6)


        # test_loss = torch.sum(test_loss) / torch.sum(gaze_inside)
        # print(test_loss)

        # pred_fov = outs["fovheatmap"]
        # pred_reldimg_filter = outs['depinduce']

        # for show img
        outimg = unnorm(img.tensors.numpy()) * 255
        outimg = np.clip(outimg, 0, 255)
        outimg = outimg.astype(np.uint8)
        outimg = outimg.squeeze()
        outimg = np.transpose(outimg, (1, 2, 0))

        # for show rel depth img
        # test_real_depthimg = test_real_depthimg.squeeze()
        # test_real_depthimg = test_real_depthimg.numpy()

        # for show heatmap
        pred_heatmap = pred_heatmap.squeeze()
        pred_heatmap = pred_heatmap.detach().numpy()

        # for show fov
        # pred_fov = pred_fov.squeeze()
        # pred_fov = pred_fov.detach().numpy()

        # for show depth induce
        # pred_reldimg_filter = pred_reldimg_filter.squeeze()
        # pred_reldimg_filter = pred_reldimg_filter.detach().numpy()

        # gt heatmap
        gt_heatmap = gaze_heatmap.squeeze()
        gt_heatmap = gt_heatmap.numpy()

        # gt
        figure, ax = plt.subplots(2, 4)
        figure.set_size_inches(20, 8)

        ax[0][0].imshow(outimg)
        # ax[0][1].imshow(test_real_depthimg, cmap='jet')

        # ax[1][2].imshow(pred_fov, cmap='jet')
        ax[1][0].imshow(pred_heatmap, cmap='jet')
        ax[1][1].imshow(gt_heatmap, cmap='jet')
        # ax[1][3].imshow(pred_reldimg_filter, cmap='jet')
        plt.show()


