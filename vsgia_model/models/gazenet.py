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


