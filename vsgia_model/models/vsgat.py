import numpy as np

import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

from vsgia_model.models.utils.utils import mlp

class EdgeApplyModule(nn.Module):

    def __init__(self,cfg):
        super(EdgeApplyModule, self).__init__()

        self.multi_attn=cfg.VSGAT.multi_attn

        self.edge_fc=mlp(cfg.VSGAT.edge_apply_layers,
                         cfg.VSGAT.edge_apply_activate,
                         cfg.VSGAT.edge_apply_bias,
                         cfg.VSGAT.edge_apply_dropout)

    def forward(self,edge):

        feat=torch.cat([edge.src['n_f'],edge.data['s_f'],edge.dst['n_f']],dim=1)

        edge_feat=self.edge_fc(feat)

        return {'e_f':edge_feat}

class EdgeAttentionApplyModule(nn.Module):

    def __init__(self,cfg):
        super(EdgeAttentionApplyModule,self).__init__()
        self.attn_fc=mlp(cfg.VSGAT.edge_attn_layers,
                         cfg.VSGAT.edge_attn_activate,
                         cfg.VSGAT.edge_attn_bias,
                         cfg.VSGAT.edge_attn_dropout)

    def forward(self,edge):

        a_feat=self.attn_fc(edge.data['e_f'])

        return {'a_feat':a_feat}

class NodeApplyModule(nn.Module):

    def __init__(self,cfg):

        super(NodeApplyModule,self).__init__()

        self.node_fc=mlp(cfg.VSGAT.node_apply_layers,
                         cfg.VSGAT.node_apply_activate,
                         cfg.VSGAT.node_apply_bias,
                         cfg.VSGAT.node_apply_dropout)

    def forward(self,node):

        feat=torch.cat([node.data['n_f'],node.data['z_f']],dim=1)
        new_feat=self.node_fc(feat)

        return {'new_n_f':new_feat}

class Vsgatmodel(nn.Module):

    def __init__(self,cfg):
        super(Vsgatmodel,self).__init__()

        self.apply_edge=EdgeApplyModule(cfg)
        self.apply_edge_attn=EdgeAttentionApplyModule(cfg)
        self.apply_node=NodeApplyModule(cfg)

        self.diff_edge=cfg.VSGAT.diff_edge

        if cfg.VSGAT.diff_edge:

            self.apply_t_o_edge=EdgeApplyModule(cfg)
            self.apply_o_o_edge=EdgeApplyModule(cfg)
            self.apply_t_node=NodeApplyModule(cfg)
            self.apply_o_node=NodeApplyModule(cfg)

        self.predictor=mlp(cfg.VSGAT.pred_layers,
                         cfg.VSGAT.pred_activate,
                         cfg.VSGAT.pred_bias,
                         cfg.VSGAT.pred_dropout)

    def _message_func(self,edges):

        return {'old_n_f':edges.src['n_f'],'e_f':edges.data['e_f'],'a_feat':edges.data['a_feat']}

    def _reduce_func(self,nodes):

        # the implement for gat (without multi-head)

        alpha=F.softmax(nodes.mailbox['a_feat'],dim=1)

        z_add_f=nodes.mailbox['old_n_f']+nodes.mailbox['e_f']

        z_f=torch.sum(alpha*z_add_f,dim=1)

        return {'z_f':z_f}

    def _interaction_feature(self,edges):

        feat=torch.cat([edges.src['new_n_f'],edges.data['s_f'],edges.dst['new_n_f']],dim=1)

        pred=self.predictor(feat)

        return {'pred':pred}


    def forward(self,graphs,t_nodes,o_nodes,t_o_edges,o_o_edges):



        if self.diff_edge:

            # concat the spatial feature and node feature to obtain the new feature
            # focus on target and other human/object
            graphs.apply_edges(self.apply_edge,graphs.edges())
            if len(t_o_edges)!=0:
                graphs.apply_edges(self.apply_t_o_edge,tuple(zip(*t_o_edges)))

            # focus on human/object with other human/object
            if len(o_o_edges)!=0:
                graphs.apply_edges(self.apply_o_o_edge,tuple(zip(*o_o_edges)))

            # calcuatate the softmax attention score
            graphs.apply_edges(self.apply_edge_attn)

            # gat to update all edge and node
            graphs.update_all(self._message_func,self._reduce_func)

            # concat the origianl feature and gat feature
            # should not be 0 (at least have one target person)
            if len(t_nodes)!=0:
                graphs.apply_nodes(self.apply_t_node,t_nodes)

            if len(o_nodes)!=0:
                graphs.apply_nodes(self.apply_o_node, o_nodes)



        else:

            # concat the spatial feature and node feature to obtain the new feature
            graphs.apply_edges(self.apply_edge,graphs.edges())
            # calcuatate the softmax attention score
            graphs.apply_edges(self.apply_edge_attn)
            # gat to update all edge and node
            graphs.update_all(self._message_func,self._reduce_func)
            # concat the origianl feature and gat feature
            graphs.apply_nodes(self.apply_node,graphs.nodes())


        # read out the interaction feature
        graphs.apply_edges(self._interaction_feature,tuple(zip(*t_o_edges)))

        inter_pred=graphs.edges[tuple(zip(*t_o_edges))].data['pred']

        return inter_pred


class GraphDealModule(nn.Module):

    def __init__(self,cfg,device):

        super(GraphDealModule,self).__init__()


        self.cfg=cfg

        self.gatmodel=Vsgatmodel(cfg)
        self.gatmodel.to(device)

        self.device=device

    def _creat_batch_graph(self,node_num_list):

        batch_graph=[]
        batch_target_h_node_list=[]
        batch_o_node_list=[]

        batch_t_o_e_list=[]
        batch_o_o_e_list=[]

        # accumulate for all graph
        node_num_cum=np.cumsum(node_num_list)
        node_num_cum[1:]=node_num_cum[0:-1]
        node_num_cum[0]=0

        for node_num,node_space in zip(node_num_list,node_num_cum) :

            s_graph,s_target_h_node_list,s_o_node_list,s_t_o_e_list, s_o_o_e_list=\
            self._deal_single_graph(node_num)

            batch_graph.append(s_graph)

            s_target_h_node_list=(np.array(s_target_h_node_list)+node_space).tolist()
            s_o_node_list=(np.array(s_o_node_list)+node_space).tolist()

            s_t_o_e_list=(np.array(s_t_o_e_list)+node_space).tolist()
            s_o_o_e_list=(np.array(s_o_o_e_list)+node_space).tolist()

            # add to batch
            batch_target_h_node_list.extend(s_target_h_node_list)
            batch_o_node_list.extend(s_o_node_list)

            batch_t_o_e_list.extend(s_t_o_e_list)
            batch_o_o_e_list.extend(s_o_o_e_list)

        batch_graph=dgl.batch(batch_graph)

        return batch_graph,batch_target_h_node_list,batch_o_node_list,batch_t_o_e_list,batch_o_o_e_list

    def _deal_single_graph(self,node_num):


        all_node_list=np.arange(node_num)
        target_h_node_list=all_node_list[0:1]
        other_node_list=all_node_list[1:]

        # fully-connected edge list
        edge_list = []
        # target person and other human or object edge list
        t_o_e_list = []
        # other human and other human edge list
        o_o_e_list = []

        # get all edge in the fully-connected graph
        for src in range(node_num):
            for dst in range(node_num):
                if src==dst:
                    continue
                else:
                    edge_list.append((src,dst))

        # get target_other edges && other_ohter edges
        for src in target_h_node_list:
            for dst in other_node_list:
                if src==dst: continue
                t_o_e_list.append((src,dst))

        for src in other_node_list:
            for dst in other_node_list:
                if src==dst: continue
                o_o_e_list.append((src,dst))

        src,dst=tuple(zip(*edge_list))
        graph=dgl.graph((src,dst),num_nodes=node_num)
        graph=graph.to(self.device)

        # graph.add_nodes(node_num)
        #
        # src,dst=tuple(zip(*edge_list))
        #
        # print(src,dst)
        # graph.add_edges(src,dst)


        return graph,target_h_node_list,other_node_list,t_o_e_list,o_o_e_list


    def forward(self,node_num_list,visual_feat=None,spatial_feat=None):

        b_graph,b_t_node_list,b_o_node_list,b_t_o_e_list,b_o_o_e_list=self._creat_batch_graph(node_num_list)

        b_graph.ndata['n_f']=visual_feat
        b_graph.edata['s_f']=spatial_feat

        feature=self.gatmodel(b_graph,b_t_node_list,b_o_node_list,b_t_o_e_list,b_o_o_e_list)

        return feature



