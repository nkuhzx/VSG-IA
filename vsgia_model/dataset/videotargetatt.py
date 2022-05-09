import glob

import numpy as np
import pandas as pd
import os
import math
import h5py

from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

from vsgia_model.utils import img_utils

from vsgia_model.models.utils.misc import nested_tensor_from_tensor_list

# from preprocess_tools.utils.utils import visualization_mask

import warnings
warnings.filterwarnings("ignore")

class VideoTargetAttLoader(object):

    def __init__(self,opt):

        self.train_gaze = VideoTargetAttDataset(opt.DATASET.train_anno, 'train', opt, show=False)
        self.val_gaze = VideoTargetAttDataset(opt.DATASET.test_anno, 'test', opt, show=False)


        self.train_loader=DataLoader(self.train_gaze,
                                     batch_size=opt.DATASET.train_batch_size,
                                     num_workers=opt.DATASET.load_workers,
                                     shuffle=True,
                                     collate_fn=collate_fn)


        self.val_loader=DataLoader(self.val_gaze,
                                   batch_size=opt.DATASET.test_batch_size,
                                   num_workers=opt.DATASET.load_workers,
                                   shuffle=False,
                                   collate_fn=collate_fn)


#deal the videoAttTarget as image
class VideoTargetAttDataset(Dataset):

    def __init__(self,csv_path,type,opt,show=False):

        test=True if type=="test" else False

        if test:

            df=pd.read_csv(os.path.join(csv_path),index_col=False)

            self.length=len(df)

            self.graph_info=h5py.File(opt.DATASET.test_graph,'r')

        else:

            df=pd.read_csv(os.path.join(csv_path),index_col=False)

            self.length=len(df)

            self.graph_info = h5py.File(opt.DATASET.train_graph, 'r')

        self.data_dir=opt.DATASET.root_dir
        self.mask_dir=opt.DATASET.mask_dir

        self.df=df

        transform_list = []
        transform_list.append(transforms.Resize((opt.TRAIN.input_size, opt.TRAIN.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(transform_list)

        self.test=test

        self.input_size = opt.TRAIN.input_size
        self.output_size = opt.TRAIN.output_size
        self.imshow = show

    def __getitem__(self, index):

        # index=self.maplist[index]

        row=self.df.loc[index]


        sub_id=row['sub_id']
        act_id=row['act_id']
        show_index=row['show_index']
        show_name=row['show_name']
        frame_scope=row['frame_scope']
        img_name=row['img_name']
        x_min=row['x_min']
        y_min=row['y_min']
        x_max=row['x_max']
        y_max=row['y_max']
        gaze_x=row['gaze_x']
        gaze_y=row['gaze_y']

        if self.test:
            pass
        else:

            b_x_min=row['bound_x_min']
            b_y_min=row['bound_y_min']
            b_x_max=row['bound_x_max']
            b_y_max=row['bound_y_max']

        # map to float
        x_min,y_min,x_max,y_max=map(float,[x_min,y_min,x_max,y_max])
        gaze_x,gaze_y=map(float,[gaze_x,gaze_y])

        # print(sub_id,act_id,show_index,show_name,frame_scope,img_name)
        # print(x_min,y_min,x_max,y_max,gaze_x,gaze_y)
        img_path=os.path.join(self.data_dir,show_name,frame_scope,img_name)
        mask_path=os.path.join(self.mask_dir,show_name,frame_scope,img_name.replace('jpg','npy'))


        # load the image
        img=Image.open(img_path)
        img=img.convert('RGB')
        width, height = img.size

        # load the mask
        maskimg=np.load(mask_path)
        # move the target person to top
        target_mask=np.expand_dims(maskimg[act_id,:,:],axis=0)

        maskimg=np.delete(maskimg,obj=act_id,axis=0)
        maskimg=np.insert(maskimg,obj=0,values=target_mask,axis=0)

        del_ids=np.where(maskimg[:,0,0]<0)[0]
        maskimg=np.delete(maskimg,obj=del_ids,axis=0)

        if maskimg.shape[0]==1:

            all_sense_mask=1-maskimg
            maskimg=np.concatenate([maskimg,all_sense_mask],axis=0)



        # get the nodenum visual feat and spatial feat
        node_num=self.graph_info[str(index)]["node_num"][()]
        visual_feat=self.graph_info[str(index)]["visual_feature"]
        spatial_feat=self.graph_info[str(index)]["spatial_feature"]

        spatial_feat=np.array(spatial_feat)

        # gaze inside or not
        if gaze_x==-1 and gaze_y==-1:

            gaze_inside=False
        else:
            if gaze_x<0: gaze_x=0
            if gaze_y<0: gaze_y=0
            gaze_inside=True

        # gaze_pixel=[math.ceil(gaze_x*width),math.ceil(gaze_y*height)]

        imsize = torch.IntTensor([width, height])

        if self.imshow:

            org_show=img.resize((224,224))
            orgmask_show=maskimg

        if self.test:

            if gaze_inside:

                gaze_x, gaze_y = gaze_x / width, gaze_y / height


        else:

            ## data augumentation

            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop
            if np.random.random_sample() <= 0.5:
            # if True:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if gaze_inside:

                    crop_x_min = np.min([gaze_x , x_min, x_max,b_x_min,b_x_max])
                    crop_y_min = np.min([gaze_y , y_min, y_max,b_y_min,b_y_max])
                    crop_x_max = np.max([gaze_x , x_min, x_max,b_x_min,b_x_max])
                    crop_y_max = np.max([gaze_y , y_min, y_max,b_y_min,b_y_max])

                else:
                    crop_x_min = np.min([x_min, x_max, b_x_min, b_x_max])
                    crop_y_min = np.min([y_min, y_max, b_y_min, b_y_max])
                    crop_x_max = np.max([x_min, x_max, b_x_min, b_x_max])
                    crop_y_max = np.max([y_min, y_max, b_y_min, b_y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                crop_list = [crop_y_min / height, (crop_y_min + crop_height) / height, crop_x_min / width,
                             (crop_x_min + crop_width) / width]
                crop_list = np.clip(crop_list, 0, 1)
                crop_list = np.array(crop_list) * maskimg.shape[1]
                crop_list = crop_list.round().astype(int)
                maskimg = maskimg[:, crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]]

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y

                # normalize to [0,1]
                if gaze_inside:
                    gaze_x, gaze_y = (gaze_x  - offset_x) / float(crop_width), \
                                     (gaze_y  - offset_y) / float(crop_height)

                    gaze_x=np.clip(gaze_x,0,1)
                    gaze_y=np.clip(gaze_y,0,1)
                # else:
                #     gaze_x = -1; gaze_y = -1


                # convert the spatial feat to cropped frame
                area_ratio = width * height / (float(crop_height) * float(crop_width))

                spatial_feat[:, [0, 2, 5, 7, 14]] = (spatial_feat[:, [0, 2, 5, 7, 14]] * width - offset_x) / float(crop_width)
                spatial_feat[:, [1, 3, 6, 8, 15]] = (spatial_feat[:, [1, 3, 6, 8, 15]] * height - offset_y) / float(crop_height)
                spatial_feat[:, [4, 9]] = spatial_feat[:, [4, 9]] * area_ratio

                width, height = crop_width, crop_height



            else:


                if gaze_inside:


                    gaze_x=gaze_x/width
                    gaze_y= gaze_y/height


            # Random flip
            if np.random.random_sample() <= 0.5:
            # if True:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                maskimg=np.flip(maskimg,axis=2)

                # depthimg=depthimg.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2

                if gaze_inside:
                    gaze_x = 1 - gaze_x



                # convert the spatial feat to filpped frame
                spatial_feat[:,[0,2,5,7,14]]=1-spatial_feat[:,[0,2,5,7,14]]



            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head_channel = img_utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        if self.imshow:
            img_show=img
            face_show=face

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)

        maskimg=maskimg.astype(np.float32)
        maskimg=torch.from_numpy(maskimg)
        maskimg=maskimg.unsqueeze(1)

        if maskimg.shape[2]!=self.input_size or maskimg.shape[3]!=self.input_size:
            maskimg=F.interpolate(maskimg,(self.input_size,self.input_size),mode="bilinear")
        maskimg=maskimg.squeeze(1)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        if gaze_inside:
            gaze_heatmap = img_utils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                   3,
                                                   type='Gaussian')

        maskimg=maskimg[1:,:,:]


        if maskimg.shape[0]!=node_num-1:
            print(maskimg.shape[0], node_num - 1)
            print(maskimg,node_num)

        all_data={}
        # X train
        all_data['img'] = img
        all_data['maskimg'] = maskimg
        all_data["face"] = face
        all_data["headloc"] = head_channel
        all_data["nodenum"] = node_num
        all_data["visfeature"] = visual_feat
        all_data["spafeature"] = spatial_feat

        # Y label
        all_data["gaze_heatmap"] = gaze_heatmap
        all_data["gaze_inside"]=gaze_inside

        cont_gaze=np.array([gaze_x,gaze_y])
        cont_gaze = torch.FloatTensor(cont_gaze)

        all_data["gaze_label"]=cont_gaze

        all_data["imsize"] = imsize

        return all_data

    def __len__(self):
        return self.length


def collate_fn(batch):

    batch_data={}

    batch_data["img"]=[]
    batch_data["maskimg"]=[]
    batch_data["headloc"]=[]
    batch_data["face"]=[]

    batch_data["node_num"]=[]
    batch_data["vis_feat"]=[]
    batch_data["spa_feat"]=[]


    batch_data["gaze_heatmap"]=[]
    batch_data["gaze_inside"]=[]
    # batch_data["gaze_vector"]=[]
    batch_data["gaze_label"]=[]

    batch_data["img_size"]=[]

    for data in batch:
        batch_data["img"].append(data["img"])
        batch_data["maskimg"].append(data["maskimg"])
        batch_data["face"].append(data["face"])
        batch_data["headloc"].append(data["headloc"])

        batch_data["node_num"].append(data["nodenum"])
        batch_data["vis_feat"].append(data["visfeature"])
        batch_data["spa_feat"].append(data["spafeature"])

        batch_data["gaze_heatmap"].append(data["gaze_heatmap"])
        batch_data["gaze_inside"].append(data["gaze_inside"])
        # batch_data["gaze_vector"].append(data["gaze_vector"])
        batch_data["gaze_label"].append(data["gaze_label"])

        batch_data["img_size"].append(data["imsize"])

    # train data
    batch_data["img"]=nested_tensor_from_tensor_list(batch_data["img"])
    batch_data["face"]=nested_tensor_from_tensor_list(batch_data["face"])
    batch_data["headloc"]=nested_tensor_from_tensor_list(batch_data["headloc"])

    batch_data["maskimg"]=torch.cat(batch_data["maskimg"],dim=0)

    batch_data['vis_feat'] = torch.FloatTensor(np.concatenate(batch_data['vis_feat'], axis=0))
    batch_data['spa_feat'] = torch.FloatTensor(np.concatenate(batch_data['spa_feat'], axis=0))

    # label data
    batch_data["gaze_heatmap"]=torch.stack(batch_data["gaze_heatmap"],0)
    batch_data["gaze_inside"] = torch.as_tensor(batch_data["gaze_inside"])
    # batch_data["gaze_vector"] = torch.stack(batch_data["gaze_vector"], 0)
    batch_data["gaze_label"] = torch.stack(batch_data["gaze_label"], 0)

    batch_data["img_size"]=torch.stack(batch_data["img_size"],0)

    return batch_data


