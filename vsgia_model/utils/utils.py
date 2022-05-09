import numpy as np

from collections import deque
from sklearn.metrics import average_precision_score
from PIL import Image
from sklearn.metrics import roc_auc_score

class AverageMeter():

    def __init__(self):

        self.reset()

    def reset(self):

        self.count=0
        self.newval=0
        self.sum=0
        self.avg=0

    def update(self,newval,n=1):

        self.newval=newval
        self.sum+=newval*n
        self.count+=n
        self.avg=self.sum/self.count

class MovingAverageMeter():

    def __init__(self,max_len=30):

        self.max_len=max_len

        self.reset()

    def reset(self):

        self.dq=deque(maxlen=self.max_len)
        self.count=0
        self.avg=0
        self.sum=0


    def update(self,newval):

        self.dq.append(newval)
        self.count=len(self.dq)
        self.sum=np.array(self.dq).sum()
        self.avg=self.sum/float(self.count)

# # Metric functions
def argmax_pts(heatmap):

    idx=np.unravel_index(heatmap.argmax(),heatmap.shape)
    pred_y,pred_x=map(float,idx)

    return pred_x,pred_y

def euclid_dist(pred,target,type='avg'):

    batch_dist=0.
    sample_dist=0.

    batch_size=pred.shape[0]
    pred_H,pred_W=pred.shape[1:]


    sample_dist_list=[]
    mean_gt_gaze_list=[]
    for b_idx in range(batch_size):

        pred_x,pred_y=argmax_pts(pred[b_idx])
        norm_p=np.array([pred_x,pred_y])/np.array([pred_W,pred_H])
        # print(norm_p,target[b_idx])
        # plt.imshow(pred[b_idx])
        # plt.show()


        b_target=target[b_idx]
        valid_target=b_target[b_target!=-1].view(-1,2)
        valid_target=valid_target.numpy()
        sample_dist=valid_target-norm_p

        sample_dist = np.sqrt(np.power(sample_dist[:, 0], 2) + np.power(sample_dist[:, 1], 2))



        if type=='avg':
            mean_gt_gaze = np.mean(valid_target, 0)
            sample_avg_dist = mean_gt_gaze - norm_p
            sample_avg_dist = np.sqrt(np.power(sample_avg_dist[0], 2) + np.power(sample_avg_dist[1], 2))

            sample_dist=float(sample_avg_dist)
        elif type=='min':
            sample_dist=float(np.min(sample_dist))

        elif type=="retained":
            mean_gt_gaze = np.mean(valid_target, 0)
            sample_avg_dist = mean_gt_gaze - norm_p
            sample_avg_dist = np.sqrt(np.power(sample_avg_dist[0], 2) + np.power(sample_avg_dist[1], 2))
            sample_dist=float(sample_avg_dist)

            mean_gt_gaze_list.append(mean_gt_gaze)
            sample_dist_list.append(sample_dist)

        else:
            raise NotImplemented

        batch_dist+=sample_dist

    euclid_dist=batch_dist/float(batch_size)

    if type=="retained":

        return mean_gt_gaze_list,sample_dist_list

    return euclid_dist

def auc(gt_gaze,pred_heatmap,imsize):
    batch_size=len(gt_gaze)

    auc_score_list=[]
    for b_idx in range(batch_size):

        multi_hot=multi_hot_targets(gt_gaze[b_idx],imsize[b_idx])
        scaled_heatmap=Image.fromarray(pred_heatmap[b_idx]).resize(size=(imsize[b_idx][0],imsize[b_idx][1]),
                                                                   resample=0)

        scaled_heatmap=np.array(scaled_heatmap)
        sample_auc_score=roc_auc_score(np.reshape(multi_hot,multi_hot.size),
                                np.reshape(scaled_heatmap,scaled_heatmap.size))
        auc_score_list.append(sample_auc_score)

    auc_score=sum(auc_score_list)/len(auc_score_list)

    return auc_score

def auc_videoatt(gt_gaze,pred_heatmap,imsize,output_resolution=64):

    batch_size=len(gt_gaze)

    valid_counter=0
    auc_score_list=[]
    for b_idx in range(batch_size):

        if -1 in  gt_gaze[b_idx]:
            continue
        else:

            multi_hot=np.zeros((output_resolution,output_resolution))
            multi_hot=draw_labelmap(multi_hot,[gt_gaze[b_idx][0]*output_resolution,gt_gaze[b_idx][1]*output_resolution],
                                    3,type="Gaussian")
            multi_hot=(multi_hot>0)
            multi_hot=multi_hot.astype(np.float)*1

            scaled_heatmap = Image.fromarray(pred_heatmap[b_idx]).resize(size=(output_resolution, output_resolution),
                                                                         resample=0)

            scaled_heatmap = np.array(scaled_heatmap)

            # plt.imshow(multi_hot,cmap='jet')
            # plt.show()
            # plt.imshow(scaled_heatmap,cmap='jet')
            # plt.show()

            sample_auc_score=roc_auc_score(np.reshape(multi_hot,multi_hot.size),
                                    np.reshape(scaled_heatmap,scaled_heatmap.size))

            auc_score_list.append(sample_auc_score)
            valid_counter+=1

    if valid_counter!=0:
        auc_score=sum(auc_score_list)/len(auc_score_list)
    else:
        auc_score=0

    return auc_score,valid_counter

def euclid_dist_videoatt(pred,target,type='avg'):

    batch_dist=0.
    sample_dist=0.

    batch_size=pred.shape[0]
    pred_H,pred_W=pred.shape[1:]


    outside_counter=0


    for b_idx in range(batch_size):

        pred_x,pred_y=argmax_pts(pred[b_idx])
        norm_p=np.array([pred_x,pred_y])/np.array([pred_W,pred_H])

        b_target=target[b_idx]
        if -1 in b_target:
            outside_counter+=1

            continue

        valid_target=b_target[b_target!=-1].view(-1,2)
        valid_target=valid_target.numpy()
        sample_dist=valid_target-norm_p

        sample_dist = np.sqrt(np.power(sample_dist[:, 0], 2) + np.power(sample_dist[:, 1], 2))



        if type=='avg':

            mean_gt_gaze = np.mean(valid_target, 0)

            sample_avg_dist = mean_gt_gaze - norm_p
            sample_avg_dist = np.sqrt(np.power(sample_avg_dist[0], 2) + np.power(sample_avg_dist[1], 2))

            sample_dist=float(sample_avg_dist)
        elif type=='min':
            sample_dist=float(np.min(sample_dist))
        else:
            raise NotImplemented

        batch_dist+=sample_dist

    if batch_size!=outside_counter:

        euclid_dist=batch_dist/(float(batch_size)-outside_counter)
    else:
        euclid_dist=0

    valid_num=batch_size-outside_counter

    return euclid_dist,valid_num

def ap(label,pred):
    return average_precision_score(label,pred)


def multi_hot_targets(gaze_pts,out_res):
    w,h= out_res
    target_map=np.zeros((h,w))
    for p in gaze_pts:
        if p[0]>=0:
            x,y=map(int,[p[0]*float(w),p[1]*float(h)])
            x=min(x,w-1)
            y=min(y,h-1)
            target_map[y,x]=1
    return target_map

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    # img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img = img/np.max(img) # normalize heatmap so it has max value of 1
    return img
