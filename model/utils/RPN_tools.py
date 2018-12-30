import numpy as np

from model.utils.bbox_ops import reg_scales, calculate_iou, reg_bbox
from model.utils.non_maximum_suppression import non_maximum_suppression


class ProposalTargetCreator(object):
    def __init__(self, samples=128, FG_ratio=0.25, FG_IOU=0.5):
        self.samples = samples
        self.FG_ratio = FG_ratio
        self.FG_IOU = FG_IOU
        
    def __call__(self, roi, BB, label):
        
        loc_normalize_mean = (0,0,0,0)
        loc_normalize_std=(0.1,0.1,0.2,0.2)
        roi = np.concatenate((roi,BB), axis = 0) #concatenating to ensure that some correct roi are also passed to classifer
        
        FG_boxes = np.round(self.samples*self.FG_ratio)
        iou = calculate_iou(roi,BB) #returns the iou for each roi with all the bboxes - usally like 1000x2 etc for 2 gt boxes
        ground_truth_index = iou.argmax(axis=1) #returns the index num of the gt selected for each gt 1000,
        max_overlap_roi = iou.max(axis=1) #maximum iou for each roi
        
        ROI_label = label[ground_truth_index]+1 #retruns the label of each roi based on the max iou
                                                #1 is added to include background class as 0
                                                #note for proposal target creator its not just back/foreground its also object class
                                                
        FG_index = np.where(max_overlap_roi >= self.FG_IOU)[0] #indices of roi meetin foreground criteria - objects
        
        n_FG = int(min(FG_boxes, FG_index.size)) #minimum between allowed positive and calculated positive quantity
        if FG_index.size > 0:#from all the obtained foreground boxes, we select the required quantity randomly
            FG_index = np.random.choice(FG_index, size = n_FG, replace = False)
        
        #similarly for background
        BG_index = np.where(max_overlap_roi < self.FG_IOU)[0]
        n_BG = self.samples - FG_boxes
        n_BG = int(min(n_BG, BG_index.size))
        
        if BG_index.size > 0:
            BG_index = np.random.choice(BG_index, size = n_BG, replace=False)
        
        keep_index = np.append(FG_index, BG_index)
        ROI_label = ROI_label[keep_index]
        ROI_label[n_FG:] = 0
        select_roi = roi[keep_index]    #the 128 roi's that we're selecting
        
        #offset and scale roi according to the gt
        offset_roi = reg_scales(select_roi, BB[ground_truth_index[keep_index]])
        offset_roi = ((offset_roi - np.array(loc_normalize_mean, np.float32))/np.array(loc_normalize_std, np.float32))
        
        return select_roi, offset_roi, ROI_label #returns the selected 128 roi, normalized roi scaled, and it's label wrt nearest gt
        

class AnchorTargetCreator(object):
    def __init__(self, samples=256, FG_threshold=0.7, BG_threshold=0.3,
                 FG_ratio=0.5):
        self.samples = samples
        self.FG_threshold = FG_threshold
        self.BG_threshold=BG_threshold
        self.FG_ratio = FG_ratio
        
    def __call__(self,BB,anchor,img_size):
        
        H, W = img_size
        anchor_length = len(anchor)
        inside_index = np.where(    #indices of all the anchors which are within limits of image
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= H) &
            (anchor[:, 3] <= W)
        )[0]
        anchor = anchor[inside_index]
        argmax_IoUs, label = self.create_label(inside_index, anchor, BB)
        
        loc = reg_scales(anchor, BB[argmax_IoUs])
        label = unmap(label, anchor_length, inside_index, fill=-1)
        loc = unmap(loc, anchor_length, inside_index, fill=0)
        
        return loc, label
    
    
    def create_label(self, inside_index, anchor, BB):
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_IoUs, max_IoUs, gt_argmax_ious = self.calc_ious(anchor, BB, inside_index)

        label[max_IoUs < self.BG_threshold] = 0
        label[gt_argmax_ious] = 1
        label[max_IoUs >= self.FG_threshold] = 1

        num_pos = int(self.FG_ratio * self.samples)
        FG_index = np.where(label == 1)[0]
        if len(FG_index) > num_pos:
            disable_index = np.random.choice(
                FG_index, size=(len(FG_index) - num_pos), replace=False)
            label[disable_index] = -1

        num_neg = self.samples - np.sum(label == 1)
        BG_index = np.where(label == 0)[0]
        if len(BG_index) > num_neg:
            disable_index = np.random.choice(
                BG_index, size=(len(BG_index) - num_neg), replace=False)
            label[disable_index] = -1

        return argmax_IoUs, label

    def calc_ious(self, anchor, BB, inside_index):
        ious = calculate_iou(anchor, BB)
        argmax_IoUs = ious.argmax(axis=1)
        max_IoUs = ious[np.arange(len(inside_index)), argmax_IoUs]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_IoUs, max_IoUs, gt_argmax_ious


def unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class ProposalCreator:

    def __init__(self,
                 parent_model,
                 threshold=0.7,
                 train_images_before_nms=12000,
                 train_images_after_nms=2000,
                 test_images_before_nms=6000,
                 test_images_after_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.threshold = threshold
        self.train_images_before_nms = train_images_before_nms
        self.train_images_after_nms = train_images_after_nms
        self.test_images_before_nms = test_images_before_nms
        self.test_images_after_nms = test_images_after_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.train_images_before_nms
            n_post_nms = self.train_images_after_nms
        else:
            n_pre_nms = self.test_images_before_nms
            n_post_nms = self.test_images_after_nms

        roi = reg_bbox(anchor, loc)

        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]

        keep = non_maximum_suppression(
            np.ascontiguousarray(np.asarray(roi)),thresh=self.threshold)
        
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi



