from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from utils import array_tool as at

from utils.config import configurations
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss',
                        'total_rpn',
                        'total_roi'
                        ])

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = configurations.rpn_sigma
        self.roi_sigma = configurations.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        lr = configurations.lr
        params = []
        for key, value in dict(faster_rcnn.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': configurations.weight_decay}]
        
        self.optimizer = t.optim.SGD(params, momentum=0.9)
        #self.optimizer = self.faster_rcnn.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale, epoch):

        pass


    def step1(self, imgs, bboxes, labels, scale, epoch): #train RPN alone
        self.optimizer.zero_grad()
        _, _, H, W = imgs.shape
        img_size = (H, W)

        ############ EXTRACTOR STEP #################
        features1 = self.faster_rcnn.extractor1(imgs)


        ############ RPN STEP #######################
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features1, img_size, scale)
        
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(at.tonumpy(bbox),anchor,img_size)

        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data,self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        roi_loc_loss = t.tensor([0]).cuda()
        roi_cls_loss = t.tensor([0]).cuda()

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] + [rpn_loc_loss + rpn_cls_loss] + [roi_loc_loss + roi_cls_loss]
        all_losses = LossTuple(*losses)
        all_losses.total_rpn.backward()
        self.optimizer.step()
        self.update_meters(all_losses)
        return all_losses




    def step2(self, imgs, bboxes, labels, scale, epoch):
        self.optimizer.zero_grad()
        _, _, H, W = imgs.shape
        img_size = (H, W)

        ############ EXTRACTOR STEP #################
        features1 = self.faster_rcnn.extractor1(imgs)
        features2 = self.faster_rcnn.extractor2(imgs)

        ############ RPN STEP #######################
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features1, img_size, scale)
        
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        ############ HEAD STEP #######################
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi,at.tonumpy(bbox),at.tonumpy(label),self.loc_normalize_mean,self.loc_normalize_std)
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(features2,sample_roi,sample_roi_index)
        
        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(),gt_roi_loc,gt_roi_label.data,self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        rpn_loc_loss = t.tensor([0]).cuda()
        rpn_cls_loss = t.tensor([0]).cuda()

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] + [rpn_loc_loss + rpn_cls_loss] + [roi_loc_loss + roi_cls_loss]
        
        all_losses = LossTuple(*losses)
        all_losses.total_roi.backward()
        self.optimizer.step()
        self.update_meters(all_losses)
        return all_losses



    def step3(self, imgs, bboxes, labels, scale, epoch):
        self.optimizer.zero_grad()
        _, _, H, W = imgs.shape
        img_size = (H, W)

        ############ EXTRACTOR STEP #################
        features2 = self.faster_rcnn.extractor2(imgs)


        ############ RPN STEP #######################
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features2, img_size, scale)
        
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(at.tonumpy(bbox),anchor,img_size)

        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data,self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        roi_loc_loss = t.tensor([0]).cuda()
        roi_cls_loss = t.tensor([0]).cuda()

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] + [rpn_loc_loss + rpn_cls_loss] + [roi_loc_loss + roi_cls_loss]
        all_losses = LossTuple(*losses)
        all_losses.total_rpn.backward()
        self.optimizer.step()
        self.update_meters(all_losses)
        return all_losses



    def step4(self, imgs, bboxes, labels, scale, epoch):
        self.optimizer.zero_grad()
        _, _, H, W = imgs.shape
        img_size = (H, W)

        ############ EXTRACTOR STEP #################
        features2 = self.faster_rcnn.extractor2(imgs)

        ############ RPN STEP #######################
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features2, img_size, scale)
        
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        ############ HEAD STEP #######################
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi,at.tonumpy(bbox),at.tonumpy(label),self.loc_normalize_mean,self.loc_normalize_std)
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(features2,sample_roi,sample_roi_index)
        
        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(),gt_roi_loc,gt_roi_label.data,self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        rpn_loc_loss = t.tensor([0]).cuda()
        rpn_cls_loss = t.tensor([0]).cuda()

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] + [rpn_loc_loss + rpn_cls_loss] + [roi_loc_loss + roi_cls_loss]
        
        all_losses = LossTuple(*losses)
        all_losses.total_roi.backward()
        self.optimizer.step()
        self.update_meters(all_losses)
        return all_losses


######################################################################################
    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}




