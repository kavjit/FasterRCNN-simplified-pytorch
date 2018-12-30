from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.RPN_tools import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch 

Losses = namedtuple('Losses',
                       ['rpn_reg_loss',
                        'rpn_classifier_loss',
                        'head_reg_loss',
                        'head_classifier_loss',
                        'total_loss',
                        'total_rpn',
                        'total_head'
                        ])

def l1loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def regressor_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = l1loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float()) 
    return loc_loss


class TrainStep(nn.Module):
    """ wrapper for conveniently training. return losses """

    def __init__(self, fasterrcnn):
        super(Trainer, self).__init__()

        self.fasterrcnn = fasterrcnn
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = fasterrcnn.loc_normalize_mean
        self.loc_normalize_std = fasterrcnn.loc_normalize_std

        lr = 1e-3
        params = []
        for key, value in dict(fasterrcnn.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        
        self.optimizer = torch.optim.SGD(params, momentum=0.9)


    def forward(self, inp_img, bboxes, labels, scale):
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

        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(at.tonumpy(bbox),anchor,img_size)

        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        
        rpn_reg_loss = _fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data,self.rpn_sigma)
        rpn_classifier_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        head_reg_loss = t.tensor([0]).cuda()
        head_classifier_loss = t.tensor([0]).cuda()

        losses = [rpn_reg_loss, rpn_classifier_loss, head_reg_loss, head_classifier_loss]
        losses = losses + [sum(losses)] + [rpn_reg_loss + rpn_classifier_loss] + [head_reg_loss + head_classifier_loss]
        all_losses = LossTuple(*losses)
        all_losses.total_rpn.backward()
        self.optimizer.step()
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

        head_reg_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(),gt_roi_loc,gt_roi_label.data,self.roi_sigma)
        head_classifier_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        rpn_reg_loss = t.tensor([0]).cuda()
        rpn_classifier_loss = t.tensor([0]).cuda()

        losses = [rpn_reg_loss, rpn_classifier_loss, head_reg_loss, head_classifier_loss]
        losses = losses + [sum(losses)] + [rpn_reg_loss + rpn_classifier_loss] + [head_reg_loss + head_classifier_loss]
        
        all_losses = LossTuple(*losses)
        all_losses.total_head.backward()
        self.optimizer.step()
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
        
        rpn_reg_loss = _fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data,self.rpn_sigma)
        rpn_classifier_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        head_reg_loss = t.tensor([0]).cuda()
        head_classifier_loss = t.tensor([0]).cuda()

        losses = [rpn_reg_loss, rpn_classifier_loss, head_reg_loss, head_classifier_loss]
        losses = losses + [sum(losses)] + [rpn_reg_loss + rpn_classifier_loss] + [head_reg_loss + head_classifier_loss]
        all_losses = LossTuple(*losses)
        all_losses.total_rpn.backward()
        self.optimizer.step()
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

        head_reg_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(),gt_roi_loc,gt_roi_label.data,self.roi_sigma)
        head_classifier_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        rpn_reg_loss = t.tensor([0]).cuda()
        rpn_classifier_loss = t.tensor([0]).cuda()

        losses = [rpn_reg_loss, rpn_classifier_loss, head_reg_loss, head_classifier_loss]
        losses = losses + [sum(losses)] + [rpn_reg_loss + rpn_classifier_loss] + [head_reg_loss + head_classifier_loss]
        
        all_losses = LossTuple(*losses)
        all_losses.total_head.backward()
        self.optimizer.step()
        return all_losses



