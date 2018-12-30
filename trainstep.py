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
                        'total_loss'
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

    def __init__(self, faster_rcnn):
        super(TrainStep, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        lr = 1e-3
        params = []
        for key, value in dict(faster_rcnn.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        
        self.optimizer = torch.optim.SGD(params, momentum=0.9)


    def forward(self, inp_img, bboxes, labels, scale):

        self.optimizer.zero_grad()

        _, _, H, W = inp_img.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(inp_img)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            bbox.detach().cpu().numpy(),
            label.detach().cpu().numpy())

        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # RPN losses 
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox.detach().cpu().numpy(),
            anchor,
            img_size)
        gt_rpn_label = torch.from_numpy(gt_rpn_label).cuda().long()
        gt_rpn_loc = torch.from_numpy(gt_rpn_loc).cuda()
        rpn_reg_loss = regressor_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)


        rpn_classifier_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score.detach().cpu().numpy()[gt_rpn_label.detach().cpu().numpy() > -1]

        #Head Losses
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                              torch.from_numpy(gt_roi_label).cuda().long()]
        gt_roi_label = torch.from_numpy(gt_roi_label).cuda().long()
        gt_roi_loc = torch.from_numpy(gt_roi_loc).cuda()

        head_reg_loss = regressor_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        head_classifier_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        losses = [rpn_reg_loss, rpn_classifier_loss, head_reg_loss, head_classifier_loss]
        losses = losses + [sum(losses)]
        all_losses = Losses(*losses)
        all_losses.total_loss.backward()
        self.optimizer.step()
        all_losses = {k: v.item() for k, v in all_losses._asdict().items()}
        return all_losses





