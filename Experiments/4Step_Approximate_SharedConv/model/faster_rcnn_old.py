from __future__ import  absolute_import
from __future__ import division
import torch 
import numpy as np

from utils import array_tool as at
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.utils.bbox_tools import loc2bbox
from model.utils.nms.non_maximum_suppression import non_maximum_suppression
from model.roi_module import RoIPooling2D

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import configurations




class FasterRCNN(nn.Module):

    """Base class for Faster R-CNN"""

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16
    
    def __init__(self,no_of_classes=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
                ):
        super(FasterRCNN, self).__init__()
        
        model1 = vgg16(not configurations.load_path)
        model2 = vgg16(not configurations.load_path)

        #getting the classifier from VGG16
        classifier = model1.classifier
        classifier = list(classifier)
        del classifier[6]
        if not configurations.use_drop:
            del classifier[5]
            del classifier[2]
        self.classifier = nn.Sequential(*classifier)

        ########### Defining two extractors ##########
        # getting the features freezing the top conv layers
        features1 = list(model1.features)[:30] 
        for layer in features1[:10]:
            for param in layer.parameters():
                param.requires_grad = False
        self.extractor1 = nn.Sequential(*features1)

        features2 = list(model2.features)[:30] 
        for layer in features2[:10]:
            for param in layer.parameters():
                param.requires_grad = False
        self.extractor2 = nn.Sequential(*features2)
        ##############################################

        self.rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )
        
        self.head = VGG16RoIHead(
            n_class=no_of_classes + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=self.classifier
        )

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.nms_thresh = 0.3
        self.score_thresh = 0.05


    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class


    def forward(self, x, scale=1.):

        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        """
        
        img_size = x.shape[2:]

        h = self.extractor2(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices



    def nms_suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                np.array(cls_bbox_l), self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    
    def predict(self, imgs,sizes=None,visualize=False):

        """Detect objects from images"""

        self.eval()
        with torch.no_grad():
            if visualize:
                self.nms_thresh = 0.3
                self.score_thresh = 0.7
                prepared_imgs = list()
                sizes = list()
                for img in imgs:
                    size = img.shape[1:]
                    img = preprocess(at.tonumpy(img))
                    prepared_imgs.append(img)
                    sizes.append(size)
            else:
                 prepared_imgs = imgs 
            bboxes = list()
            labels = list()
            scores = list()
            for img, size in zip(prepared_imgs, sizes):
                img = at.totensor(img[None]).float()
                scale = img.shape[3] / size[1]
                roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
                # We are assuming that batch size is 1.
                roi_score = roi_scores.data
                roi_cls_loc = roi_cls_loc.data
                roi = at.totensor(rois) / scale

                # Convert predictions to bounding boxes in image coordinates.
                # Bounding boxes are scaled to the scale of the input images.
                mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
                    repeat(self.n_class)[None]
                std = torch.Tensor(self.loc_normalize_std).cuda(). \
                    repeat(self.n_class)[None]

                roi_cls_loc = (roi_cls_loc * std + mean)
                roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
                roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
                cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                    at.tonumpy(roi_cls_loc).reshape((-1, 4)))
                cls_bbox = at.totensor(cls_bbox)
                cls_bbox = cls_bbox.view(-1, self.n_class * 4)
                # clip bounding box
                cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
                cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

                prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

                raw_cls_bbox = at.tonumpy(cls_bbox)
                raw_prob = at.tonumpy(prob)

                bbox, label, score = self.nms_suppress(raw_cls_bbox, raw_prob)
                bboxes.append(bbox)
                labels.append(label)
                scores.append(score)

            self.nms_thresh = 0.3
            self.score_thresh = 0.05
            self.train()
        return bboxes, labels, scores

    

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
    

class VGG16RoIHead(nn.Module):

    """Faster R-CNN Head for VGG-16 based implementation"""

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):

        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        
        return roi_cls_locs, roi_scores




