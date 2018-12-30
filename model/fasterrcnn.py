from __future__ import  absolute_import
from __future__ import division
import torch 
import numpy as np
from torch import nn
from datloader import preprocess
from torch.nn import functional as F
from torchvision.models import vgg16
from model.RPN import RPN
from model.utils.bbox_ops import reg_bbox
from model.utils.non_maximum_suppression import non_maximum_suppression
from model.roi_module import RoIPooling2D


def normalizer(m, mean, stddev):
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
    

class Head(nn.Module):

    """Faster R-CNN Head for VGG-16 based implementation"""

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(Head, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normalizer(self.cls_loc, 0, 0.001)
        normalizer(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):

        if isinstance(roi_indices, np.ndarray):
            roi_indices = torch.from_numpy(roi_indices).cuda().float()
        if isinstance(roi_indices, torch.Tensor):
            roi_indices = roi_indices.detach().cuda()

        rois = torch.from_numpy(rois).cuda().float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        
        return roi_cls_locs, roi_scores


class FasterRCNN(nn.Module):
    stride = 16
    def __init__(self,no_of_classes=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        super(FasterRCNN, self).__init__()

        #Loading the pretrained VGG16 model        
        model = vgg16(pretrained = True)

        #getting the classifier from VGG16
        classifier = model.classifier
        classifier = list(classifier)
        del classifier[6]
        del classifier[5]
        del classifier[2]
        self.classifier = nn.Sequential(*classifier)

        features = list(model.features)[:30] 
        for layer in features[:10]:
                layer.requires_grad = False
        self.extractor = nn.Sequential(*features)
        
        self.rpn = RPN(
            512, 512,
            asp_ratios=ratios,
            scales=anchor_scales,
            stride=self.stride,
        )
        
        self.head = Head(
            n_class=no_of_classes + 1,
            roi_size=7,
            spatial_scale=(1. / self.stride),
            classifier=self.classifier
        )

        # mean and std
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.nms_thresh = 0.3
        self.score_thresh = 0.05
        self.n_class = no_of_classes + 1

    def forward(self, x, scale=1.):

        """ Forward Faster R-CNN """
        
        img_size = x.shape[2:]

        features = self.extractor(x)
        rpn_coords, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)
        
        roi_cls_locs, roi_scores = self.head(features, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices


    def nonmaxsup_execution(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(np.array(cls_bbox_l), self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
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
                    img = preprocess(img)
                    prepared_imgs.append(img)
                    sizes.append(size)
            else:
                 prepared_imgs = imgs 
            bboxes = list()
            labels = list()
            scores = list()
            for img, size in zip(prepared_imgs, sizes):
                img = torch.from_numpy(img[None]).cuda().float()
                scale = img.shape[3] / size[1]
                roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
                roi_score = roi_scores.data
                roi_cls_loc = roi_cls_loc.data
                roi = torch.from_numpy(rois).cuda() / scale

                mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
                    repeat(self.n_class)[None]
                std = torch.Tensor(self.loc_normalize_std).cuda(). \
                    repeat(self.n_class)[None]

                roi_cls_loc = (roi_cls_loc * std + mean)
                roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
                roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
                cls_bbox = reg_bbox(roi.detach().cpu().numpy().reshape((-1, 4)),
                                    roi_cls_loc.detach().cpu().numpy().reshape((-1, 4)))
                cls_bbox = torch.from_numpy(cls_bbox).cuda()
                cls_bbox = cls_bbox.view(-1, self.n_class * 4)

                cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
                cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

                prob = (F.softmax(roi_score.cuda(), dim=1)).detach().cpu().numpy()

                raw_cls_bbox = cls_bbox.detach().cpu().numpy()
                raw_prob = prob

                bbox, label, score = self.nonmaxsup_execution(raw_cls_bbox, raw_prob)
                bboxes.append(bbox)
                labels.append(label)
                scores.append(score)

            self.nms_thresh = 0.3
            self.score_thresh = 0.05
            self.train()
        return bboxes, labels, scores

    






