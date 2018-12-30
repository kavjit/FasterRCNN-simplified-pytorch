import numpy as np
from torch.nn import functional as F
from torch import nn
import numpy as np

from model.utils.bbox_ops import anchor_base_generator
from model.utils.RPN_tools import ProposalCreator


def normalize(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class RPN(nn.Module):

    def __init__(
            self, in_channels=512, mid_channels=512, asp_ratios=[0.5, 1, 2],
            scales=[8, 16, 32], stride=16,
            proposal_creator_params=dict(),
    ):
        super(RPN, self).__init__()
        self.gen_anchor = anchor_base_generator(
            scales=scales, ratios=asp_ratios)
        self.stride = stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        num_anchors = self.gen_anchor.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, num_anchors * 2, 1, 1, 0)
        self.cord = nn.Conv2d(mid_channels, num_anchors * 4, 1, 1, 0)
        normalize(self.conv1, 0, 0.01)
        normalize(self.score, 0, 0.01)
        normalize(self.cord, 0, 0.01)

    def forward(self, map_, img_size, scale=1.):
        n, _, hh, ww = map_.shape
        anchor = shift_anchor(
            np.array(self.gen_anchor),
            self.stride, hh, ww)

        num_anchors = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(map_))

        rpn_cords = self.cord(h)
        rpn_cords = rpn_cords.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        prob_scores = F.softmax(rpn_scores.view(n, hh, ww, num_anchors, 2), dim=4)
        rpn_fg_scores = prob_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_cords[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_cords, rpn_scores, rois, roi_indices, anchor


def shift_anchor(gen_anchor, stride, height, width):

    shift_y = np.arange(0, height * stride, stride)
    shift_x = np.arange(0, width * stride, stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = gen_anchor.shape[0]
    K = shift.shape[0]
    anchor = gen_anchor.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor




