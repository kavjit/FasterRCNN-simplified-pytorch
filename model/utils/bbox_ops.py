
import numpy as np


def calculate_iou(bbox_a, bbox_b):

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    iou = area_i / (area_a[:, None] + area_b - area_i)
    return iou


def reg_bbox(BBcoordinates, scales):
    '''
    BBcoordinates are the coordinates of the bounding boxes. It has the following four values(in order):
    1)minimum value in y-axis
    2)minimum value in x-axis
    3)maximum value in y-axis
    4)maximum value in x-axis
    
    The 'scales' array contains the regression coefficients to offset the existing bounding boxes. We use these values to get a better bounding box for the images.
    It has the following four values(in order):
    1)regression coefficient for y-axis value of the centre point
    2)regression coefficient for x-axis value of the centre point
    3)regression coefficient for width of bounding box
    4)regression coefficient for height of bounding box
    '''
    if BBcoordinates.shape[0] == 0:
        return np.zeros((0, 4), dtype=scales.dtype)

# original bounding boxes
    height = BBcoordinates[:, 2] - BBcoordinates[:, 0] #difference between the two y values
    width = BBcoordinates[:, 3] - BBcoordinates[:, 1] #difference between two x values
    centre_y = BBcoordinates[:, 0] + (height)/2  
    centre_x = BBcoordinates[:, 1] + (width)/2

    dy = scales[:, 0::4] #taking 1st, 5th, 9th elements and so on
    dx = scales[:, 1::4] #taking 2nd,6th, 10th elements and so on
    dh = scales[:, 2::4]
    dw = scales[:, 3::4]

# target bounding boxes
    y = dy * height[:, np.newaxis] + centre_y[:, np.newaxis]
    x = dx * width[:, np.newaxis] + centre_x[:, np.newaxis]
    h = np.exp(dh) * height[:, np.newaxis] 
    w = np.exp(dw) * width[:, np.newaxis]

    targetBB = np.zeros(scales.shape, dtype=scales.dtype)
    targetBB[:, 0::4] = y - h/2
    targetBB[:, 1::4] = x - w/2
    targetBB[:, 2::4] = y + h/2
    targetBB[:, 3::4] = x + w/2

    return targetBB


def reg_scales(BBcoordinates, targetBB):

# function to get the regression coefficents

    height = BBcoordinates[:, 2] - BBcoordinates[:, 0]
    width = BBcoordinates[:, 3] - BBcoordinates[:, 1]
    y = BBcoordinates[:, 0] + height/2
    x = BBcoordinates[:, 1] + width/2

    base_height = targetBB[:, 2] - targetBB[:, 0]
    base_width = targetBB[:, 3] - targetBB[:, 1]
    base_y = targetBB[:, 0] + 0.5 * base_height
    base_x = targetBB[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_y - y)/height
    dx = (base_x - x)/width
    dh = np.log(base_height/height)
    dw = np.log(base_width/width)

    scales = np.vstack((dy, dx, dh, dw)).transpose()

    return scales

if __name__ == '__main__':
    pass

def anchor_base_generator(base_size=16, ratios=[0.5, 1, 2],
                         scales=[8, 16, 32]):
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = base_size * scales[j] * np.sqrt(ratios[i])
            w = base_size * scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
