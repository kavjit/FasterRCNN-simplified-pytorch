import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



labels = (
    'aeroplane',
    'bike',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'motorbike',
    'human',
    'plant',
    'sheep',
    'sofa',
    'train',
    'tv',
)


def image_vis(img, ax=None):
    """ Visualize a color image """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax



def image_bbox(img, bbox, label=None, score=None, ax=None):
    """ Visualize bounding boxes inside image """

    label_names = list(labels) + ['bg']
    ax = image_vis(img, ax=ax)


    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def normRGBA(fig):
    """
    Convert a matploblib.axes.Axes figure to a 3D numpy array with RGBA 
    channels and normalize it
    """
    fig = fig.get_figure()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    buf = buf.reshape(h, w, 4)
    img_data = buf.astype(np.int32)
    plt.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.



def visualize(*args, **kwargs):
    fig = image_bbox(*args, **kwargs)
    data = normRGBA(fig)
    return data





