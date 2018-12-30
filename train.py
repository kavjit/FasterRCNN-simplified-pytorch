from __future__ import  absolute_import
from datloader import Dataset, TestDataset
from model.fasterrcnn import FasterRCNN
from torch.utils import data as data_
from trainstep import TrainStep
from visualization import visualize
from evaluation import evaluate
import cupy as cp
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch


def append_loss(d):
    for hl, new_data in d.items():
        if new_data is not None:
            eval(hl).append(new_data)
        

def plot_loss(d,plot_dir):
    for hl in d:
        plt.figure()
        plt.plot(np.asarray(eval(hl)))
        plt.title(hl)
        plt.savefig('{}/{}.png'.format(plot_dir,hl))
        plt.close()


dataset = Dataset()
print('data has been loaded')
dataloader = data_.DataLoader(dataset, \
                              batch_size=1, \
                              shuffle=True, \
                              pin_memory=True,
                              num_workers=8)

faster_rcnn = FasterRCNN()
print('model has been loaded')
trainer = TrainStep(faster_rcnn).cuda()


plot_dir = 'loss_plots'
img_dir = 'Images'

rpn_reg_loss = []
rpn_classifier_loss = []
head_reg_loss = []
head_classifier_loss = []
total_loss = []


loss_list = ['rpn_reg_loss',
'rpn_classifier_loss', 
'head_reg_loss', 
'head_classifier_loss', 
'total_loss']

lr_ = 1e-3
num_epochs = 10
plot_every = 2000
precisions = np.zeros([num_epochs,20])
recall = np.zeros([num_epochs,20])

for epoch in range(num_epochs):     
    for ii, (img, bbox, label, scale) in tqdm(enumerate(dataloader)):
        scale = scale.item()
        img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
        loss = trainer.forward(img, bbox, label, scale)

        if((ii+1) % 500 == 0):
            append_loss(loss)


        if (ii + 1) % plot_every == 0:

            #plot loss 
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            plot_loss(loss_list,plot_dir)

            # plot groud truth bboxes
            img_cpu = img[0].detach().cpu().numpy()
            ori_img_ = (img_cpu * 0.225 + 0.45).clip(min=0, max=1) * 255
            gt_img = visualize(ori_img_,
                                 bbox[0],
                                 label[0])
            gt_img = gt_img.transpose(1,2,0)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            plt.imsave('{}/actual_image_{}_{}.jpg'.format(img_dir, epoch, ii), gt_img)
            
            # plot prediction bboxes
            _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
            pred_img = visualize(ori_img_,
                                   _bboxes[0],
                                   _labels[0],
                                   _scores[0])
            pred_img = pred_img.transpose(1,2,0)
            plt.imsave('{}/predicted_image_{}_{}.jpg'.format(img_dir,epoch, ii), pred_img)



    torch.save(faster_rcnn.state_dict(),'faster_rcnn_model_{}.ckpt'.format(epoch+1))

    all_losses = np.zeros((5,len(total_loss)))
    all_losses[0,:] = rpn_reg_loss
    all_losses[1,:] = rpn_classifier_loss
    all_losses[2,:] = head_reg_loss
    all_losses[3,:] = head_classifier_loss
    all_losses[4,:] = total_loss
    
    save_dir = 'prec_rec_loss/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(save_dir+'all_losses_'+str(epoch)+'.npy',all_losses)
    print("Epoch {} completed".format(epoch+1))
