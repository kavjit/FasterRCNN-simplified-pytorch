from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch

from utils.config import configurations
from data.dataset import Dataset, TestDataset#, inverse_normalize
from model.faster_rcnn_update import FasterRCNN
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import fetch_image
from utils.eval_tool import eval_detection_voc
import datetime

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))



plot_dir = 'loss_plots'
img_dir = 'Images'

rpn_loc_loss = []
rpn_cls_loss = []
roi_loc_loss = []
roi_cls_loss = []
total_loss = []
total_rpn = []
total_roi = []

loss_list = ['rpn_loc_loss',
'rpn_cls_loss', 
'roi_loc_loss', 
'roi_cls_loss', 
'total_loss',
'total_rpn',
'total_roi']

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


def inverse_normalize(image):
    return (image * 0.225 + 0.45).clip(min=0, max=1) * 255

def eval_model(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        #print('Eval start time - {}'.format(datetime.datetime.now().time()))
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    precisions = np.zeros([configurations.epoch,20])
    recall = np.zeros([configurations.epoch,20])
    configurations._parse(kwargs)

    dataset = Dataset(configurations)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  pin_memory=True,
                                  num_workers=configurations.num_workers)
    testset = TestDataset(configurations)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=configurations.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNN()
    #faster_rcnn.load_state_dict(torch.load('faster_rcnn_model_0.ckpt'))
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if configurations.load_path:
        trainer.load(configurations.load_path)
        print('load pretrained model from %s' % configurations.load_path)


    best_map = 0
    lr_ = configurations.lr
    for epoch in range(configurations.epoch):
        trainer.reset_meters()
        ########### FREEZING REQD MODEL #####################################
        if epoch==0: ##freezing ex2 and head
            for param in trainer.faster_rcnn.extractor2[10:].parameters():
                param.requires_grad = False
            for param in trainer.faster_rcnn.head.parameters():
                param.requires_grad = False
            trainer.faster_rcnn.extractor1.train()
            trainer.faster_rcnn.rpn.train()

        elif epoch==3: ##freezing ex1 and rpn, unfreeze ex2 and head
            #unfreeze ex2 and head
            for param in trainer.faster_rcnn.extractor2[10:].parameters():
                param.requires_grad = True
            for param in trainer.faster_rcnn.head.parameters():
                param.requires_grad = True
            #make ex1 and rpn eval and frozen
            for param in trainer.faster_rcnn.extractor1[10:].parameters():
                param.requires_grad = False
            for param in trainer.faster_rcnn.rpn.parameters():
                param.requires_grad = False
            trainer.faster_rcnn.extractor1.eval()
            trainer.faster_rcnn.rpn.eval()  
            trainer.faster_rcnn.head.train()
            trainer.faster_rcnn.extractor2.train()                    

        elif epoch==7:
            trainer.faster_rcnn.rpn.train()
            for param in trainer.faster_rcnn.rpn.parameters():
                param.requires_grad = True
            for param in trainer.faster_rcnn.extractor2[10:].parameters():
                param.requires_grad = False
            for param in trainer.faster_rcnn.head.parameters():
                param.requires_grad = False
            trainer.faster_rcnn.extractor2.eval() 
            trainer.faster_rcnn.head.eval()

        
        elif epoch==9:
            for param in trainer.faster_rcnn.rpn.parameters():
                param.requires_grad = False
            for param in trainer.faster_rcnn.head.parameters():
                param.requires_grad = True   
            trainer.faster_rcnn.rpn.eval()  
            trainer.faster_rcnn.head.train()       


        #######################################################################

        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            if epoch<=2:
                trainer.step1(img, bbox, label, scale, epoch)
            elif epoch>=3 and epoch<=6:
                trainer.step2(img, bbox, label, scale, epoch)
            elif epoch>=7 and epoch<=8:
                trainer.step3(img, bbox, label, scale, epoch)
            elif epoch>=9 and epoch<=10:
                trainer.step4(img, bbox, label, scale, epoch)                


            if((ii+1) % 500 == 0):
                append_loss(trainer.get_meter_data())


            if (ii + 1) % configurations.plot_every == 0:
                if os.path.exists(configurations.debug_file):
                    ipdb.set_trace()

                #plot loss 
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                plot_loss(loss_list,plot_dir)

            # if ii == 7000:
            #     # plot groud truth bboxes
            #     ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            #     gt_img = fetch_image(ori_img_,
            #                          at.tonumpy(bbox_[0]),
            #                          at.tonumpy(label_[0]))
            #     gt_img = gt_img.transpose(1,2,0)
            #     if not os.path.exists(img_dir):
            #         os.mkdir(img_dir)
            #     plt.imsave('{}/actual_image_{}_{}.jpg'.format(img_dir, epoch, ii), gt_img)
                
            #     # plot prediction bboxes
            #     _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
            #     pred_img = fetch_image(ori_img_,
            #                            at.tonumpy(_bboxes[0]),
            #                            at.tonumpy(_labels[0]).reshape(-1),
            #                            at.tonumpy(_scores[0]))
            #     pred_img = pred_img.transpose(1,2,0)
            #     plt.imsave('{}/predicted_image_{}_{}.jpg'.format(img_dir,epoch, ii), pred_img)



        torch.save(faster_rcnn.state_dict(),'faster_rcnn_model_{}.ckpt'.format(epoch+1))
        all_losses = np.zeros((7,len(total_loss)))
        all_losses[0,:] = rpn_loc_loss
        all_losses[1,:] = rpn_cls_loss
        all_losses[2,:] = roi_loc_loss
        all_losses[3,:] = roi_cls_loss
        all_losses[4,:] = total_loss
        all_losses[5,:] = total_rpn
        all_losses[6,:] = total_roi
        
        save_dir = 'prec_rec_loss/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(save_dir+'all_losses_'+str(epoch)+'.npy',all_losses)
        print("Epoch {} completed".format(epoch+1))



if __name__ == '__main__':
    train(plot_every=2000, num_workers=8, test_num_workers=0, test_num = 2500, epoch = 11)