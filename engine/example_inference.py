# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


from utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, get_min_HTER

def do_eval(val_loader, model, device):
    """
        Eval the FAS model on the target data.
        The function evaluates the model on the target data and return TPR@FPR, HTER
        and AUC.
        Used in train.py to eval the model after each epoch; used in test.py to eval
        the final model.
    """

    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()

    f_prob_list = []
    f_label_list = []

    val_dict = {}
    val_dict['f_HTER'] = None
    val_dict['f_AUC'] = None

    model.eval()

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            # for idx, (img, aug1_img, aug2_img, freq, label) in enumerate(val_loader):
            for idx, (img, label) in enumerate(val_loader):
                img = img.to(device)
                # freq = freq.to(device)
                label = label.to(device)

                input_dict = {}
                input_dict['img'] = img
                # input_dict['aug1_img'] = img
                # input_dict['aug2_img'] = img
                input_dict['label'] = label
                # input_dict['freq'] = freq
                input_dict['isTrain'] = False

                output_dict = model(input_dict)

                logits = output_dict['logits']

                valid_loss = criterion(logits, label)
                valid_losses.update(valid_loss.item())

                acc = accuracy(logits, label, topk = (1, ))
                valid_top1.update(acc[0].item())

                prob = F.softmax(logits, dim = 1).cpu().data.numpy()[:, 1]
                label = label.cpu().data.numpy()  

                f_prob_list = np.append(f_prob_list, prob)
                f_label_list = np.append(f_label_list, label)
    

    f_auc_score = roc_auc_score(f_label_list, f_prob_list)


    # cur_EER_valid, threshold, _, _ = get_EER_states(f_prob_list, f_label_list, grid_density = 1000)
    # cur_f_HTER_valid = get_HTER_at_thr(f_prob_list, f_label_list, threshold)
    cur_f_HTER_valid = get_min_HTER(f_prob_list, f_label_list, grid_density = 1000)
    
        
    val_dict['f_HTER'] = cur_f_HTER_valid
    val_dict['f_AUC'] = f_auc_score
    #####

    val_dict['acc'] = valid_top1.avg
    val_dict['loss'] = valid_losses.avg

    val_dict['t_HTER'] = val_dict['f_HTER']
    val_dict['t_AUC'] = val_dict['f_AUC']


    return val_dict

    
