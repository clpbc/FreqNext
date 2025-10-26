# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import torch

from modeling import BuildModel
from solver import make_optimizer, create_lr_scheduler
# from data import BuildLoader
from data_new import BuildLoader
from engine.example_trainer import do_train
import losses

def train(cfg, log):

    # 数据集加载
    train_set, train_loader = BuildLoader(cfg, isTrain = True, log = log)
    val_set, val_loader = BuildLoader(cfg, isTrain = False, log = log)
        
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    model = BuildModel(cfg, log).to(device)

    criterion = {}
    for key, loss_info in cfg['loss'].items():
        criterion[key] = losses.__dict__[loss_info['name']](**loss_info['params'])

    
    optimizer = make_optimizer(cfg, model, log)
    scheduler = create_lr_scheduler(optimizer, **cfg['scheduler']['params'])
    scaler = torch.cuda.amp.GradScaler() if cfg['train']['amp'] else None

    # Load if checkpoint is provided
    if cfg['ckpt']:
        ckpt = torch.load(cfg['ckpt'])
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        iter_num_start = epoch * cfg['iter_per_epoch']
        if cfg['train']['amp']:
            scaler.load_state_dict(ckpt["scaler"])
        log.write(f'\nLoaded checkpoint from epoch {epoch} at iteration : {iter_num_start}\n', is_file = 1)
    else:
        epoch = 1
        iter_num_start = 0
        log.write(f'\nStarting training from epoch {epoch} at iteration : {iter_num_start}\n', is_file = 1)


    f_hter, f_auc, d_hter, d_auc, t_hter, t_auc = do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
        criterion,
        device,
        log,
        epoch,
        iter_num_start
    )

    return f_hter, f_auc, d_hter, d_auc, t_hter, t_auc

