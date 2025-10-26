# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
from datetime import datetime
from timeit import default_timer as timer

import torch

from utils import AverageMeter, accuracy, save_checkpoint, time_to_str

from .example_inference import do_eval


def do_train(
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
    iter_num_start,
):
    best_ACC = 0.0
    best_f_HTER = 1.0
    best_f_AUC = 0.0
    best_d_HTER = 1.0
    best_d_AUC = 0.0
    best_t_HTER = 1.0
    best_t_AUC = 0.0

    f_HTER = None
    d_HTER = None
    f_AUC = None
    d_AUC = None

    loss_classifier = AverageMeter()
    loss_simclr = AverageMeter()
    loss_l2 = AverageMeter()
    loss_total = AverageMeter()
    classifier_top1 = AverageMeter()

    # clpbc
    # loss_ortho = AverageMeter()
    # loss_contra = AverageMeter()

    loss_recce_recons = AverageMeter()
    loss_recce_contra = AverageMeter()

    log.write(
        f"{'-' * 53} [START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {'-' * 53}\n"
    )
    log.write(f"{'** start training model! **':^135}")

    message = (
        f"|{'-' * 7}|"
        f"{' VALID ':-^48}|"
        f"{' Train ':-^36}|"
        f"{' Current Best ':-^24}|"
        f"{'-' * 14}|"
    )
    log.write(message)

    # FLIP,resent  message
    message = (
        f"|{'epoch':^7}|"
        f"{'loss':^8}{'top-1':^8}{'f_HTER':^8}{'f_AUC':^8}{'d_HTER': ^8}{'d_AUC': ^8}|"
        f"{'cls-loss':^10}{'Sim-loss':^10}{'or-loss':^9}{'top-1':^7}|"
        f"{'top-1':^8}{'t_HTER':^8}{'t_AUC':^8}|"
        f"{'time':^14}|"
    )

    log.write(message)

    log.write(f"|{'-' * 133}|")

    ### 训练过程
    train_iter = iter(train_loader)
    iter_per_epoch_train = len(train_iter)

    start = timer()

    for iter_num in range(iter_num_start, cfg["train"]["iters"] + 1):
        if iter_num % iter_per_epoch_train == 0:
            train_iter = iter(train_loader)

        if iter_num != 0 and iter_num % cfg["train"]["iter_per_epoch"] == 0:
            epoch += 1

        model.train(True)

        ###### data prepare ######
        # img, aug1_img, aug2_img, freq, label = train_iter.next()

        # img, aug1_img, aug2_img, freq, label = (
        #     img.to(device),
        #     aug1_img.to(device),
        #     aug2_img.to(device),
        #     freq.to(device),
        #     label.to(device),
        # )
        
        img, label = train_iter.next()
        
        # img, label = (
        #     img.to(device),
        #     label.to(device),
        # )
        for i in range(len(img)):
            img[i] = img[i].to(device)
        label = label.to(device)

        ###### forward ######
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            input_dict = {}
            input_dict["img"] = img
            # input_dict["aug1_img"] = aug1_img
            # input_dict["aug2_img"] = aug2_img
            input_dict["label"] = label
            # input_dict["freq"] = freq
            input_dict["isTrain"] = True
            
            import time
            start_time = time.time()

            output_dict = model(input_dict)
            
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")

            try:
                logits = output_dict["logits"]

                # recce_loss_input = output_dict["recce_loss_input"]
                # aug1_feat = output_dict["aug1_feat"]
                # aug2_feat = output_dict["aug2_feat"]
                # aug1_text_dot = output_dict["aug1_text_dot"]
                # aug2_text_dot = output_dict["aug2_text_dot"]
                # lowfreq = output_dict["lowfreq"]
                # midfreq = output_dict["midfreq"]
                # highfreq = output_dict["highfreq"]
                # diff_freq_cls = output_dict["diff_freq_cls"]
                # diff_freq_text = output_dict["diff_freq_text"]

            except ValueError as e:
                print(f"output_dict Value Error: {e}")

            try:
                cls_loss = criterion["loss_1"](logits, label)
                # recce_recons_loss = criterion["loss_2"](
                #     recce_loss_input["recons"], img, label
                # )
                # recce_ml_loss = criterion["loss_3"](recce_loss_input["contra"], label)
                # simclr_loss = criterion["loss_2"](aug1_feat, aug2_feat)
                # mse_loss = criterion["loss_3"](aug1_text_dot, aug2_text_dot)
                # ortho_loss = criterion["loss_4"](lowfreq, midfreq, highfreq)
                # contra_loss = criterion["loss_5"](diff_freq_cls, diff_freq_text)

            except ValueError as e:
                print(f"Loss Calculate Error: {e}")

            try:
                cfg["loss"]["loss_1"]["value"] = cls_loss
                # cfg["loss"]["loss_2"]["value"] = recce_recons_loss
                # cfg["loss"]["loss_3"]["value"] = recce_ml_loss
                # cfg["loss"]["loss_2"]["value"] = simclr_loss
                # cfg["loss"]["loss_3"]["value"] = mse_loss
                # cfg["loss"]["loss_4"]["value"] = ortho_loss
                # cfg["loss"]["loss_5"]["value"] = contra_loss
            except ValueError as e:
                print(f"Loss assignment Error: {e}")

            loss = torch.tensor(0.0).to(device)

            for key, loss_info in cfg["loss"].items():
                loss += loss_info["weight"] * loss_info["value"]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        try:
            loss_classifier.update(cls_loss.item())
            # loss_recce_recons.update(recce_recons_loss.item())
            # loss_recce_contra.update(recce_ml_loss.item())
            # loss_simclr.update(simclr_loss.item())
            # loss_l2.update(mse_loss.item())
            # loss_contra.update(contra_loss.item())
            # loss_ortho.update(ortho_loss.item())

        except ValueError as e:
            print(f"Loss Update Error{e}")

        loss_total.update(loss.item())

        acc = accuracy(logits, label, topk=(1,))
        classifier_top1.update(acc[0].item())

        if iter_num != 0 and (iter_num + 1) % cfg["train"]["iter_per_epoch"] == 0:

            valid_dict = do_eval(val_loader, model, device)

            f_HTER = valid_dict["f_HTER"]
            f_AUC = valid_dict["f_AUC"]
            # d_HTER = valid_dict['d_HTER']
            # d_AUC = valid_dict['d_AUC']
            if valid_dict["f_HTER"] is None:
                f_HTER = 0
                f_AUC = 0
            # if valid_dict['d_HTER'] is None:
            #     d_HTER = 0
            #     d_AUC = 0
            d_HTER = 0
            d_AUC = 0

            # judge model according to t_HTER
            is_best = valid_dict["t_HTER"] <= best_t_HTER
            if is_best:
                best_ACC = valid_dict["acc"]
                best_t_HTER = valid_dict["t_HTER"]
                best_t_AUC = valid_dict["t_AUC"]
                best_f_HTER = valid_dict["f_HTER"]
                best_f_AUC = valid_dict["f_AUC"]
                # best_d_HTER = valid_dict['d']
                # best_d_AUC = valid_dict['d_AUC']
                if valid_dict["f_HTER"] is None:
                    best_f_HTER = 0
                    best_f_AUC = 0

                save_list = [epoch, best_t_HTER, best_t_AUC, best_ACC]
                # save_checkpoint(save_list, is_best, model, optimizer, scheduler, os.path.join(cfg['op_dir'], cfg['exp_name'], "best_checkpoint.pt"))

            print("\r", end="", flush=True)

            message = (
                f"|{epoch:^7d}|"
                f"{valid_dict['loss']:^8.2f}{valid_dict['acc']:^8.2f}{f_HTER * 100:^8.2f}{f_AUC * 100:^8.2f}{d_HTER * 100:^8.2f}{d_AUC * 100:^8.2f}|"
                f"{loss_classifier.avg:^10.2f}{loss_recce_recons.avg:^10.2f}{loss_recce_contra.avg:^9.2f}{classifier_top1.avg:^7.2f}|"
                f"{float(best_ACC):^8.2f}{float(best_t_HTER * 100):^8.2f}{float(best_t_AUC * 100):^8.2f}|"
                f"{time_to_str(timer() - start, 'sec'):^14}|"
            )
            log.write(message)

    return (
        best_f_HTER * 100,
        best_f_AUC * 100,
        best_d_HTER * 100,
        best_d_AUC * 100,
        best_t_HTER * 100,
        best_t_AUC * 100,
    )
