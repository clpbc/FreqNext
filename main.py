# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import os
import json
import yaml
import torch
import random
import shutil
import warnings
import numpy as np

from utils import Logger
from tools import train, GetCfg

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", category=UserWarning)

torch.autograd.set_detect_anomaly(True)


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


if __name__ == "__main__":
    cfg = GetCfg()  # get all config

    expSavePath = os.path.join(cfg["op_dir"], cfg["exp_name"])
    if not os.path.exists(expSavePath):
        os.makedirs(expSavePath, exist_ok=True)

    ### 备份当前运行code
    rootPath = r"./"
    desPath = os.path.join(expSavePath, "code/")
    itemsExclude = ["op_dir", "RunProcess.ipynb", "pretrained", "scripts" "cam.ipynb"]

    for root, dirs, files in os.walk(rootPath):
        for file in files:
            filePath = os.path.join(root, file)
            if any(item in filePath for item in itemsExclude):
                continue

            desFilePath = filePath.replace(rootPath, desPath)
            os.makedirs(os.path.dirname(desFilePath), exist_ok=True)

            shutil.copy2(filePath, desFilePath)

    print(f"Backup of '{rootPath}' completed to '{desPath}'.\n")
    ###

    ### 保存当前config
    cfgName = f"{cfg['exp_name']}.yaml"
    cfgSavePath = os.path.join(expSavePath, cfgName)

    with open(cfgSavePath, "w", encoding="utf-8") as file:
        yaml.dump(cfg, file, allow_unicode=True, default_flow_style=False)
    ###

    ### 保存当前log文件与result文件
    logName = f"{cfg['exp_name']}_log.txt"
    logSavePath = os.path.join(expSavePath, logName)
    log = Logger()
    log.open(logSavePath)
    log.write(json.dumps(cfg, indent=4), is_terminal=False)

    resultName = f"{cfg['exp_name']}_result.csv"
    resultSavePath = os.path.join(expSavePath, resultName)
    with open(resultSavePath, "a") as f:
        f.write(
            f"{'Run': ^10}{'F_HTER': ^10}{'F_AUC': ^10}{'D_HTER': ^10}{'D_AUC': ^10}{'T_HTER': ^10}{'T_AUC': ^10}\n"
        )
    ###

    f_hters, f_aucs, d_hters, d_aucs, t_hters, t_aucs = [], [], [], [], [], []

    for i in range(cfg["base"]["repeat_num"]):
        # To reproduce results
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
        torch.cuda.manual_seed(i)

        f_hter, f_auc, d_hter, d_auc, t_hter, t_auc = train(cfg, log)

        with open(resultSavePath, "a") as f:
            f.write(
                f"{i: ^10d}{f_hter: ^10.3f}{f_auc: ^10.3f}{d_hter: ^10.3f}{d_auc: ^10.3f}{t_hter: ^10.3f}{t_auc: ^10.3f}\n"
            )

        f_hters.append(f_hter)
        f_aucs.append(f_auc)
        d_hters.append(d_hter)
        d_aucs.append(d_auc)
        t_hters.append(t_hter)
        t_aucs.append(t_auc)

    f_hter_mean = np.mean(f_hters)
    f_auc_mean = np.mean(f_aucs)
    d_hter_mean = np.mean(d_hters)
    d_auc_mean = np.mean(d_aucs)
    t_hter_mean = np.mean(t_hters)
    t_auc_mean = np.mean(t_aucs)

    f_hter_best = np.min(f_hters)
    f_auc_best = np.max(f_aucs)
    d_hter_best = np.min(d_hters)
    d_auc_best = np.max(d_aucs)
    t_hter_best = np.min(t_hters)
    t_auc_best = np.max(t_aucs)

    f_hter_std = np.std(f_hters)
    f_auc_std = np.std(f_aucs)
    d_hter_std = np.std(d_hters)
    d_auc_std = np.std(d_aucs)
    t_hter_std = np.std(t_hters)
    t_auc_std = np.std(t_aucs)

    with open(resultSavePath, "a") as f:
        f.write(
            f"\n{'Best': ^10}{f_hter_best: ^10.3f}{f_auc_best: ^10.3f}{d_hter_best: ^10.3f}{d_auc_best: ^10.3f}{t_hter_best: ^10.3f}{t_auc_best: ^10.3f}\n"
        )
        f.write(
            f"{'Mean': ^10}{f_hter_mean: ^10.3f}{f_auc_mean: ^10.3f}{d_hter_mean: ^10.3f}{d_auc_mean: ^10.3f}{t_hter_mean: ^10.3f}{t_auc_mean: ^10.3f}\n"
        )
        f.write(
            f"{'Std': ^10}{f_hter_std: ^10.3f}{f_auc_std: ^10.3f}{d_hter_std: ^10.3f}{d_auc_std: ^10.3f}{t_hter_std: ^10.3f}{t_auc_std: ^10.3f}\n"
        )
