# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import sys, yaml, json
from datetime import datetime

sys.path.append('.')
from args import get_parser

def update_moreparams_from_args(cfg, updates):
    for update in updates:
        key, value = update.split('=')

        keys = key.split('.')    
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)

        d = cfg
        for k in keys[: -1]:
            d = d[k]
        if d[keys[-1]] is None:
            continue
        d[keys[-1]] = value
    return cfg
    

def GetCfg():
    parser = get_parser()
    args = parser.parse_args()

    args_dict = vars(args)
    # print(json.dumps(args_dict, indent=4))

    cfgPath = args_dict['config']

    with open(cfgPath, 'r') as cfgFile:
        cfg = yaml.safe_load(cfgFile)
    
    cfg['config'] = args_dict['config']
    cfg['device'] = args_dict['device']
    cfg['ckpt'] = args_dict['ckpt']
    cfg['op_dir'] = args_dict['op_dir']
    cfg['add_info'] = args_dict['add_info']

    if args_dict['train_set']:
        cfg['dataset']['train_set'] = args_dict['train_set']
        cfg['dataset_new']['train_dataset'] = args_dict['train_set']
    if args_dict['val_set']:
        cfg['dataset']['val_set'] = args_dict['val_set']
        cfg['dataset_new']['test_dataset'] = args_dict['val_set']
    if args_dict['bs']:
        cfg['dataset']['batch_size'] = args_dict['bs']
        cfg['dataset_new']['train_batchSize'] = args_dict['bs']
        cfg['dataset_new']['test_batchSize'] = args_dict['bs']


    cfg['now_time'] = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args_dict['add_info']:
        cfg['exp_name'] = f"{cfg['model']['mode']}_{cfg['now_time']}_{cfg['add_info']}"
    else:
        cfg['exp_name'] = f"{cfg['model']['mode']}_{cfg['now_time']}"

    if args.set:
        cfg = update_moreparams_from_args(cfg, args.set)

    # print(json.dumps(cfg, indent = 4))

    return cfg