from __future__ import print_function, absolute_import
import argparse
import os
import sys

from torch.backends import cudnn
import torch.nn as nn
import random
from config import cfg
# from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet import make_model
from reid.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
from reid.evaluation.fast_test import fast_test_p_s
from reid.models.rehearser import KernelLearning
# from reid.models.cm import ClusterMemory
from datetime import datetime
import pandas as pd

import torch
import numpy as np
from reid.utils.fourier_style import FourierStyleTransfer


def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def cur_timestamp_str():
    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)
    content = "{}-{}{}-{}{}{}".format(year, month, day, hour, minute, second)
    return content


def main():
    args = parser.parse_args()

    if args.seed is not None:
        print("setting the seed to", args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg.merge_from_file(args.config_file)

    current_dir = os.path.dirname(__file__)  # /home/Newdisk/chenlong/ADSR
    args.data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))  # /home/Newdisk/chenlong/data
    logs_dir = os.path.join(current_dir, 'logs', str(args.logs_dir))  # /home/Newdisk/chenlong/ADSR/logs/exp
    os.makedirs(logs_dir, exist_ok=True)  # 创建日志目录，若已存在则不报错
    args.logs_dir = logs_dir  # /home/Newdisk/chenlong/ADSR/logs/exp

    timestamp = cur_timestamp_str()
    log_name = f'log_{timestamp}.txt'  # 日志文件名，包含时间戳
    if not args.evaluate:  # 训练模式
        sys.stdout = Logger(os.path.join(args.logs_dir, log_name))
    else:  # 测试模式
        sys.stdout = Logger(os.path.join(os.path.dirname(args.test_folder), log_name))

    print("==========\nArgs:{}\n==========".format(args))

    log_res_name = f'log_res_{timestamp}.txt'  # 测试结果日志文件名，包含时间戳
    if not args.evaluate:  # 训练模式
        logger_res = Logger_res(os.path.join(args.logs_dir, log_res_name))  # record the test results
    else:  # 测试模式
        logger_res = Logger_res(os.path.join(os.path.dirname(args.test_folder), log_res_name))

    train_start_time = datetime.now()  # 记录训练总开始时间
    dataset_train_times = []  # 用来存储每个数据集训练完成的时间点

    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']
    elif 2 == args.setting:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    elif 51 == args.setting:
        training_set = ['msmt17', 'cuhk_sysu', 'dukemtmc', 'market1501', 'cuhk03']
    elif 52 == args.setting:
        training_set = ['dukemtmc', 'market1501', 'cuhk03', 'msmt17', 'cuhk_sysu']
    elif 53 == args.setting:
        training_set = ['cuhk_sysu', 'dukemtmc', 'cuhk03', 'msmt17', 'market1501']
    elif 54 == args.setting:
        training_set = ['cuhk03', 'msmt17', 'dukemtmc', 'market1501', 'cuhk_sysu']
    elif 55 == args.setting:
        training_set = ['market1501', 'msmt17', 'dukemtmc', 'cuhk_sysu', 'cuhk03']
    else:
        print(f"Warning: Invalid setting {args.setting}, using default setting 1.")
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']

    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']
    testing_only_set = [x for x in all_set if x not in training_set]
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)

    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0)
    model.cuda()
    model = DataParallel(model)

    writer = SummaryWriter(log_dir=args.logs_dir)

    if args.resume:  # 断点续训，已经训练完第一个数据集
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")

    fourier_style = FourierStyleTransfer(args, num_domains=len(training_set))
    train_loader_list = [x[2] for x in all_train_sets]  # 最好提前都算好，不然的话一直占用显存，产生额外的显存占用！
    fourier_style.collect_domain_style(train_loader_list)  # 计算并保存每个域的多样风格特征到 memory_bank

    # 在数据集上顺序训练
    for set_index in range(0, len(training_set)):
        if args.resume != '' and set_index == 0:  # 断点续训，已经训练完第一个数据集
            continue
        model_old = copy.deepcopy(model) # 深拷贝当前模型
        model = train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer, logger_res=logger_res, fourier_style=fourier_style)

        # 记录该数据集训练完成时间
        finish_time = datetime.now()
        dataset_train_times.append(finish_time)

        if set_index > 0:
            # best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, set_index)
            if args.fix_EMA >= 0:
                best_alpha = args.fix_EMA  # 使用固定的融合参数
            print('****************** After training on the {}-th dataset, the alpha is: {:.4f} ******************'.format(set_index + 1, best_alpha))
            model = linear_combination(model, model_old, best_alpha)  # 融合模型
            fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res, args=args, writer=writer)  # 融合模型测试

    train_end_time = datetime.now()  # 训练结束时间
    print('finished')

    if not args.evaluate:  # 训练模式，保存测试结果到 Excel 文件
        input_file = os.path.join(args.logs_dir, log_res_name)
        rows = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('|'):
                    parts = [cell.strip() for cell in line.strip('|').split('|')]
                    rows.append(parts)
        df = pd.DataFrame(rows)
        df.to_excel(os.path.join(os.path.dirname(input_file), 'log_res.xlsx'), index=False, header=False)

        time_diffs = []  # 计算时间差（秒）
        prev_time = train_start_time
        for t in dataset_train_times:
            diff_sec = (t - prev_time).total_seconds()
            time_diffs.append(diff_sec)
            prev_time = t
        time_diffs.append((train_end_time - prev_time).total_seconds())
        time_diffs.append((train_end_time - train_start_time).total_seconds())
        print("各阶段训练时间统计：", [seconds_to_hms(td) for i, td in enumerate(time_diffs)])


def obtain_old_types(args, all_train_sets, set_index, model):
    # trainloader of old dataset
    dataset_old, num_classes_old, train_loader_old, _, init_loader_old, name_old = all_train_sets[set_index]
    # init_loader is original designed for classifer init
    features_all_old, labels_all_old, fnames_all, camids_all, features_mean, labels_named, vars_mean = extract_features_proto(model, init_loader_old, get_mean_feature=True)
    features_all_old = torch.stack(features_all_old)
    labels_all_old = torch.tensor(labels_all_old).to(features_all_old.device)
    features_all_old.requires_grad = False
    # savename = os.path.join(args.logs_dir, f"proto_{name_old}.pt")
    # torch.save(,savename)
    return features_all_old, labels_all_old, features_mean, labels_named, vars_mean


def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer, logger_res=None, fourier_style=None):
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[set_index]
    Epochs = args.epochs0 if set_index == 0 else args.epochs

    # set_index == 0 时，第一个数据集训练
    add_num = 0
    old_model = None
    if set_index > 0:
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()  # old_model不训练
        # 计算 add_num 旧任务类别数之和
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])
        # 扩展分类器的维度
        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()
        # 用类中心初始化新分类器
        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        model.cuda()

    # 重新初始化优化器
    params = []
    for key, value in model.named_params(model): # 是模型中可训练参数的直接引用，而非副本! 优化器的更新会直接作用于模型model本身
        if not value.requires_grad:
            print('冻结参数：', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)
    else:
        print(f"Warning: Optimizer {args.optimizer} not supported, using SGD instead.")
        optimizer = torch.optim.SGD(params, momentum=args.momentum)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    trainer = Trainer(cfg, args, model, add_num + num_classes, writer=writer, fourier_style=fourier_style)  # 每次训练都会重新实例化训练器！！！
    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):
        train_loader.new_epoch() # 重置迭代器, 以便后续训练时从头开始读取数据
        trainer.train(epoch, train_loader, optimizer, training_phase=set_index + 1, train_iters=len(train_loader), add_num=add_num, old_model=old_model)
        lr_scheduler.step() # 更新学习率

        if (epoch + 1) % args.eval_epoch == 0 or epoch + 1 == Epochs:  # 每 eval_epoch 轮或最后一轮进行测试和保存
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=os.path.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))  # 保存模型！！！
            logger_res.append('epoch: {}'.format(epoch + 1))  # 记录当前 epoch
            mAP = 0.
            if args.middle_test or epoch + 1 == Epochs:  # 中间测试或者最后一个epoch一定测试
                mAP = fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res, args=args, writer=writer)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=os.path.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))  # 保存模型！！！

    return model


def linear_combination(model, model_old, alpha, model_old_id=-1):
    # print("*******combining the models with alpha: {}*******".format(alpha))
    # 新模型参数 = alpha × 新模型 + (1 - alpha) × 旧模型

    model_old_state_dict = model_old.state_dict() # 产生副本
    model_state_dict = model.state_dict() # 产生副本
    model_new = copy.deepcopy(model) # 产生副本
    model_new_state_dict = model_new.state_dict() # 产生副本

    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
            model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')  # 未融合的参数
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]  # 融合分类器参数
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4, help="each identity has num_instances instances in a minibatch")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x', choices=['50x'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help="optimizer")
    parser.add_argument('--lr', type=float, default=0.008, help="learning rate of new parameters, for pretrained")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30], help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')  # 指定第一个数据集的模型路径，从第二个数据集开始训练
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='data')  # 不用指定，程序会自动创建
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='exp')  # 指定log文件名，路径会自动创建
    parser.add_argument('--config_file', type=str, default='config/base.yml', help="config_file")
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2, 51, 52, 53, 54, 55], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")
    parser.add_argument('--fix_EMA', default=0.5, type=float, help="model fusion weight")
    parser.add_argument('--global_alpha', type=float, default=100, help="")
    parser.add_argument('--absolute_feat', action='store_true', help="")
    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")
    parser.add_argument('--absolute_delta', action='store_true', default=True, help="only use dual teacher")
    parser.add_argument('--trans', action='store_true', default=True, help="only use dual teacher")
    parser.add_argument('--blur', action='store_true', help="adopt blur augmentation")
    parser.add_argument('--n_kernel', default=1, type=int, help="number of Distribution Transfer kernel")
    parser.add_argument('--groups', default=1, type=int, help="convolution group number of each Distribution Transfer kernel")
    parser.add_argument('--joint_test', action='store_true', help="use the AKPNet model during testing")
    parser.add_argument('--aux_weight', default=4.5, type=float, help="the loss weight of rehearsed data, e.g. β in the paper")

    # 添加到参数解析部分
    parser.add_argument('--style_L', type=float, default=0.01, help='Low frequency ratio for Fourier style transfer')
    parser.add_argument('--ssn_weight', default=0.5, type=float, help="the loss weight of self norm data")
    main()
