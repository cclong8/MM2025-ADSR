from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *

from reid.utils.make_loss import make_loss
import copy
from reid.utils.color_transformer import ColorTransformer
from reid.metric_learning.distance import cosine_similarity


class Trainer(object):
    def __init__(self, cfg, args, model, num_classes, writer=None, fourier_style=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
        self.loss_ce = nn.CrossEntropyLoss(reduction='batchmean')
        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")
        self.MSE = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

        self.fourier_style = fourier_style
        self.ce_criterion = nn.CrossEntropyLoss(reduction='none')  # 保留每个样本的损失

    def train(self, epoch, data_loader_train, optimizer, training_phase, train_iters=200, add_num=0, old_model=None):
        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad is False and m.bias.requires_grad is False:
                    m.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()

        losses_kd_orig = AverageMeter()
        losses_kd_ssn = AverageMeter()
        losses_kd_fst = AverageMeter()

        end = time.time()
        total_weight = [1.0, self.args.ssn_weight, self.args.aux_weight]

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)
            s_inputs, imgs_origin, targets, cids, domains = self._parse_data(train_inputs)

            if training_phase == 1:
                inputs_norm = self.fourier_style.self_style_norm(imgs_origin, training_phase)
                inputs_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs_norm)
                inputs_norm = T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])(inputs_norm)
                datas = [
                    [s_inputs, imgs_origin, targets, cids, domains],
                    [inputs_norm, imgs_origin, targets, cids, domains],
                ]
            elif training_phase > 1:
                inputs_norm = self.fourier_style.self_style_norm(imgs_origin, training_phase)
                inputs_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs_norm)
                inputs_norm = T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])(inputs_norm)

                inputs_r = self.fourier_style.transfer_style(imgs_origin, training_phase)
                inputs_r = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs_r)
                inputs_r = T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])(inputs_r)

                datas = [
                    [s_inputs, imgs_origin, targets, cids, domains],
                    [inputs_norm, imgs_origin, targets, cids, domains],
                    [inputs_r, imgs_origin, targets, cids, domains],
                ]

            loss = 0
            for idx, (s_inputs, imgs_origin, targets, cids, domains) in enumerate(datas):
                targets = targets + add_num
                s_features, bn_feat, cls_outputs, feat_final_layer = self.model(s_inputs)
                loss_ce, loss_tp = self.loss_fn(cls_outputs, s_features, targets, target_cam=None)  # base loss
                losses_ce.update(loss_ce.mean().item())
                losses_tr.update(loss_tp.item())

                divergence = 0.
                if old_model is not None:
                    with torch.no_grad():
                        s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
                    if isinstance(s_features_old, tuple):
                        s_features_old = s_features_old[0]
                    affinity_matrix_new = self.get_normal_affinity(s_features)
                    affinity_matrix_old = self.get_normal_affinity(s_features_old)
                    divergence = self.KLDivLoss(torch.log(affinity_matrix_new), affinity_matrix_old)
                    divergence = divergence * self.AF_weight
                    if idx == 0:
                        losses_kd_orig.update(divergence.item())
                    elif idx == 1:
                        losses_kd_ssn.update(divergence.item())
                    elif idx == 2:
                        losses_kd_fst.update(divergence.item())

                loss = loss + (loss_ce + loss_tp + divergence) * total_weight[idx]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer is not None:
                if training_phase == 1:
                    self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val, global_step=epoch * train_iters + i)
                    self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val, global_step=epoch * train_iters + i)
                    self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val, global_step=epoch * train_iters + i)
                elif training_phase > 1:
                    self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val, global_step=epoch * train_iters + i)
                    self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val, global_step=epoch * train_iters + i)
                    self.writer.add_scalar(tag="loss/Loss_kd_orig_{}".format(training_phase), scalar_value=losses_kd_orig.val, global_step=epoch * train_iters + i)
                    self.writer.add_scalar(tag="loss/Loss_kd_ssn_{}".format(training_phase), scalar_value=losses_kd_ssn.val, global_step=epoch * train_iters + i)
                    self.writer.add_scalar(tag="loss/Loss_kd_fst_{}".format(training_phase), scalar_value=losses_kd_fst.val, global_step=epoch * train_iters + i)
                    self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val, global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
                if training_phase == 1:
                    print('Epoch: [{}][{}/{}]  Time {:.3f} ({:.3f})  Loss_ce {:.3f} ({:.3f})  Loss_tp {:.3f} ({:.3f})'
                          .format(epoch, i + 1, train_iters, batch_time.val, batch_time.avg, losses_ce.val, losses_ce.avg, losses_tr.val, losses_tr.avg))
                elif training_phase > 1:
                    print('Epoch: [{}][{}/{}]  Time {:.3f} ({:.3f})  Loss_ce {:.3f} ({:.3f})  Loss_tp {:.3f} ({:.3f})  Loss_kd_orig {:.3f} ({:.3f})  Loss_kd_ssn {:.3f} ({:.3f})  Loss_kd_fst {:.3f} ({:.3f})'
                          .format(epoch, i + 1, train_iters, batch_time.val, batch_time.avg, losses_ce.val, losses_ce.avg, losses_tr.val, losses_tr.avg,
                                  losses_kd_orig.val, losses_kd_orig.avg, losses_kd_ssn.val, losses_kd_ssn.avg, losses_kd_fst.val, losses_kd_fst.avg))

    def get_normal_affinity(self, x, norm=0.1):
        pre_matrix_origin = cosine_similarity(x, x)
        pre_affinity_matrix = F.softmax(pre_matrix_origin / norm, dim=1)  # 转化为概率分布
        return pre_affinity_matrix

    def _parse_data(self, inputs):  # 解析输入数据，并将其移动到GPU上！
        # CPU 加载数据→取出数据→转移到 GPU→模型计算
        imgs, imgs_origin, _, pids, cids, domains = inputs
        # pids：行人 ID（person IDs），即训练的标签（用于分类损失计算）
        # cids：摄像头 ID（camera IDs），记录图像采集的设备信息
        # domains：域 ID（domain IDs），表示图像所属的不同数据域或环境
        inputs = imgs.cuda()
        targets = pids.cuda()
        imgs_origin = imgs_origin.cuda()
        # cids, domains 不需要放到 GPU 上，是因为它们在后续的训练过程中并不直接参与计算
        return inputs, imgs_origin, targets, cids, domains
