import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import copy
import os.path
import os
from reid.utils.feature_tools import *
import lreid_dataset.datasets as datasets
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data import IterLoader
import numpy as np


def get_data(name, data_dir, height, width, batch_size, workers, num_instances, select_num=0):
    root = data_dir
    dataset = datasets.create(name, root)
    if select_num > 0:  # 选择部分ID进行训练
        train = []
        for instance in dataset.train:
            if instance[1] < select_num:  # pid 是随机未排序的，所以要判断一下
                # new_id=id_2_id[instance[1]]
                train.append((instance[0], instance[1], instance[2], instance[3]))  # img_path, pid, camid, domain-id
        dataset.train = train
        dataset.num_train_pids = select_num
        dataset.num_train_imgs = len(train)

    # 根据ImageNet数据集的均值和标准差进行标准化
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)  # 按照文件名排序，pid从小到大排序
    iters = int(len(train_set) / batch_size)  # 每个epoch的迭代次数，向下取整
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        # T.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.03),  # 新增的随机颜色打乱
        T.ToTensor(),  # 先归一化
        normalizer,  # 后标准化
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    # 傅里叶变换，不加Pad，不加RandomCrop，不能normalizer和RandomErasing
    # 傅里叶变换函数（torch.fft.fft2）仅接受张量输入，直接对 PIL 图像操作会抛出类型错误
    # 水平翻转不影响图像语义，随后生成旧风格化图像之后我再加normalizer和RandomErasing用于辅助训练
    basic_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor()
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])  # 无数据增强（确保测试数据的一致性）

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)  # 采用随机多重采样
    else:
        sampler = None  # 否则使用普通随机打乱（shuffle=not rmgs_flag）
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, transform_basic=basic_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)  # 通过 IterLoader 固定迭代次数为 iters

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),  # 并集（去重后）
                     root=dataset.images_dir, transform=test_transformer, transform_basic=basic_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)  # drop_last默认为False，没有IterLoader

    # 通常用于模型初始化阶段（如预训练、特征提取初始化等），加载训练集数据但采用测试级别的变换。
    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=test_transformer, transform_basic=basic_transformer),
                             batch_size=128, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

    return [dataset, num_classes, train_loader, test_loader, init_loader, name]


def build_data_loaders(args, training_set, testing_only_set, toy_num=0):
    training_loaders = [get_data(name, args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances, select_num=500) for name in training_set]
    testing_loaders = [get_data(name, args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances) for name in testing_only_set]
    return training_loaders, testing_loaders
