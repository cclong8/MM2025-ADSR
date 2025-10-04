import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans


class FourierStyleTransfer:
    def __init__(self, args, num_domains):
        self.style_L = args.style_L
        C, H, W = 3, args.height, args.width
        b_h = 0
        b_w = 0
        c_h, c_w = H // 2, W // 2  # 128, 64
        self.h1, self.h2 = c_h - b_h, c_h + b_h + 1
        self.w1, self.w2 = c_w - b_w, c_w + b_w + 1
        print(f"style_L: {self.style_L}, b_h: {b_h}, b_w: {b_w}")
        print(f"Low frequency region: h[{self.h1}:{self.h2}], w[{self.w1}:{self.w2}]")

        self.K = 5  # 每个域保存的风格特征数量
        self.memory_bank = torch.empty(num_domains * self.K, C, self.h2 - self.h1, self.w2 - self.w1, dtype=torch.float64)  # 存储各域的平均风格特征
        print("the shape of memory_bank:", self.memory_bank.shape)  # (num_domains * K, C, h2-h1, w2-w1)

        self.domain_memory_path = os.path.join(args.logs_dir, "domain_styles")  # /home/Newdisk/chenlong/ADSR/logs/exp/domain_styles
        os.makedirs(self.domain_memory_path, exist_ok=True)

    def fft_transform(self, img_tensor):
        r"""对图像进行傅里叶变换，返回振幅谱和相位谱
        img_tensor： Tensor of shape (B, C, H, W) dtype: torch.float32 -> torch.float64
        """
        # 转换为双精度以提高计算稳定性
        img_tensor = img_tensor.to(torch.float64)
        fft = torch.fft.fft2(img_tensor, norm='ortho', dim=(-2, -1))  # complex128，norm参数不太重要，不改变结果
        amp = torch.abs(fft)  # 振幅谱: (B, C, H, W)
        pha = torch.angle(fft)  # 相位谱: (B, C, H, W)
        return amp, pha

    def ifft_transform(self, amp, pha):
        r"""根据振幅谱和相位谱进行傅里叶逆变换，返回图像
        amp, pha: Tensor of shape (B, C, H, W) dtype: torch.float64 -> torch.float32
        """
        fft = amp * torch.exp(1j * pha)
        img = torch.fft.ifft2(fft, norm='ortho', dim=(-2, -1)).real.float()  # complex128，norm参数不太重要，不改变结果
        return img

    def transfer_style(self, img, training_phase):
        r"""将源图像转换为目标风格
        img: Tensor of shape (B, C, H, W)   device=GPU
        training_phase: 当前训练阶段（整数）1-5
        """
        B, C, H, W = img.shape
        device = img.device

        amp, _ = self.fft_transform(img)  # (B, C, H, W)
        amp_shifted = torch.fft.fftshift(amp, dim=(-2, -1))  # (B, C, H, W)
        amp_center = amp_shifted[:, :, self.h1:self.h2, self.w1:self.w2]  # (B, C, h2-h1, w2-w1)
        target_amps = self.memory_bank.to(device)  # (num_domains * K, C, h2-h1, w2-w1)
        target_amps = target_amps[:(training_phase - 1) * self.K]  # (num_old_domains * K, C, h2-h1, w2-w1)
        amp_center_exp = amp_center.unsqueeze(1)  # (B, 1, C, h2-h1, w2-w1)
        target_amps_exp = target_amps.unsqueeze(0)  # (1, num_old_domains * K, C, h2-h1, w2-w1)
        l1_distances = torch.mean(torch.abs(amp_center_exp - target_amps_exp), dim=(2, 3, 4))  # (B, num_old_domains * K)
        indices = torch.argmin(l1_distances, dim=1)

        selected_amp = target_amps[indices]  # (B, C, h2-h1, w2-w1)
        amp, pha = self.fft_transform(img)  # (B, C, H, W)
        amp_shifted = torch.fft.fftshift(amp, dim=(-2, -1))  # (B, C, H, W)
        amp_shifted[:, :, self.h1:self.h2, self.w1:self.w2] = selected_amp  # 替换高频区域（并行处理整个batch）
        amp_ = torch.fft.ifftshift(amp_shifted, dim=(-2, -1))
        img_ = self.ifft_transform(amp_, pha)  # (B, C, H, W)
        return img_

    def self_style_norm(self, img, training_phase):
        r"""将当前域图像进行自我风格归一化（使用memory_bank中当前域的平均振幅谱）
        img: Tensor of shape (B, C, H, W)   device=GPU
        training_phase: 当前训练阶段（整数）1-5
        """
        B, C, H, W = img.shape
        device = img.device
        indices = torch.randint((training_phase - 1) * self.K, training_phase * self.K, (B,), device=device)
        target_amps = self.memory_bank.to(device)
        selected_amp = target_amps[indices]  # (B, C, h2-h1, w2-w1)
        amp, pha = self.fft_transform(img)  # (B, C, H, W)
        amp_shifted = torch.fft.fftshift(amp, dim=(-2, -1))  # (B, C, H, W)
        amp_shifted[:, :, self.h1:self.h2, self.w1:self.w2] = selected_amp  # 替换高频区域（并行处理整个batch）
        amp_ = torch.fft.ifftshift(amp_shifted, dim=(-2, -1))
        img_ = self.ifft_transform(amp_, pha)  # (B, C, H, W)
        return img_

    def load_domain_style(self, path):
        # 从文件加载预先计算好的各域风格特征（振幅谱）
        for domain_idx in range(len(self.memory_bank) // self.K):
            style_path = os.path.join(path, f"domain_{domain_idx + 1}.pth")
            assert os.path.exists(style_path), f"Style file for domain_{domain_idx + 1} not found at {style_path}"
            centroids = torch.load(style_path)  # (K, C, h2-h1, w2-w1)
            assert centroids.dim() == 4 and centroids.shape[0] == self.K
            self.memory_bank[domain_idx * self.K:(domain_idx + 1) * self.K] = centroids
            print(f"Loaded domain_{domain_idx + 1} style from {style_path}")
            print(centroids)

    def collect_domain_style(self, train_loader_list):
        for domain_idx, train_loader in enumerate(train_loader_list):
            amp_list = []
            for i in range(len(train_loader)):
                train_inputs = train_loader.next()
                _, imgs_origin, _, _, _ = self._parse_data(train_inputs)
                amp = self.fft_transform(imgs_origin)[0]  # 仅取振幅谱 (B, C, H, W), dtype=torch.float64
                amp_shifted = torch.fft.fftshift(amp, dim=(-2, -1))  # 频谱中心化
                amp_shifted = amp_shifted[:, :, self.h1:self.h2, self.w1:self.w2]  # 仅保留中心区域 (B, C, h2-h1, w2-w1)
                amp_shifted = amp_shifted.reshape(amp_shifted.shape[0], -1).cpu().numpy()  # 展平
                amp_list.append(amp_shifted)
            amp_all = np.concatenate(amp_list, axis=0)
            sk_kmeans = KMeans(n_clusters=self.K, random_state=0).fit(amp_all)
            centroids = sk_kmeans.cluster_centers_
            centroids = torch.tensor(centroids, dtype=torch.float64).reshape(self.K, 3, self.h2 - self.h1, self.w2 - self.w1)  # (K, C, h2-h1, w2-w1)
            self.memory_bank[domain_idx * self.K:(domain_idx + 1) * self.K] = centroids  # 存储到 memory bank
            torch.save(centroids, os.path.join(self.domain_memory_path, f"domain_{domain_idx + 1}.pth"))
            print(f"Saved domain_{domain_idx + 1} kmeans centroids to memory bank using method 4 (kmeans++ sklearn)")
            train_loader.new_epoch()  # 重置迭代器, 以便后续训练时从头开始读取数据

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

    def low_freq_mutate(self, amp_src, amp_trg, L=0.01):
        # 频谱中心化必不可少，否则低频部分不在中心位置！！！
        a_src = torch.fft.fftshift(amp_src, dim=(-2, -1))
        # a_trg = torch.fft.fftshift(amp_trg, dim=(-2, -1))
        _, _, h, w = amp_src.shape
        # b = int(np.floor(min(h, w) * L))
        b = 0
        c_h, c_w = h // 2, w // 2
        h1, h2 = c_h - b, c_h + b + 1
        w1, w2 = c_w - b, c_w + b + 1
        # a_src[:, :, h1:h2, w1:w2] = a_trg[:, :, h1:h2, w1:w2]
        a_src[:, :, h1:h2, w1:w2] = amp_trg
        a_src = torch.fft.ifftshift(a_src, dim=(-2, -1))
        return a_src

    def FDA_source_to_amp(self, src_img, amp_trg, L=0.01):
        # src_img: [1, 3, 256, 128], float32 -> float64
        # amp_trg: [1, 3, 256, 128], float64 未中心化的全频谱
        amp_src, pha_src = self.fft_transform(src_img)  # [1, 3, 256, 128], float64
        if amp_trg.dim() == 3:
            amp_trg = amp_trg.unsqueeze(0)  # [1, 3, 256, 128], float64
        # assert amp_src.shape == amp_trg.shape, f"源图像和目标振幅谱的形状不匹配: {amp_src.shape} vs {amp_trg.shape}"
        amp_src_ = self.low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)
        src_in_trg = self.ifft_transform(amp_src_, pha_src)
        return src_in_trg

    def image_amp_only_rec(self, img):  # 提取一个图片的振幅谱，并将相位设为零，进行傅里叶逆变换
        # img: [1, 3, 256, 128], float32 -> float64
        amp_img, _ = self.fft_transform(img)  # [1, 3, 256, 128], float64
        # amp_img * torch.exp(1j * 0) 会报错：torch.exp() 期望输入是一个 Tensor，而不是 Python 的复数 1j * 0
        # 相位设为零，相当于复数谱只有实部
        fft_reconstruct = torch.complex(amp_img, torch.zeros_like(amp_img))
        img_reconstruct = torch.fft.ifft2(fft_reconstruct, norm='ortho', dim=(-2, -1)).real.float()
        _, _, H, W = img.shape
        img_reconstruct = torch.roll(img_reconstruct, shifts=(H // 2, W // 2), dims=(-2, -1))  # 低频成分被移到四个角落，将低频成分移回图像的中心。
        return img_reconstruct

    def image_pha_only_rec(self, img):  # 提取一个图片的相位谱，并将振幅设为1，进行傅里叶逆变换
        # img: [1, 3, 256, 128], float32 -> float64
        _, pha_img = self.fft_transform(img)  # [1, 3, 256, 128], float64
        fft_reconstruct = 1 * torch.exp(1j * pha_img)  # 幅度设为1，相当于复数谱只有虚部
        img_reconstruct = torch.fft.ifft2(fft_reconstruct, norm='ortho', dim=(-2, -1)).real.float()
        return img_reconstruct

    def amp_only_rec(self, amp):  # 仅根据振幅谱进行傅里叶逆变换，返回图像
        # amp: [3, 256, 128] or [1, 3, 256, 128], float64
        # amp = torch.fft.ifftshift(amp, dim=(-2, -1))  # 把频谱重新排列，低频移到角落。
        if amp.dim() == 3:
            amp = amp.unsqueeze(0)  # 保持 batch 维度
        fft_reconstruct = torch.complex(amp, torch.zeros_like(amp))
        img_reconstruct = torch.fft.ifft2(fft_reconstruct, norm='ortho', dim=(-2, -1)).real.float()
        # _, _, H, W = amp.shape
        # img_reconstruct = torch.roll(img_reconstruct, shifts=(H // 2, W // 2), dims=(-2, -1))
        return img_reconstruct

    def pha_only_rec(self, pha):  # 仅根据相位谱进行傅里叶逆变换，返回图像
        # pha: [3, 256, 128] or [1, 3, 256, 128], float64
        if pha.dim() == 3:
            pha = pha.unsqueeze(0)
        fft_reconstruct = 1 * torch.exp(1j * pha)
        img_reconstruct = torch.fft.ifft2(fft_reconstruct, norm='ortho', dim=(-2, -1)).real.float()
        return img_reconstruct

    def load_image_as_tensor(self, image_path, size=(128, 256)):  # 加载图像并转换为 Tensor
        img = Image.open(image_path).convert('RGB')  # img.size 只会返回图像的 (宽度, 高度)
        img = img.resize(size, Image.BICUBIC)
        img_np = np.asarray(img, np.float32).transpose((2, 0, 1))  # HWC -> CHW，(3, 256, 128), float32
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # 加入 batch 维度
        return img_tensor  # torch.Size([1, 3, 256, 128]), dtype: torch.float32

    def tensor_to_image(self, tensor):  # 将 Tensor 转换为 PIL 图像
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0).detach().cpu().numpy()  # 去除 batch 维度，numpy 数组
        elif tensor.dim() == 3:
            tensor = tensor.detach().cpu().numpy()
        tensor = np.transpose(tensor, (1, 2, 0))  # CHW -> HWC
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        tensor = ((tensor - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(tensor)
        return img