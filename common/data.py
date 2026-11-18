"""
GPU-resident CIFAR data pipeline.

Why: on Windows, PyTorch DataLoader workers are spawned (not forked) and the
per-image PIL transforms become the bottleneck for small 32x32 images - the GPU
sits idle most of the time. CIFAR is tiny (50k x 3 x 32 x 32 uint8 = 150 MB),
so we load the whole dataset onto the GPU once and do augmentation
(random-crop with 4px padding + horizontal flip) with batched tensor ops.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# Per-channel normalisation statistics
CIFAR_STATS = {
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def load_cifar_numpy(name, root="data", train=True):
    """Return (images uint8 [N,32,32,3], labels int64 [N]) using torchvision's reader."""
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root, train=train, download=True)
    return ds.data, np.asarray(ds.targets, dtype=np.int64)


class GPULoader:
    """Iterates over a GPU-resident dataset in mini-batches with on-GPU augmentation.

    images : uint8 tensor [N,3,H,W] (kept as uint8 to save memory)
    labels : int64 tensor [N]
    """

    def __init__(self, images, labels, batch_size, mean, std, train=True,
                 resolution=32, device="cuda", drop_last=None, seed=0):
        self.x = images.to(device)
        self.y = labels.to(device)
        self.bs = batch_size
        self.train = train
        self.res = resolution
        self.device = device
        self.drop_last = train if drop_last is None else drop_last
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    @property
    def num_samples(self):
        return len(self.y)

    def _augment(self, xb):
        """Random crop (pad=4, reflect) + random horizontal flip, fully batched."""
        b, c, h, w = xb.shape
        pad = 4
        xp = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
        ox = torch.randint(0, 2 * pad + 1, (b,), device=xb.device, generator=self.gen)
        oy = torch.randint(0, 2 * pad + 1, (b,), device=xb.device, generator=self.gen)
        ar = torch.arange(h, device=xb.device)
        rows = (oy[:, None] + ar[None, :])                     # [B,H]
        cols = (ox[:, None] + ar[None, :])                     # [B,W]
        bidx = torch.arange(b, device=xb.device)[:, None, None]
        # advanced indexing -> [B,H,W,C]; then back to NCHW
        xc = xp.permute(0, 2, 3, 1)[bidx, rows[:, :, None], cols[:, None, :]]
        xc = xc.permute(0, 3, 1, 2)
        flip = torch.rand(b, device=xb.device, generator=self.gen) < 0.5
        xc = torch.where(flip[:, None, None, None], xc.flip(3), xc)
        return xc

    def __iter__(self):
        n = len(self.y)
        if self.train:
            perm = torch.randperm(n, device=self.device, generator=self.gen)
        else:
            perm = torch.arange(n, device=self.device)
        for i in range(len(self)):
            idx = perm[i * self.bs:(i + 1) * self.bs]
            xb = self.x[idx].float().div_(255.0)
            if self.train:
                xb = self._augment(xb)
            if self.res != xb.shape[-1]:
                xb = F.interpolate(xb, size=(self.res, self.res), mode="bilinear",
                                   align_corners=False, antialias=self.res < xb.shape[-1])
            xb = (xb - self.mean) / self.std
            yield xb.contiguous(memory_format=torch.channels_last), self.y[idx]


def get_loaders(name, root="data", batch_size=128, eval_batch_size=512, resolution=32,
                val_split=0, device="cuda", seed=0):
    """Build GPU loaders.

    If val_split > 0, the last `val_split` images of a seeded shuffle of the
    *training* set are held out as validation (used by NAS so the test set is
    never touched during search). Returns (train_loader, eval_loader) where
    eval_loader is the validation split if val_split>0 else the official test set.
    """
    mean, std = CIFAR_STATS[name]
    xtr, ytr = load_cifar_numpy(name, root, train=True)
    xtr = torch.from_numpy(xtr).permute(0, 3, 1, 2).contiguous()
    ytr = torch.from_numpy(ytr)
    if val_split > 0:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ytr), generator=g)
        tr_idx, va_idx = perm[:-val_split], perm[-val_split:]
        xev, yev = xtr[va_idx], ytr[va_idx]
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    else:
        xte, yte = load_cifar_numpy(name, root, train=False)
        xev = torch.from_numpy(xte).permute(0, 3, 1, 2).contiguous()
        yev = torch.from_numpy(yte)
    train = GPULoader(xtr, ytr, batch_size, mean, std, train=True,
                      resolution=resolution, device=device, seed=seed)
    evl = GPULoader(xev, yev, eval_batch_size, mean, std, train=False,
                    resolution=resolution, device=device, seed=seed)
    return train, evl


def preprocess_numpy(images_uint8_nhwc, name, resolution=32):
    """CPU preprocessing used by the inference / ONNX pipelines.
    images: uint8 array [N,32,32,3] -> float32 array [N,3,res,res] (normalised)."""
    mean, std = CIFAR_STATS[name]
    x = torch.from_numpy(np.ascontiguousarray(images_uint8_nhwc)).permute(0, 3, 1, 2).float() / 255.0
    if resolution != x.shape[-1]:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear",
                          align_corners=False, antialias=resolution < x.shape[-1])
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return x.numpy().astype(np.float32)
