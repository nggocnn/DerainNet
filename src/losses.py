import torch
from torch import nn
from torch.nn import functional as F


class LossBasic(nn.Module):
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return \
            self.l2_loss(pred, ground_truth) + \
            self.l1_loss(self.gradient(pred), self.gradient(ground_truth))


class LossAnneal(nn.Module):
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


class LossFunc(nn.Module):
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        return \
            self.coeff_basic * self.loss_basic(pred_img, ground_truth), \
            self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)


class TensorGradient(nn.Module):
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])

        if self.L1:
            return torch.abs((l - r)[..., 0: w, 0: h]) + torch.abs((u - d)[..., 0: w, 0: h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0: w, 0: h], 2) + torch.pow((u - d)[..., 0: w, 0: h], 2)
            )
