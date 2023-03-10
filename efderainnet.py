import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def weights_init(network, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    """

    def init_function(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xaiver':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # Apply the initialization function <init_function>
    print('Initialize network with %s type' % init_type)
    network.apply(init_function)


# ----------------------------------------
#      Kernel Prediction Network (KPN)
# ----------------------------------------
class Basic(nn.Module):
    def __init__(self, in_channels, out_channels, att_rate=16, channel_att=False, spatial_att=False) -> None:
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att

        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

        if self.channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels//att_rate, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels//att_rate, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )

        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):

        fm = self.conv1(data)

        if self.channel_att:
            fm_pool = torch.cat([
                F.adaptive_avg_pool2d(fm, (1, 1)),
                F.adaptive_max_pool2d(fm, (1, 1))
            ], dim=1)

            att = self.att_c(fm_pool)
            fm = fm * att

        if self.spatial_att:
            fm_pool = torch.cat([
                torch.mean(fm, dim=1, keepdim=True),
                torch.max(fm, dim=1, keepdim=True)[0]
            ], dim=1)

            att = self.att_s(fm_pool)
            fm = fm * att

        return fm


class KernelConv(nn.Module):
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False) -> None:
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)

        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)

        core_out = {}
        current = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, current: current + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, current: current + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            current += K

        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0: self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        elif len(frames.size()) == 4:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        else:
            raise ValueError('Frames size is invalid: [%d]' % len(frames.size()))

        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        
        image_stack = []
        pred_image = []
        kernel = self.kernel_size[::-1]

        for index, K in enumerate(kernel):
            if not image_stack:
                padding_num = (K // 2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                
                for i in range(0, K):
                    for j in range(0, K):
                        image_stack.append(frame_pad[..., i*rate: i*rate+height, j*rate: j*rate+width])
                
                image_stack = torch.stack(image_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                image_stack = image_stack[:, :, k_diff: -k_diff, ...]

            pred_image.append(torch.sum(
                core[K].mul(image_stack), dim=2, keepdim=False
            ))

        pred_image = torch.stack(pred_image, dim=0)
        pred_img_i = torch.mean(pred_image, dim=0, keepdim=False)
        pred_img_i = pred_img_i.squeeze(2)
        
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            
            pred_img_i += bias

        pred_img_i = pred_img_i / white_level

        return pred_img_i



class KPN(nn.Module):
    def __init__(
            self, 
            color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
            channel_att=False, spatial_att=False, up_mode='bilinear', core_bias=False
    ) -> None:
        super(KPN, self).__init__()
        self.color_channel = 3 if color else 1
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.up_mode = up_mode

        in_channels = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)
        out_channels = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length

        if self.core_bias:
            out_channels += (3 if color else 1) * burst_length

        self.conv1 = Basic(in_channels, out_channels=64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(in_channels=64, out_channels=128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(in_channels=128, out_channels=256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(in_channels=256, out_channels=512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(in_channels=512, out_channels=512, channel_att=False, spatial_att=False)

        self.conv6 = Basic(in_channels=512+512, out_channels=512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(in_channels=256+512, out_channels=256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(in_channels=256+128, out_channels=out_channels, channel_att=channel_att, spatial_att=spatial_att)
        
        self.outc = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size=kernel_size, sep_conv=sep_conv, core_bias=self.core_bias)
        
        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, data_with_est, data, white_level=1.0):
        
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))

        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.up_mode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.up_mode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.up_mode)], dim=1))
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.up_mode))
        
        pred1 = self.kernel_pred(data, core, white_level, rate=1)
        pred2 = self.kernel_pred(data, core, white_level, rate=2)
        pred3 = self.kernel_pred(data, core, white_level, rate=3)
        pred4 = self.kernel_pred(data, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)
        
        pred = self.conv_final(pred_cat)
        
        return pred


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

if __name__ == '__main__':
    
    kpn = KPN().cuda()
    a = torch.randn(4, 3, 224, 224).cuda()
    b = kpn(a, a)
    print(b.shape)
