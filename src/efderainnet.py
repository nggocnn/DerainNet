import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def weights_init(network, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        network (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    """

    def init_function(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method {init_type} is not implemented')

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # Apply the initialization function <init_function>
    print(f'Initialize network with {init_type} type')
    network.apply(init_function)


# ----------------------------------------
#      Kernel Prediction Network (KPN)
# ----------------------------------------
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, att_rate=16, channel_att=False, spatial_att=False) -> None:
        super(BasicConv, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        if self.channel_att:
            self.channel_att_block = nn.Sequential(
                nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels//att_rate, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels//att_rate, out_channels=out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        if self.spatial_att:
            self.spatial_att_block = nn.Sequential(
                nn.Conv2d(in_channels=2,  out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):

        fw = self.conv_block(data)

        if self.channel_att:
            fw_pool = torch.cat([
                F.adaptive_avg_pool2d(fw, (1, 1)),
                F.adaptive_max_pool2d(fw, (1, 1)),
            ], dim=1)

            att = self.channel_att_block(fw_pool)
            fw = fw * att

        if self.spatial_att:
            fw_pool = torch.cat([
                torch.mean(fw, dim=1, keepdim=True),
                torch.max(fw, dim=1, keepdim=True)[0]
            ], dim=1)

            att = self.spatial_att_block(fw_pool)
            fw = fw * att

        return fw


class KernelConv(nn.Module):
    def __init__(self, kernel_size=5, sep_conv=False, core_bias=False) -> None:
        super(KernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        core = core.view(batch_size, N, -1, color, height, width)

        core_1, core_2, core_3 = None, None, None

        if not self.core_bias:
            core_1, core_2 = torch.split(core, self.kernel_size, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, self.kernel_size, dim=2)

        t1 = core_1[:, :, 0: 0 + self.kernel_size, ...].view(batch_size, N, self.kernel_size, 1, height, width)
        t2 = core_2[:, :, 0: 0 + self.kernel_size, ...].view(batch_size, N, 1, self.kernel_size, height, width)
        core_ = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, self.kernel_size**2, color, height, width)

        return core_, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        core = core.view(batch_size, N, -1, color, height, width)
        core_ = core[:, :, 0: self.kernel_size**2, ...]

        return core_, None if not self.core_bias else core[:, :, -1, ...].squeeze()

    def forward(self, frames, core, white_level=1.0, rate=1):
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)

        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        
        image_stack = []

        padding_num = (self.kernel_size//2) * rate
        frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])

        for i in range(0, self.kernel_size):
            for j in range(0, self.kernel_size):
                image_stack.append(frame_pad[..., i*rate: i*rate+height, j*rate: j*rate+width])

        image_stack = torch.stack(image_stack, dim=2)

        pred_image = torch.sum(core.mul(image_stack), dim=2, keepdim=False)
        pred_image = pred_image.squeeze()
        
        if self.core_bias:
            pred_image += bias

        pred_image = pred_image / white_level

        return pred_image


class KPN(nn.Module):
    def __init__(
            self, 
            color=True, burst_length=1, blind_est=True, kernel_size=5, sep_conv=False,
            channel_att=False, spatial_att=False, up_mode='bilinear', core_bias=False
    ) -> None:
        super(KPN, self).__init__()
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.up_mode = up_mode

        in_channels = (3 if color else 1) * \
                      (burst_length if blind_est else burst_length + 1)
        out_channels = (3 if color else 1) * \
                       ((2 * kernel_size) if sep_conv else (kernel_size**2 * burst_length))

        if self.core_bias:
            out_channels += (3 if color else 1) * burst_length

        self.conv1 = BasicConv(in_channels, out_channels=64,  channel_att=False, spatial_att=False)
        self.conv2 = BasicConv(in_channels=64, out_channels=128, channel_att=False, spatial_att=False)
        self.conv3 = BasicConv(in_channels=128, out_channels=256, channel_att=False, spatial_att=False)
        self.conv4 = BasicConv(in_channels=256, out_channels=512, channel_att=False, spatial_att=False)
        self.conv5 = BasicConv(in_channels=512, out_channels=512, channel_att=False, spatial_att=False)

        self.conv6 = BasicConv(in_channels=512+512, out_channels=512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = BasicConv(in_channels=256+512, out_channels=256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = BasicConv(in_channels=256+128, out_channels=out_channels, channel_att=channel_att, spatial_att=spatial_att)

        self.conv9 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.kernel_pred = KernelConv(kernel_size=kernel_size, sep_conv=sep_conv, core_bias=self.core_bias)
        
        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, data, white_level=1.0):
        
        conv1 = self.conv1(data)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))

        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.up_mode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.up_mode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.up_mode)], dim=1))
        core = self.conv9(F.interpolate(conv8, scale_factor=2, mode=self.up_mode))

        pred1 = self.kernel_pred(data, core, white_level, rate=1)
        pred2 = self.kernel_pred(data, core, white_level, rate=2)
        pred3 = self.kernel_pred(data, core, white_level, rate=3)
        pred4 = self.kernel_pred(data, core, white_level, rate=4)

        pred_cat = torch.cat([pred1, pred2, pred3, pred4], dim=1)
        
        pred = self.conv_final(pred_cat)
        
        return pred


if __name__ == '__main__':
    
    kpn = KPN(channel_att=True, spatial_att=True, core_bias=True).cuda()
    a = torch.randn(4, 3, 224, 224).cuda()
    b = kpn(a)
    print(b.shape)
