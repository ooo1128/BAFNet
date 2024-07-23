from ..builder import DETECTORS
from .two_stage import TwoStageDetector
#from ..bn import InPlaceABNSync as BatchNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses.detail_loss import DetailAggregateLoss
BatchNorm2d = nn.BatchNorm2d
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.bn.train(True)
        self.bn.track_running_stats = False
        #self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes=3, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, 3, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.bn_atten.train(True)
        self.bn_atten.track_running_stats = False
        # self.bn_atten = BatchNorm2d(out_chan, activation='none')

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        #atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
class Fusion(nn.Module):
    def __init__(self, in_c=512, out_c=256):
        super().__init__()
        self.point_conv = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                    BNReLU(out_c),
                                    nn.Conv2d(out_c, 1, 1))

    def forward(self, low, high):
        high = F.interpolate(high, size=(low.size(2), low.size(3)),
                                    mode="bilinear", align_corners=False)
        attmap = torch.cat([high, low], dim=1)
        attmap = self.point_conv(attmap)
        attmap = torch.sigmoid(attmap)
        return attmap * low + high
def BNReLU(num_features):
    return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU())
@DETECTORS.register_module()
class CascadeRCNN_STDC(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CascadeRCNN_STDC, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.context1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(1, 5), dilation=2, padding=(0, 4), groups=256),
                                      nn.Conv2d(256, 256, kernel_size=(5, 1), dilation=1, padding=(2, 0), groups=256),
                                      nn.Conv2d(256, 256, kernel_size=1),
                                      BNReLU(256))

        self.context2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(1, 5), dilation=1, padding=(0, 2), groups=256),
                                      nn.Conv2d(256, 256, kernel_size=(5, 1), dilation=1, padding=(2, 0), groups=256),
                                      nn.Conv2d(256, 256, kernel_size=1),
                                      BNReLU(256))
        self.fusion = Fusion()
        self.conv_out_sp8 = BiSeNetOutput(256, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(256, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(256, 64, 1)
        self.arm16 = AttentionRefinementModule(256, 256)
        self.arm32 = AttentionRefinementModule(256, 256)
        self.conv_head32 = ConvBNReLU(256, 256, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(256, 256, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(256, 256, ks=1, stride=1, padding=0)
        self.ffm = FeatureFusionModule(512, 256)
        self.boundary_loss_func = DetailAggregateLoss()
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        # H8, W8 = x[0].size()[2:]
        # H16, W16 = x[1].size()[2:]
        # H32, W32 = x[2].size()[2:]
        # avg = F.avg_pool2d(x[4],x[4].size()[2:])
        # avg = self.conv_avg(avg)
        # avg_up = F.interpolate(avg, (H32, W32), mode='nearest')
        # feat32_arm = self.arm32(x[2])
        # feat32_sum = feat32_arm + avg_up
        # feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        # feat32_up = self.conv_head32(feat32_up)
        #
        # feat16_arm = self.arm16(x[1])
        # feat16_sum = feat16_arm + feat32_up
        # feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        # feat16_up = self.conv_head16(feat16_up)
        # feat_fuse = self.ffm(x[0], feat16_up )
        # x1[0] = feat_fuse
        x = list(x)
        x_c = self.context1(x[-1]) + x[-1]
        x_4 = self.context2(x_c) + x_c
        # selective fusion module
        x_f = self.fusion(x[3], x_4)
        x[3] = x[3] + x_f
        x_c1 = self.context1(x[3]) + x[3]
        x_3 = self.context2(x_c1) + x_c1
        # selective fusion module
        x_f = self.fusion(x[2], x_3)
        x[2] = x[2] + x_f
        x_c2 = self.context1(x[2]) + x[2]
        x_2 = self.context2(x_c2) + x_c2
        # selective fusion module
        x_f = self.fusion(x[1], x_2)
        x[1] = x[1] + x_f
        x_c3 = self.context1(x[1]) + x[1]
        x_1 = self.context2(x_c3) + x_c3
        # selective fusion module
        x_f = self.fusion(x[0], x_1)
        x[0] = x[0] + x_f
        feat_out_sp2 = self.conv_out_sp2(x[0])
        # feat_out_sp4 = self.conv_out_sp4(x[1])
        # feat_out_sp8 = self.conv_out_sp8(x[2])
        #return x,feat_out_sp2,feat_out_sp4,feat_out_sp8
        return x, feat_out_sp2
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        x, detail2= self.extract_feat(img)
        boundery_bce_loss = 0.
        boundery_dice_loss = 0.
        losses = dict()
        lb = torch.squeeze(img, 1)
        boundery_bce_loss2, boundery_dice_loss2 = self.boundary_loss_func(detail2, lb)
        boundery_bce_loss += boundery_bce_loss2
        boundery_dice_loss += boundery_dice_loss2
        # boundery_bce_loss4, boundery_dice_loss4 = self.boundary_loss_func(detail4, lb)
        # boundery_bce_loss += boundery_bce_loss4
        # boundery_dice_loss += boundery_dice_loss4
        # boundery_bce_loss8, boundery_dice_loss8 = self.boundary_loss_func(detail8, lb)
        # boundery_bce_loss += boundery_bce_loss8
        # boundery_dice_loss += boundery_dice_loss8
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        #
        losses.update(roi_losses)
        losses['loss_boundery_bce'] = 0.4*boundery_bce_loss
        losses['loss_boundery_dice'] = boundery_bce_loss*0.4
        return losses
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x, detail2 = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_STDC, self).show_result(data, result, **kwargs)
