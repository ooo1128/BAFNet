from ...utils.criterion import BondaryLoss
import torch
from torch import nn
from torch.nn import functional as F

class DetailAggregateLoss(nn.Module):
	def __init__(self, *args, **kwargs):
	super(DetailAggregateLoss, self).__init__()
	
	self.laplacian_kernel = torch.tensor(
	[-1, -1, -1, -1, 8, -1, -1, -1, -1],
	dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
	self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
	dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor).requires_grad_())
	self.b_loss = BondaryLoss()
	
	def forward(self, boundary_logits, gtmasks):
		boundary_targets = F.conv2d(gtmasks.type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
		boundary_targets = boundary_targets.clamp(min=0)
		boundary_targets[boundary_targets > 0.1] = 1
		boundary_targets[boundary_targets <= 0.1] = 0
		boundary_targets_x2 = F.conv2d(gtmasks.type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
		boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
		boundary_targets_x4 = F.conv2d(gtmasks.type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
		boundary_targets_x4 = boundary_targets_x4.clamp(min=0)
		# boundary_targets_x8 = F.conv2d(gtmasks.type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
		# boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
		# boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
		boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
		boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
		boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
		boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
		boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
		boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
		# boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
		# boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
		boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
		boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
		boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)
		boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
		boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
		if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
		boundary_logits = F.interpolate(
		boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
		B_loss = self.b_loss(boundary_logits, boudary_targets_pyramid)
	return B_loss
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
                nowd_params += list(module.parameters())
        return nowd_params
def get_boundary(gtmasks):

    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets

