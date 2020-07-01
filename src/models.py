import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
  def __init__(self, c, n1, n2, n3, f1, f2, f3):
    super(SRCNN, self).__init__()

    # patch extraction
    self.F1 = nn.Conv2d(
      in_channels=c,
      out_channels=n1,
      kernel_size=f1,
      stride=1,
      padding=4,
      )
    
    # non-linear mapping
    self.F2 = nn.Conv2d(
      in_channels=n1,
      out_channels=n2,
      kernel_size=f2,
      stride=1,
      padding=1,       
      )
    
    # reconstruction
    self.F3 = nn.Conv2d(
      in_channels=n2,
      out_channels=n3,
      kernel_size=f3,
      stride=1,
      padding=2,
      )
    
  def forward(self, low_res_img):
    patches = F.relu(self.F1(low_res_img))
    mapping = F.relu(self.F2(patches))
    high_res = self.F3(mapping)

    return high_res