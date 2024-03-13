import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_model import *
    
class differ_agg(nn.Module):
    def __init__(self, inter_type:str):
        super(differ_agg, self).__init__()
        self.inter_type = inter_type
    def forward(self, x, x_differ, T, S):

        if S == 1:
            x = x.unsqueeze(1)
            x_differ = x_differ.permute(0, 2, 1, 3, 4)

        inter_x = x
        # 주어진 입력 x를 그대로 반환합니다.
        if self.inter_type == 'JB':
            # print('inter_x:     ',inter_x.size())
            inter_x[:, :, 1:, :, :] = inter_x[:, :, :1, :, :]
            inter_x[:, :, 1:, 1:, :] += x_differ
            mean_difference = x_differ.mean(dim=2, keepdim=True)
        elif self.inter_type == 'DM':
            mean_difference = x_differ.mean(dim=2, keepdim=True)
            for i in range(T-1):
                inter_x[:, :, i:i+1, 1:, :] += mean_difference * i
        elif self.inter_type == 'DMR':
            mean_difference = x_differ.mean(dim=2, keepdim=True)
            for i in range(T-1):
                inter_x[:, :, i:i+1, 1:, :] += mean_difference * i
            inter_x = x + inter_x # 여기서 사라지는거 같은디
            inter_x = inter_x / 2
        elif self.inter_type == 'DM_make_mean_image':
            mean_difference = x_differ.mean(dim=2, keepdim=True)
            for i in range(T-1):
                inter_x[:, :, i:i+1, 1:, :] -= mean_difference * i

        if S == 1:
            video_cls = inter_x[:, 0, :, 0, :]
        else:
            video_cls = inter_x[:, :, :, 0, :].flatten(1, 2)
        
        inter_x = inter_x.view(-1, inter_x.size(-2), inter_x.size(-1)).permute(1, 0, 2)
        return inter_x, video_cls, mean_difference
    
class differ_block(nn.Module):
    # interpolation_type: 1-> just_before(JB)   2-> Differ Mean(DM)     3-> Differ Mean residual(DMR)
    def __init__(self, input_channel, kernel_size, interpolation_type, substitute_frame):
        super(differ_block, self).__init__()
        self.fc1 = nn.Linear(input_channel, input_channel//2)
        self.conv = nn.Conv3d(input_channel//2, input_channel//2,
                              kernel_size=kernel_size,
                              stride=(1, 1, 1),
                              padding=tuple(x // 2 for x in kernel_size))
        self.gelu = QuickGELU()
        self.fc2 = nn.Linear(input_channel//2, input_channel)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)

        self.substitute_frame = substitute_frame
        self.differ_agg = differ_agg(interpolation_type)

    def forward(self, x, T):
        # print('differ_block1:   ',x.size())
        L, BT, C = x.size()
        S = self.substitute_frame
        Clip = T // S

        if S == 1:
            diff_S = S
            diff_Clip = Clip -1
        else:
            diff_S = S-1
            diff_Clip = Clip

        B = BT // T
        Ca = x.size(-1)//2 # Down Sample을 거친 다음의 2
        H = W = round(math.sqrt(L - 1))
        x = x.permute(1, 0, 2).contiguous()
        # print('differ_block2:   ',x.size())
        x = x.view(B, Clip, S, L, C)

        if S == 1:
            x = x.squeeze(2)
            x_differ = x[:, 1:, :, :] - x[:, :-1, :, :]
        else:
            x_differ = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        
        # Differ modeling
        x_differ = x_differ.view(-1, L, C)
        x_differ = x_differ[:, 1:, :] # CLS token 제거
        x_differ = self.fc1(x_differ)
        x_differ = self.gelu(x_differ)

        if S == 1:
            x_differ = x_differ.view(B, diff_Clip, diff_S, H, W, Ca).permute(0, 2, 5, 1, 3, 4).flatten(0, 1).contiguous()
        else:
            x_differ = x_differ.view(B, diff_Clip, diff_S, H, W, Ca).permute(0, 1, 5, 2, 3, 4).flatten(0, 1).contiguous()

        x_differ = self.conv(x_differ)
        x_differ = x_differ.permute(0, 2, 3, 4, 1).contiguous().view(B*diff_Clip*diff_S, L - 1, Ca)
        

        x_differ = self.fc2(x_differ)
        x_differ = self.gelu(x_differ)
        x_differ = x_differ.view(B, diff_Clip, diff_S, L-1, C)
        # Interpolation
        # 1. 전 frame 값을 대입 한 다음에, differ만 변화 시키는 방향
        inter_x, video_cls, mean_difference = self.differ_agg(x, x_differ, T, S)
        # print(asd)
        return inter_x, mean_difference, video_cls