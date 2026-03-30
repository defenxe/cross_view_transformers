
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y


'''



import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # [수정] 부모 모듈(Decoder)에서 이미 입력을 float32로 변환해서 내려보내므로
        # 여기서는 별도의 .to(torch.float32) 없이 연산해도 가중치(FP32)와 타입이 일치합니다.
        
        x = self.conv(x)
        
        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])
            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)
            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

        # [수정] 생성 시점에 Decoder 내부의 모든 가중치(Conv, BN 등)를 float32로 고정합니다.
        # 전체 모델이 .bfloat16()으로 변환된 이후라도, 이 명령어가 마지막에 실행되면 FP32를 유지합니다.
        self.float()

    # [수정] 매우 중요: 만약 외부에서 autocast(dtype=torch.bfloat16)를 사용 중이라면,
    # 해당 영역 내의 연산을 강제로 FP32로 수행하도록 autocast를 일시 비활성화합니다.
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        # [수정] 입력 데이터의 원래 타입(예: bf16)을 저장해둡니다.
        orig_dtype = x.dtype
        
        # [수정] 입력을 FP32로 변환합니다. (가중치가 FP32이므로 입력도 맞춰야 함)
        x = x.float()
        y = x

        for layer in self.layers:
            # [수정] 각 레이어에 들어가는 x(skip용)와 y(입력용) 모두 이미 FP32 상태입니다.
            y = layer(y, x)

        # [수정] Decoder 연산이 끝난 후, 다음 레이어와의 호환성을 위해 
        # 결과를 다시 원래 타입(예: bf16)으로 돌려줍니다.
        return y.to(orig_dtype)
