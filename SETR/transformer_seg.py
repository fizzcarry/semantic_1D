import logging
import math
import os
import numpy as np 

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from SETR.transformer_model import TransModel2d, TransConfig
import math 

class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size * config.hidden_size % sample_v== 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size, config.patch_size * config.hidden_size // sample_v)
        self.patch_size = config.patch_size
        self.ll = self.patch_size // sample_v

        self.is_segmentation = is_segmentation
    def forward(self, x):
        ## x:(b, c, w, h)
        b, c, l = x.shape
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p = self.patch_size

        if l % p != 0:
            print("请重新输入len size 参数 必须整除")
            os._exit(0)

        ll = l // p

        x = rearrange(x, 'b c (ll p)  -> b ll (p c)', p = p)
        
        encode_x = self.bert_model(x)[-1] # 取出来最后一层
        if not self.is_segmentation:
            return encode_x

        x = self.final_dense(encode_x)
        x = rearrange(x, "b l (p c) -> b c (l p) ", p = self.ll, l = ll, c = self.config.hidden_size)
        return encode_x, x 


class PreTrainModel(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=2048,
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

class Vit(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=2048,
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        sample_rate=4,
                        ):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[1024,512, 256, 128]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv1d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm1d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(in_channels=features[0], out_channels=features[0], kernel_size=2, stride=2),
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv1d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm1d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv1d(features[1], features[2], 3, padding=1),
            nn.BatchNorm1d(features[2]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=features[2], out_channels=features[2], kernel_size=2, stride=2)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv1d(features[2], features[3], 3, padding=1),
            nn.BatchNorm1d(features[3]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=features[3], out_channels=features[3], kernel_size=2, stride=2)
        )

        self.final_out = nn.Conv1d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x

class SETRModel(nn.Module):
    def __init__(self, patch_size=16,
                        in_channels=3, 
                        out_channels=1, 
                        hidden_size=2048,
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64],
                        sample_rate=4,):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config)
        self.decoder_2d = Decoder2D(in_channels=config.hidden_size, out_channels=config.out_channels, features=decode_features)

    def forward(self, x):
        _, final_x = self.encoder_2d(x)
        x = self.decoder_2d(final_x)
        return x 

   

