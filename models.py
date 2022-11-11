# Models
import torch
import tensorflow as tf
from torch import nn, Tensor
import math
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from e3nn import o3
import numpy as np
from attention import MultiHeadAttention, ChannelAttentionGG
import utils.key_signatures as key_sig
from torchmetrics import Accuracy
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple
import torch.utils.checkpoint as cp

class EquivariantDropout(torch.nn.Module):
    # Implementation for this Dropout class from: https://github.com/e3nn/e3nn/blob/main/e3nn/nn/_dropout.py
    """Equivariant Dropout
    :math:`A_{zai}` is the input and :math:`B_{zai}` is the output where
    - ``z`` is the batch index
    - ``a`` any non-batch and non-irrep index
    - ``i`` is the irrep index, for instance if ``irreps="0e + 2x1e"`` then ``i=2`` select the *second vector*
    .. math::
        B_{zai} = \frac{x_{zi}}{1-p} A_{zai}
    where :math:`p` is the dropout probability and :math:`x` is a Bernoulli random variable with parameter :math:`1-p`.
    Parameters
    ----------
    irreps : `o3.Irreps`
        representation
    p : float
        probability to drop
    """

    def __init__(self, irreps, p):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, p={self.p})"

    def forward(self, x):
        """evaluate
        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        if not self.training:
            return x

        batch = x.shape[0]

        noises = []
        for mul, (l, _p) in self.irreps:
            dim = 2 * l + 1
            noise = x.new_empty(batch, mul)

            if self.p >= 1:
                noise.fill_(0)
            elif self.p <= 0:
                noise.fill_(1)
            else:
                noise.bernoulli_(1 - self.p).div_(1 - self.p)

            noise = noise[:, :, None].expand(-1, -1, dim).reshape(batch, mul * dim)
            noises.append(noise)

        noise = torch.cat(noises, dim=-1)
        while noise.dim() < x.dim():
            noise = noise[:, None]
        return x * noise

class EquivariantPitchClassConvolutionSimple(nn.Module):
    def __init__(self, pitch_classes: int, in_channels: int, out_channels: int, kernel_depth: int,
                 same_depth_padding=False):
        super().__init__()
        if out_channels > 0:
            self.conv2d = nn.Conv2d(in_channels, out_channels, (pitch_classes, kernel_depth),
                                    padding=(0, kernel_depth // 2 if same_depth_padding else 0))
        self.pitch_classes = pitch_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_depth = kernel_depth
        self._same_depth_padding = same_depth_padding
        

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: tensor if dimension   (N, in_channels,  pitch_classes, depth)
        :return:  tensor of dimension   (N, out_channels, pitch_classes, depth - kernel_depth + 1)
        """
        assert len(x.shape) == 4
        assert x.shape[1:3] == torch.Size([self.in_channels, self.pitch_classes])
        x_wrap = torch.cat([x, x[::, ::, 0:self.pitch_classes - 1, ::]], dim=2)
        if self.out_channels > 0:
            return self.conv2d(x_wrap)
        else:
            return torch.empty((x.shape[0], 1, self.pitch_classes,
                                x.shape[3] if self._same_depth_padding else x.shape[3] - self.kernel_depth + 1),
                               device=x.device)[::, 0:0]
            # TODO: Find better way to create tensor with dimensions that have zero length
    
    def generate_random_bias(self):
        '''
        This is for test purpose, since usually a bias is initialized to zeros
        :return:
        '''
        for i, param in enumerate(self.conv2d.parameters()):
            if i == 1:
                param.data = torch.rand(param.shape)
    
    def set_weights(self, new_values):
        for i, param in enumerate(self.conv2d.parameters()):
            if i == 0:
                param.data = new_values

    def set_bias(self, new_values):
        for i, param in enumerate(self.conv2d.parameters()):
            if i == 1:
                param.data = new_values

    def get_weights(self):
        for i, param in enumerate(self.conv2d.parameters()):
            if i == 0:
                return param

    def get_bias(self):
        for i, param in enumerate(self.conv2d.parameters()):
            if i == 1:
                return param
            
class Pitch2PitchClassPool(nn.Module):
    def __init__(self, pitch_classes: int, pitches_in: int, kernel_depth: int, padding_value):
        super().__init__()
        self.pitch_classes = pitch_classes
        self.pitches_in = pitches_in
        self.kernel_depth = kernel_depth
        self.padding_value = padding_value
        self.kernel_size = math.ceil(self.pitches_in / self.pitch_classes)
        self.padding = self.kernel_size * self.pitch_classes - self.pitches_in
        self.pool = nn.MaxPool2d((self.kernel_size, self.kernel_depth), (1, 1), dilation=(self.pitch_classes, 1))
        

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: tensor if dimension   (N, in_channels,  pitch_in, depth)
        :return:  tensor of dimension   (N, out_channels, pitch_classes, depth / kernel_depth)
        """
        assert len(x.shape) == 4
        
        shape = x.shape
        x_padded = torch.cat(
            [x, torch.full(size=(shape[0], shape[1], self.padding, shape[3]), fill_value=self.padding_value, device=x.device)], dim=2)
        return self.pool(x_padded)
    
class Pitch2PitchClassAlternative(nn.Module):
    def __init__(self, pitch_classes: int, pitches_in: int, kernel_depth: int, padding_value, in_channels: int, n_filters):
        super().__init__()
        self.pitch_classes = pitch_classes
        self.pitches_in = pitches_in
        self.kernel_depth = kernel_depth
        self.padding_value = padding_value
        self.in_channels = in_channels
        self.kernel_size = math.ceil(self.pitches_in / self.pitch_classes)
        self.n_filters = n_filters
        self.filters = n_filters
        self.p2pc = nn.Sequential(
            
            nn.Conv2d(self.in_channels, self.filters, kernel_size=self.kernel_depth,padding=self.kernel_depth//2, bias=False),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(),
            
            nn.Conv2d(self.filters, self.filters, kernel_size=self.kernel_depth,padding=self.kernel_depth//2,bias=False),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(),
            
            nn.Conv2d(self.filters, self.filters, kernel_size=(5,1),dilation=(36,1),bias=False),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(),
            
            nn.Conv2d(self.filters, 2*self.filters, kernel_size=self.kernel_depth,padding=self.kernel_depth//2, bias=False),
            nn.BatchNorm2d(2*self.filters),
            nn.LeakyReLU(),
            
            nn.Conv2d(2*self.filters, 2*self.filters, kernel_size=self.kernel_depth,padding=self.kernel_depth//2,bias=False),
            nn.BatchNorm2d(2*self.filters),
            nn.LeakyReLU(),
            
            #nn.Conv2d(2*self.filters, 4*self.filters, kernel_size=(self.kernel_depth,1)),
            #nn.BatchNorm2d(4*self.filters),
            #nn.LeakyReLU(),
            
            #nn.Conv2d(4*self.filters, 4*self.filters, kernel_size=(self.kernel_depth,1)),
            #nn.BatchNorm2d(4*self.filters),
            #nn.LeakyReLU(),
            
            #nn.Conv2d(4*self.filters, 4*self.filters, kernel_size=(self.kernel_depth,1)),
            #nn.BatchNorm2d(4*self.filters),
            #nn.LeakyReLU(),
            
            Pitch2PitchClassPool(self.pitch_classes,36, 1, float('-inf'))
            
            
            ).double().cuda()
        shape = self.p2pc[6].weight.shape
        self.p2pc[6].weight = torch.nn.Parameter(torch.ones(shape).double().cuda())
        #print(self.p2pc[6].weight)
        
    def forward(self, x: Tensor) -> Tensor:
        
        x = self.p2pc(x)
        
        return x
        
    
class PitchClass2Pitch(nn.Module):
    def __init__(self, pitches: int):
        super().__init__()
        self.target_length = pitches

    def forward(self, x: Tensor):
        repetitions = math.ceil(self.target_length / x.shape[2])
        x_repeated = x.repeat(1, 1, repetitions, 1)
        return x_repeated[::, ::, 0:self.target_length, ::]
    
class PitchClass2PitchClass(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, conv_layers: int, resblock: bool, denseblock: bool):
        super().__init__()
        self.pitch_classes = 12
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.resblock = resblock
        self.denseblock = denseblock
        
        self.modules = []
        if self.resblock:
            for i in range(conv_layers):
                if i==0:
                    self.modules.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels=self.in_channels, out_channels=self.out_channels, kernel_depth=self.kernel_size, same_depth_padding=True))
                    self.modules.append(nn.BatchNorm2d(self.out_channels))
                    self.modules.append(nn.LeakyReLU())
                self.modules.append(ResBlockEquivariant(self.kernel_size, self.conv_layers, self.out_channels))
        elif self.denseblock:
            self.modules.append(DenseBlockEquivariant(num_layers=self.conv_layers, num_input_features=self.in_channels, bn_size=self.in_channels//2 if self.in_channels>1 else 1, growth_rate=self.out_channels, drop_rate=0.0))
        else:
            for i in range(conv_layers):
                if i==0:
                    self.modules.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels=self.in_channels, out_channels=self.out_channels, kernel_depth=self.kernel_size, same_depth_padding=True))
                else:
                    self.modules.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels=self.out_channels, out_channels=self.out_channels, kernel_depth=self.kernel_size, same_depth_padding=True))
                self.modules.append(nn.BatchNorm2d(self.out_channels))
                self.modules.append(nn.LeakyReLU())
        
        self.layer = nn.Sequential(*self.modules).double().cuda()

    def forward(self, pc: Tensor):
        pc = self.layer(pc)
        return pc
    
class Pitch2Pitch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, conv_layers: int, resblock: bool, denseblock: bool):
        super().__init__()
        self.pitch_classes = 12
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.resblock = resblock
        self.denseblock = denseblock
        
        self.modules = []
        if self.resblock:
            for i in range(conv_layers):
                if i==0:
                    self.modules.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2, padding_mode="circular"))
                    self.modules.append(nn.BatchNorm2d(self.out_channels))
                    self.modules.append(nn.LeakyReLU())
                self.modules.append(ResBlock(self.kernel_size, self.conv_layers, self.out_channels))
        elif self.denseblock:
            self.modules.append(DenseBlock(num_layers=self.conv_layers, num_input_features=self.in_channels, bn_size=self.in_channels//2 if self.in_channels>1 else 1, growth_rate=self.out_channels, drop_rate=0.0))
        else:
            for i in range(conv_layers):
                if i==0:
                    self.modules.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2, padding_mode="circular"))
                else:
                    self.modules.append(nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2, padding_mode="circular"))
                self.modules.append(nn.BatchNorm2d(self.out_channels))
                self.modules.append(nn.LeakyReLU())
            
        
        self.layer = nn.Sequential(*self.modules).double().cuda()

    def forward(self, p: Tensor):
        
        p = self.layer(p)
        
        return p
    
    
class PitchClassNetLayer(nn.Module):
    def __init__(self, pitches, pitch_classes, num_layers, kernel_size, layer_num, window_size, conv_layers, num_filters, resblock, denseblock):
        super().__init__()
        self.pitch_classes = pitch_classes
        self.pitches = pitches
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        self.num_layers = num_layers
        self.window_size = window_size
        self.a_heads = 1
        self.a_depth = 1
        self.conv_layers = conv_layers
        self.num_filters = num_filters
        self.final_window_size = self.window_size-((self.kernel_size-1)*(self.num_layers-1)*(self.conv_layers))
        self.final = 16-self.kernel_size+1#(592//(2**(self.num_layers-1)))-1*(self.kernel_size-1)#self.final_window_size
        self.final_window_size = 1
        self.resblock = resblock
        self.denseblock = denseblock
        
        # define input & output channels for each layer:
        if self.denseblock:
            # filters in layer 1
            self.all_previous_channels_p = 1
            self.all_previous_channels_pc = 1+self.num_filters*self.conv_layers
            # filters in intermediate layers
            for i in range(self.layer_num-1):
                self.all_previous_channels_p += self.num_filters*self.conv_layers+self.all_previous_channels_pc
                self.all_previous_channels_pc += self.num_filters*self.conv_layers+self.all_previous_channels_p
                
            self.out_channels_p = self.all_previous_channels_p + self.num_filters*self.conv_layers + self.all_previous_channels_pc
            self.out_channels_pc = self.all_previous_channels_pc + self.num_filters*self.conv_layers + self.all_previous_channels_p
            self.out_channels_dense = self.num_filters # growth rate for denseblocks
        else:
            if self.layer_num==0:
                self.all_previous_channels_p = 0
                self.all_previous_channels_pc = 0
            elif self.layer_num==1:
                self.all_previous_channels_p = 1
                self.all_previous_channels_pc = self.num_filters
            elif self.layer_num==2:
                self.all_previous_channels_p = self.num_filters*2
                self.all_previous_channels_pc = 2*self.all_previous_channels_p
            else:
                self.all_previous_channels_p = (self.num_filters*2)*(4**(self.layer_num-2))
                self.all_previous_channels_pc = 2*self.all_previous_channels_p

            if self.layer_num==0:
                self.out_channels_p = 1
                self.out_channels_pc = 4
            elif self.layer_num==1:
                self.out_channels_p = 2*self.num_filters
                self.out_channels_pc = 2*self.out_channels_p
            else:
                self.out_channels_p = 4*self.all_previous_channels_p
                self.out_channels_pc = 4*self.all_previous_channels_pc

        # Layer Content Definition:
        if self.layer_num==0:
            #self.pre = Pitch2Pitch(in_channels = 1, out_channels = 1, kernel_size = self.kernel_size, conv_layers = self.conv_layers, resblock=self.resblock)
            #self.pool = Pitch2PitchClassPool(self.pitch_classes, self.pitches-(self.kernel_size-1)*self.layer_num, 1, float('-inf'))
            self.pool_semi = nn.Conv2d(1, 1, 3, stride=(3,1), padding=(0,1),padding_mode="circular")
            self.pool_semi_b = nn.BatchNorm2d(1)
            self.pool_semi_a = nn.LeakyReLU()
            #self.pool_semi = nn.MaxPool2d(kernel_size=(3,1))#nn.AvgPool2d(kernel_size=(3,1))
            self.pool = Pitch2PitchClassPool(self.pitch_classes, self.pitches//3, 1, float('-inf'))
            #self.pool = EquivariantDilatedLayer(nf=self.num_filters, in_channels=self.num_filters, out_channels=self.num_filters)
            self.pc2pc = PitchClass2PitchClass(in_channels = 1, out_channels = self.num_filters, kernel_size = self.kernel_size, conv_layers = self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
            #self.pc2pc = PitchClass2PitchClass(in_channels = 80, out_channels = 80, kernel_size = self.kernel_size, conv_layers = self.conv_layers, resblock=self.resblock)
        else:
            #self.up = PitchClass2Pitch(self.pitches-(self.kernel_size-1)*(self.layer_num-1)*self.conv_layers)
            self.up_sixth = nn.ConvTranspose2d(in_channels=self.all_previous_channels_pc, out_channels=self.all_previous_channels_pc, kernel_size=(3,1), stride=(3,1))
            self.up_sixth_b = nn.BatchNorm2d(self.all_previous_channels_pc)
            self.up_sixth_a = nn.LeakyReLU()
            self.up = PitchClass2Pitch(self.pitches)
            if self.denseblock:
                self.p2p = Pitch2Pitch(in_channels=self.all_previous_channels_pc+self.all_previous_channels_p, out_channels=self.out_channels_dense, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
            else:
                self.p2p = Pitch2Pitch(in_channels=self.all_previous_channels_pc+self.all_previous_channels_p, out_channels=self.out_channels_p, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
            #self.pool = Pitch2PitchClassPool(self.pitch_classes, self.pitches-(self.kernel_size-1)*self.layer_num*self.conv_layers, 1, float('-inf'))
            self.pool_semi = nn.Conv2d(in_channels=self.out_channels_p, out_channels=self.out_channels_p, kernel_size=(3,3), stride=(3,1), padding=(0,1), padding_mode="circular")
            self.pool_semi_b = nn.BatchNorm2d(self.out_channels_p)
            self.pool_semi_a = nn.LeakyReLU()
            self.pool = Pitch2PitchClassPool(self.pitch_classes, self.pitches//3, 1, float('-inf'))
            #self.pool = EquivariantDilatedLayer(nf=(self.layer_num+1)*self.num_filters, in_channels=(self.layer_num+1)*self.num_filters, out_channels=(self.layer_num+1)*self.num_filters)
            if self.denseblock:
                self.pc2pc = PitchClass2PitchClass(in_channels=self.out_channels_p+self.all_previous_channels_pc, out_channels=self.out_channels_dense, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
            else:
                self.pc2pc = PitchClass2PitchClass(in_channels=self.out_channels_p+self.all_previous_channels_pc, out_channels=self.out_channels_pc, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
            self.time_pool_p = nn.MaxPool2d(kernel_size=(1,2))
            self.time_pool_pc = nn.MaxPool2d(kernel_size=(1,2))

    def forward(self, x: Tensor) -> Tensor:
        
        (p, pc) = x
        if self.layer_num>0:
            assert len(pc.shape) == 4
        assert len(p.shape) == 4
        
        if self.layer_num==0:
            #p = self.pre(p)
            p_semi = self.pool_semi(p) # Pool from Sixth of a tone to semitone
            p_semi = self.pool_semi_b(p_semi)
            p_semi = self.pool_semi_a(p_semi)
            pc = self.pool(p_semi) # Pitch2PitchClass
            pc = self.pc2pc(pc) # Equivariant Layer
            #print("PC2PC Layer1: "+str(pc.shape))
        else:
            p_sixth = self.up_sixth(pc)
            p_sixth = self.up_sixth_b(p_sixth)
            p_sixth = self.up_sixth_a(p_sixth)
            p2 = self.up(p_sixth) # PitchClass2Pitch
            p = torch.cat([p,p2],dim=1) # PitchConcat
            #print("P after Concat: "+str(p.shape))
            p = self.p2p(p) # Pitch2PitchConv
            #print("P after P2P: "+str(p.shape))
            pc2 = self.pool_semi(p) # Pool from Sixth of a tone to semitone
            pc2 = self.pool_semi_b(pc2)
            pc2 = self.pool_semi_a(pc2)
            pc2 = self.pool(pc2) # Pitch2PitchClass
            #print("PC after Pool: "+str(pc2.shape))
            pc = torch.cat([pc, pc2],dim=1) # PitchClassConcat
            #print("PC after Concat: "+str(pc.shape))
            pc = self.pc2pc(pc) # Equivariant Layer
            #print("PC after PC2PC: "+str(pc.shape))
            p = self.time_pool_p(p) # Time Pooling for pitch wise level
            pc = self.time_pool_pc(pc) # Time Pooling for pitchClass wise level
        if self.layer_num==(self.num_layers-1):
            #tonic = self.tonic_classifier(pc)
            #tonic = torch.mean(tonic, axis=-1)
            #tonic = nn.Flatten()(tonic)
            #pc = self.eq(pc)
            #pc = self.global_pool(pc)
            #pc = torch.mean(pc, axis=-1)
            #pc = nn.Flatten()(pc)
            #pc = self.sig(pc)
            pass
            
            
        return (p, pc)
    
    
class ResBlock(nn.Module):
    def __init__(self, kernel_size, conv_layers, num_filters):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.num_filters = num_filters
        
        self.conv1 = nn.Conv2d(self.num_filters, 2*self.num_filters, self.kernel_size, padding=(self.kernel_size//2, self.kernel_size//2), padding_mode='circular')
        self.b1 = nn.BatchNorm2d(2*self.num_filters)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(2*self.num_filters, self.num_filters, self.kernel_size, padding=(self.kernel_size//2, self.kernel_size//2), padding_mode='circular')
        self.b2 = nn.BatchNorm2d(self.num_filters)
        self.act2 = nn.LeakyReLU()
        
        
    def forward(self, x: Tensor) -> Tensor:
        
        x_res = self.conv1(x)
        x_res = self.b1(x_res)
        x_res = self.act1(x_res)
        x_res = self.conv2(x_res)
        x_res = self.b2(x_res)
        x = x+x_res
        x = self.act2(x)
        
        return x
    
class ResBlockEquivariant(nn.Module):
    def __init__(self, kernel_size, conv_layers, num_filters):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.num_filters = num_filters
        
        self.conv1 = EquivariantPitchClassConvolutionSimple(pitch_classes=12, in_channels=self.num_filters, out_channels=2*self.num_filters, kernel_depth=self.kernel_size, same_depth_padding=True)
        self.b1 = nn.BatchNorm2d(2*self.num_filters)
        self.act1 = nn.LeakyReLU()
        self.conv2 = EquivariantPitchClassConvolutionSimple(pitch_classes=12, in_channels=2*self.num_filters, out_channels=self.num_filters, kernel_depth=self.kernel_size, same_depth_padding=True)
        self.b2 = nn.BatchNorm2d(self.num_filters)
        self.act2 = nn.LeakyReLU()
        
        
    def forward(self, x: Tensor) -> Tensor:
        
        x_res = self.conv1(x)
        x_res = self.b1(x_res)
        x_res = self.act1(x_res)
        x_res = self.conv2(x_res)
        x_res = self.b2(x_res)
        x = x+x_res
        x = self.act2(x)
        
        return x

class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseLayerEquivariant(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = EquivariantPitchClassConvolutionSimple(pitch_classes=12, in_channels=num_input_features, out_channels=bn_size * growth_rate, kernel_depth=1)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = EquivariantPitchClassConvolutionSimple(pitch_classes=12, in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_depth=3, same_depth_padding=True)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features
    
class DenseBlock(nn.ModuleDict):

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
    
class DenseBlockEquivariant(nn.ModuleDict):

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayerEquivariant(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class AttentionPitchClassNet(pl.LightningModule):
    
    def __init__(self, pitches, pitch_classes, num_layers, kernel_size, opt=None, window_size=23, batch_size=4, train_set=None, val_set=None):
        super().__init__()
        # set hyperparams
        #self.hparams.update(hparams)
        self.pitches = pitches
        self.pitch_classes = pitch_classes
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.opt = opt
        self.conv_layers = opt.conv_layers
        self.n_filters = opt.n_filters
        self.resblock = opt.resblock
        self.denseblock = opt.denseblock
        self.data = {'train': train_set,
                     'val': val_set}
        
        ########################################################################
        # Initialize Model:                                               #
        ########################################################################
        self.modules = []
        for i in range(self.num_layers):
            self.modules.append(PitchClassNetLayer(self.pitches, self.pitch_classes, self.num_layers, self.kernel_size, i, self.window_size, self.conv_layers, self.n_filters, self.resblock, self.denseblock))
        
        
        self.model = nn.Sequential(*self.modules)
        # define input channels for output layer:
        if self.denseblock:
            # filters in layer 1
            self.all_previous_channels_p = 1
            self.all_previous_channels_pc = 1+self.n_filters*self.conv_layers
            # filters in intermediate layers
            for i in range(self.num_layers-2):
                self.all_previous_channels_p += self.n_filters*self.conv_layers+self.all_previous_channels_pc
                self.all_previous_channels_pc += self.n_filters*self.conv_layers+self.all_previous_channels_p
            if self.num_layers>1:  
                self.out_channels_p = self.all_previous_channels_p + self.n_filters*self.conv_layers + self.all_previous_channels_pc
                self.final_channels = self.all_previous_channels_pc + self.n_filters*self.conv_layers + self.out_channels_p
            else:
                self.final_channels = self.all_previous_channels_pc
        else:
            if self.num_layers==1:
                self.all_previous_channels_p = 0
                self.all_previous_channels_pc = 0
            elif self.num_layers==2:
                self.all_previous_channels_p = 1
                self.all_previous_channels_pc = self.n_filters
            elif self.num_layers==3:
                self.all_previous_channels_p = self.n_filters*2
                self.all_previous_channels_pc = 2*self.all_previous_channels_p
            else:
                self.all_previous_channels_p = (self.n_filters*2)*(4**(self.num_layers-3))
                self.all_previous_channels_pc = 2*self.all_previous_channels_p

            if self.num_layers==1:
                self.final_channels = self.n_filters
            else:
                self.final_channels = 4*self.all_previous_channels_pc

        #self.final_channels_tonic = self.final_channels if self.num_layers>1 else 1
        #self.final_channels_tonic_int = self.final_channels_tonic if self.num_layers>1 else self.n_filters
        self.tonic_classifier = nn.Sequential(
            EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels = self.final_channels, out_channels=1, kernel_depth=self.kernel_size),
            ).double().cuda()
        self.key_classifier = nn.Sequential(
            EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels = self.final_channels, out_channels=1, kernel_depth=self.kernel_size),
            ).double().cuda()
        #self.eq = EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels=80, out_channels=1, kernel_depth=self.kernel_size).double().cuda()
        #self.global_pool = nn.MaxPool2d(kernel_size=(1,self.final))
        self.sig = nn.Sigmoid()

    
    def forward(self, mel):
        
        #mel = self.feature_extractor(mel)
        (p, pc) = self.model((mel,None))
        tonic = self.tonic_classifier(pc)
        tonic = torch.mean(tonic, axis=-1)
        tonic = nn.Flatten()(tonic)
        pc = self.key_classifier(pc)
        #pc = self.global_pool(pc)
        pc = torch.mean(pc, axis=-1)
        pc = nn.Flatten()(pc)
        pc = self.sig(pc)
        if self.opt.local: # for local key estimation
            x = pc.reshape(pc.shape[0], pc.shape[2], pc.shape[3])
        else: # for global key estimation
            #x = pc.reshape(pc.shape[0], pc.shape[2])
            x = (pc, tonic)
        
        
        return x

    def general_step(self, batch, batch_idx, mode):
        
        mel = batch['mel']
        key_signature_id = batch['key_signature_id']
        key_labels = batch['key_labels'].double()
        tonic_labels = batch['tonic_labels'].long()
        tonic_labels_idx = torch.argmax(tonic_labels,axis=1)
        
        if self.opt.local:
            seq_length = batch['seq_length']
        # forward pass
        out = self.forward(mel)
        key_out, tonic_out = out
        
        # loss
        loss_func = nn.BCELoss()
        loss_func_tonic = nn.CrossEntropyLoss()
        bce_loss = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                
                bce_loss = bce_loss + loss_func(key_out[i,:,:(seq_length[i]-self.window_size+1)], key_labels[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            bce_loss = loss_func(key_out, key_labels) # for global key estimation
            tonic_loss = loss_func_tonic(tonic_out.detach().cpu(), tonic_labels_idx.detach().cpu())
            
        cosine_sim_func = nn.CosineSimilarity(dim=1)
        #cosine_sim = cosine_sim_func(key_out, key_labels)
        loss = bce_loss + self.opt.tonic_weight*tonic_loss# + (1 - torch.sum(cosine_sim)/key_out.shape[0])
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                accuracy = accuracy + self.all_key_accuracy_local(key_labels[i,:,:(seq_length[i]-self.window_size+1)], key_out[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            #accuracy = self.all_key_accuracy(key_labels, key_out)# for global key estimation
            mirex, correct, fifths, relative, parallel, other, accuracy = self.mirex_score(key_labels, key_out, tonic_labels, tonic_out, key_signature_id)
            Accuracy_tonic = Accuracy().cuda()
            accuracy_tonic = Accuracy_tonic(torch.argmax(tonic_out,axis=1), tonic_labels_idx)
            
            
        return loss, accuracy, mirex, correct, fifths, relative, parallel, other, accuracy_tonic

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        acc = torch.stack(
            [x[mode + '_accuracy'] for x in outputs]).mean()
        mirex_score = torch.stack(
            [x[mode + '_mirex_score'] for x in outputs]).mean()
        correct = torch.stack(
            [x[mode + '_correct'] for x in outputs]).mean()
        fifths = torch.stack(
            [x[mode + '_fifths'] for x in outputs]).mean()
        relative = torch.stack(
            [x[mode + '_relative'] for x in outputs]).mean()
        parallel = torch.stack(
            [x[mode + '_parallel'] for x in outputs]).mean()
        other = torch.stack(
            [x[mode + '_other'] for x in outputs]).mean()
        acc_tonic = torch.stack(
            [x[mode + '_accuracy_tonic'] for x in outputs]).mean()
        return avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic

    def training_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_accuracy': accuracy, 'train_mirex_score': mirex_score, 'train_correct': correct, 'train_fifths': fifths, 'train_relative': relative, 'train_parallel': parallel, 'train_other': other, 'train_accuracy_tonic': acc_tonic, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_accuracy': accuracy, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic}

    def test_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy, 'test_mirex_score': mirex_score, 'train_correct': correct, 'test_fifths': fifths, 'test_relative': relative, 'test_parallel': parallel, 'test_other': other, 'test_accuracy_tonic': acc_tonic}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-Acc_Tonic={}".format(acc_tonic))
        print("Val-Mirex_Score={}".format(mirex_score))
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", acc)
        self.log("val_mirex_score", mirex_score)
        self.log("val_correct", correct)
        self.log("val_fifths", fifths)
        self.log("val_relative", relative)
        self.log("val_parallel", parallel)
        self.log("val_other", other)
        self.log("val_accuracy_tonic", acc_tonic)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy", acc, self.global_step)
        self.logger.experiment.add_scalar("val_mirex_score", mirex_score, self.global_step)
        self.logger.experiment.add_scalar("val_correct", correct, self.global_step)
        self.logger.experiment.add_scalar("val_fifths", fifths, self.global_step)
        self.logger.experiment.add_scalar("val_relative", relative, self.global_step)
        self.logger.experiment.add_scalar("val_parallel", parallel, self.global_step)
        self.logger.experiment.add_scalar("val_other", other, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy_tonic", acc_tonic, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.batch_size)
    
    def configure_optimizers(self):

        params = list(self.model.parameters()) + list(self.tonic_classifier.parameters()) + list(self.key_classifier.parameters())# + list(self.classifier.parameters())
            
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.opt.lr, weight_decay=self.opt.reg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.opt.gamma)

        return [optim], [scheduler]
    
    def all_key_accuracy(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        for i in range(len(y_true)):
            y_pred_keys = y_pred[i]
            y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
            label_key = y_true[i]
            correct_keys = torch.sum(y_pred_keys == label_key)
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
        return torch.tensor((accuracy / samples))
    
    def all_key_accuracy_local(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        #for i in range(len(y_true)):
        y_pred_keys = y_pred
        y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
        label_key = y_true
        correct_keys = torch.sum(y_pred_keys == label_key)
        score = score + correct_keys
        accuracy = accuracy + (1 if correct_keys == 12 else 0)
        samples = samples + 1
        return torch.tensor((accuracy / samples))
    
    # not functional yet!!!
    def single_key_accuracy(y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        for i in range(len(y_true)):
            y_pred_keys = y_pred[i]
            y_pred_keys = tf.cast(y_pred_keys >= tf.sort(y_pred_keys)[-7], tf.float32)
            label_key = y_true[i]
            correct_keys = tf.math.reduce_sum(tf.cast(y_pred_keys == label_key[:12], tf.int32))
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
            #print("Label: "+str(label_key[:12]))
            #print("Pred: "+str(y_pred_keys))
        return tf.convert_to_tensor((score / (samples*12)))
    
    # needs to be tested!!!
    def genre_accuracy(y_true, y_pred):
        samples, score = 0, 0
        for i in range(len(y_true)):
            y_pred_genre = y_pred[i]
            y_true_genre = y_true[i]
            if torch.argmax(y_pred_genre)==torch.argmax(y_true_genre):
                score += 1
            samples += 1
        
        return torch.tensor((score/samples))

    def mirex_score(self, key_labels, key_preds, tonic_labels, tonic_preds, key_signature_id):
        score, accuracy, samples = 0, 0, 0
        similarity = 0
        correct_tonics_total = 0
        mirex_score, correct, fifths, parallel, relative, other = 0, 0, 0, 0, 0, 0
    
        for i in range(len(key_labels)):
            category = 0
            key_label = key_labels[i]
            key_pred_values = key_preds[i]
            tonic_label = tonic_labels[i]
            tonic_pred = tonic_preds[i]
            KEY_SIGNATURE_MAP = torch.tensor(key_sig.KEY_SIGNATURE_MAP.numpy()).cuda()
    
            # Every Key signature contains 7 pitch classes so use top 7 pitch class predictions
            #   Only major and minro are relevant for dataset
    
            # Find most similar cosine key signature given key predictions:
            cosinesim = nn.CosineSimilarity(dim=1)
            pred_key_id = torch.argmax(cosinesim(key_pred_values.reshape(1,key_pred_values.shape[0]), KEY_SIGNATURE_MAP))
            key_pred = KEY_SIGNATURE_MAP[pred_key_id]
            key_sig_label_id = torch.argmax(key_signature_id[i])
            
            # Check for relative fifths:
            cosinesim2 = nn.CosineSimilarity(dim=0)
            correct_keys = torch.sum(key_pred == key_label)
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
            similarity = similarity + cosinesim2(key_pred_values, key_label)
            diff = torch.abs(pred_key_id - key_sig_label_id)
            correct_tonic = 1 if torch.argmax(tonic_label) == torch.argmax(tonic_pred) else 0
            tonic_diff = torch.abs(torch.argmax(tonic_label) - torch.argmax(tonic_pred))
            correct_tonics_total = correct_tonics_total + correct_tonic
    
            if (diff == 1) and not(correct_tonic == 1 and correct_keys == 12):
                fifths = fifths + 1
                category = 1
            if correct_tonic == 1 and correct_keys == 12 and category == 0:
                correct = correct + 1
                category = 1
            if correct_keys == 12 and correct_tonic == 0 and category == 0:
                relative = relative + 1
                category = 1
            if correct_tonic == 1 and correct_keys != 12 and category == 0:
                parallel = parallel + 1
                category = 1
            if category == 0:
                other = other + 1
        mirex_score = (1 * correct) + (0.5 * fifths) + (0.3 * relative) + (0.2 * parallel)
        
        '''
        score = score / (samples * 12)
        accuracy = accuracy / samples
        print(f'dataset accuracy:')
        print(f'Correctly classified single keys: {score:.2%}')
        print(f'Correctly classified songs (all keys): {accuracy:.2%}')
        print(f'Avg. Cosine Similarity: {similarity / samples:.2%}')
        print(f'Fifths: {fifths / samples:.2%}')
        print(f'Correct tonics: {correct_tonics_total / samples:.2%}')
        print(f'All correct: {correct / samples:.2%}')
        print(f'Others: {others / samples:.2%}')
        print(f'MiReX Score: {mirex_score / samples:.2%}')
        '''
        
        return torch.tensor(mirex_score/samples).float(), torch.tensor(correct/samples).float(), torch.tensor(fifths/samples).float(), torch.tensor(relative/samples).float(), torch.tensor(parallel/samples).float(), torch.tensor(other/samples).float(), torch.tensor(accuracy/samples).float()


    
class TestNet(pl.LightningModule):
    
    def __init__(self, pitches, pitch_classes, num_layers, kernel_size, opt=None, window_size=23, batch_size=4, train_set=None, val_set=None):
        super().__init__()
        # set hyperparams
        #self.hparams.update(hparams)
        self.pitches = pitches
        self.pitch_classes = pitch_classes
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.opt = opt
        self.conv_layers = opt.conv_layers
        self.n_filters = opt.n_filters
        self.resblock = opt.resblock
        self.data = {'train': train_set,
                     'val': val_set}
        
        ########################################################################
        # Initialize Model:                                               #
        ########################################################################
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(1,3)),
            #Pitch2PitchClassPool(pitch_classes = 60, pitches_in = 180, kernel_depth = 1, padding_value = float('inf')),
            nn.AvgPool2d(kernel_size=(3,1)),
            #nn.MaxPool2d(kernel_size=(3,1)),
            
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(1,3)),
            Pitch2PitchClassPool(pitch_classes = 12, pitches_in = 60, kernel_depth = 1, padding_value = float('inf')),
            
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(1,4)),
            #Pitch2PitchClassPool(pitch_classes = 12, pitches_in = 30, kernel_depth = 1, padding_value = float('inf')),
            
            nn.Conv2d(in_channels=64,out_channels=80,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=80,out_channels=80,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=80,out_channels=80,kernel_size=self.kernel_size,padding=self.kernel_size//2,padding_mode="circular"),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=80, out_channels=1, kernel_size=1),
            
            )

    
    def forward(self, mel):
        
        pc = self.model(mel)
        if self.opt.local: # for local key estimation
            x = pc.reshape(pc.shape[0], pc.shape[2], pc.shape[3])
        else: # for global key estimation
            #x = pc.reshape(pc.shape[0], pc.shape[2])
            pc = torch.mean(pc, axis=-1)
            pc = nn.Flatten()(pc)
            pc = nn.Sigmoid()(pc)
            x = pc
        
        return x

    def general_step(self, batch, batch_idx, mode):
        
        mel = batch['mel']
        key_signature_id = batch['key_signature_id']
        key_labels = batch['key_labels'].double()
        if self.opt.local:
            seq_length = batch['seq_length']
        # forward pass
        out = self.forward(mel)
        
        # loss
        loss_func = nn.BCELoss()
        bce_loss = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                
                bce_loss = bce_loss + loss_func(out[i,:,:(seq_length[i]-self.window_size+1)], key_labels[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            bce_loss = loss_func(out, key_labels) # for global key estimation
            
        cosine_sim_func = nn.CosineSimilarity(dim=1)
        cosine_sim = cosine_sim_func(out, key_labels)
        loss = bce_loss# + (1 - torch.sum(cosine_sim)/out.shape[0])
        accuracy = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                accuracy = accuracy + self.all_key_accuracy_local(key_labels[i,:,:(seq_length[i]-self.window_size+1)], out[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            accuracy = self.all_key_accuracy(key_labels, out)# for global key estimation
        
            
        return loss, accuracy

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        acc = torch.stack(
            [x[mode + '_accuracy'] for x in outputs]).mean()
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_accuracy': accuracy, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", acc)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy", acc, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.batch_size)
    
    def configure_optimizers(self):

        params = list(self.model.parameters())# + list(self.classifier.parameters())
            
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.opt.lr, weight_decay=self.opt.reg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.opt.gamma)

        return [optim], [scheduler]
    
    def all_key_accuracy(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        for i in range(len(y_true)):
            y_pred_keys = y_pred[i]
            y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
            label_key = y_true[i]
            correct_keys = torch.sum(y_pred_keys == label_key)
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
        return torch.tensor((accuracy / samples))
    
    def all_key_accuracy_local(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        #for i in range(len(y_true)):
        y_pred_keys = y_pred
        y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
        label_key = y_true
        correct_keys = torch.sum(y_pred_keys == label_key)
        score = score + correct_keys
        accuracy = accuracy + (1 if correct_keys == 12 else 0)
        samples = samples + 1
        return torch.tensor((accuracy / samples))
    
class TestCNN(pl.LightningModule):
     
    def __init__(self, pitches, pitch_classes, num_layers, kernel_size, opt=None, window_size=23, batch_size=4, train_set=None, val_set=None):
         super().__init__()
         self.pitches = pitches
         self.pitch_classes = pitch_classes
         self.num_layers = num_layers
         self.kernel_size = kernel_size
         self.batch_size = batch_size
         self.window_size = window_size
         self.opt = opt
         self.data = {'train': train_set,
                      'val': val_set}

         self.model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, kernel_size=(2,2), dilation=(36,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=(2,2), dilation=(36,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(2,2), dilation=(36,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(37,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(3,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            
            #nn.Conv2d(32, 32, kernel_size=(3,1)), ###### Addition start
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(32),
            #nn.Conv2d(32, 32, kernel_size=(3,1), stride=(4,1)),
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(32),
            #nn.Conv2d(32, 32, kernel_size=(3,1)),
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(32),
            #nn.Conv2d(32, 32, kernel_size=(3,1)),
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(32),
            #nn.Conv2d(32, 32, kernel_size=(3,1)),
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(32), ######### Addition end
            
            nn.Conv2d(32, 1, kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=(1,34)),
            nn.Flatten(),
            nn.Sigmoid(),
            ).double().cuda()
        
    def forward(self, mel):
        x = self.model(mel)
        #x = x.reshape(x.shape[0], x.shape[2]) # for global key estimation
        #x = nn.Sigmoid()(x)
        
        
        
        return x

    def general_step(self, batch, batch_idx, mode):
        
        mel = batch['mel']
        key_signature_id = batch['key_signature_id']
        key_labels = batch['key_labels'].double()
        if self.opt.local:
            seq_length = batch['seq_length']
        # forward pass
        out = self.forward(mel)
        
        # loss
        loss_func = nn.BCELoss()
        bce_loss = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                
                bce_loss = bce_loss + loss_func(out[i,:,:(seq_length[i]-self.window_size+1)], key_labels[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            bce_loss = loss_func(out, key_labels) # for global key estimation
            
        cosine_sim_func = nn.CosineSimilarity(dim=1)
        cosine_sim = cosine_sim_func(out, key_labels)
        loss = bce_loss + (1 - torch.sum(cosine_sim)/out.shape[0])
        accuracy = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                accuracy = accuracy + self.all_key_accuracy_local(key_labels[i,:,:(seq_length[i]-self.window_size+1)], out[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            accuracy = self.all_key_accuracy(key_labels, out)# for global key estimation
            
        return loss, accuracy

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        acc = torch.stack(
            [x[mode + '_accuracy'] for x in outputs]).mean()
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_accuracy': accuracy, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", acc)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy", acc, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.batch_size)
    
    def configure_optimizers(self):

        params = list(self.model.parameters())
            
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.opt.lr, weight_decay=self.opt.reg)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.opt.gamma)
        #scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=40, gamma=0.9)
        
        return [optim]#, [scheduler]
    
    def all_key_accuracy(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        for i in range(len(y_true)):
            y_pred_keys = y_pred[i]
            y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
            label_key = y_true[i]
            correct_keys = torch.sum(y_pred_keys == label_key)
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
        return torch.tensor((accuracy / samples))
    
    
class KeyEncoder(nn.Module):
    def __init__(self, nf, in_channels):
        super().__init__()
        self.in_channels = in_channels if in_channels>0 else 1
        self.preprocess1 = nn.Sequential(
            #nn.Conv2d(1, nf, kernel_size=(3, 3),padding=1, padding_mode='circular'),
            nn.Conv2d(self.in_channels, nf, kernel_size=(3, 3),padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(nf),
            nn.Conv2d(nf, nf, kernel_size=(3, 3),padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(nf),
        ).double().cuda()
        self.combineOctaves1 = CombineOctavesNet(nf).double().cuda()
        
        self.preprocess2 = nn.Sequential(
            nn.Conv2d(nf, 2*nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(2*nf),
            nn.Conv2d(2*nf, 2*nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(2*nf)
        ).double().cuda()
        self.combineOctaves2 = CombineOctavesNet(2 * nf).double().cuda()

        self.midiNoteBinNet = nn.Sequential(
            nn.Conv2d(2*nf, 4*nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(4*nf),
            nn.Conv2d(4*nf, 4*nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(4*nf),
            #nn.Conv2d(4*nf, 4*nf, kernel_size=(3, 3), stride=(3, 1)),
            nn.Conv2d(4*nf, 4*nf, kernel_size=(3, 3), stride=(3, 1), padding=(0,1),padding_mode='circular'),
            nn.ELU(),
            #nn.MaxPool2d(kernel_size=(1, 2)),
        ).double().cuda()
        
    def forward(self, x: Tensor) -> Tensor:
        
        x = self.preprocess1(x)

        z1 = self.combineOctaves1(x[:, :, :72, :])
        z2 = self.combineOctaves1(x[:, :, 72:144, :])
        z3 = self.combineOctaves1(x[:, :, 144:216, :])
        z4 = self.combineOctaves1(x[:, :, 216:, :])
        x = torch.cat([z1, z2, z3, z4], 2)

        x = self.preprocess2(x)

        z1 = self.combineOctaves2(x[:, :, :72, :])
        z2 = self.combineOctaves2(x[:, :, 72:144, :])
        x = torch.cat([z1, z2], 2)

        return self.midiNoteBinNet(x)
    
class CombineOctavesNet(nn.Module):
    def __init__(self, nf):
        super().__init__()
        octaves = 2

        self.model = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=(octaves, 1), dilation=(36, 1), bias=False),
            nn.ELU(),
            #nn.MaxPool2d(kernel_size=(1, 2)),
        ).double().cuda()
        shape = self.model[0].weight.shape
        self.model[0].weight = torch.nn.Parameter(torch.ones(shape).double().cuda())
        #print(self.p2pc[6].weight)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class KeySignatureEncoder(nn.Module):
    def __init__(self, nf, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(4*nf, 4 * nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(4*nf),
            nn.Conv2d(4*nf, 4 * nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(4*nf),
            nn.Conv2d(4*nf, 8 * nf, kernel_size=(13, 1)),
            nn.ELU(),
            nn.BatchNorm2d(8*nf),
            nn.Conv2d(8*nf, 1, 1,),
            #nn.Conv2d(8*nf, self.out_channels, 1,),
            #nn.LeakyReLU(),
            nn.ELU(),
            #layers.Lambda(lambda x: tf.reduce_mean(x, 2)),
            #nn.Flatten(),
            #nn.Sigmoid(),
        ).double().cuda()
        shape = self.model[6].weight.shape
        initializer = tf.keras.initializers.Orthogonal()
        weights = torch.nn.Parameter(torch.tensor(initializer(shape=shape).numpy()).double().cuda())
        self.model[6].weight = weights
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = torch.mean(x,3)
        x = nn.Flatten()(x)
        x = nn.Sigmoid()(x)
        return x
    
class TonicEncoder(nn.Module):
    def __init__(self, nf, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(4*nf, 4 * nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(4*nf),
            nn.Conv2d(4*nf, 4 * nf, kernel_size=(3, 3), padding=1, padding_mode='circular'),
            nn.ELU(),
            nn.BatchNorm2d(4*nf),
            nn.Conv2d(4*nf, 8 * nf, kernel_size=(13, 1)),
            nn.ELU(),
            nn.BatchNorm2d(8*nf),
            nn.Conv2d(8*nf, 1, 1,),
            #nn.Conv2d(8*nf, self.out_channels, 1,),
            #nn.LeakyReLU(),
            nn.ELU(),
            #layers.Lambda(lambda x: tf.reduce_mean(x, 2)),
            #nn.Flatten(),
            #nn.Sigmoid(),
        ).double().cuda()
        shape = self.model[6].weight.shape
        initializer = tf.keras.initializers.Orthogonal()
        weights = torch.nn.Parameter(torch.tensor(initializer(shape=shape).numpy()).double().cuda())
        self.model[6].weight = weights
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = torch.mean(x,3)
        x = nn.Flatten()(x)
        x = nn.Sigmoid()(x)
        return x
    
    
class EquivariantDilatedLayer(nn.Module):
    def __init__(self, nf, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nf = nf
        
        self.key_encoder = KeyEncoder(self.nf, self.in_channels)
        self.key_signature_encoder = KeySignatureEncoder(self.nf, self.out_channels)
        #self.tonic_encoder = TonicEncoder(activation, nf, d_rate)
        self.tonic_encoder = TonicEncoder(self.nf, self.out_channels)
        
    def forward(self, x: Tensor) -> Tensor:
        # The shared network part
        z = self.key_encoder(x)
        # The two network heads
        key_signature = self.key_signature_encoder(z)
        tonic = self.tonic_encoder(z)
        
        return key_signature, tonic
    
class EquivariantDilatedModel(pl.LightningModule):
    def __init__(self, nf, batch_size=None, opt=None, train_set=None,val_set=None):
        super().__init__()
        self.batch_size = batch_size
        self.opt = opt
        self.data = {'train': train_set,
                     'val': val_set}
        
        self.key_encoder = KeyEncoder(nf, 0)
        self.key_signature_encoder = KeySignatureEncoder(nf, 0)
        #self.tonic_encoder = TonicEncoder(activation, nf, d_rate)
        self.tonic_encoder = TonicEncoder(nf, 0)
        
    def forward(self, x: Tensor) -> Tensor:
        # The shared network part
        z = self.key_encoder(x)
        # The two network heads
        key_signature = self.key_signature_encoder(z)
        tonic = self.tonic_encoder(z)
        
        return key_signature, tonic
    
    def general_step(self, batch, batch_idx, mode):
        
        mel = batch['mel']
        key_signature_id = batch['key_signature_id']
        key_labels = batch['key_labels'].double()
        tonic_labels = batch['tonic_labels'].long()
        tonic_labels_idx = torch.argmax(tonic_labels,axis=1)
        
        if self.opt.local:
            seq_length = batch['seq_length']
        # forward pass
        out = self.forward(mel)
        key_out, tonic_out = out
        
        # loss
        loss_func = nn.BCELoss()
        loss_func_tonic = nn.CrossEntropyLoss()
        bce_loss = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                
                bce_loss = bce_loss + loss_func(key_out[i,:,:(seq_length[i]-self.window_size+1)], key_labels[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            bce_loss = loss_func(key_out, key_labels) # for global key estimation
            tonic_loss = loss_func_tonic(tonic_out.detach().cpu(), tonic_labels_idx.detach().cpu())
            
        cosine_sim_func = nn.CosineSimilarity(dim=1)
        cosine_sim = cosine_sim_func(key_out, key_labels)
        loss = bce_loss + tonic_loss + (1 - torch.sum(cosine_sim)/key_out.shape[0])
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                accuracy = accuracy + self.all_key_accuracy_local(key_labels[i,:,:(seq_length[i]-self.window_size+1)], key_out[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            #accuracy = self.all_key_accuracy(key_labels, key_out)# for global key estimation
            mirex, correct, fifths, relative, parallel, other, accuracy = self.mirex_score(key_labels, key_out, tonic_labels, tonic_out, key_signature_id)
            Accuracy_tonic = Accuracy().cuda()
            accuracy_tonic = Accuracy_tonic(torch.argmax(tonic_out,axis=1), tonic_labels_idx)
            
            
        return loss, accuracy, mirex, correct, fifths, relative, parallel, other, accuracy_tonic

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        acc = torch.stack(
            [x[mode + '_accuracy'] for x in outputs]).mean()
        mirex_score = torch.stack(
            [x[mode + '_mirex_score'] for x in outputs]).mean()
        correct = torch.stack(
            [x[mode + '_correct'] for x in outputs]).mean()
        fifths = torch.stack(
            [x[mode + '_fifths'] for x in outputs]).mean()
        relative = torch.stack(
            [x[mode + '_relative'] for x in outputs]).mean()
        parallel = torch.stack(
            [x[mode + '_parallel'] for x in outputs]).mean()
        other = torch.stack(
            [x[mode + '_other'] for x in outputs]).mean()
        acc_tonic = torch.stack(
            [x[mode + '_accuracy_tonic'] for x in outputs]).mean()
        return avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic

    def training_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_accuracy': accuracy, 'train_mirex_score': mirex_score, 'train_correct': correct, 'train_fifths': fifths, 'train_relative': relative, 'train_parallel': parallel, 'train_other': other, 'train_accuracy_tonic': acc_tonic, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_accuracy': accuracy, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic}

    def test_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy, 'test_mirex_score': mirex_score, 'train_correct': correct, 'test_fifths': fifths, 'test_relative': relative, 'test_parallel': parallel, 'test_other': other, 'test_accuracy_tonic': acc_tonic}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-Acc_Tonic={}".format(acc_tonic))
        print("Val-Mirex_Score={}".format(mirex_score))
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", acc)
        self.log("val_mirex_score", mirex_score)
        self.log("val_correct", correct)
        self.log("val_fifths", fifths)
        self.log("val_relative", relative)
        self.log("val_parallel", parallel)
        self.log("val_other", other)
        self.log("val_accuracy_tonic", acc_tonic)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy", acc, self.global_step)
        self.logger.experiment.add_scalar("val_mirex_score", mirex_score, self.global_step)
        self.logger.experiment.add_scalar("val_correct", correct, self.global_step)
        self.logger.experiment.add_scalar("val_fifths", fifths, self.global_step)
        self.logger.experiment.add_scalar("val_relative", relative, self.global_step)
        self.logger.experiment.add_scalar("val_parallel", parallel, self.global_step)
        self.logger.experiment.add_scalar("val_other", other, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy_tonic", acc_tonic, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.batch_size)
    
    def configure_optimizers(self):

        params = list(self.key_encoder.parameters()) + list(self.tonic_encoder.parameters()) + list(self.key_signature_encoder.parameters())# + list(self.classifier.parameters())
            
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.opt.lr, weight_decay=self.opt.reg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.opt.gamma)

        return [optim], [scheduler]
    
    def all_key_accuracy(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        for i in range(len(y_true)):
            y_pred_keys = y_pred[i]
            y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
            label_key = y_true[i]
            correct_keys = torch.sum(y_pred_keys == label_key)
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
        return torch.tensor((accuracy / samples))
    
    def all_key_accuracy_local(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        #for i in range(len(y_true)):
        y_pred_keys = y_pred
        y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
        label_key = y_true
        correct_keys = torch.sum(y_pred_keys == label_key)
        score = score + correct_keys
        accuracy = accuracy + (1 if correct_keys == 12 else 0)
        samples = samples + 1
        return torch.tensor((accuracy / samples))
    
    # not functional yet!!!
    def single_key_accuracy(y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        for i in range(len(y_true)):
            y_pred_keys = y_pred[i]
            y_pred_keys = tf.cast(y_pred_keys >= tf.sort(y_pred_keys)[-7], tf.float32)
            label_key = y_true[i]
            correct_keys = tf.math.reduce_sum(tf.cast(y_pred_keys == label_key[:12], tf.int32))
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
            #print("Label: "+str(label_key[:12]))
            #print("Pred: "+str(y_pred_keys))
        return tf.convert_to_tensor((score / (samples*12)))
    
    # needs to be tested!!!
    def genre_accuracy(y_true, y_pred):
        samples, score = 0, 0
        for i in range(len(y_true)):
            y_pred_genre = y_pred[i]
            y_true_genre = y_true[i]
            if torch.argmax(y_pred_genre)==torch.argmax(y_true_genre):
                score += 1
            samples += 1
        
        return torch.tensor((score/samples))

    def mirex_score(self, key_labels, key_preds, tonic_labels, tonic_preds, key_signature_id):
        score, accuracy, samples = 0, 0, 0
        similarity = 0
        correct_tonics_total = 0
        mirex_score, correct, fifths, parallel, relative, other = 0, 0, 0, 0, 0, 0
    
        for i in range(len(key_labels)):
            category = 0
            key_label = key_labels[i]
            key_pred_values = key_preds[i]
            tonic_label = tonic_labels[i]
            tonic_pred = tonic_preds[i]
            KEY_SIGNATURE_MAP = torch.tensor(key_sig.KEY_SIGNATURE_MAP.numpy()).cuda()
    
            # Every Key signature contains 7 pitch classes so use top 7 pitch class predictions
            #   Only major and minro are relevant for dataset
    
            # Find most similar cosine key signature given key predictions:
            cosinesim = nn.CosineSimilarity(dim=1)
            pred_key_id = torch.argmax(cosinesim(key_pred_values.reshape(1,key_pred_values.shape[0]), KEY_SIGNATURE_MAP))
            key_pred = KEY_SIGNATURE_MAP[pred_key_id]
            key_sig_label_id = torch.argmax(key_signature_id[i])
            
            # Check for relative fifths:
            cosinesim2 = nn.CosineSimilarity(dim=0)
            correct_keys = torch.sum(key_pred == key_label)
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
            similarity = similarity + cosinesim2(key_pred_values, key_label)
            diff = torch.abs(pred_key_id - key_sig_label_id)
            correct_tonic = 1 if torch.argmax(tonic_label) == torch.argmax(tonic_pred) else 0
            tonic_diff = torch.abs(torch.argmax(tonic_label) - torch.argmax(tonic_pred))
            correct_tonics_total = correct_tonics_total + correct_tonic
    
            if (diff == 1) and not(correct_tonic == 1 and correct_keys == 12):
                fifths = fifths + 1
                category = 1
            if correct_tonic == 1 and correct_keys == 12 and category == 0:
                correct = correct + 1
                category = 1
            if correct_keys == 12 and correct_tonic == 0 and category == 0:
                relative = relative + 1
                category = 1
            if correct_tonic == 1 and correct_keys != 12 and category == 0:
                parallel = parallel + 1
                category = 1
            if category == 0:
                other = other + 1
        mirex_score = (1 * correct) + (0.5 * fifths) + (0.3 * relative) + (0.2 * parallel)
        
        '''
        score = score / (samples * 12)
        accuracy = accuracy / samples
        print(f'dataset accuracy:')
        print(f'Correctly classified single keys: {score:.2%}')
        print(f'Correctly classified songs (all keys): {accuracy:.2%}')
        print(f'Avg. Cosine Similarity: {similarity / samples:.2%}')
        print(f'Fifths: {fifths / samples:.2%}')
        print(f'Correct tonics: {correct_tonics_total / samples:.2%}')
        print(f'All correct: {correct / samples:.2%}')
        print(f'Others: {others / samples:.2%}')
        print(f'MiReX Score: {mirex_score / samples:.2%}')
        '''
        
        return torch.tensor(mirex_score/samples).float(), torch.tensor(correct/samples).float(), torch.tensor(fifths/samples).float(), torch.tensor(relative/samples).float(), torch.tensor(parallel/samples).float(), torch.tensor(other/samples).float(), torch.tensor(accuracy/samples).float()

    
    
class JXC1(pl.LightningModule):
    
    def __init__(self, pitches, pitch_classes, num_layers, kernel_size, opt=None, window_size=23, batch_size=4, train_set=None, val_set=None):
        super().__init__()
        # set hyperparams
        #self.hparams.update(hparams)
        self.pitches = pitches
        self.pitch_classes = pitch_classes
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.opt = opt
        self.conv_layers = opt.conv_layers
        self.n_filters = opt.n_filters
        self.resblock = opt.resblock
        self.data = {'train': train_set,
                     'val': val_set}
        
        ########################################################################
        # Initialize Model:                                               #
        ########################################################################
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(1,3)),
            
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(1,3)),
            
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(1,4)),
            
            nn.Conv2d(in_channels=64,out_channels=80,kernel_size=3, padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=80,out_channels=80,kernel_size=3, padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            )
        
        self.bilstm = nn.LSTM(input_size=80*(opt.octaves*3*12-4), hidden_size=128, batch_first=True,bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*128,12),
            nn.Sigmoid(),
            )
    
    def forward(self, mel):
        
        x = self.feature_extractor(mel)
        x = x.reshape(x.shape[0], x.shape[3], x.shape[1]*x.shape[2])
        x, _ = self.bilstm(x)
        x = x[:,-1] # last hidden state
        x = self.classifier(x)
        
        
        return x

    def general_step(self, batch, batch_idx, mode):
        
        mel = batch['mel']
        key_signature_id = batch['key_signature_id']
        key_labels = batch['key_labels'].double()
        if self.opt.local:
            seq_length = batch['seq_length']
        # forward pass
        out = self.forward(mel)
        
        # loss
        loss_func = nn.BCELoss()
        bce_loss = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                
                bce_loss = bce_loss + loss_func(out[i,:,:(seq_length[i]-self.window_size+1)], key_labels[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            bce_loss = loss_func(out, key_labels) # for global key estimation
            
        cosine_sim_func = nn.CosineSimilarity(dim=1)
        cosine_sim = cosine_sim_func(out, key_labels)
        loss = bce_loss# + (1 - torch.sum(cosine_sim)/out.shape[0])
        accuracy = 0
        
        if self.opt.local: # for local key estimation
            for i in range(mel.shape[0]):
                accuracy = accuracy + self.all_key_accuracy_local(key_labels[i,:,:(seq_length[i]-self.window_size+1)], out[i,:,:(seq_length[i]-self.window_size+1)])
        else:
            accuracy = self.all_key_accuracy(key_labels, out)# for global key estimation
        
            
        return loss, accuracy

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        acc = torch.stack(
            [x[mode + '_accuracy'] for x in outputs]).mean()
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_accuracy': accuracy, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", acc)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy", acc, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.batch_size)
    
    def configure_optimizers(self):

        params = list(self.feature_extractor.parameters()) + list(self.bilstm.parameters()) + list(self.classifier.parameters())
            
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.opt.lr, weight_decay=self.opt.reg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.opt.gamma)

        return [optim], [scheduler]
    
    def all_key_accuracy(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        for i in range(len(y_true)):
            y_pred_keys = y_pred[i]
            y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
            label_key = y_true[i]
            correct_keys = torch.sum(y_pred_keys == label_key)
            score = score + correct_keys
            accuracy = accuracy + (1 if correct_keys == 12 else 0)
            samples = samples + 1
        return torch.tensor((accuracy / samples))
    
    def all_key_accuracy_local(self, y_true, y_pred):
        score, accuracy, samples = 0, 0, 0
        #for i in range(len(y_true)):
        y_pred_keys = y_pred
        y_pred_keys = y_pred_keys >= torch.sort(y_pred_keys).values[-7]
        label_key = y_true
        correct_keys = torch.sum(y_pred_keys == label_key)
        score = score + correct_keys
        accuracy = accuracy + (1 if correct_keys == 12 else 0)
        samples = samples + 1
        return torch.tensor((accuracy / samples))