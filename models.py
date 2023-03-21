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
import copy
import numpy as np
import utils.key_signatures as key_sig
from torchmetrics import Accuracy
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple
import torch.utils.checkpoint as cp
from torch.autograd import Variable


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
        # Adapted from https://www.dropbox.com/sh/k8be3v8nc8sz6p6/AAB8seucuLq1dI-26_7KA6P7a?dl=0
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
    # Adapted from https://www.dropbox.com/sh/k8be3v8nc8sz6p6/AAB8seucuLq1dI-26_7KA6P7a?dl=0
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
    
class Pitch2PitchClassConv(nn.Module):
    def __init__(self, pitch_classes: int, pitches_in: int, kernel_depth: int, padding_value, in_channels):
        super().__init__()
        self.pitch_classes = pitch_classes
        self.pitches_in = pitches_in
        self.kernel_depth = kernel_depth
        self.padding_value = padding_value
        self.kernel_size = math.ceil(self.pitches_in / self.pitch_classes)
        self.padding = self.kernel_size * self.pitch_classes - self.pitches_in # not really needed as padding is avoided
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(self.kernel_size, self.kernel_depth), dilation=(self.pitch_classes, 1))
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: tensor if dimension   (N, in_channels,  pitch_in, depth)
        :return:  tensor of dimension   (N, out_channels, pitch_classes, depth / kernel_depth)
        """
        assert len(x.shape) == 4
        
        shape = x.shape
        x_padded = torch.cat(
            [x, torch.full(size=(shape[0], shape[1], self.padding, shape[3]), fill_value=self.padding_value, device=x.device)], dim=2)
        return self.act(self.bn(self.conv(x_padded)))
    
class PitchClass2Pitch(nn.Module):
    def __init__(self, pitches: int):
        super().__init__()
        self.target_length = pitches

    def forward(self, x: Tensor):
        repetitions = math.ceil(self.target_length / x.shape[2])
        x_repeated = x.repeat(1, 1, repetitions, 1)
        return x_repeated[::, ::, 0:self.target_length, ::]
    
class PitchClass2Pitch_MemoryVariant(nn.Module):
    def __init__(self, pitches: int, pitch_classes: int):
        super().__init__()
        self.target_length = pitches
        self.pitch_classes = pitch_classes

    def forward(self, pitches: Tensor, pitch_classes: Tensor):
        
        # Sum over pitch_class wise feature channels to gain same channel number
        pitch_classes_prep = pitch_classes.reshape(pitch_classes.shape[0], pitches.shape[1], int(pitch_classes.shape[1]/pitches.shape[1]), pitch_classes.shape[2], pitch_classes.shape[3])
        pitch_classes = torch.sum(pitch_classes_prep,axis=2)
        # Adjust pitch and pitch class-wise feature shapes
        pitches_int = pitches.reshape(pitches.shape[0], pitches.shape[1], self.pitch_classes, int(pitches.shape[2]/self.pitch_classes), pitches.shape[3])
        pitch_classes_int = pitch_classes.reshape(pitch_classes.shape[0], pitch_classes.shape[1], pitch_classes.shape[2], 1, pitch_classes.shape[3])

        # Combine pitch and pitch class wise features via addition
        pitches_int = pitches_int + pitch_classes_int
        
        # Reshape pitch features to original form
        pitches_out = pitches_int.reshape(pitches.shape[0], pitches.shape[1], pitches.shape[2], pitches.shape[3])
        
        return pitches_out
    
class PitchClass2PitchClass(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, conv_layers: int, resblock: bool, denseblock: bool, multi_path: bool=False):
        super().__init__()
        self.pitch_classes = 12
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.resblock = resblock
        self.denseblock = denseblock
        self.multi_path = multi_path
        
        self.modules = []
        if self.resblock:
            for i in range(conv_layers):
                if i==0:
                    self.modules.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels=self.in_channels, out_channels=self.out_channels, kernel_depth=self.kernel_size, same_depth_padding=True))
                    self.modules.append(nn.BatchNorm2d(self.out_channels))
                    self.modules.append(nn.LeakyReLU())
                self.modules.append(ResBlockEquivariant(self.kernel_size, self.conv_layers, self.out_channels))
        elif self.denseblock:
            self.modules.append(DenseBlockEquivariant(num_layers=self.conv_layers, num_input_features=self.in_channels, bn_size=self.in_channels//2 if self.in_channels>1 else 1, growth_rate=self.out_channels, drop_rate=0.0, multi_path=self.multi_path, kernel_size=self.kernel_size))
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, conv_layers: int, resblock: bool, denseblock: bool, multi_path: bool=False):
        super().__init__()
        self.pitch_classes = 12
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.resblock = resblock
        self.denseblock = denseblock
        self.multi_path = multi_path
        
        self.modules = []
        if self.resblock:
            for i in range(conv_layers):
                if i==0:
                    self.modules.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2, padding_mode="circular"))
                    self.modules.append(nn.BatchNorm2d(self.out_channels))
                    self.modules.append(nn.LeakyReLU())
                self.modules.append(ResBlock(self.kernel_size, self.conv_layers, self.out_channels))
        elif self.denseblock:
            self.modules.append(DenseBlock(num_layers=self.conv_layers, num_input_features=self.in_channels, bn_size=self.in_channels//2 if self.in_channels>1 else 1, growth_rate=self.out_channels, drop_rate=0.0, multi_path=self.multi_path, kernel_size=self.kernel_size))
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
    def __init__(self, pitches, pitch_classes, num_layers, kernel_size, layer_num, window_size, conv_layers, num_filters, opt):
        super().__init__()
        self.pitch_classes = pitch_classes
        self.pitches = pitches
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        self.num_layers = num_layers
        self.window_size = window_size
        self.conv_layers = conv_layers
        self.num_filters = num_filters
        self.final_window_size = self.window_size-((self.kernel_size-1)*(self.num_layers-1)*(self.conv_layers))
        self.final = 16-self.kernel_size+1#(592//(2**(self.num_layers-1)))-1*(self.kernel_size-1)#self.final_window_size
        self.final_window_size = 1
        self.resblock = opt.resblock
        self.denseblock = opt.denseblock
        self.stay_sixth = opt.stay_sixth
        self.opt = opt
        self.dense_multi_path = False
        
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
            # previous filters in layer 1
            if self.layer_num==0:
                self.all_previous_channels_p = 0
                self.all_previous_channels_pc = 0
            # previous filters in layer 2
            elif self.layer_num==1:
                self.all_previous_channels_p = 1
                self.all_previous_channels_pc = self.num_filters
            # previous filters in layer 3
            elif self.layer_num==2:
                self.all_previous_channels_p = self.num_filters*2
                self.all_previous_channels_pc = 2*self.all_previous_channels_p
            # previous filters in layer 4 and higher
            else:
                self.all_previous_channels_p = (self.num_filters*2)*(4**(self.layer_num-2))
                self.all_previous_channels_pc = 2*self.all_previous_channels_p
            
            # filters in layer 1
            if self.layer_num==0:
                self.out_channels_p = 1
                self.out_channels_pc = 4
            # filters in layer 2
            elif self.layer_num==1:
                self.out_channels_p = 2*self.num_filters
                self.out_channels_pc = 2*self.out_channels_p
            # filters in layer 3 and higher
            else:
                self.out_channels_p = 4*self.all_previous_channels_p
                self.out_channels_pc = 4*self.all_previous_channels_pc

        # Layer Content Definition:
        if self.layer_num==0:
            if not self.opt.only_semitones:
                self.pool_semi = nn.Conv2d(1, 1, 3, stride=(3,1), padding=(0,1),padding_mode="circular")
                self.pool_semi_b = nn.BatchNorm2d(1)
                self.pool_semi_a = nn.LeakyReLU()
            if self.opt.p2pc_conv:
                self.pool = Pitch2PitchClassConv(self.pitch_classes, self.pitches//3, 1, float('-inf'), in_channels=1) 
            else:
                self.pool = Pitch2PitchClassPool(self.pitch_classes, self.pitches//3, 1, float('-inf'))
            self.pc2pc = PitchClass2PitchClass(in_channels = 1, out_channels = self.num_filters, kernel_size = self.kernel_size, conv_layers = self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
        else:
            if self.stay_sixth or (self.opt.only_semitones):
                self.up = PitchClass2Pitch(self.pitches//3)
            else:
                self.up_sixth = nn.ConvTranspose2d(in_channels=self.all_previous_channels_pc, out_channels=self.all_previous_channels_pc, kernel_size=(3,1), stride=(3,1))
                self.up_sixth_b = nn.BatchNorm2d(self.all_previous_channels_pc)
                self.up_sixth_a = nn.LeakyReLU()
                if self.opt.pc2p_mem:
                    self.up = PitchClass2Pitch_MemoryVariant(self.pitches, self.pitch_classes if self.opt.only_semitones else self.pitch_classes*3)
                else:
                    self.up = PitchClass2Pitch(self.pitches)
            if self.denseblock:
                self.p2p = Pitch2Pitch(in_channels=self.all_previous_channels_pc+self.all_previous_channels_p, out_channels=self.out_channels_dense, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock, multi_path=self.dense_multi_path)
            else:
                self.p2p = Pitch2Pitch(in_channels=self.all_previous_channels_p if self.opt.pc2p_mem else self.all_previous_channels_pc+self.all_previous_channels_p, out_channels=self.out_channels_p, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
            if not self.stay_sixth and (not self.opt.only_semitones):
                self.pool_semi = nn.Conv2d(in_channels=self.out_channels_p, out_channels=self.out_channels_p, kernel_size=(3,3), stride=(3,1), padding=(0,1), padding_mode="circular")
                self.pool_semi_b = nn.BatchNorm2d(self.out_channels_p)
                self.pool_semi_a = nn.LeakyReLU()
            if self.opt.p2pc_conv:
                self.pool = Pitch2PitchClassConv(self.pitch_classes, self.pitches//3, 1, float('-inf'), in_channels=self.out_channels_p)
            else:
                self.pool = Pitch2PitchClassPool(self.pitch_classes, self.pitches//3, 1, float('-inf'))
            if self.denseblock:
                self.pc2pc = PitchClass2PitchClass(in_channels=self.out_channels_p+self.all_previous_channels_pc, out_channels=self.out_channels_dense, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock, multi_path=self.dense_multi_path)
            else:
                self.pc2pc = PitchClass2PitchClass(in_channels=self.out_channels_p+self.all_previous_channels_pc, out_channels=self.out_channels_pc, kernel_size=self.kernel_size, conv_layers=self.conv_layers, resblock=self.resblock, denseblock=self.denseblock)
            if not self.opt.local:
                self.time_pool_p = nn.MaxPool2d(kernel_size=(1,self.opt.time_pool_size))
                self.time_pool_pc = nn.MaxPool2d(kernel_size=(1,self.opt.time_pool_size))

    def forward(self, x: Tensor) -> Tensor:
        
        (p, pc) = x
        if self.layer_num>0:
            assert len(pc.shape) == 4
        assert len(p.shape) == 4
        
        if self.layer_num==0:
            if not self.opt.only_semitones:
                p_semi = self.pool_semi(p) # Pool from a third of a semitone to semitone
                p_semi = self.pool_semi_b(p_semi)
                p_semi = self.pool_semi_a(p_semi)
            else:
                p_semi = p
            if self.stay_sixth:
                p = p_semi
            pc = self.pool(p_semi) # Pitch2PitchClass
            pc = self.pc2pc(pc) # Equivariant Layer
        else:
            if (not self.stay_sixth) and (not self.opt.only_semitones):
                p_sixth = self.up_sixth(pc)
                p_sixth = self.up_sixth_b(p_sixth)
                p_sixth = self.up_sixth_a(p_sixth)
                if self.opt.pc2p_mem:
                    p = self.up(p, p_sixth)
                else:
                    p2 = self.up(p_sixth) # PitchClass2Pitch
            else:
                if not self.opt.pc2p_mem:
                    p2 = self.up(pc)
            if not self.opt.pc2p_mem:
                p = torch.cat([p,p2],dim=1) # PitchConcat
            p = self.p2p(p) # Pitch2PitchConv
            if (not self.stay_sixth) and (not self.opt.only_semitones):
                pc2 = self.pool_semi(p) # Pool from Third of a semitone to semitone
                pc2 = self.pool_semi_b(pc2)
                pc2 = self.pool_semi_a(pc2)
                pc2 = self.pool(pc2) # Pitch2PitchClass
            else:
                pc2 = self.pool(p) # Pitch2PitchClass
            pc = torch.cat([pc, pc2],dim=1) # PitchClassConcat
            pc = self.pc2pc(pc) # Equivariant Layer
            if not self.opt.local:
                p = self.time_pool_p(p) # Time Pooling for pitch wise level
                pc = self.time_pool_pc(pc) # Time Pooling for pitchClass wise level
            
            
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
    # Code adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, kernel_size: int, memory_efficient: bool = True
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

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
    # Code adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, kernel_size: int, memory_efficient: bool = True
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = EquivariantPitchClassConvolutionSimple(pitch_classes=12, in_channels=num_input_features, out_channels=bn_size * growth_rate, kernel_depth=1)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = EquivariantPitchClassConvolutionSimple(pitch_classes=12, in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_depth=kernel_size, same_depth_padding=True)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

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
    # Code adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = True,
        # gradually increase kernel_size by 2 to imitate mulit_path idea
        multi_path: bool = True,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                kernel_size=(2*i+3) if multi_path else kernel_size,
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
    # Code adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = True,
        # gradually increase kernel_size by 2 to imitate mulit_path idea
        multi_path: bool = True,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayerEquivariant(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                kernel_size=(2*i+3) if multi_path else kernel_size,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class PitchClassNet(pl.LightningModule):
    
    def __init__(self, pitches, pitch_classes, num_layers, kernel_size, opt=None, window_size=23, batch_size=4, train_set=None, val_set=None):
        super().__init__()
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
        self.best_mirex_score = 0
        self.data = {'train': train_set,
                     'val': val_set}

        ########################################################################
        # Initialize Model:                                               #
        ########################################################################
        self.modules = []
        for i in range(self.num_layers):
            self.modules.append(PitchClassNetLayer(self.pitches, self.pitch_classes, self.num_layers, self.kernel_size, i, self.window_size, self.conv_layers, self.n_filters, self.opt))
        
        
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
        
        # Define Classifier heads:
        self.t_head_m = []
        self.k_head_m = []
        self.g_head_m = []
        for i in range(self.opt.head_layers):
            if i == (self.opt.head_layers-1):
                self.t_head_m.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels = self.final_channels, out_channels=1, kernel_depth=self.kernel_size))
                self.k_head_m.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels = self.final_channels, out_channels=1, kernel_depth=self.kernel_size))
                if self.opt.local:
                    self.t_head_m.append(nn.MaxPool2d(kernel_size=(1,(opt.frames*opt.loc_window_size)-self.opt.head_layers*(self.kernel_size-1)),stride=1,dilation=1))
                    self.k_head_m.append(nn.MaxPool2d(kernel_size=(1,(opt.frames*opt.loc_window_size)-self.opt.head_layers*(self.kernel_size-1)),stride=1,dilation=1))
                if self.opt.genre:
                    self.g_head_m.append(nn.Conv2d(in_channels = self.final_channels, out_channels = 1, kernel_size=(2, self.kernel_size)))
            else:
                self.t_head_m.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels = self.final_channels, out_channels=2*self.final_channels if i==0 else self.final_channels, kernel_depth=self.kernel_size))
                self.t_head_m.append(nn.BatchNorm2d(2*self.final_channels if i==0 else self.final_channels))
                self.t_head_m.append(nn.LeakyReLU())
                self.k_head_m.append(EquivariantPitchClassConvolutionSimple(pitch_classes=self.pitch_classes, in_channels = self.final_channels, out_channels=2*self.final_channels if i==0 else self.final_channels, kernel_depth=self.kernel_size))
                self.k_head_m.append(nn.BatchNorm2d(2*self.final_channels if i==0 else self.final_channels))
                self.k_head_m.append(nn.LeakyReLU())
                if self.opt.genre:
                    self.g_head_m.append(nn.Conv2d(in_channels = self.final_channels, out_channels = 2*self.final_channels if i==0 else self.final_channels, kernel_size = (1, self.kernel_size)))
                    self.g_head_m.append(nn.BatchNorm2d(2*self.final_channels if i==0 else self.final_channels))
                    self.g_head_m.append(nn.LeakyReLU())
                if i==0:
                    self.final_channels = 2*self.final_channels
                
        self.tonic_classifier = nn.Sequential(*self.t_head_m).double().cuda()
        self.key_classifier = nn.Sequential(*self.k_head_m).double().cuda()
        if self.opt.genre:
            self.genre_classifier = nn.Sequential(*self.g_head_m).double().cuda()
        
        self.sig = nn.Sigmoid()

    
    def forward(self, mel, seq_length):

        (p, pc) = self.model((mel,None))
        tonic = self.tonic_classifier(pc)
        key = self.key_classifier(pc)
        if self.opt.genre:
            genre = self.genre_classifier(pc)
        if seq_length!=None and not self.opt.local:
            # calculates actual seq_length of model output
            # takes model operations into account
            actual_seq_length = seq_length
            for i in range(self.num_layers-1):
                actual_seq_length = torch.floor(actual_seq_length/self.opt.time_pool_size)
            actual_seq_length = actual_seq_length.int()-(self.kernel_size-1)*self.opt.head_layers
            # computes mean of final model ouput given only the actual seq length
            # needs to be done seperately for each element in the batch

            for j in range(tonic.shape[0]):
                if j==0:
                    if self.opt.max_pool:
                        tonic_out = torch.max(tonic[j,:,:,:actual_seq_length[j]],axis=-1).values
                        key_out = torch.max(key[j,:,:,:actual_seq_length[j]],axis=-1).values
                    else:
                        tonic_out = torch.mean(tonic[j,:,:,:actual_seq_length[j]],axis=-1)
                        key_out = torch.mean(key[j,:,:,:actual_seq_length[j]],axis=-1)
                    tonic_out = tonic_out.reshape(1, tonic_out.shape[0],tonic_out.shape[1])
                    key_out = key_out.reshape(1, key_out.shape[0],key_out.shape[1])
                    if self.opt.genre:
                        if self.opt.max_pool:
                            genre_out = torch.max(genre[j,:,:,:actual_seq_length[j]],axis=-1).values
                        else:
                            genre_out = torch.mean(genre[j,:,:,:actual_seq_length[j]],axis=-1)
                        
                        genre_out = genre_out.reshape(1, genre_out.shape[0],genre_out.shape[1])
                else:
                    tonic_out = torch.cat((tonic_out, torch.mean(tonic[j,:,:,:actual_seq_length[j]],axis=-1).reshape(1, tonic_out.shape[1],tonic_out.shape[2])))
                    key_out = torch.cat((key_out, torch.mean(key[j,:,:,:actual_seq_length[j]],axis=-1).reshape(1, key_out.shape[1],key_out.shape[2])))
                    if self.opt.genre:
                        genre_out = torch.cat((genre_out, torch.mean(genre[j,:,:,:actual_seq_length[j]],axis=-1).reshape(1, genre_out.shape[1],genre_out.shape[2])))
        elif (not self.opt.local):
            if self.opt.max_pool:
                tonic_out = torch.max(tonic, axis=-1).values
                key_out = torch.max(key, axis=-1).values
            else:   
                tonic_out = torch.mean(tonic, axis=-1)
                key_out = torch.mean(key, axis=-1)
            if self.opt.genre:
                if self.opt.max_pool:
                    genre_out = torch.max(genre, axis=-1).values
                else:
                    genre_out = torch.mean(genre, axis=-1)

        if not self.opt.local:
            tonic_out = nn.Flatten()(tonic_out)
            key_out = nn.Flatten()(key_out)
            key_out = self.sig(key_out)
            if self.opt.genre:
                genre_out = nn.Flatten()(genre_out)
        else:
            tonic_out = tonic.reshape(tonic.shape[0],tonic.shape[3],tonic.shape[2])
            key_out = key.reshape(key.shape[0],key.shape[3],key.shape[2])
            key_out = self.sig(key_out)
            if self.opt.genre:
                genre_out = genre.reshape(genre.shape[0],genre.shape[3],genre.shape[2])

        if self.opt.genre:
            x = (key_out, tonic_out, genre_out)
        else:
            x = (key_out, tonic_out)

        return x

    def general_step(self, batch, batch_idx, mode):
        
        mel = batch['mel']
        key_signature_id = batch['key_signature_id']
        key_labels = batch['key_labels'].double()
        tonic_labels = batch['tonic_labels'].long()
        if self.opt.genre:
            genre_labels = batch['genre'].long()
        if self.opt.local:
            tonic_labels_idx = torch.argmax(tonic_labels,axis=2)
            if self.opt.genre:
                genre_labels_idx = torch.argmax(genre_labels,axis=2)
        else:
            tonic_labels_idx = torch.argmax(tonic_labels, axis=1)
            if self.opt.genre:
                genre_labels_idx = torch.argmax(genre_labels, axis=1)

        if self.opt.genre:
            # create mask for genre loss
            # shall ignore missing genre labels
            genre_mask = torch.sum(genre_labels, axis=1)==1
            genre_mask = genre_mask.cuda()

        if self.opt.local or self.opt.frames>0:
            seq_length = batch['seq_length']
        # forward pass
        if self.opt.frames>0:
            out = self.forward(mel,seq_length)
        else:
            out = self.forward(mel, None)
        if self.opt.genre:
            key_out, tonic_out, genre_out = out
        else:
            key_out, tonic_out = out
        
        # loss
        loss_func = nn.BCELoss()
        loss_func_tonic = nn.CrossEntropyLoss()
        loss_func_genre = nn.CrossEntropyLoss()
        bce_loss = 0
        tonic_loss = 0
        genre_loss = 0
        if self.opt.local: # for local key estimation
            if self.opt.genre:
                genre_out = genre_out.reshape(genre_out.shape[0], genre_out.shape[2], genre_out.shape[1])
            for i in range(mel.shape[0]):
                bce_loss = bce_loss + loss_func(key_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:], key_labels[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:])
                tonic_loss = tonic_loss + loss_func_tonic(tonic_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:], tonic_labels_idx[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1)])
                if self.opt.genre:
                    genre_out = genre_out[genre_mask]
                    genre_labels_idx = genre_labels_idx[genre_mask]
                    if torch.sum(genre_mask)==0:
                        genre_loss = torch.zeros(1)
                    else:
                        genre_loss = genre_loss + loss_func_genre(genre_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:], genre_labels_idx[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1)])
                        genre_loss = genre_loss/mel.shape[0]
            bce_loss = bce_loss/mel.shape[0]
            tonic_loss = tonic_loss/mel.shape[0]
        else:
            bce_loss = loss_func(key_out, key_labels) # for global key estimation
            tonic_loss = loss_func_tonic(tonic_out, tonic_labels_idx)
            if self.opt.genre:
                genre_out = genre_out[genre_mask]
                genre_labels_idx = genre_labels_idx[genre_mask]
                genre_loss = loss_func_genre(genre_out, genre_labels_idx)

        if self.opt.use_cos:
            cosine_sim_func = nn.CosineSimilarity(dim=1)
            cosine_sim = cosine_sim_func(key_out, key_labels)
            
        loss = self.opt.key_weight*bce_loss + self.opt.tonic_weight*tonic_loss
        
        if self.opt.genre:
            if torch.sum(genre_mask)!=0:
                loss += self.opt.genre_weight*genre_loss
    
        if self.opt.use_cos:
            loss += (1 - torch.sum(cosine_sim)/key_out.shape[0])
        
        if self.opt.local: # for local key estimation
            mirex, correct, fifths, relative, parallel, other, accuracy, accuracy_tonic, accuracy_genre = 0,0,0,0,0,0,0,0,0
            for i in range(mel.shape[0]):
                mirex_sub, correct_sub, fifths_sub, relative_sub, parallel_sub, other_sub, accuracy_sub = self.mirex_score(key_labels[i,:(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], key_out[i,:(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], tonic_labels[i, :(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], tonic_out[i, :(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], key_signature_id[i, :(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:])
                mirex, correct, fifths, relative, parallel, other, accuracy = mirex+mirex_sub, correct+correct_sub, fifths+fifths_sub, relative+relative_sub, parallel+parallel_sub, other+other_sub, accuracy+accuracy_sub
                Accuracy_tonic = Accuracy().cuda()
                accuracy_tonic = accuracy_tonic + Accuracy_tonic(torch.argmax(tonic_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1)),:],axis=1), tonic_labels_idx[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1))]) 
                if self.opt.genre:
                    if torch.sum(genre_mask)==0:
                        accuracy_genre = torch.tensor(0.0).float()
                    else:
                        Accuracy_genre = Accuracy().cuda()
                        accuracy_genre = accuracy_genre + Accuracy_genre(torch.argmax(genre_out[i,:,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1))],axis=1), genre_labels_idx[i,:,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1))])
            mirex, correct, fifths, relative, parallel, other, accuracy, accuracy_tonic, accuracy_genre = torch.tensor(mirex/mel.shape[0]).clone().float(), torch.tensor(correct/mel.shape[0]).clone().float(), torch.tensor(fifths/mel.shape[0]).clone().float(), torch.tensor(relative/mel.shape[0]).clone().float(), torch.tensor(parallel/mel.shape[0]).clone().float(), torch.tensor(other/mel.shape[0]).clone().float(), torch.tensor(accuracy/mel.shape[0]).clone().float(), torch.tensor(accuracy_tonic/mel.shape[0]).clone().float(), torch.tensor(accuracy_genre/mel.shape[0]).clone().float()
        else:
            mirex, correct, fifths, relative, parallel, other, accuracy = self.mirex_score(key_labels, key_out, tonic_labels, tonic_out, key_signature_id)
            Accuracy_tonic = Accuracy().cuda()
            accuracy_tonic = Accuracy_tonic(torch.argmax(tonic_out,axis=1), tonic_labels_idx)
            if self.opt.genre:
                # shall prevent error for datasets with missing genre labels
                # simply ignore the genre accuracy for those
                if torch.sum(genre_mask)==0:
                    accuracy_genre = torch.tensor(0.0).float()
                else:
                    Accuracy_genre = Accuracy().cuda()
                    accuracy_genre = Accuracy_genre(torch.argmax(genre_out,axis=1), genre_labels_idx)
            else:
                accuracy_genre = torch.tensor(0.0).float()

        return loss, accuracy, mirex, correct, fifths, relative, parallel, other, accuracy_tonic, accuracy_genre

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
        acc_genre = torch.stack(
            [x[mode + '_accuracy_genre'] for x in outputs]).mean()
        return avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre

    def training_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_accuracy': accuracy, 'train_mirex_score': mirex_score, 'train_correct': correct, 'train_fifths': fifths, 'train_relative': relative, 'train_parallel': parallel, 'train_other': other, 'train_accuracy_tonic': acc_tonic, 'train_accuracy_genre': acc_genre, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_accuracy': accuracy, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'val_accuracy_genre': acc_genre}

    def test_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy, 'test_mirex_score': mirex_score, 'train_correct': correct, 'test_fifths': fifths, 'test_relative': relative, 'test_parallel': parallel, 'test_other': other, 'test_accuracy_tonic': acc_tonic, 'test_accuracy_genre': acc_genre}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-Acc_Tonic={}".format(acc_tonic))
        if self.opt.genre:
            print("Val-Acc_Genre={}".format(acc_genre))
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
        self.log("val_accuracy_genre", acc_genre)
        if mirex_score > self.best_mirex_score and not self.opt.no_ckpt:
            self.best_mirex_score = mirex_score
            torch.save(self.state_dict(), 'Model_logs/lightning_logs/version_'+str(self.logger.version)+'/best_model.pt')
        if not self.opt.no_ckpt:
            self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
            self.logger.experiment.add_scalar("val_accuracy", acc, self.global_step)
            self.logger.experiment.add_scalar("val_mirex_score", mirex_score, self.global_step)
            self.logger.experiment.add_scalar("val_correct", correct, self.global_step)
            self.logger.experiment.add_scalar("val_fifths", fifths, self.global_step)
            self.logger.experiment.add_scalar("val_relative", relative, self.global_step)
            self.logger.experiment.add_scalar("val_parallel", parallel, self.global_step)
            self.logger.experiment.add_scalar("val_other", other, self.global_step)
            self.logger.experiment.add_scalar("val_accuracy_tonic", acc_tonic, self.global_step)
            self.logger.experiment.add_scalar("val_accuracy_genre", acc_genre, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'val_accuracy_genre': acc_genre}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'val_accuracy_genre': acc_genre, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.batch_size)
    
    def configure_optimizers(self):

        if self.opt.genre:
            params = list(self.model.parameters()) + list(self.tonic_classifier.parameters()) + list(self.key_classifier.parameters()) + list(self.genre_classifier.parameters())
        else:
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
        return tf.convert_to_tensor((score / (samples*12)))

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
            #   Only major and minor are relevant for the dataset
    
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

        return torch.tensor(mirex_score/samples).float(), torch.tensor(correct/samples).float(), torch.tensor(fifths/samples).float(), torch.tensor(relative/samples).float(), torch.tensor(parallel/samples).float(), torch.tensor(other/samples).float(), torch.tensor(accuracy/samples).float()

class PitchClassNet_Multi(pl.LightningModule):
    
    def __init__(self, pitches1, pitches2, pitch_classes, num_layers, kernel_size, opt=None, window_size=23, batch_size=4, train_set=None, val_set=None):
        super().__init__()
        self.pitches1 = pitches1
        self.pitches2 = pitches2
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
        # Two models for two different scales:
            # First scale only on tone level input
            # Second inputs is encoded on semitone level
        self.model1 = PitchClassNet(self.pitches1, self.pitch_classes, self.num_layers, self.kernel_size, opt=self.opt, window_size=window_size, batch_size=batch_size, train_set=train_set, val_set=val_set).double().cuda()
        opt2 = copy.deepcopy(opt)
        opt2.no_semitones = True
        self.model2 = PitchClassNet(self.pitches2, self.pitch_classes, self.num_layers, self.kernel_size, opt=opt2, window_size=window_size, batch_size=batch_size, train_set=train_set, val_set=val_set).double().cuda()
        
        if self.opt.linear_reg_multi:
            self.wk = torch.randn(2, 12, requires_grad=True, device="cuda")
            self.wt = torch.randn(2, 12, requires_grad=True, device="cuda")
            self.bk = torch.randn(12, requires_grad=True, device="cuda")
            self.bt = torch.randn(12, requires_grad=True, device="cuda")
            if self.opt.genre:
                self.wg = torch.randn(2, 12, requires_grad=True, device="cuda")
                self.bg = torch.randn(12, requires_grad=True, device="cuda")
            self.sig = nn.Sigmoid()
    
    def forward(self, mel1, mel2, seq_length):
        
        x1 = self.model1.forward(mel1, seq_length)
        x2 = self.model2.forward(mel2, seq_length)
        
        if self.opt.genre:
            key_out1, tonic_out1, genre_out1 = x1
            key_out2, tonic_out2, genre_out2 = x2
        else:
            key_out1, tonic_out1 = x1
            key_out2, tonic_out2 = x2
        
        # Combine x1 and x2 to x!!!!
        # Linear Regression version:
        if self.opt.linear_reg_multi:
            key_out = self.wk[0]*key_out1 + self.wk[1]*key_out2 + self.bk
            tonic_out = self.wt[0]*tonic_out1 + self.wt[1]*tonic_out2 + self.bt
            key_out = self.sig(key_out)
            if self.opt.genre:
                genre_out = self.wg[0]*genre_out1 + self.wg[1]*genre_out2 + self.bg
        else: # Simple averaging variant
            key_out = (key_out1+key_out2)/2
            tonic_out = (tonic_out1+tonic_out2)/2
            if self.opt.genre:
                genre_out = (genre_out1+genre_out2)/2
        
        if self.opt.genre:
            x = (key_out, tonic_out, genre_out)
        else:
            x = (key_out, tonic_out)

        return x

    def general_step(self, batch, batch_idx, mode):
        
        mel1 = batch['mel']
        mel2 = batch['mel2']
        key_signature_id = batch['key_signature_id']
        key_labels = batch['key_labels'].double()
        tonic_labels = batch['tonic_labels'].long()
        genre_labels = batch['genre'].long()
        tonic_labels_idx = torch.argmax(tonic_labels,axis=1)
        genre_labels_idx = torch.argmax(genre_labels,axis=1)

        if self.opt.genre:
            # create mask for genre loss
            # shall ignore missing genre labels
            genre_mask = torch.sum(genre_labels, axis=1)==1
        
        if self.opt.local or self.opt.frames>0:
            seq_length = batch['seq_length']
        # forward pass
        if self.opt.frames>0:
            out = self.forward(mel1, mel2, seq_length)
        else:
            out = self.forward(mel1, mel2, None)
        if self.opt.genre:
            key_out, tonic_out, genre_out = out
        else:
            key_out, tonic_out = out
        
        # loss
        loss_func = nn.BCELoss()
        loss_func_tonic = nn.CrossEntropyLoss()
        loss_func_genre = nn.CrossEntropyLoss()
        bce_loss = 0
        tonic_loss = 0
        genre_loss = 0
        
        if self.opt.local: # for local key estimation
            if self.opt.genre:
                genre_out = genre_out.reshape(genre_out.shape[0], genre_out.shape[2], genre_out.shape[1])
            for i in range(mel.shape[0]):
                bce_loss = bce_loss + loss_func(key_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:], key_labels[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:])
                tonic_loss = tonic_loss + loss_func_tonic(tonic_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:], tonic_labels_idx[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1)])
                if self.opt.genre:
                    genre_out = genre_out[genre_mask]
                    genre_labels_idx = genre_labels_idx[genre_mask]
                    if torch.sum(genre_mask)==0:
                        genre_loss = torch.zeros(1)
                    else:
                        genre_loss = genre_loss + loss_func_genre(genre_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1),:], genre_labels_idx[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames)+1)])
                        genre_loss = genre_loss/mel.shape[0]
            bce_loss = bce_loss/mel.shape[0]
            tonic_loss = tonic_loss/mel.shape[0]
        else:
            bce_loss = loss_func(key_out, key_labels) # for global key estimation
            tonic_loss = loss_func_tonic(tonic_out, tonic_labels_idx)
            if self.opt.genre:
                genre_out = genre_out[genre_mask]
                genre_labels_idx = genre_labels_idx[genre_mask]
                genre_loss = loss_func_genre(genre_out, genre_labels_idx)

        if self.opt.use_cos:
            cosine_sim_func = nn.CosineSimilarity(dim=1)
            cosine_sim = cosine_sim_func(key_out, key_labels)
            
        loss = self.opt.key_weight*bce_loss + self.opt.tonic_weight*tonic_loss
        
        if self.opt.genre:
            if torch.sum(genre_mask)!=0:
                loss += self.opt.genre_weight*genre_loss
    
        if self.opt.use_cos:
            loss += (1 - torch.sum(cosine_sim)/key_out.shape[0])
        
        if self.opt.local: # for local key estimation
            mirex, correct, fifths, relative, parallel, other, accuracy, accuracy_tonic, accuracy_genre = 0,0,0,0,0,0,0,0,0
            for i in range(mel.shape[0]):
                mirex_sub, correct_sub, fifths_sub, relative_sub, parallel_sub, other_sub, accuracy_sub = self.mirex_score(key_labels[i,:(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], key_out[i,:(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], tonic_labels[i, :(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], tonic_out[i, :(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:], key_signature_id[i, :(seq_length[i]-self.opt.loc_window_size*self.opt.frames+1),:])
                mirex, correct, fifths, relative, parallel, other, accuracy = mirex+mirex_sub, correct+correct_sub, fifths+fifths_sub, relative+relative_sub, parallel+parallel_sub, other+other_sub, accuracy+accuracy_sub
                Accuracy_tonic = Accuracy().cuda()
                accuracy_tonic = accuracy_tonic + Accuracy_tonic(torch.argmax(tonic_out[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1)),:],axis=1), tonic_labels_idx[i,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1))]) 
                if self.opt.genre:
                    if torch.sum(genre_mask)==0:
                        accuracy_genre = torch.tensor(0.0).float()
                    else:
                        Accuracy_genre = Accuracy().cuda()
                        accuracy_genre = accuracy_genre + Accuracy_genre(torch.argmax(genre_out[i,:,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1))],axis=1), genre_labels_idx[i,:,:(seq_length[i]-(self.opt.loc_window_size*self.opt.frames+1))])
            mirex, correct, fifths, relative, parallel, other, accuracy, accuracy_tonic, accuracy_genre = torch.tensor(mirex/mel.shape[0]).clone().float(), torch.tensor(correct/mel.shape[0]).clone().float(), torch.tensor(fifths/mel.shape[0]).clone().float(), torch.tensor(relative/mel.shape[0]).clone().float(), torch.tensor(parallel/mel.shape[0]).clone().float(), torch.tensor(other/mel.shape[0]).clone().float(), torch.tensor(accuracy/mel.shape[0]).clone().float(), torch.tensor(accuracy_tonic/mel.shape[0]).clone().float(), torch.tensor(accuracy_genre/mel.shape[0]).clone().float()
        else:
            mirex, correct, fifths, relative, parallel, other, accuracy = self.mirex_score(key_labels, key_out, tonic_labels, tonic_out, key_signature_id)
            Accuracy_tonic = Accuracy().cuda()
            accuracy_tonic = Accuracy_tonic(torch.argmax(tonic_out,axis=1), tonic_labels_idx)
            if self.opt.genre:
                # shall prevent error for datasets with missing genre labels
                # simply ignore the genre accuracy for those
                if torch.sum(genre_mask)==0:
                    accuracy_genre = torch.tensor(0.0).float()
                else:
                    Accuracy_genre = Accuracy().cuda()
                    accuracy_genre = Accuracy_genre(torch.argmax(genre_out,axis=1), genre_labels_idx)
            else:
                accuracy_genre = torch.tensor(0.0).float()
            
        return loss, accuracy, mirex, correct, fifths, relative, parallel, other, accuracy_tonic, accuracy_genre

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
        acc_genre = torch.stack(
            [x[mode + '_accuracy_genre'] for x in outputs]).mean()
        return avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre

    def training_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_accuracy': accuracy, 'train_mirex_score': mirex_score, 'train_correct': correct, 'train_fifths': fifths, 'train_relative': relative, 'train_parallel': parallel, 'train_other': other, 'train_accuracy_tonic': acc_tonic, 'train_accuracy_genre': acc_genre, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_accuracy': accuracy, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'val_accuracy_genre': acc_genre}

    def test_step(self, batch, batch_idx):
        loss, accuracy, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy, 'test_mirex_score': mirex_score, 'train_correct': correct, 'test_fifths': fifths, 'test_relative': relative, 'test_parallel': parallel, 'test_other': other, 'test_accuracy_tonic': acc_tonic, 'test_accuracy_genre': acc_genre}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, mirex_score, correct, fifths, relative, parallel, other, acc_tonic, acc_genre = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-Acc_Tonic={}".format(acc_tonic))
        if self.opt.genre:
            print("Val-Acc_Genre={}".format(acc_genre))
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
        self.log("val_accuracy_genre", acc_genre)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy", acc, self.global_step)
        self.logger.experiment.add_scalar("val_mirex_score", mirex_score, self.global_step)
        self.logger.experiment.add_scalar("val_correct", correct, self.global_step)
        self.logger.experiment.add_scalar("val_fifths", fifths, self.global_step)
        self.logger.experiment.add_scalar("val_relative", relative, self.global_step)
        self.logger.experiment.add_scalar("val_parallel", parallel, self.global_step)
        self.logger.experiment.add_scalar("val_other", other, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy_tonic", acc_tonic, self.global_step)
        self.logger.experiment.add_scalar("val_accuracy_genre", acc_genre, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'val_accuracy_genre': acc_genre}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_mirex_score': mirex_score, 'val_correct': correct, 'val_fifths': fifths, 'val_relative': relative, 'val_parallel': parallel, 'val_other': other, 'val_accuracy_tonic': acc_tonic, 'val_accuracy_genre': acc_genre, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.batch_size)
    
    def configure_optimizers(self):

        params = list(self.model1.parameters()) + list(self.model2.parameters())
        if self.opt.linear_reg_multi:
            params  = params + [self.wk, self.bk, self.wt, self.bt] + list(self.sig.parameters())
            if self.opt.genre:
                params = params + [self.wg, self.bg]
        
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

        return torch.tensor(mirex_score/samples).float(), torch.tensor(correct/samples).float(), torch.tensor(fifths/samples).float(), torch.tensor(relative/samples).float(), torch.tensor(parallel/samples).float(), torch.tensor(other/samples).float(), torch.tensor(accuracy/samples).float()