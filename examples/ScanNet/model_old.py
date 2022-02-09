import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

import sys, os, time
import logging
import json
import pdb

class ResidualBlock(nn.Module):
    def __init__(self, dimension, a,b,leakiness=0):
        nn.Module.__init__(self)
        self.dimension = dimension
        self.a = a
        self.b = b
        self.BN1 = scn.BatchNormLeakyReLU(a,leakiness=leakiness)
        self.SCN1 = scn.SubmanifoldConvolution(dimension, a, b, 3, False)
        self.BN2 = scn.BatchNormLeakyReLU(b,leakiness=leakiness)
        self.SCN2 = scn.SubmanifoldConvolution(dimension, b, b, 3, False)
        if a==b:
            self.direct = scn.Identity()
        else:
            self.direct = scn.NetworkInNetwork(a, b, False)
    def forward(self,x):

        y1 = self.direct(x)
        y2 = self.SCN2(self.BN2(self.SCN1(self.BN1(x))))
        return scn.add_feature_planes([y1,y2])

class SeperableResidualBlock(nn.Module):
    def __init__(self, dimension, a,b,leakiness=0):
        nn.Module.__init__(self)
        self.dimension = dimension
        # assumption here: a or b can be divided by 16
        self.a = a
        self.b = b
        self.input_element_num = int(self.a / 16)
        self.output_element_num = int(self.b / 16)
        self.linearInput = nn.ModuleList()
        self.bn = scn.BatchNormLeakyReLU(a,leakiness=leakiness)
        self.bn1 =  nn.ModuleList()
        self.scn1 =  nn.ModuleList()
        self.bn2 =  nn.ModuleList()
        self.scn2 =  nn.ModuleList()
        self.linearOutput = nn.ModuleList()
        for i in range(self.input_element_num):
            self.linearInput.append(LinearSCN(a,16))
            self.bn1.append(scn.BatchNormLeakyReLU(16,leakiness=leakiness))
            self.scn1.append(scn.SubmanifoldConvolution(dimension, 16, 16, 3, False))
            self.bn2.append(scn.BatchNormLeakyReLU(16,leakiness=leakiness))
            self.scn2.append(scn.SubmanifoldConvolution(dimension, 16, 16, 3, False))
            self.linearOutput.append(LinearSCN(16,b))
        if a==b:
            self.direct = scn.Identity()
        else:
            self.direct = scn.NetworkInNetwork(a, b, False)
    def forward(self,x):
        y1 = self.direct(x)
        x = self.bn(x)
        for i in range(self.input_element_num):
            y1 = scn.add_feature_planes([y1, self.linearOutput[i](self.scn2[i](self.bn2[i](self.scn1[i](self.bn1[i](self.linearInput[i](x))))))])
        return y1


class RepResidualBlock(nn.Module):
    def __init__(self,reps,dimension, a,b,leakiness=0):
        nn.Module.__init__(self)
        assert(reps > 0)
        self.reps = reps
        self.res = nn.ModuleList()
        self.res.append(ResidualBlock(dimension, a,b,leakiness=leakiness))
        for _ in range(reps-1):
            self.res.append(ResidualBlock(dimension, b,b,leakiness=leakiness))
    def forward(self, x):
        for i in range(self.reps ):
            x = self.res[i](x)
        return x


class LinearSCN(nn.Module):
    def __init__(self,a,b):
        nn.Module.__init__(self)
        self.linear = nn.Linear(a, b)
    def forward(self, input):
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = self.linear(input.features)
        return output


class SpatialDropOut(nn.Module):
    def __init__(self,p):
        nn.Module.__init__(self)
        self.p = p
    def forward(self, input):

        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        feature_channels = input.features.shape[-1]
        dropout = torch.bernoulli(torch.ones(feature_channels) * (1 - self.p)).view(1,feature_channels).cuda().repeat(input.features.shape[0],1)
        if self.training:
            output.features = input.features * dropout
        else:
            output.features = input.features * (1 - self.p)
        return output


class CrossScaleFusion(nn.Module):
    def __init__(self,a,b,leakiness=0):
        nn.Module.__init__(self)
#        self.bn = scn.BatchNormLeakyReLU(a,leakiness)
        self.linear = nn.Linear(a, b)
        self.reweight = nn.Linear(a, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.output_width = b
    def forward(self, input):
#        input = self.bn(input)
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features  = self.linear((input).features)  * self.sigmoid(self.reweight(input.features)).repeat(1,self.output_width)
        return output


#we borrow the idea of resnet and
class DenseUNet_BN(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        #seven layer convolution is here!
        self.leakiness = 0
        self.downsample=[2, 2]
        self.nPlanes = config['unet_structure']
        self.reps = config['block_reps']
        self.dimension = 3

        self.res = nn.ModuleList()
        self.bn0 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.deconv = nn.ModuleList()
        self.res2 = nn.ModuleList()

        self.linear = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.linear3 = nn.ModuleList()
        self.linear4 = nn.ModuleList()
        self.linear5 = nn.ModuleList()

        self.linearReweight = nn.ModuleList()
        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        for idx, nPlane in enumerate(nPlanes):
            if(idx < len(nPlanes) - 1):
                self.res.append(RepResidualBlock(reps,dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))
                self.bn0.append(scn.BatchNormLeakyReLU(nPlanes[idx],leakiness=leakiness))
                self.conv.append(scn.Convolution(dimension, nPlanes[idx], nPlanes[idx+1], downsample[0], downsample[1], False))
                self.bn1.append(scn.BatchNormLeakyReLU(nPlanes[idx+1],leakiness=leakiness))
                self.bn2.append(scn.BatchNormLeakyReLU(nPlanes[idx],leakiness=leakiness))
                self.deconv.append(scn.Deconvolution(dimension, nPlanes[idx+1], nPlanes[idx], downsample[0], downsample[1], False))
                self.linearReweight.append(LinearSCN(nPlanes[idx],nPlanes[idx]))
                self.res2.append(RepResidualBlock(reps,dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))
                self.linear.append(CrossScaleFusion(nPlanes[len(nPlanes) - 1],nPlanes[idx]))
                self.linear1.append(CrossScaleFusion(nPlanes[len(nPlanes) - 2],nPlanes[idx]))
                self.linear2.append(CrossScaleFusion(nPlanes[len(nPlanes) - 3],nPlanes[idx]))
                self.linear3.append(CrossScaleFusion(nPlanes[len(nPlanes) - 4],nPlanes[idx]))
                self.linear4.append(CrossScaleFusion(nPlanes[len(nPlanes) - 5],nPlanes[idx]))
                self.linear5.append(CrossScaleFusion(nPlanes[len(nPlanes) - 6],nPlanes[idx]))
            else:
                self.res.append(RepResidualBlock(reps, dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))



        self.outputFeatureLvl = 0 # 0 for original resolution, 1 for 4cm resolution
    def forward(self,x):
        outputFeatureLvl = self.outputFeatureLvl
        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        idx = 0
        features = []
        d_pyramid = []
        u_pyramid = []
        layer_idx = 0
        features2 = []
        context_feature = []
        features.append(self.res[layer_idx](x))
        d_pyramid_feature = self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx]))


        for count in range(len(nPlanes) - 2):
            layer_idx = layer_idx + 1
            features.append(self.res[layer_idx](d_pyramid_feature))
            d_pyramid_feature = self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx]))


        layer_idx = layer_idx + 1

        features.append(self.res[layer_idx](d_pyramid_feature))

        layer_idx = layer_idx - 1
        u_pyramid = self.deconv[layer_idx](self.bn1[layer_idx](features[layer_idx+1]))
        a = self.res2[layer_idx](scn.add_feature_planes([features[layer_idx], self.linearReweight[layer_idx](u_pyramid)]))

#        print(features[-1].shape, a.shape)
        b = self.linear[layer_idx](features[-1])
        b = scn.upsample_feature(lr = b, hr = a, stride = 2)
        features2.append(self.bn2[layer_idx](scn.add_feature_planes([a,b])))

        for count in range(len(nPlanes) - 2 - outputFeatureLvl):
            layer_idx = layer_idx - 1
            u_pyramid = self.deconv[layer_idx](self.bn1[layer_idx](features2[len(nPlanes) - 3 - layer_idx]))
            a = self.res2[layer_idx](scn.add_feature_planes([features[layer_idx], self.linearReweight[layer_idx](u_pyramid)]))

            b = self.linear[layer_idx](features[-1])
            b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 1 - layer_idx))
            a = scn.add_feature_planes([a,b])
            if(count >= 0 ):
                b = self.linear1[layer_idx](features2[0])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 2 - layer_idx))
                a = scn.add_feature_planes([a,b])
            if(count >= 1 ):
                b = self.linear2[layer_idx](features2[1])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 3 - layer_idx))
                a = scn.add_feature_planes([a,b])
            if(count >= 2 ):
                b = self.linear3[layer_idx](features2[2])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 4 - layer_idx))
                a = scn.add_feature_planes([a,b])
            if(count >= 3 ):
                b = self.linear4[layer_idx](features2[3])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 5 - layer_idx))
                a = scn.add_feature_planes([a,b])
            if(count >= 4 ):
                b = self.linear5[layer_idx](features2[4])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 6 - layer_idx))
                a = scn.add_feature_planes([a,b])

            features2.append(self.bn2[layer_idx](a))

        output = features2[-1]
        if(outputFeatureLvl > 0):
            output  = scn.upsample_feature(lr = output , hr = features[0], stride = 2 ** (outputFeatureLvl))
        return output




#we borrow the idea of resnet and
class DenseUNet_Concate(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        #seven layer convolution is here!
        self.leakiness = 0
        self.downsample=[2, 2]
        self.nPlanes = config['unet_structure']
        self.reps = config['block_reps']
        self.dimension = 3

        self.res = nn.ModuleList()
        self.bn0 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.deconv = nn.ModuleList()
        self.res2 = nn.ModuleList()

        self.linear = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.linear3 = nn.ModuleList()
        self.linear4 = nn.ModuleList()

        self.concateLinear = nn.ModuleList()

        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        for idx, nPlane in enumerate(nPlanes):
            if(idx < len(nPlanes) - 1):
                self.res.append(RepResidualBlock(reps,dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))
                self.bn0.append(scn.BatchNormLeakyReLU(nPlanes[idx],leakiness=leakiness))
                self.conv.append(scn.Convolution(dimension, nPlanes[idx], nPlanes[idx+1], downsample[0], downsample[1], False))
                self.bn1.append(scn.BatchNormLeakyReLU(nPlanes[idx+1],leakiness=leakiness))
                self.bn2.append(scn.BatchNormLeakyReLU(nPlanes[idx],leakiness=leakiness))
                self.deconv.append(scn.Deconvolution(dimension, nPlanes[idx+1], nPlanes[idx], downsample[0], downsample[1], False))
                self.res2.append(RepResidualBlock(reps,dimension, nPlanes[idx]*2,nPlanes[idx],leakiness=leakiness))
                self.linear.append(LinearSCN(nPlanes[len(nPlanes) - 1],nPlanes[idx]))
                self.linear1.append(LinearSCN(nPlanes[len(nPlanes) - 2],nPlanes[idx]))
                self.linear2.append(LinearSCN(nPlanes[len(nPlanes) - 3],nPlanes[idx]))
                self.linear3.append(LinearSCN(nPlanes[len(nPlanes) - 4],nPlanes[idx]))
                self.linear4.append(LinearSCN(nPlanes[len(nPlanes) - 5],nPlanes[idx]))

                concate_length = sum(nPlanes[idx:])
                self.concateLinear.append(LinearSCN(concate_length ,nPlanes[idx]))


            else:
                self.res.append(ResidualBlock(dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))



    def forward(self,x):

        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        idx = 0
        features = []
        d_pyramid = []
        u_pyramid = []
        layer_idx = 0
        features2 = []
        context_feature = []
        features.append(self.res[layer_idx](x))
        d_pyramid.append(self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx])))


        for count in range(len(nPlanes) - 2):
            layer_idx = layer_idx + 1
            features.append(self.res[layer_idx](d_pyramid[layer_idx - 1]))
            d_pyramid.append(self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx])))

        layer_idx = layer_idx + 1
        features.append(self.res[layer_idx](d_pyramid[layer_idx-1]))

        layer_idx = layer_idx - 1
        u_pyramid.append(self.deconv[layer_idx](self.bn1[layer_idx](features[layer_idx+1])))
        a = self.res2[layer_idx](scn.concatenate_feature_planes([features[layer_idx], u_pyramid[len(nPlanes) - 2 - layer_idx]]))

#        print(features[-1].shape, a.shape)
        b = features[-1]
        b = scn.upsample_feature(lr = b, hr = a, stride = 2)
        features2.append(self.bn2[layer_idx](self.concateLinear[layer_idx](scn.concatenate_feature_planes([a,b]))))

        for count in range(len(nPlanes) - 2):
            layer_idx = layer_idx - 1
            u_pyramid.append(self.deconv[layer_idx](self.bn1[layer_idx](features2[len(nPlanes) - 3 - layer_idx])))
            feature_candidate = []
            a = self.res2[layer_idx](scn.concatenate_feature_planes([features[layer_idx], u_pyramid[len(nPlanes) - 2 - layer_idx]]))
            b = features[-1]
            #b = self.linear[layer_idx](features[-1])
            b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 1 - layer_idx))
            feature_candidate.append(a)
            feature_candidate.append(b)
            if(count >= 0):
                b = features2[0]
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 2 - layer_idx))
                feature_candidate.append(b)
            if(count >= 1):
                b = features2[1]
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 3 - layer_idx))
                feature_candidate.append(b)
            if(count >= 2):
                b = features2[2]
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 4 - layer_idx))
                feature_candidate.append(b)
            if(count >= 3):
                b = features2[3]
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 5 - layer_idx))
                feature_candidate.append(b)
            if(count >= 4):
                b = features2[4]
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 6 - layer_idx))
                feature_candidate.append(b)

            features2.append(self.bn2[layer_idx](self.concateLinear[layer_idx](scn.concatenate_feature_planes(feature_candidate))))
        return features2[-1]
#we borrow the idea of resnet and
class DenseUNet_CompactConcate(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        #seven layer convolution is here!
        self.leakiness = 0
        self.downsample=[2, 2]
        self.nPlanes = config['unet_structure']
        self.reps = config['block_reps']
        self.dimension = 3

        self.res = nn.ModuleList()
        self.bn0 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.deconv = nn.ModuleList()
        self.res2 = nn.ModuleList()

        self.linear = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.linear3 = nn.ModuleList()
        self.linear4 = nn.ModuleList()

        self.linear5 = nn.ModuleList()
        self.concateLinear = nn.ModuleList()

        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        for idx, nPlane in enumerate(nPlanes):
            if(idx < len(nPlanes) - 1):
                self.res.append(RepResidualBlock(reps,dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))
                self.bn0.append(scn.BatchNormLeakyReLU(nPlanes[idx],leakiness=leakiness))
                self.conv.append(scn.Convolution(dimension, nPlanes[idx], nPlanes[idx+1], downsample[0], downsample[1], False))
                self.bn1.append(scn.BatchNormLeakyReLU(nPlanes[idx+1],leakiness=leakiness))
                self.bn2.append(scn.BatchNormLeakyReLU(nPlanes[idx],leakiness=leakiness))
                self.deconv.append(scn.Deconvolution(dimension, nPlanes[idx+1], nPlanes[idx], downsample[0], downsample[1], False))
                self.res2.append(RepResidualBlock(reps,dimension, nPlanes[idx]*2,nPlanes[idx],leakiness=leakiness))
                self.linear.append(LinearSCN(nPlanes[len(nPlanes) - 1],nPlanes[idx]))
                self.linear1.append(LinearSCN(nPlanes[len(nPlanes) - 2],nPlanes[idx]))
                self.linear2.append(LinearSCN(nPlanes[len(nPlanes) - 3],nPlanes[idx]))
                self.linear3.append(LinearSCN(nPlanes[len(nPlanes) - 4],nPlanes[idx]))
                self.linear4.append(LinearSCN(nPlanes[len(nPlanes) - 5],nPlanes[idx]))
                self.linear5.append(LinearSCN(nPlanes[len(nPlanes) - 6],nPlanes[idx]))

                concate_length = sum(nPlanes[idx:])
                self.concateLinear.append(LinearSCN(nPlanes[idx] * (len(nPlanes)  -  idx) ,nPlanes[idx]))


            else:
                self.res.append(ResidualBlock(dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))



    def forward(self,x):

        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        idx = 0
        features = []
        d_pyramid = []
        u_pyramid = []
        layer_idx = 0
        features2 = []
        context_feature = []
        features.append(self.res[layer_idx](x))
        d_pyramid.append(self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx])))


        for count in range(len(nPlanes) - 2):
            layer_idx = layer_idx + 1
            features.append(self.res[layer_idx](d_pyramid[layer_idx - 1]))
            d_pyramid.append(self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx])))

        layer_idx = layer_idx + 1
        features.append(self.res[layer_idx](d_pyramid[layer_idx-1]))

        layer_idx = layer_idx - 1
        u_pyramid.append(self.deconv[layer_idx](self.bn1[layer_idx](features[layer_idx+1])))
        a = self.res2[layer_idx](scn.concatenate_feature_planes([features[layer_idx], u_pyramid[len(nPlanes) - 2 - layer_idx]]))

#        print(features[-1].shape, a.shape)
        b = self.linear[layer_idx](features[-1])
        b = scn.upsample_feature(lr = b, hr = a, stride = 2)
        features2.append(self.bn2[layer_idx](self.concateLinear[layer_idx](scn.concatenate_feature_planes([a,b]))))

        for count in range(len(nPlanes) - 2):
            layer_idx = layer_idx - 1
            u_pyramid.append(self.deconv[layer_idx](self.bn1[layer_idx](features2[len(nPlanes) - 3 - layer_idx])))
            feature_candidate = []
            a = self.res2[layer_idx](scn.concatenate_feature_planes([features[layer_idx], u_pyramid[len(nPlanes) - 2 - layer_idx]]))
            b = self.linear[layer_idx](features[-1])
            b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 1 - layer_idx))
            feature_candidate.append(a)
            feature_candidate.append(b)
            if(count >= 0):
                b = self.linear1[layer_idx](features2[0])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 2 - layer_idx))
                feature_candidate.append(b)
            if(count >= 1):
                b = self.linear2[layer_idx](features2[1])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 3 - layer_idx))
                feature_candidate.append(b)
            if(count >= 2):
                b = self.linear3[layer_idx](features2[2])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 4 - layer_idx))
                feature_candidate.append(b)
            if(count >= 3):
                b = self.linear4[layer_idx](features2[3])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 5 - layer_idx))
                feature_candidate.append(b)
            if(count >= 4):
                b = self.linear5[layer_idx](features2[4])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 6 - layer_idx))
                feature_candidate.append(b)
            features2.append(self.bn2[layer_idx](self.concateLinear[layer_idx](scn.concatenate_feature_planes(feature_candidate))))
        return features2[-1]


class UNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        #seven layer convolution is here!
        self.leakiness = 0
        self.downsample=[2, 2]
        self.nPlanes = config['unet_structure']
        self.reps = config['block_reps']
        self.dimension = 3

        self.res = nn.ModuleList()
        self.bn0 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.deconv = nn.ModuleList()
        self.res2 = nn.ModuleList()

        self.linear = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.linear3 = nn.ModuleList()
        self.linear4 = nn.ModuleList()
        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        for idx, nPlane in enumerate(nPlanes):
            if(idx < len(nPlanes) - 1):
                self.res.append(RepResidualBlock(reps,dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))
                self.bn0.append(scn.BatchNormLeakyReLU(nPlanes[idx],leakiness=leakiness))
                self.conv.append(scn.Convolution(dimension, nPlanes[idx], nPlanes[idx+1], downsample[0], downsample[1], False))
                self.bn1.append(scn.BatchNormLeakyReLU(nPlanes[idx+1],leakiness=leakiness))
                self.deconv.append(scn.Deconvolution(dimension, nPlanes[idx+1], nPlanes[idx], downsample[0], downsample[1], False))
                self.res2.append(RepResidualBlock(reps,dimension, nPlanes[idx]*2,nPlanes[idx],leakiness=leakiness))
                self.linear.append(LinearSCN(nPlanes[len(nPlanes) - 1],nPlanes[idx]))
                self.linear1.append(LinearSCN(nPlanes[len(nPlanes) - 2],nPlanes[idx]))
                self.linear2.append(LinearSCN(nPlanes[len(nPlanes) - 3],nPlanes[idx]))
                self.linear3.append(LinearSCN(nPlanes[len(nPlanes) - 4],nPlanes[idx]))
                self.linear4.append(LinearSCN(nPlanes[len(nPlanes) - 5],nPlanes[idx]))
            else:
                self.res.append(ResidualBlock(dimension, nPlanes[idx],nPlanes[idx],leakiness=leakiness))



    def forward(self,x):

        downsample = self.downsample
        leakiness = self.leakiness
        nPlanes = self.nPlanes
        reps = self.reps
        dimension = self.dimension
        idx = 0
        features = []
        d_pyramid = []
        u_pyramid = []
        layer_idx = 0
        features2 = []
        context_feature = []
        features.append(self.res[layer_idx](x))
        d_pyramid.append(self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx])))


        for count in range(len(nPlanes) - 2):
            layer_idx = layer_idx + 1
            features.append(self.res[layer_idx](d_pyramid[layer_idx - 1]))
            d_pyramid.append(self.conv[layer_idx](self.bn0[layer_idx](features[layer_idx])))

        layer_idx = layer_idx + 1
        features.append(self.res[layer_idx](d_pyramid[layer_idx-1]))

        layer_idx = layer_idx - 1
        u_pyramid.append(self.deconv[layer_idx](self.bn1[layer_idx](features[layer_idx+1])))
        a = self.res2[layer_idx](scn.concatenate_feature_planes([features[layer_idx], u_pyramid[len(nPlanes) - 2 - layer_idx]]))

#        print(features[-1].shape, a.shape)
        b = self.linear[layer_idx](features[-1])
        b = scn.upsample_feature(lr = b, hr = a, stride = 2)
        features2.append(scn.add_feature_planes([a,b]))

        for count in range(len(nPlanes) - 2):
            layer_idx = layer_idx - 1
            u_pyramid.append(self.deconv[layer_idx](self.bn1[layer_idx](features2[len(nPlanes) - 3 - layer_idx])))
            feature_candidate = []
            a = self.res2[layer_idx](scn.concatenate_feature_planes([features[layer_idx], u_pyramid[len(nPlanes) - 2 - layer_idx]]))
            b = self.linear[layer_idx](features[-1])
            b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 1 - layer_idx))
            feature_candidate.append(a)
            feature_candidate.append(b)
            if(count >= 0):
                b = self.linear1[layer_idx](features2[0])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 2 - layer_idx))
                feature_candidate.append(b)
            if(count >= 1):
                b = self.linear2[layer_idx](features2[1])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 3 - layer_idx))
                feature_candidate.append(b)
            if(count >= 2):
                b = self.linear3[layer_idx](features2[2])
                b = scn.upsample_feature(lr = b, hr = a, stride = 2 ** (len(nPlanes) - 4 - layer_idx))
                feature_candidate.append(b)


            features2.append(scn.add_feature_planes(feature_candidate))
        return features2[-1]



class DenseUNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.config = config
        self.input = scn.InputLayer(config['dimension'], config['full_scale'], mode=4,
                                    normal_guide_scale= (config['full_scale'] >> config['rotation_guide_level']) + 1)
        self.sub = scn.SubmanifoldConvolution(config['dimension'], config['input_feature_number'], config['unet_structure'][0], 3, False)
        self.unet = ThreeVoxelKernel(config)

        self.output_feature_dim = config['unet_structure'][self.unet.outputFeatureLvl]
        self.bn = scn.BatchNormReLU(self.output_feature_dim)
        self.output = scn.OutputLayer(self.output_feature_dim)
        self.linear = nn.Linear(self.output_feature_dim , config['class_num'])

        self.siamesenet = nn.Linear(self.output_feature_dim , self.output_feature_dim)
        self.linear_regularize = nn.Linear(self.output_feature_dim, 2)
#        self.bn = scn.BatchNormReLU(config['unet_structure'][0])
#        self.output = scn.OutputLayer(config['dimension'])
#        self.linear = nn.Linear(config['unet_structure'][0], config['class_num'])
    def similarity(self, f1, f2):
        f1 = torch.nn.functional.relu(self.siamesenet(f1))
        f2 = torch.nn.functional.relu(self.siamesenet(f2))
        return(self.linear_regularize(torch.abs(f1 - f2)))
    def forward(self, x):
#        f = self.sparseModel(x)
#        y = self.linear(f)

        input = self.input(x)
        sub = self.sub(input)
        feature = self.output(self.bn(self.unet(sub)))
        y = self.linear(feature)

        # random permutation test

        return y, feature

class InstanceDenseUNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.config = config
        self.input = scn.InputLayer(config['dimension'], config['full_scale'], mode=4,
                                    normal_guide_scale= (config['full_scale'] >> config['rotation_guide_level']) + 1)
        self.sub = scn.SubmanifoldConvolution(config['dimension'], config['input_feature_number'], config['unet_structure'][0], 3, False)
#        self.unet = ThreeVoxelKernel(config)
        self.unet = scn.UNet(config['dimension'], config['block_reps'], config['unet_structure'], config['residual_blocks'])
        self.output_feature_dim = config['unet_structure'][0]
        self.bn = scn.BatchNormReLU(self.output_feature_dim)
        self.output = scn.OutputLayer(self.output_feature_dim)
        self.linear = nn.Linear(self.output_feature_dim , config['class_num'])

        self.fc_regress = nn.Linear(self.output_feature_dim, self.output_feature_dim)
        self.linear_regress = nn.Linear(self.output_feature_dim , 1)
        self.sigmoid_regress = nn.Sigmoid()

        self.fc_embedding = nn.Linear(self.output_feature_dim , self.output_feature_dim)
        self.linear_embedding = nn.Linear(self.output_feature_dim , self.output_feature_dim)

        self.fc_displacement = nn.Linear(self.output_feature_dim , self.output_feature_dim)
        self.linear_displacement = nn.Linear(self.output_feature_dim , config['dimension'])


    def forward(self, x):
        input = self.input(x)
        sub = self.sub(input)
        feature = self.output(self.bn(self.unet(sub)))
        y = self.linear(feature)
        embedding = self.linear_embedding(self.fc_embedding(feature))
        offset = self.sigmoid_regress(self.linear_regress(self.fc_regress(feature)))
        displacement = self.linear_displacement(self.fc_displacement(feature))
        return y, feature, embedding, offset, displacement

class LearningBWDenseUNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.config = config
        self.backbone = InstanceDenseUNet(config)
#        if(os.path.exists(config['backbone_network'])):
#            self.backbone.load_state_dict(torch.load(config['backbone_network']))


        self.fc_bw = nn.Linear(self.backbone.output_feature_dim , self.backbone.output_feature_dim)
        self.linear_bw = nn.Linear(self.backbone.output_feature_dim , 2)
        self.relu_bw = nn.Softplus()

        self.fc_occupancy = nn.Linear(self.backbone.output_feature_dim , self.backbone.output_feature_dim)
        self.linear_occupancy = nn.Linear(self.backbone.output_feature_dim , 1)
        self.relu_occupancy = nn.Softplus()

    def forward(self, x):

        semantics, feature, embedding, offset, displacement = self.backbone(x)
        bw = self.relu_bw(self.linear_bw(self.fc_bw(feature)))
        occupancy = self.relu_occupancy(self.linear_occupancy(self.fc_occupancy(feature)))
        return semantics, embedding, offset, displacement, bw, occupancy



class ThreeVoxelKernel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.config = config

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(config['dimension'], config['full_scale'], mode=4,
                           normal_guide_scale= (config['full_scale'] >> config['rotation_guide_level']) + 1)
        ).add(
            scn.SubmanifoldConvolution(config['dimension'], config['input_feature_number'], config['m'], 3, False)
        ).add(
            scn.UNet(config['dimension'], config['block_reps'], config['unet_structure'], config['residual_blocks'])
        ).add(
            scn.BatchNormReLU(config['m'])
        ).add(
            scn.OutputLayer(config['dimension'])
        )
        self.output_feature_dim = config['m']
        self.linear = nn.Linear(config['m'], config['class_num'])

        self.fc_regress = nn.Linear(self.output_feature_dim, self.output_feature_dim)
        self.linear_regress = nn.Linear(self.output_feature_dim , 1)
        self.sigmoid_regress = nn.Sigmoid()
        self.fc_embedding = nn.Linear(self.output_feature_dim , self.output_feature_dim)
        self.linear_embedding = nn.Linear(self.output_feature_dim , self.output_feature_dim)


    def forward(self, x):
#        feature = self.sparseModel(x)
#        y = self.linear(f)

        feature = self.sparseModel(x)
        y = self.linear(feature)
        embedding = self.linear_embedding(self.fc_embedding(feature))
        offset = self.sigmoid_regress(self.linear_regress(self.fc_regress(feature)))
        return y, embedding, offset

