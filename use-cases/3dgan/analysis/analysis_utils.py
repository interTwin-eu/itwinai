from __future__ import print_function
import os

import glob

from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import argparse
import sys
import h5py
import numpy as np
import time
import math
import torch

# import tensorflow as tf

# import tensorflow.keras.backend as K
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
# from tensorflow.keras.utils import Progbar

# from tensorflow.compat.v1.keras.layers import BatchNormalization
# from tensorflow.keras.layers import (
#     Input,
#     Dense,
#     Reshape,
#     Flatten,
#     Lambda,
#     Dropout,
#     Activation,
#     Embedding,
# )
# from tensorflow.keras.layers import LeakyReLU
# from tensorflow.keras.layers import (
#     UpSampling3D,
#     Conv3D,
#     ZeroPadding3D,
#     AveragePooling3D,
# )
# from tensorflow.keras.models import Model, Sequential
import math

import json

# def GetDataAngle(datafile, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1):
#     print ('Loading Data from .....', datafile)
#     f=h5py.File(datafile,'r')
#     X=np.array(f.get('ECAL'))* xscale
#     Y=np.array(f.get('energy'))/yscale
#     X[X < thresh] = 0
#     X = X.astype(np.float32)
#     Y = Y.astype(np.float32)
#     ecal = np.sum(X, axis=(1, 2, 3))
#     indexes = np.where(ecal > 10.0)
#     X=X[indexes]
#     Y=Y[indexes]
#     if angtype in f:
#       ang = np.array(f.get(angtype))[indexes]
#     #else:
#       #ang = gan.measPython(X)
#     X = np.expand_dims(X, axis=daxis)
#     ecal=ecal[indexes]
#     ecal=np.expand_dims(ecal, axis=daxis)
#     if xpower !=1.:
#         X = np.power(X, xpower)
#     return X, Y, ang, ecal

def GetDataAngle(batch, xscale=1, xpower=1, yscale=100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1):
    X = batch['X'] * xscale
    Y = batch['Y'] / yscale
    X[X < thresh] = 0
    X = X.to(torch.float32) #X.astype(np.float32)
    Y = Y.to(torch.float32) #Y.astype(np.float32)
    ecal = torch.sum(X, axis=(1, 2, 3))
    indexes = torch.where(ecal > 10.0)
    X = X[indexes]
    Y = Y[indexes]

    if angtype in batch:
        ang = batch[angtype][indexes]
    else:
        ang = batch['ang']

    X = torch.unsqueeze(X, axis=daxis)
    ecal = ecal[indexes]
    ecal = torch.unsqueeze(ecal, axis=daxis)
    if xpower != 1.:
        X = X.pow(xpower)

    return X, Y, ang, ecal


# def sortEnergy(data, ecal, energies, ang=1):
#     var={}
#     tolerance =5
#     for energy in energies:
#         if energy==0:
#             var["events_act" + str(energy)]=data[0][:5000]
#             var["energy" + str(energy)]=data[1][:5000]
#             if ang: var["angle_act" + str(energy)]=data[2][:5000]
#             var["ecal_act" + str(energy)]=ecal[:5000]
#             var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
#         else:
#             var["indexes" + str(energy)] = np.where((data[1] > (energy - tolerance)/100. ) & ( data[1] < (energy + tolerance)/100.))
#             var["events_act" + str(energy)]=data[0][var["indexes" + str(energy)]]
#             var["energy" + str(energy)]=data[1][var["indexes" + str(energy)]]
#             if ang:  var["angle_act" + str(energy)]=data[2][var["indexes" + str(energy)]]
#             var["ecal_act" + str(energy)]=ecal[var["indexes" + str(energy)]]
#             var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
#     return var

# def sortEnergy(data, ecal, energies, ang=1):
#     var = {}
#     tolerance = 5
#     for energy in energies:
#         energy = torch.tensor(energy, dtype=ecal.dtype, device=ecal.device)
#         tolerance = torch.tensor(tolerance, dtype=ecal.dtype, device=ecal.device)

#         if energy.item() == 0:
#             var["events_act" + str(energy.item())] = data[0][:5000]
#             var["energy" + str(energy.item())] = data[1][:5000]
#             if ang: var["angle_act" + str(energy.item())] = data[2][:5000]
#             var["ecal_act" + str(energy.item())] = ecal[:5000]
#             var["index" + str(energy.item())] = var["events_act" + str(energy.item())].shape[0]
#         else:
#             var["indexes" + str(energy.item())] = np.where((data[1] > (energy - tolerance)/100. ) & ( data[1] < (energy + tolerance)/100.))
#             var["events_act" + str(energy.item())]=data[0][var["indexes" + str(energy.item())]]
#             var["energy" + str(energy.item())]=data[1][var["indexes" + str(energy.item())]]
#             if ang:  var["angle_act" + str(energy.item())]=data[2][var["indexes" + str(energy.item())]]
#             var["ecal_act" + str(energy.item())]=ecal[var["indexes" + str(energy.item())]]
#             var["index" + str(energy.item())] = var["events_act" + str(energy.item())].shape[0]
#     return var

def sortEnergy(data, ecal, energies, ang=1):
    var={}
    tolerance = 5
    for energy in energies:
        if energy==0:
            var["events_act" + str(energy)]=data[0][:5000]
            var["energy" + str(energy)]=data[1][:5000]
            if ang: var["angle_act" + str(energy)]=data[2][:5000]
            var["ecal_act" + str(energy)]=ecal[:5000]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
        else:
            var["indexes" + str(energy)] = torch.where((data[1] > (energy - tolerance)/100. ) & ( data[1] < (energy + tolerance)/100.))
            
            if len(var["indexes" + str(energy)]) > 0:  # Ensure indexes are not empty
                index_tensor = var["indexes" + str(energy)][0]  # Assuming torch.where returns a tuple with a tensor at [0]
                var["events_act" + str(energy)] = torch.index_select(data[0], 0, index_tensor)
                var["energy" + str(energy)] = torch.index_select(data[1], 0, index_tensor)
                if ang:  var["angle_act" + str(energy)] = torch.index_select(data[2], 0, index_tensor)
                var["ecal_act" + str(energy)] = torch.index_select(ecal, 0, index_tensor)
                var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]

            else:
                print("indexes are empty!")
                sys.exit()

    return var


# def OptAnalysisAngle(var, g, energies, ascale=None, xscale=None, yscale=100, xpower=None, latent=256, concat=2):
#     m=2
#     for energy in energies:
#         var["events_act" + str(energy)] = np.squeeze(var["events_act" + str(energy)])
#         if ascale: var["angle_act"+ str(energy)]= var["angle_act"+ str(energy)] * ascale
#         if yscale: var["energy" + str(energy)]=var["energy" + str(energy)]/yscale
#         print(var["events_act" + str(energy)].shape)
#         num = var["events_act" + str(energy)].shape[0]
#         x = var["events_act" + str(energy)].shape[1]
#         y = var["events_act" + str(energy)].shape[2]
#         z = var["events_act" + str(energy)].shape[3]
#         var["events_gan" + str(energy)] = generate(g, num, [var["energy" + str(energy)], (var["angle_act"+ str(energy)])], latent, concat)
#         var["events_gan" + str(energy)] = np.squeeze(var["events_gan" + str(energy)])
#         if xpower: var["events_gan" + str(energy)] = np.power(var["events_gan" + str(energy)], 1.0/xpower)
#         if xscale: var["events_gan" + str(energy)] = var["events_gan" + str(energy)]/xscale
        
#         var["ecal_act"+ str(energy)] = torch.sum(var["events_act" + str(energy)], dim = (1, 2, 3))
#         var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
#         var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = torch_get_sums(var["events_act" + str(energy)])
#         var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
#         var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
#         var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
#         var["angle_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])
#     return metric(var, energies, m, angtype='angle', x=x, y=y, z=z, ang=1)

def OptAnalysisAngle(var, g, energies, ascale=None, xscale=None, yscale=None, xpower=None, latent=256, concat=2):
    m = 2
    for energy in energies:
        
        #print("before squeezing shape: " + str(var["events_act" + str(energy)].shape))
        var["events_act" + str(energy)] = torch.squeeze(var["events_act" + str(energy)])
        if ascale: var["angle_act"+ str(energy)] = var["angle_act"+ str(energy)] * ascale
        if yscale: var["energy" + str(energy)] = var["energy" + str(energy)] / yscale

        num = var["events_act" + str(energy)].shape[0]
        x = var["events_act" + str(energy)].shape[1]
        y = var["events_act" + str(energy)].shape[2]
        #print(f"Shape before accessing z for energy {energy}: {var['events_act' + str(energy)].shape}")
        z = var["events_act" + str(energy)].shape[3]

        print(num, x, y, z)

        var["events_gan" + str(energy)] = generate(g, num, [var["energy" + str(energy)], (var["angle_act"+ str(energy)])], latent, concat)
        if isinstance(var["events_gan" + str(energy)], np.ndarray):
            var["events_gan" + str(energy)] = torch.from_numpy(var["events_gan" + str(energy)]).float()
        var["events_gan" + str(energy)] = torch.squeeze(var["events_gan" + str(energy)])
        if xpower: var["events_gan" + str(energy)] = torch.pow(var["events_gan" + str(energy)], 1.0 / xpower)
        if xscale: var["events_gan" + str(energy)] = var["events_gan" + str(energy)] / xscale
        
        var["ecal_act"+ str(energy)] = torch.sum(var["events_act" + str(energy)], dim = (1, 2, 3))
        var["ecal_gan"+ str(energy)] = torch.sum(var["events_gan" + str(energy)], dim = (1, 2, 3))

        var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = torch_get_sums(var["events_act" + str(energy)])
        var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = torch_get_sums(var["events_gan" + str(energy)])

        var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)] = get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
        var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)

        var["angle_gan"+ str(energy)] = measPython(var["events_gan" + str(energy)])

    return metric(var, energies, m, angtype='angle', x=x, y=y, z=z, ang=1)

#Optimization metric
# def metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1):
#     metricp = 0
#     metrice = 0
#     metrica = 0
#     for energy in energies:
#         #Relative error on mean moment value for each moment and each axis
#         x_act= np.mean(var["momentX_act"+ str(energy)], axis=0)
#         x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
#         y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
#         y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
#         z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
#         z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
#         var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act
#         var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
#         var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
#         #Taking absolute of errors and adding for each axis then scaling by 3
#         var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)])+ np.absolute(var["posz_error"+ str(energy)]))/3
#         #Summing over moments and dividing for number of moments
#         var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
#         metricp += var["pos_total"+ str(energy)]
#         #Take profile along each axis and find mean along events
#         sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis=0), np.mean(var["sumsz_act" + str(energy)], axis=0)
#         sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
#         var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
#         var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
#         var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
#         #Take absolute of error and mean for all events                                                           
#         var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
#         #Summing over moments and dividing for number of moments
#         var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
#         metricp += var["pos_total"+ str(energy)]
#         #Take profile along each axis and find mean along events
#         sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis= 0), np.mean(var["sumsz_act" + str(energy)], axis=0)
#         sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
#         var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
#         var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
#         var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
#         #Take absolute of error and mean for all events
#         var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)]))/x
#         var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)]))/y
#         var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/z

#         var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
#         metrice += var["eprofile_total"+ str(energy)]
#         if ang:
#             var["angle_error"+ str(energy)] = np.mean(np.absolute((var[angtype + "_act" + str(energy)] - var[angtype + "_gan" + str(energy)])/var[angtype + "_act" + str(energy)]))
#             metrica += var["angle_error"+ str(energy)]
#     metricp = metricp/len(energies)
#     metrice = metrice/len(energies)
#     if ang:metrica = metrica/len(energies)
#     tot = metricp + metrice
#     if ang:tot = tot +metrica
#     result = [tot, metricp, metrice]
#     if ang: result.append(metrica)
#     return result

def metric(var, energies, m, angtype='mtheta', x=51, y=51, z=25, ang=1):
    metricp = 0
    metrice = 0
    metrica = 0

    for energy in energies:
        # Relative error on mean moment value for each moment and each axis
        x_act = torch.mean(var["momentX_act" + str(energy)], dim=0)
        x_gan = torch.mean(var["momentX_gan" + str(energy)], dim=0)
        y_act = torch.mean(var["momentY_act" + str(energy)], dim=0)
        y_gan = torch.mean(var["momentY_gan" + str(energy)], dim=0)
        z_act = torch.mean(var["momentZ_act" + str(energy)], dim=0)
        z_gan = torch.mean(var["momentZ_gan" + str(energy)], dim=0)

        var["posx_error" + str(energy)] = (x_act - x_gan) / x_act
        var["posy_error" + str(energy)] = (y_act - y_gan) / y_act
        var["posz_error" + str(energy)] = (z_act - z_gan) / z_act

        var["pos_error" + str(energy)] = (torch.abs(var["posx_error" + str(energy)]) + torch.abs(var["posy_error" + str(energy)]) + torch.abs(var["posz_error" + str(energy)])) / 3
        var["pos_total" + str(energy)] = torch.sum(var["pos_error" + str(energy)]) / m
        metricp += var["pos_total" + str(energy)]

        # Profile along each axis and mean along events
        sumxact = torch.mean(var["sumsx_act" + str(energy)], dim=0)
        sumxgan = torch.mean(var["sumsx_gan" + str(energy)], dim=0)
        sumyact = torch.mean(var["sumsy_act" + str(energy)], dim=0)
        sumygan = torch.mean(var["sumsy_gan" + str(energy)], dim=0)
        sumzact = torch.mean(var["sumsz_act" + str(energy)], dim=0)
        sumzgan = torch.mean(var["sumsz_gan" + str(energy)], dim=0)

        var["eprofilex_error" + str(energy)] = (sumxact - sumxgan) / sumxact
        var["eprofiley_error" + str(energy)] = (sumyact - sumygan) / sumyact
        var["eprofilez_error" + str(energy)] = (sumzact - sumzgan) / sumzact

        var["eprofilex_total" + str(energy)] = torch.sum(torch.abs(var["eprofilex_error" + str(energy)])) / x
        var["eprofiley_total" + str(energy)] = torch.sum(torch.abs(var["eprofiley_error" + str(energy)])) / y
        var["eprofilez_total" + str(energy)] = torch.sum(torch.abs(var["eprofilez_error" + str(energy)])) / z

        var["eprofile_total" + str(energy)] = (var["eprofilex_total" + str(energy)] + var["eprofiley_total" + str(energy)] + var["eprofilez_total" + str(energy)]) / 3
        metrice += var["eprofile_total" + str(energy)]

        if ang:
            var["angle_error" + str(energy)] = torch.mean(torch.abs((var[angtype + "_act" + str(energy)] - var[angtype + "_gan" + str(energy)]) / var[angtype + "_act" + str(energy)]))
            metrica += var["angle_error" + str(energy)]

    metricp /= len(energies)
    metrice /= len(energies)
    if ang: metrica /= len(energies)

    tot = metricp + metrice
    if ang: tot += metrica
    result = [tot, metricp, metrice]
    if ang: result.append(metrica)

    return result

# get sums along different axis
def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

def torch_get_sums(images):
    # Ensure images is a PyTorch tensor
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # Use torch.sum instead of np.sum
    sumsx = torch.sum(images, dim=(2, 3)).squeeze()
    sumsy = torch.sum(images, dim=(1, 3)).squeeze()
    sumsz = torch.sum(images, dim=(1, 2)).squeeze()

    return sumsx, sumsy, sumsz


# get moments
# def get_moments(sumsx, sumsy, sumsz, totalE, m, x=51, y=51, z=25):
#     old_err_state = np.seterr(divide='raise')
#     ignored_states = np.seterr(**old_err_state)
#     totalE = np.squeeze(totalE)
#     index = sumsx.shape[0]
#     momentX = np.zeros((index, m))
#     momentY = np.zeros((index, m))
#     momentZ = np.zeros((index, m))
#     ECAL_midX = np.zeros(index)
#     ECAL_midY = np.zeros(index)
#     ECAL_midZ = np.zeros(index)
#     for i in range(m):
#       relativeIndices = np.tile(np.arange(x), (index,1))
#       moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
#       #ECAL_momentX = np.divide(umath.inner1d(sumsx, moments) ,totalE)
#       ECAL_momentX = np.einsum('ij,ij->i', sumsx, moments) / totalE
#       if i==0: ECAL_midX = ECAL_momentX.transpose()
#       momentX[:,i] = ECAL_momentX
#     for i in range(m):
#       relativeIndices = np.tile(np.arange(y), (index,1))
#       moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
#       #ECAL_momentY = np.divide(umath.inner1d(sumsy, moments), totalE)
#       ECAL_momentY = np.einsum('ij,ij->i', sumsy, moments) / totalE
#       if i==0: ECAL_midY = ECAL_momentY.transpose()
#       momentY[:,i]= ECAL_momentY
#     for i in range(m):
#       relativeIndices = np.tile(np.arange(z), (index,1))
#       moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
#       #ECAL_momentZ = np.divide(umath.inner1d(sumsz, moments), totalE)
#       ECAL_momentZ = np.einsum('ij,ij->i', sumsz, moments) / totalE
#       if i==0: ECAL_midZ = ECAL_momentZ.transpose()
#       momentZ[:,i]= ECAL_momentZ
#     return momentX, momentY, momentZ

def get_moments(sumsx, sumsy, sumsz, totalE, m, x=51, y=51, z=25):
#  # Check for NaNs and Infs in the result
#     assert not torch.isnan(result).any(), "Result contains NaNs"
#     assert not torch.isinf(result).any(), "Result contains Infs"

    totalE = torch.squeeze(totalE)
    index = sumsx.shape[0]
    momentX = torch.zeros((index, m), device=sumsx.device)
    momentY = torch.zeros((index, m), device=sumsy.device)
    momentZ = torch.zeros((index, m), device=sumsz.device)

    ECAL_midX = torch.zeros(index, device=sumsx.device)
    ECAL_midY = torch.zeros(index, device=sumsy.device)
    ECAL_midZ = torch.zeros(index, device=sumsz.device)

    for i in range(m):
        relativeIndices = torch.arange(x, device=sumsx.device).expand(index, -1)
        moments = torch.pow((relativeIndices.transpose(0, 1) - ECAL_midX).transpose(0, 1), i + 1)
        ECAL_momentX = torch.einsum('ij,ij->i', sumsx, moments) / totalE
        if i == 0: ECAL_midX = ECAL_momentX.clone()
        momentX[:, i] = ECAL_momentX

    for i in range(m):
        relativeIndices = torch.arange(y, device=sumsy.device).expand(index, -1)
        moments = torch.pow((relativeIndices.transpose(0, 1) - ECAL_midY).transpose(0, 1), i + 1)
        ECAL_momentY = torch.einsum('ij,ij->i', sumsy, moments) / totalE
        if i == 0: ECAL_midY = ECAL_momentY.clone()
        momentY[:, i] = ECAL_momentY

    for i in range(m):
        relativeIndices = torch.arange(z, device=sumsz.device).expand(index, -1)
        moments = torch.pow((relativeIndices.transpose(0, 1) - ECAL_midZ).transpose(0, 1), i + 1)
        ECAL_momentZ = torch.einsum('ij,ij->i', sumsz, moments) / totalE
        if i == 0: ECAL_midZ = ECAL_momentZ.clone()
        momentZ[:, i] = ECAL_momentZ

    return momentX, momentY, momentZ


def generate(g, index, cond, latent=256, concat=2, batch_size=64):
    energy_labels=np.expand_dims(cond[0], axis=1)
    energy_labels = np.squeeze(energy_labels)  # Squeeze out singleton dimensions
    # Ensure energy_labels is 2D
    if energy_labels.ndim == 1:
        energy_labels = energy_labels.reshape(-1, 1)
    if len(cond)> 1: # that means we also have angle
        angle_labels = cond[1]
        if isinstance(angle_labels, torch.Tensor):
            angle_labels = angle_labels.numpy()  # Convert PyTorch tensor to NumPy array
        angle_labels = angle_labels.reshape(-1, 1)  # Ensure angle_labels is 2D
        if concat==1:
            noise = np.random.normal(0, 1, (index, latent-1))  
            noise = energy_labels * noise
            gen_in = np.concatenate((angle_labels.reshape(-1, 1), noise), axis=1)
        elif concat==2:
            noise = np.random.normal(0, 1, (index, latent-2))
            #print(energy_labels.shape, angle_labels.shape, noise.shape)
            gen_in = np.concatenate((energy_labels, angle_labels.reshape(-1, 1), noise), axis=1)
        else:  
            noise = np.random.normal(0, 1, (index, 2, latent))
            angle_labels=np.expand_dims(angle_labels, axis=1)
            gen_in = np.concatenate((energy_labels, angle_labels), axis=1)
            gen_in = np.expand_dims(gen_in, axis=2)
            gen_in = gen_in * noise
    else:
      noise = np.random.normal(0, 1, (index, latent))
      #energy_labels=np.expand_dims(energy_labels, axis=1)
      gen_in = energy_labels * noise

    # Prepare data for model input (assuming gen_in is a NumPy array)
    gen_in_tensor = torch.from_numpy(gen_in).float()  # Convert to PyTorch tensor and ensure type is float

    # If your model expects data on a specific device (e.g., GPU), move the tensor to that device
    device = next(g.parameters()).device
    gen_in_tensor = gen_in_tensor.to(device)
    #gen_in_tensor = gen_in_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

    # Disable gradient calculations for inference
    with torch.no_grad():
        # Directly call the model to get predictions
        generated_images = g(gen_in_tensor)

    # Convert predictions back to a suitable format if necessary (e.g., NumPy array)
    generated_images = generated_images.cpu().numpy()  # Assuming you want a NumPy array

    #generated_images = g.predict(gen_in, verbose=False, batch_size=batch_size)
    return generated_images

# def measPython(image): # Working version:p1 and p2 are not used. 3D angle with barycenter as reference point
#     image = np.squeeze(image)
#     x_shape= image.shape[1]
#     y_shape= image.shape[2]
#     z_shape= image.shape[3]

#     sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
#     indexes = np.where(sumtot > 0)
#     amask = np.ones_like(sumtot)
#     amask[indexes] = 0

#     masked_events = np.sum(amask) # counting zero sum events

#     x_ref = np.sum(np.sum(image, axis=(2, 3)) * np.expand_dims(np.arange(x_shape) + 0.5, axis=0), axis=1)
#     y_ref = np.sum(np.sum(image, axis=(1, 3)) * np.expand_dims(np.arange(y_shape) + 0.5, axis=0), axis=1)
#     z_ref = np.sum(np.sum(image, axis=(1, 2)) * np.expand_dims(np.arange(z_shape) + 0.5, axis=0), axis=1)

#     x_ref[indexes] = x_ref[indexes]/sumtot[indexes]
#     y_ref[indexes] = y_ref[indexes]/sumtot[indexes]
#     z_ref[indexes] = z_ref[indexes]/sumtot[indexes]

#     sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z

#     x = np.expand_dims(np.arange(x_shape) + 0.5, axis=0)
#     x = np.expand_dims(x, axis=2)
#     y = np.expand_dims(np.arange(y_shape) + 0.5, axis=0)
#     y = np.expand_dims(y, axis=2)
#     x_mid = np.sum(np.sum(image, axis=2) * x, axis=1)
#     y_mid = np.sum(np.sum(image, axis=1) * y, axis=1)
#     indexes = np.where(sumz > 0)

#     zmask = np.zeros_like(sumz)
#     zmask[indexes] = 1
#     zunmasked_events = np.sum(zmask, axis=1)

#     x_mid[indexes] = x_mid[indexes]/sumz[indexes]
#     y_mid[indexes] = y_mid[indexes]/sumz[indexes]
#     z = np.arange(z_shape) + 0.5# z indexes
#     x_ref = np.expand_dims(x_ref, 1)
#     y_ref = np.expand_dims(y_ref, 1)
#     z_ref = np.expand_dims(z_ref, 1)

#     zproj = np.sqrt((x_mid-x_ref)**2.0  + (z - z_ref)**2.0)
#     m = (y_mid-y_ref)/zproj
#     z = z * np.ones_like(z_ref)
#     indexes = np.where(z<z_ref)
#     m[indexes] = -1 * m[indexes]
#     ang = (math.pi/2.0) - np.arctan(m)
#     ang = ang * zmask

#     #ang = np.sum(ang, axis=1)/zunmasked_events #mean
#     ang = ang * z # weighted by position
#     sumz_tot = z * zmask
#     ang = np.sum(ang, axis=1)/np.sum(sumz_tot, axis=1)

#     indexes = np.where(amask>0)
#     ang[indexes] = 100.
#     return ang


def measPython(image):
    print("this the image shape: " + str(image.shape))
    image = torch.squeeze(image)
    x_shape, y_shape, z_shape = image.shape[1], image.shape[2], image.shape[3]

    sumtot = torch.sum(image, dim=(1, 2, 3))  # sum of events
    indexes = torch.where(sumtot > 0)
    amask = torch.ones_like(sumtot)
    amask[indexes] = 0
    masked_events = torch.sum(amask) # counting zero sum events

    x_ref = torch.sum(torch.sum(image, dim=(2, 3)) * torch.arange(x_shape, device=image.device).float() + 0.5, dim=1)
    y_ref = torch.sum(torch.sum(image, dim=(1, 3)) * torch.arange(y_shape, device=image.device).float() + 0.5, dim=1)
    z_ref = torch.sum(torch.sum(image, dim=(1, 2)) * torch.arange(z_shape, device=image.device).float() + 0.5, dim=1)

    x_ref[indexes] = x_ref[indexes] / sumtot[indexes]
    y_ref[indexes] = y_ref[indexes] / sumtot[indexes]
    z_ref[indexes] = z_ref[indexes] / sumtot[indexes]

    sumz = torch.sum(image, dim=(1, 2))  # sum for x,y planes going along z

    x = (torch.arange(x_shape, device=image.device).float() + 0.5).unsqueeze(0).unsqueeze(-1)
    y = (torch.arange(y_shape, device=image.device).float() + 0.5).unsqueeze(0).unsqueeze(-1)

    x_mid = torch.sum(torch.sum(image, dim=2) * x, dim=1)
    y_mid = torch.sum(torch.sum(image, dim=1) * y, dim=1)

    indexes = torch.where(sumz > 0)
    zmask = torch.zeros_like(sumz)
    zmask[indexes] = 1
    zunmasked_events = torch.sum(zmask, dim=1)

    x_mid[indexes] = x_mid[indexes] / sumz[indexes]
    y_mid[indexes] = y_mid[indexes] / sumz[indexes]
    z = torch.arange(z_shape, device=image.device).float() + 0.5

    x_ref = x_ref.unsqueeze(1)
    y_ref = y_ref.unsqueeze(1)
    z_ref = z_ref.unsqueeze(1)

    zproj = torch.sqrt((x_mid - x_ref) ** 2.0 + (z - z_ref) ** 2.0)
    m = (y_mid - y_ref) / zproj
    z = z * torch.ones_like(z_ref)
    indexes = torch.where(z < z_ref)
    m[indexes] = -1 * m[indexes]
    ang = (math.pi / 2.0) - torch.atan(m)
    ang = ang * zmask

    # Weighted by position and calculating mean
    ang = ang * z
    sumz_tot = z * zmask
    ang = torch.sum(ang, dim=1) / torch.sum(sumz_tot, dim=1)

    indexes = torch.where(amask>0)
    ang[indexes] = 100.

    return ang


# Get all files Without Dividing into Test and Train
def GetDataFiles(FileSearch="/data/LCD/*/*.h5",
                 Particles=[], MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))
    print ("Found {} files. ".format(len(Files)))
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]
                if MaxFiles>0:
                    if FileCount>MaxFiles:
                        break
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)
    return Sample


# get variable angle data
def GetAngleData(datafile, thresh=0, angtype='eta', offset=0.0, num_events=10000):
    #get data for training                                                                                        
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL')[:num_events])
    Y=np.array(f.get('energy')[:num_events])
    if angtype in f:
      ang = np.array(f.get(angtype)[:num_events])
    else:
      if angtype=='mtheta':
        ang = measPython(X)
    ang = ang + offset
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X=X[indexes]
    Y=Y[indexes]
    ang=ang[indexes]
    X = np.expand_dims(X, axis=-1)
    return X, Y, ang 

# Get sorted data for variable angle
def get_sorted_angle(datafiles, energies, flag=False, num_events1=5000, num_events2=2000, tolerance1=5, tolerance2=0.5, Data=GetAngleData, angtype='theta', thresh=0.0, offset=0.0):
    srt = {}
    for index, datafile in enumerate(datafiles):
       data = Data(datafile, thresh = thresh, angtype=angtype, offset= offset)
       X = data[0]
       sumx = np.sum(np.squeeze(X), axis=(1, 2, 3))
       indexes= np.where(sumx>0)
       X=X[indexes]
       Y = data[1]
       Y=Y[indexes]
       angle = data[2]
       angle=angle[indexes]
       for energy in energies:
           if index== 0:
              if energy == 0:
                 srt["events_act" + str(energy)] = X # More events in random bin                                                                                                                                  
                 srt["energy" + str(energy)] = Y
                 srt["angle" + str(energy)] = angle
                 if srt["events_act" + str(energy)].shape[0] > num_events1:
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                    srt["angle" + str(energy)]= srt["angle" + str(energy)][:num_events1]
                    print('For {} energy {} events were found in first file'.format(energy, srt["events_act" + str(energy)].shape[0]))
                    flag=False
              else:
                 indexes = np.where((Y > energy - tolerance1 ) & ( Y < energy + tolerance1))
                 srt["events_act" + str(energy)] = X[indexes]
                 srt["energy" + str(energy)] = Y[indexes]
                 srt["angle" + str(energy)]= angle[indexes]
                 print('For {} energy {} events were found in first file'.format(energy, srt["events_act" + str(energy)].shape[0]))
           else:
              if energy == 0:
                 if flag:
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X, axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y, axis=0)
                    srt["angle" + str(energy)]=np.append(srt["angle" + str(energy)], angle, axis=0)
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                       srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                       srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                       srt["angle" + str(energy)]=srt["angle" + str(energy)][:num_events1]
                       flag=False
              else:
                 if srt["events_act" + str(energy)].shape[0] < num_events2:
                    indexes = np.where((Y > energy - tolerance1 ) & ( Y < energy + tolerance1))
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X[indexes], axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y[indexes], axis=0)
                    srt["angle" + str(energy)]=np.append(srt["angle" + str(energy)], angle[indexes], axis=0)
                 else:
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                    srt["angle" + str(energy)]=srt["angle" + str(energy)][:num_events1]
              print('For {} energy {} events were loaded'.format(energy, srt["events_act" + str(energy)].shape[0]))
    return srt


    # make a directory
def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception


