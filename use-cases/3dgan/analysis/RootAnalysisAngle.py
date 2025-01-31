#!/usr/bin/env python
# -*- coding: utf-8 -*-
## This script loads weights into architectures for generator and discriminator. Different Physics quantities are then calculated and plotted for a 100-200 GeV events from LCD variable angle dataset##

from utils.GANutils import perform_calculations_angle, perform_calculations_multi, generate_pt   # to calculate different Physics quantities
import utils.GANutils as gan
from utils.RootPlotsGAN import get_plots_angle, get_plots_multi         # to make plots with ROOT
import os
import h5py
import numpy as np
import math
import sys
import argparse
import torch
from torch.autograd import profiler
from pl_3dgan_models_v1 import *
#from memory_profiler import profile
if os.environ.get('HOSTNAME') == 'tlab-gpu-oldeeptector.cern.ch': # Here a check for host can be used        
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech                                                                                
except:
    pass

#sys.path.insert(0,'../')

def main():
   parser = get_parser()
   params = parser.parse_args()

   datapath =params.datapath
   latent = params.latentsize
   particle= params.particle
   angtype= params.angtype
   plotdir= params.outdir + '/'
   sortdir= params.sortdir
   gendir= params.gendir
   discdir= params.discdir
   nbEvents= params.nbEvents
   binevents= params.binevents
   moments= params.moments
   addloss= params.addloss
   angloss= params.angloss
   concat= params.concat
   cell= params.cell
   corr= params.corr
   test= params.test
   stest= params.stest
   save_data= params.save_data
   read_data= params.read_data
   save_gen= params.save_gen
   read_gen= params.read_gen
   save_disc= params.save_disc
   read_disc= params.read_disc
   ifpdf= params.ifpdf
   grid = params.grid
   leg= 1 if stest else params.leg
   statbox= params.statbox
   mono= params.mono
   labels = params.labels if isinstance(params.labels, list) else [params.labels]
   gweights= params.gweights if isinstance(params.gweights, list) else [params.gweights]
   #gweights= params.gweights 
   dweights= params.dweights if isinstance(params.dweights, list) else [params.dweights]
   #dweights= params.dweights 
   xscales= params.xscales 
   ascales= params.ascales 
   yscale= params.yscale
   xpowers = params.xpower 
   thresh = params.thresh
   dformat = params.dformat
   ang = params.ang

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f'Using device: {device}')

   
   #Architecture 
   if ang:
     from pl_3dgan_models_v1 import Generator, Discriminator
     #from AngleArch3dGAN import generator, discriminator
     dscale=50.
     if not xscales:
       xscales=1.
     if not xpowers:
       xpowers = 0.85
     if not latent:
       latent = 256
     if not ascales:
       ascales = 1
       
     if datapath=='reduced':
       datapath = "/eos/user/k/ktsolaki/data/3dgan_100_200_data/*.h5"   #"/storage/group/gpu/bigdata/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV
       events_per_file = 5000
       energies = [0, 110, 150, 190]
     elif datapath=='full':
       #datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
       datapath = "/eos/user/k/ktsolaki/data/2_500_fullEnergy_data/*scan_RandomAngle_*.h5"
       events_per_file = 10000
       energies = [0, 50, 100, 200, 250, 300, 400, 500]
       #energies =[0, 160, 200, 250, 290]

   else:
     from EcalEnergyGan import generator, discriminator
     dscale=1
     if not xscales:
       xscales=100.
     if not xpowers:
       xpowers = 1.
     if not latent:
       latent =200
     if not ascales:
       ascales = 1

     if datapath=='full':
       datapath ='/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5'
     events_per_file = 10000
     energies = [0, 50, 100, 200, 250, 300, 400, 500]  
   if tlab: 
     #Weights
     dweights=["/gkhattak/weights/3dgan_weights_gan_training/params_discriminator_epoch_059.hdf5"]
     gweights= ["/gkhattak/weights/3dgan_weights_gan_training/params_generator_epoch_059.hdf5"]
     datapath = '/eos/user/g/gkhattak/VarAngleData/*Measured3ThetaEscan/*.h5'
     events_per_file = 5000
   xscales = xscales if isinstance(xscales, list) else [xscales]*len(gweights)
   ascales = ascales if isinstance(ascales, list) else [ascales]*len(gweights)
   xpowers = xpowers if isinstance(xpowers, list) else [xpowers]*len(gweights)
  #  if not isinstance(xpowers, list):
  #   # If xpowers isn't already a list, replicate it len(gweights) times
  #   xpowers = [xpowers] * len(gweights)

  #  # If there is exactly one generator weight, make xpowers a float
  #  if len(xpowers) == 1:
  #   xpowers = xpowers[0]
   angles = [62, 90, 118]
   flags =[test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]

   if ang:
    #  d = discriminator(xpowers[0], dformat=dformat)
    #  g = generator(latent, dformat=dformat)
     g = Generator(latent)
     g.to(device)
     g.eval()
     # Load PyTorch weights from .pth (or .pt)
     print(f"Loading PyTorch weights from {gweights}")
     weights_file = gweights[0]  # Grab the first element
     state_dict = torch.load(weights_file, map_location=device)
     g.load_state_dict(state_dict)

     d = Discriminator(xpowers[0])
     d.to(device)
     d.eval()
     # Load PyTorch weights from .pth (or .pt)
     print(f"Loading PyTorch disc weights from {dweights}")
     dweights_file = dweights[0]  # Grab the first element
     state_dict = torch.load(dweights_file, map_location=device)
     d.load_state_dict(state_dict)
   
     var= perform_calculations_angle(g, d, gweights, dweights, energies, angles, 
                datapath, sortdir, gendir, discdir, nbEvents, binevents, moments, xscales, xpowers,
                ascales, dscale, flags, latent, particle, events_per_file=events_per_file, thresh=thresh, angtype=angtype, offset=0.0,
                angloss=angloss, addloss=addloss, concat=concat#, Data=GetDataAngle2 
                , pre =taking_power, post =inv_power  # Adding other preprocessing, Default is simple scaling                 
    )
   
     get_plots_angle(var, labels, plotdir, energies, angles, angtype, moments, 
               len(gweights), ifpdf=ifpdf, grid=grid, stest=stest, angloss=angloss, 
                addloss=addloss, cell=cell, corr=corr, leg=leg, statbox=statbox, mono=mono)
   else:
    #  d = discriminator( dformat=dformat)
    #  g = generator(latent,  dformat=dformat)
     var= perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendir, discdir, num_data=nbEvents
         , num_events=binevents, m=moments, scales=xscales, thresh=thresh, flags=flags, latent=latent, particle=particle, dformat=dformat)
     get_plots_multi(var, labels, plotdir, energies, moments, len(gweights), cell=cell, corr=corr,
                     ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)

def sqrt(n, scale=1):
   return np.sqrt(n * scale)

def square(n, scale=1):
   return np.square(n)/scale
        
def taking_power(n, scale=1.0, power=1.0):
   return(np.power(n * scale, power))

def inv_power(n, scale=1.0, power=1.0):
   return(np.power(n, 1.0/power))/scale

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, help='size of random N(0, 1) latent space to sample')    #parser.add_argument('--model', action='store', default=AngleArch3dgan, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='full', help='HDF5 files to train from.')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle used.')
    parser.add_argument('--outdir', action='store', type=str, default='/eos/user/k/ktsolaki/misc/3dgan_pytorch/root_analysis_plots/', help='Directory to store the analysis plots.')
    parser.add_argument('--sortdir', action='store', type=str, default='/eos/user/k/ktsolaki/misc/3dgan_pytorch/root_analysis_plots/sorted/', help='Directory to store sorted data.')
    parser.add_argument('--gendir', action='store', type=str, default='/eos/user/k/ktsolaki/misc/3dgan_pytorch/root_analysis_plots/gen_results/', help='Directory to store the generated images.')
    parser.add_argument('--discdir', action='store', type=str, default='/eos/user/k/ktsolaki/misc/3dgan_pytorch/root_analysis_plots/disc_results/', help='Directory to store the discriminator outputs.')
    parser.add_argument('--nbEvents', action='store', type=int, default=10, help='Max limit for events used for Testing')
    parser.add_argument('--eventsperfile', action='store', type=int, default=5000, help='Number of events in a file')
    parser.add_argument('--binevents', action='store', type=int, default=50, help='Number of events in each bin')
    parser.add_argument('--moments', action='store', type=int, default=3, help='Number of moments to compare')
    parser.add_argument('--addloss', action='store', type=int, default=0, help='If using bin count loss')
    parser.add_argument('--angloss', action='store', type=int, default=1, help='Number of loss terms related to angle')
    parser.add_argument('--concat', action='store', type=int, default=2, help='Modes related to combining conditions with latent 0)not cancatenated.. 1)concatenate angle...3) concatenate energy and angle')
    parser.add_argument('--cell', action='store', type=int, default=0, help='Whether to plot cell energies..0)Not plotted...1)Only for bin with uniform spectrum.....2)For all energy bins')
    parser.add_argument('--corr', action='store', type=int, default=2, help='Plot correlation plots..0)Not plotted...1)detailed features ...2) reduced features.. 3) reduced features for each energy bin')
    parser.add_argument('--test', action='store', type=int, default=1,  help='Use Test data')
    parser.add_argument('--stest', action='store', type=int, default=0, help='Statistics test for shower profiles')
    parser.add_argument('--save_data', default=False, action='store_true', help='Save sorted data')
    parser.add_argument('--read_data', default=False, action='store_true', help='Get saved and sorted data')
    parser.add_argument('--save_gen', default=False, action='store_true', help='Save generated images')
    parser.add_argument('--read_gen', default=False, action='store_true', help='Get saved generated images')
    parser.add_argument('--save_disc', default=False, action='store_true', help='Save discriminator output')
    parser.add_argument('--read_disc', default=False, action='store_true', help='Get discriminator output')
    parser.add_argument('--ifpdf', default=1, type=int, action='store', help='Whether generate pdf plots or .C plots')
    parser.add_argument('--grid', default=0, type=int, action='store', help='set grid')
    parser.add_argument('--leg', default=0, type=int, action='store', help='add legends')
    parser.add_argument('--statbox', default=0, type=int, action='store', help='add statboxes')
    parser.add_argument('--mono',  default=False, action='store_true', help='changing line style as well as color for comparison')
    parser.add_argument('--labels',  default='', type=str, nargs='+', help='id for particular weights when comparing multiple training')
    parser.add_argument('--gweights', action='store', type=str, nargs='+', default=['../weights/3dgan_weights_wt_aux/params_generator_epoch_035.hdf5'], help='comma delimited list for paths to Generator weights.')
    parser.add_argument('--dweights', action='store', type=str, nargs='+', default=['../weights/3dgan_weights_wt_aux/params_discriminator_epoch_035.hdf5'], help='comma delimited list for paths to Discriminator weights')
    parser.add_argument('--xscales', action='store', type=int, nargs='+', help='Multiplication factors for cell energies')
    parser.add_argument('--ascales', action='store', type=int, nargs='+', help='Multiplication factors for angles')
    parser.add_argument('--yscale', action='store', default=100., help='Division Factor for Primary Energy.')
    parser.add_argument('--xpower', action='store', help='Power of cell energies')
    parser.add_argument('--thresh', action='store', default=0, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
    parser.add_argument('--ang', action='store', default=1, type=int, help='if variable angle')
    return parser
   
# If using reduced Ecal 25x25x25 then use the following function as argument to perform_calculations_angle, Data=GetAngleDataEta_reduced
def GetAngleDataEta_reduced(datafile, thresh=1e-6):
    #get data for training                                                                                        
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))[:, 13:38, 13:38, :]
    Y=np.array(f.get('energy'))
    eta = np.array(f.get('eta')) + 0.6
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, eta, ecal
#get data for training                                                                                                       
def GetDataAngle2(datafile, xscale =1, xpower=1, yscale = 1, angscale=1, angtype='theta', offset=0.0, thresh=1e-4, daxis=-1):
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))* xscale
    Y=np.array(f.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where((ecal > 10.0) & (Y > 150) & (Y < 350))
    print('From {} events {} passed'.format(Y.shape[0], indexes[0].shape[0]))
    X=X[indexes]
    Y=Y[indexes]
    ecal = ecal[indexes]
    if angtype in f:
      ang = np.array(f.get(angtype))[indexes]
    else:
      ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal

if __name__ == "__main__":
    main()
