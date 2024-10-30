# 2D projections for 3 D events from GAN and GEANT4
from os import path
import os
import sys
import numpy as np
import argparse
sys.path.insert(0,'../')

#from pl_3dgan_models_v1 import *
# I do not need those 2 lines if the files are in the same directory
import sys
sys.path.append('/p/project1/intertwin/tsolaki1/itwinai/use-cases/3dgan/')
from model import *

from analysis_utils import *
import torch
import utils.GANutils as gan
import utils.RootPlotsGAN as pl
# try:
#     import setGPU #if Caltech
# except:
#     pass

def main():
   parser = get_parser()
   params = parser.parse_args()

   datapath =params.datapath
   events_per_file = params.eventsperfile
   energies = params.energies if isinstance(params.energies, list) else [params.energies]
   latent = params.latentsize
   particle= params.particle
   angtype= params.angtype
   plotsdir= params.outdir+'/'
   concat= params.concat
   gweight= params.gweight 
   xscale= params.xscale
   ascale= params.ascale
   yscale= params.yscale
   xpower= params.xpower 
   thresh = params.thresh
   dformat = params.dformat
   ang = params.ang
   ifC = params.ifC
   num = params.num
   gan.safe_mkdir(plotsdir) # make plot directory
   tolerance2=0.05
   opt="colz"
   if ang:
     #from AngleArch3dGAN import generator
     dscale=50.
     if not xscale:
       xscale=1.
     if not xpower:
       xpower = 0.85
     if not latent:
       latent = 256
     if not ascale:
       ascale = 1

     if datapath=='reduced':
       #datapath = "/storage/group/gpu/bigdata/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV
       datapath = "/p/scratch/intertwin/datasets/cern/*.h5" # just one file for testing the script
       events_per_file = 5000
       energies = [0, 110, 150, 190]
     elif datapath=='full':
       datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
       events_per_file = 10000
       energies = [50, 100, 200, 300, 400, 500]
     else: 
       datapath = datapath + "/*scan/*scan_RandomAngle_*.h5"
     thetas = [62, 90, 118]      
   else:
     #from EcalEnergyGan import generator
     dscale=1
     if not xscale:
       xscale=100.
     if not xpower:
       xpower = 1
     if not latent:
       latent = 200
     if not ascale:
       ascale = 1

     if datapath=='full':
       datapath ='/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5'
       energies = [50, 100, 200, 300, 400, 500]
     else:
       datapath =  datapath+ "/*scan/*scan_*.h5"
     events_per_file = 10000    

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f'Using device: {device}')
   
   datafiles = GetDataFiles(datapath, Particles=[particle]) # get list of files
   if ang:
     var = get_sorted_angle(datafiles[-2:], energies, True, num_events1=1000, num_events2=1000, angtype=angtype, thresh=0.0)#get data from last two files
     g = Generator(latent) #generator(latent, dformat=dformat)
     #g.load_weights(gweight)
     g = g.to(device)
     g.load_state_dict(torch.load(gweight))
     for energy in energies: # for each energy bin
        edir = os.path.join(plotsdir, 'energy{}'.format(energy))
        safe_mkdir(edir)
        rad = np.radians(thetas)
        for index, a in enumerate(rad): # for each angle bin
          adir = os.path.join(edir, 'angle{}'.format(thetas[index]))
          safe_mkdir(adir)
          if a==0:
            var["events_act" + str(energy) + "ang_" + str(index)] = var["events_act" + str(energy)]/dscale # data in units of GeV * dscale
            var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)] # energy labels
            var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)]  # angle labels
            var["index" + str(energy)+ "ang_" + str(index)] = var["events_act" + str(energy) + "ang_" + str(index)].shape[0] # number of events
          else:
            indexes = np.where(((var["angle" + str(energy)]) > a - tolerance2) & ((var["angle" + str(energy)]) < a + tolerance2)) # all events with angle within a bin                                     
            var["events_act" + str(energy) + "ang_" + str(index)] = var["events_act" + str(energy)][indexes]/dscale
            var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)][indexes]
            var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)][indexes]
            var["index" + str(energy)+ "ang_" + str(index)] = var["events_act" + str(energy) + "ang_" + str(index)].shape[0]

          var["events_act" + str(energy) + "ang_" + str(index)] = applythresh(var["events_act" + str(energy) + "ang_" + str(index)], thresh) # remove energies below threshold
          var["events_gan" + str(energy) + "ang_" + str(index)]= generate(g, var["index" + str(energy)+ "ang_" + str(index)],  # generate events
                                                                           [var["energy" + str(energy)+ "ang_" + str(index)]/yscale,
                                                                            (var["angle"+ str(energy)+ "ang_" + str(index)]) * ascale], latent, concat=2)
          var["events_gan" + str(energy) + "ang_" + str(index)]= inv_power(var["events_gan" + str(energy) + "ang_" + str(index)], xpower=xpower)/dscale # post processing
          var["events_gan" + str(energy) + "ang_" + str(index)]= applythresh(var["events_gan" + str(energy) + "ang_" + str(index)], thresh) # remove energies below threshold
          for n in np.arange(min(num, var["index" + str(energy)+ "ang_" + str(index)])): # plot events
            pl.PlotEvent2(var["events_act" + str(energy) + "ang_" + str(index)][n], var["events_gan" + str(energy) + "ang_" + str(index)][n],
                         var["energy" + str(energy) + "ang_" + str(index)][n],
                         var["angle" + str(energy) + "ang_" + str(index)][n],
                          os.path.join(adir, 'Event{}'.format(n)), n, opt=opt, logz=1, ifC=ifC)

  #  else:
  #    g = generator(latent, dformat=dformat)
  #    g.load_weights(gweight)
  #    var = gan.get_sorted(datafiles[-2:], energies, True, num_events1=50, num_events2=50, thresh=0.0)#get data from last two files
  #    for energy in energies: # for each energy bin
  #       edir = os.path.join(plotsdir, 'energy{}'.format(energy))
  #       gan.safe_mkdir(edir)
  #       var["events_act" + str(energy)] = var["events_act" + str(energy)]/dscale # data in units of GeV * dscale
  #       var["energy" + str(energy)] = var["energy" + str(energy)] # energy labels
  #       var["index" + str(energy)] = var["events_act" + str(energy)].shape[0] # number of events
  #       var["events_act" + str(energy)] = applythresh(var["events_act" + str(energy)], thresh)
  #       var["events_gan" + str(energy)]= gan.generate(g, var["index" + str(energy)],
  #                                                     [var["energy" + str(energy)]/yscale], latent=latent)
  #       var["events_gan" + str(energy)]= var["events_gan" + str(energy)]/(xscale* dscale) # post processing
  #       var["events_gan" + str(energy)]= applythresh(var["events_gan" + str(energy)], thresh)# remove energies below threshold
  #       for n in np.arange(min(num, var["index" + str(energy)])): # plot events
  #           pl.PlotEvent2(var["events_act" + str(energy)][n], var["events_gan" + str(energy)][n],
  #                        var["energy" + str(energy)][n],
  #                        None,
  #                        os.path.join(edir, 'Event{}'.format(n)), n, opt=opt, logz=1)

   print('Plots are saved in {}'.format(plotsdir))

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, help='size of random N(0, 1) latent space to sample')    #parser.add_argument('--model', action='store', default=AngleArch3dgan, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='reduced', help='HDF5 files to train from.') #'full'
    parser.add_argument('--eventsperfile', action='store', type=int, default=1000, help='Number of events in a file')
    parser.add_argument('--energies', action='store', type=int, nargs='+', default=[0], help='Energy bins')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle used.')
    parser.add_argument('--outdir', action='store', type=str, default='/eos/user/k/ktsolaki/misc/results/2d_projections/torch', help='Directory to store the analysis plots.') #'results/2d_projections'
    parser.add_argument('--nbEvents', action='store', type=int, default=100000, help='Max limit for events used for Testing')
    parser.add_argument('--concat', action='store', type=int, default=2, help='Modes related to combining conditions with latent 0)not cancatenated.. 1)concatenate angle...3) concatenate energy and angle')
    parser.add_argument('--gweight', action='store', type=str, default='../weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5', help='Generator weights')
    parser.add_argument('--xscale', action='store', type=int, help='Multiplication factors for cell energies')
    parser.add_argument('--ascale', action='store', type=int, help='Multiplication factors for angles')
    parser.add_argument('--yscale', action='store', default=100., help='Division Factor for Primary Energy')
    parser.add_argument('--xpower', action='store', help='Power of cell energies')
    parser.add_argument('--thresh', action='store', default=0, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
    parser.add_argument('--ang', action='store', default=1, type=int, help='if variable angle')
    parser.add_argument('--ifC', action='store', default=0, type=int, help='Generate .C files')
    parser.add_argument('--num', action='store', default=10, type=int, help='number of events to plot')
    return parser

def power(n, xscale=1, xpower=1):
   return np.power(n/xscale, xpower)

def inv_power(n, xscale=1, xpower=1):
   return np.power(n, 1./xpower) / xscale

def applythresh(n, thresh):
   n[n<thresh]=0
   return n

if __name__ == "__main__":
    main()


   
