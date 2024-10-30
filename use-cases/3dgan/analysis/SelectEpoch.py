#!/usr/bin/env python
# -*- coding: utf-8 -*-   

# This file tests the training results with the optimization function

from __future__ import print_function
import sys
import h5py
import os
import numpy as np
import glob
#import numpy.core.umath_tests as umath
import time
import math
import ROOT
import argparse
from scipy import stats
from scipy.stats import wasserstein_distance as wass
from pl_3dgan_models_v1 import *
from analysis_utils import *
# if '.cern.ch' in os.environ.get('HOSTNAME'): # Here a check for host can be used
#     tlab = True
# else:
#     tlab= False

# try:
#     import setGPU #if Caltech
# except:
#     pass

import utils.GANutils as gan
import utils.ROOTutils as my
sys.path.insert(0,'../')
import re

def extract_epoch_number(name):
    match = re.search(r'epoch_(\d+)', name)  # Look for 'epoch_' followed by digits
    if match:
        return int(match.group(1))  # Return the captured number
    else:
        print(f"Could not find epoch number in the file name: {name}")
        return None

def main():
   parser = get_parser()
   params = parser.parse_args()

   latent = params.latentsize
   datapath =params.datapath
   particle= params.particle
   angtype= params.angtype
   outdir= params.outdir
   sortdir= params.sortdir
   nbEvents= params.nbEvents
   binevents= params.binevents
   start = params.start
   stop = params.stop
   sl = params.sl
   ang= params.ang
   concat= params.concat
   test= params.test
   ifpdf= params.ifpdf
   grid = params.grid
   leg= params.leg
   statbox= params.statbox
   mono= params.mono
   gweightdir= params.gweightsdir if isinstance(params.gweightsdir, list) else [params.gweightsdir]
   labels = params.labels if isinstance(params.labels, list) else [params.labels]
   xscale= params.xscale 
   yscale= params.yscale
   xpower = params.xpower
   thresh = params.thresh 
   dformat = params.dformat
   fits = params.fits if isinstance(params.fits, list) else [params.fits]
   opt = params.opt if isinstance(params.opt, list) else [params.opt]
   if ang:
     #from AngleArch3dGAN import generator # architecture
     dscale = 50.
     if not xscale:
       xscale=1.
     if not xpower:
       xpower = 0.85
     if not latent:
       latent = 256
     if datapath=='reduced':
      #  if tlab:
      #    datapath = '/gkhattak/data/*Measured3ThetaEscan/*.h5'
      #  else:
       datapath = "/p/scratch/intertwin/datasets/cern/*.h5"  # Data path for 100-200 GeV must be changed here
       events_per_file = 5000
       energies = [0, 110, 150, 190]
     else:
       if datapath=='full':
         datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech
       else:
         datapath = datapath + "/*scan/*scan_RandomAngle_*.h5"
       events_per_file = 10000
       energies =[0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
   else:
     #from EcalEnergyGan import generator
     dscale = 1.
     if not xscale:
       xscale=100.
     if not xpower:
       xpower = 1.
     if not latent:
       latent =200
     if datapath=='full':
       datapath = "/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5"
     else:
       datapath = datapath + "/*scan/*scan_*.h5"
     events_per_file = 10000
     energies =[0, 100, 200, 300, 400]

   device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
   print(f'Using device: {device}')

  #  genpaths =[]
  #  # get a list of paths for all weight directories from different trainings
  #  for gweight in gweightdir:
  #     genpaths.append( gweight +'/*generator*.hdf5')# path to weights
  #  sorted_path = 'Anglesorted'  # where sorted data is to be placed
  #  print(latent)
  #  #g= generator(latent_size=latent, dformat=dformat) # generator
  #  g = Generator(latent)
  #  g = g.to(device)
  #  #g.load_state_dict(torch.load(gweight))
  #  gen_weights=[]
  #  #gan.safe_mkdir(outdir) # make a diretory for results
  #  # list of lists of weights for each path
  #  for index, genpath in enumerate(genpaths):
  #     gen_weights.append(sorted(glob.glob(genpath))) # append all weights for a particular path
  #     gen_weights[index]=gen_weights[index][start:stop] # select weights based on start and stop
  #  epochs = []
  #  for index, gen_weight in enumerate(gen_weights):
  #    epoch=[]
  #    for i in np.arange(len(gen_weight)):
  #       name = os.path.basename(gen_weight[i])
  #       #num = int(filter(str.isdigit, name)[:-1])# get the epoch number from the weights
  #       num = int(''.join(list(filter(str.isdigit, name)))[:-1]) #converted the line above as it was causing a problem
  #       epoch.append(num) # make a list of epochs from one path
  #    epochs.append(epoch)# append this list to a list for all paths
  #    print("{} weights are found in {}".format(len(gen_weight), gweightdir[index])) 

   genpaths = []
  # get a list of paths for all weight directories from different trainings
   for gweight in gweightdir:
       genpaths.append(os.path.join(gweight, '*generator*.pth'))  # Modify to use full path

   sorted_path = 'Anglesorted'  # where sorted data is to be placed
   print(latent)

   g = Generator(latent)
   g = g.to(device)

   gen_weights = []
  # list of lists of weights for each path
   for index, genpath in enumerate(genpaths):
       gen_weights.append(sorted(glob.glob(genpath)))  # append all weights for a particular path
       gen_weights[index] = gen_weights[index][start:stop]  # select weights based on start and stop

   epochs = []
   for index, gen_weight in enumerate(gen_weights):
       epoch = []
       for i in np.arange(len(gen_weight)):
           name = os.path.basename(gen_weight[i])  # Get the base name of the file
           num = extract_epoch_number(name)  # Extract epoch number using the regex function
           if num is not None:
               epoch.append(num)  # Append the epoch number if found
       epochs.append(epoch)  # Append this list to a list for all paths
       print(f"{len(gen_weight)} weights are found in {gweightdir[index]}")
   
   # Get results and errors
   result, error_bar = GetResults(opt, outdir, gen_weights, g, datapath, sorted_path, particle
            , scale=xscale, power=xpower, thresh=thresh, ang=ang, concat=concat, latent=latent
            , energies=energies, nbEvents=nbEvents, eventsperfile=events_per_file, dscale=dscale
             )
   # make plots
   for i, op in enumerate(opt):
      r=[]
      for res in result:
        r.append(res[i::len(opt)])     
      if op=='mre':
         PlotResultsErrors(r, error_bar, op, outdir, epochs, fits, sl=sl, labels=labels, ang=ang)
      else:
         PlotResultsRoot(r, op, outdir, epochs, fits, sl=sl, labels=labels, ang=ang)
 
def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='reduced', help='HDF5 files to train from.')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle used.')
    parser.add_argument('--outdir', action='store', type=str, default='results/best_epoch_gan_training/', help='Directory to store the analysis plots.')
    parser.add_argument('--sortdir', action='store', type=str, default='SortedData', help='Directory to store sorted data.')
    parser.add_argument('--nbEvents', action='store', type=int, default=5000, help='Max limit for events used for Testing')
    parser.add_argument('--eventsperfile', action='store', type=int, default=5000, help='Number of events in a file')
    parser.add_argument('--binevents', action='store', type=int, default=10000, help='Number of events in each bin')
    parser.add_argument('--start', action='store', type=int, default=0, help='plot beginning from epoch')
    parser.add_argument('--stop', action='store', type=int, default=500, help='plot till epoch')
    parser.add_argument('--sl', action='store', type=int, default=10, help='select from last n epochs')
    parser.add_argument('--opt', default='mre', type=str, nargs='+', help='selction criterion: mre, chi2, ks')
    parser.add_argument('--ang', action='store', type=int, default=1, help='if inclusing angle angle')
    parser.add_argument('--concat', action='store', type=int, default=2, help='Modes related to combining conditions with latent 0)not cancatenated.. 1)concatenate angle...3) concatenate energy and angle')
    parser.add_argument('--test', action='store_false', default=True, help='Use Test data')
    parser.add_argument('--ifpdf', action='store_false', default=True, help='Whether generate pdf plots or .C plots')
    parser.add_argument('--grid', action='store_true', default=False, help='set grid')
    parser.add_argument('--leg', action='store_true', default=False, help='add legends')
    parser.add_argument('--statbox', action='store_true', default=False, help='add statboxes')
    parser.add_argument('--mono', action='store_true', default=False, help='changing line style as well as color for comparison')
    parser.add_argument('--gweightsdir', action='store', type=str, nargs='+', default='../weights/3dgan_weights_gan_training_epsilon_2_500GeV/', help='paths to Generator weights.')
    parser.add_argument('--labels', action='store', type=str, nargs='+', default='', help='labels for Generator weights.')
    parser.add_argument('--xscale', action='store', type=int, help='Multiplication factors for cell energies')
    parser.add_argument('--yscale', action='store', help='Division Factor for Primary Energy.')
    parser.add_argument('--xpower', action='store', help='Power of cell energies')
    parser.add_argument('--thresh', action='store', type=int, default=0, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
    parser.add_argument('--fits', action='store', type=str, nargs='+', default=[], help='fits to use')
    return parser


def sqrt(n, scale=1):
    return np.sqrt(n * scale)

def square(n, scale=1):
    return np.square(n)/scale

def taking_power(n, xscale=1, power=1):
    return np.power(n * xscale, power)

def inv_power(n, xscale=1, power=1.):
    return np.power(n, 1./power)/xscale

def PlotResultsErrors(results, error_bars, opt, resultdir, epochs, fits, labels=[''], sl=10, ang=1):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    legend = ROOT.TLegend(.4, .7, .89, .89)
    legend.SetTextSize(0.028)
    legend.SetBorderSize(0)
    mg=ROOT.TMultiGraph()
    colors = [4, 6, 7, 8, 3]
    graphs=[]
    for index, result in enumerate(results):
       num = len(result)
       sf_e  = np.zeros((num))
       er = np.zeros((num))
       epoch = np.zeros((num))
      
       mins =1000
       mins_n =0
       j=0
       for i, item in enumerate(result):
         if item!=0:
           epoch[j] = epochs[index][i]       
           sf_e[j]=item
           er[j] = error_bars[index][i]
           j+=1
       if  num<sl: sl=num
       for i in range(num):
        if epoch[i] > epoch[num-1]-sl:
          if sf_e[i] <= mins:
            mins = sf_e[i]
            mins_n = epoch[i]
       er_x = 0.5 * np.zeros((num))                                         
       graphs.append(ROOT.TGraphErrors(j, epoch[:j], sf_e[:j], er_x[:j], er[:j]))
       gs = graphs[index]
       gs.SetLineColor(colors[index])
       mg.Add(gs)
       legend.AddEntry(gs, "{} (min epoch {}= {:.4f})".format(labels[index], mins_n, mins), "l")
       c1.Update()
    if opt=='chi2' or opt=='ks':
       title = "-log[{}] for sampling fraction (selecting from last {} epochs);Epochs;{}"
    else:
       title = "{} for sampling fraction (selecting from last {} epochs);Epochs;{}"
    print(title.format(opt, sl, opt))
    mg.SetTitle(title.format(opt, sl, opt))
    mg.Draw('APL')
    mg.GetYaxis().SetRangeUser(0, 0.2)
    c1.Update()
    legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "{}_result.pdf".format(opt)))


#Plots results in a root file
def PlotResultsRoot(results, opt, resultdir, epochs, fits, labels=[''], sl=10, ang=1):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    legend = ROOT.TLegend(.4, .7, .89, .89)
    legend.SetBorderSize(0)
    mg=ROOT.TMultiGraph()
    colors = [4, 6, 7, 8, 3]
    graphs=[]
    print(epochs)
    for index, result in enumerate(results):
      num = len(result)
      sf_e  = np.zeros((num))
      epoch = np.zeros((num))
      
      mins =1000
      mins_n =0
      j=0
      for i, item in enumerate(result):
        if item!=0:
          epoch[j] = epochs[index][i]       
          sf_e[j]=item
          j+=1
      if  num<sl: sl=num
      for i in range(num):
        if epoch[i] > num-sl:
          if sf_e[i] <= mins:
            mins = sf_e[i]
            mins_n = epoch[i]
                                         
      graphs.append(ROOT.TGraph(j, epoch[:j], sf_e[:j]))
      gs = graphs[index]
      gs.SetLineColor(colors[index])
      gs.SetMarkerColor(colors[index])
      gs.SetMarkerStyle(5)
      mg.Add(gs)
      legend.AddEntry(gs, "{} (min epoch {}= {:.4f})".format(labels[index], mins_n, mins), "l")
      c1.Update()
    if opt=='chi2' or opt=='ks':
      title = "-log[{}] for sampling fraction (selecting from last {} epochs);Epochs;{}"
      ymax = 1.2 * np.amax(sf_e)
    else:
      title = "{} for sampling fraction (selecting from last {} epochs);Epochs;{}"
      ymax= 0.02
    mg.SetTitle(title.format(opt, sl, opt))
    mg.Draw('ALP')
    
    mg.GetYaxis().SetRangeUser(0, ymax)
    c1.Update()
    legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "{}_result.pdf".format(opt)))

    for i, fit in enumerate(fits):
      mg.SetTitle(title.format(opt, sl, opt))  
      for index, gs in enumerate(graphs):
        gs.Fit(fit)
        gs.GetFunction(fit).SetLineColor(colors[index])
        gs.GetFunction(fit).SetLineStyle(2)

      if i == 0:
        legend.AddEntry(gs.GetFunction(fit), '{} ({} fit)'.format(opt, fit), "l")
      legend.Draw()
      c1.Update()
      c1.Print(os.path.join(resultdir, "{}_result_{}.pdf".format(opt, fit)))
    print ('The plot is saved to {}'.format(resultdir))

def preproc(n, scale=1):
    return n * scale

def postproc(n, scale=1):
    return n/scale
        
# results are obtained using metric and saved to a log file
def GetResults(opt, resultdir, gen_weights, g, datapath, sorted_path, particle="Ele", scale=100, power=1, thresh=1e-6, ang=1, concat=1, latent = 256, energies=[110, 150, 190], nbEvents=100000, eventsperfile=5000, dscale=1.):
    n = len(opt)
    results=[]
    error_bars=[]
    for index, gen_weight in enumerate(gen_weights):
       resultfile = os.path.join(resultdir,  'result_log{}'.format(index))
       resultfile = resultfile + '.txt'
       file = open(resultfile,'w')
       result, error_bar = analyse(g, False,True, gen_weight, datapath, sorted_path, opt, scale, power, particle, 
                     thresh=thresh, ang=ang, concat=concat, latent=latent, energies=energies,
                     nbEvents= nbEvents, eventsperfile=eventsperfile, dscale=dscale)
       for i, r in enumerate(result):
         if i !=0 and i % n==0:
           file.write('\n')
         file.write(str(r)+'\t')
       file.close
       #print all results together at end                                                                               
       for i in range(len(gen_weights)):                                                                                            
         print ('The results for ......',gen_weights[index][i])
         reslog = " The result = {}".format(result[n*i:n*i + n])
         if 'mre' in opt:
            reslog = reslog +' mre error {}'.format(error_bar[i])
         print (reslog)
       print ('The results are saved to {}.txt'.format(resultfile))
       results.append(result)
       error_bars.append(error_bar)
    return results, error_bars

def GetAngleData_reduced(datafile, thresh=1e-6):
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

def preproc(n, scale=1):
    return n * scale

def postproc(n, scale=1):
    return n/scale

# This function will calculate two errors derived from position of maximum along an axis and the sum of ecal along the axis
def analyse(g, read_data, save_data, gen_weights, datapath, sorted_path, optimizer, xscale=100, power=1, particle="Ele", thresh=1e-6, ang=1, latent = 256, concat=1, energies=[110, 150, 190], nbEvents=100000, eventsperfile=5000, dscale=1.):
   print ("Started")
   
   Test = True
   m = 2
   var = {}
   sorted_path= sorted_path 
   total=0   
   Trainfiles, Testfiles = gan.DivideFiles(datapath, Fractions=[.5,.5], datasetnames=["ECAL"], Particles =[particle])
   if Test:
     data_files = Testfiles
   else:
     data_files = Trainfiles
   nb_files = int(math.ceil(nbEvents/eventsperfile))
   data_files = data_files[-1* nb_files:]
   print(data_files)
   start = time.time()
   if ang:
     var = gan.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)     
   else:
     var = gan.get_sorted(data_files, energies, flag=False, num_events1=20000, num_events2=2000, thresh=thresh)
   data_time = time.time() - start
   print ("The events were loaded in {} seconds".format(data_time))
   result = []
   error_bar = []
   for energy in energies:
     var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
     var["events_act" + str(energy)]=  np.squeeze(var["events_act" + str(energy)])/dscale
     total += var["index" + str(energy)]
     data_time = time.time() - start
   print ("{} events were put in {} bins".format(total, len(energies)))
   for gen_weight in gen_weights:
     g.load_state_dict(torch.load(gen_weight))
     #g.load_weights(gen_weight)            
     start = time.time()
     print("Weights are loaded from {}".format(gen_weight))
     for energy in energies:
       if ang:
          angle = measPython(var["events_act"+ str(energy)])
          var["events_gan" + str(energy)] = gan.generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100, var["angle" + str(energy)] ], concat=concat, latent=latent)
          var["events_gan" + str(energy)] = np.power(var["events_gan" + str(energy)], 1./power)
       else:
          var["events_gan" + str(energy)] = gan.generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100], concat=concat, latent=latent) 
          var["events_gan" + str(energy)] = var["events_gan" + str(energy)]/xscale

       var["events_gan" + str(energy)]=  np.squeeze(var["events_gan" + str(energy)])/dscale
                     
       gen_time = time.time() - start
       ecal_act = np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))
       ecal_gan = np.sum(var["events_gan"+ str(energy)], axis=(1, 2, 3))

       var["sf_act" + str(energy)] = np.divide(ecal_act, var["energy"+ str(energy)])
       var["sf_gan" + str(energy)] = np.divide(ecal_gan, var["energy"+ str(energy)])

     print ("{} events were generated in {} seconds".format(total, gen_time))
     for opt in optimizer:
       if opt == 'mre':
         res, d = mre(var, energies[1:], ang=ang)
         error_bar.append(d)
       elif opt == 'chi2':
         res = stat_test(var, [energies[0]], ang=ang, test=opt, p=[100, 400])
       elif opt == 'ks':
         res = stat_test(var, [energies[0]], ang=ang, test=opt, p=[100, 400])
       elif opt == 'wass':
         if 0 in energies:
           res = wass( var["sf_act0"],  var["sf_act0"])
       print('{} error = {}'.format(opt, res))
         
       result.append(res)
   return result, error_bar                                        
 
def mre(var, energies, ang=1):
   metrics = 0
   d = 0
   for energy in energies:
     #Relative error on mean moment value for each moment and each axis
     mean_sf_g4 = np.mean(var["sf_act"+ str(energy)])
     mean_sf_gan = np.mean(var["sf_gan"+ str(energy)])
     N = var["sf_act"+ str(energy)].shape[0]
     sf_error = np.divide(np.absolute(mean_sf_g4 - mean_sf_gan), mean_sf_g4)
     er = (np.square(np.std(var["sf_gan"+ str(energy)])/mean_sf_g4)/N) + (np.square(np.std(var["sf_act"+ str(energy)])/mean_sf_gan)/N)
     metrics +=sf_error
     d += er
   d = np.sqrt(d/len(energies))
   metrics = metrics/len(energies)
   return metrics, d

def stat_test(var, energies, ang, bins=[10, 5], p=[100, 300], r = [0.018, 0.022], test='chi2', d=0):
    error=0
    bsize=5
    for energy in energies:
      if len(p)!=2:
        pmin = np.amin(var["energy"+ str(energy)])
        pmax = np.amax(var["energy"+ str(energy)])
      else:
        pmin = p[0]
        pmax = p[1]
      b2 = int((pmax-pmin)/bsize)
      if d==0:
        g4hist = ROOT.TProfile('g4hist', 'g4hist', b2, pmin, pmax)
        ganhist = ROOT.TProfile('ganhist', 'ganhist', b2, pmin, pmax)
        g4hist.Sumw2()
        ganhist.Sumw2()
        my.fill_profile(g4hist, var["energy"+ str(energy)], var["sf_act"+ str(energy)])
        my.fill_profile(ganhist, var["energy"+ str(energy)], var["sf_gan"+ str(energy)])
      elif d==2:
        g4hist = ROOT.TH2F('g4hist', 'g4hist', bins[0], pmin, pmax, bins[1], r[0], r[1])
        ganhist = ROOT.TH2F('ganhist', 'ganhist', bins[0], pmin, pmax, bins[1], r[0], r[1])
        g4hist.Sumw2()
        ganhist.Sumw2()
        my.fill_hist_2D(g4hist, var["energy"+ str(energy)], var["sf_act"+ str(energy)])
        my.fill_hist_2D(ganhist, var["energy"+ str(energy)], var["sf_gan"+ str(energy)])

      else:
        g4hist = ROOT.TH1F('g4hist', 'g4hist', bins[1], r[0], r[1])
        ganhist = ROOT.TH1F('ganhist', 'ganhist', bins[1], r[0], r[1])
        g4hist.Sumw2()
        ganhist.Sumw2()
        print(var["sf_act"+ str(energy)][:10])
        print(var["sf_gan"+ str(energy)][:10])      
        my.fill_hist(g4hist, var["sf_act"+ str(energy)])
        my.fill_hist(ganhist, var["sf_gan"+ str(energy)])
      
      if test == 'ks':
        if d==0:
          serror=g4hist.KolmogorovTest(ganhist, 'WW')
        else:
          serror=g4hist.KolmogorovTest(ganhist)
        print(test, serror)
      if test == 'chi2':
        if d==0:
          serror = g4hist.Chi2Test(ganhist, 'WW')
        else:
          serror = g4hist.Chi2Test(ganhist)
        print(test, serror)
      if serror> 0:
        serror = -1 * np.log10(serror)
      
      error+=serror
    return error/len(energies)


def measPython(image): # Working version:p1 and p2 are not used. 3D angle with barycenter as reference point
    image = np.squeeze(image)
    x_shape= image.shape[1]
    y_shape= image.shape[2]
    z_shape= image.shape[3]

    sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
    indexes = np.where(sumtot > 0)
    amask = np.ones_like(sumtot)
    amask[indexes] = 0

    masked_events = np.sum(amask) # counting zero sum events

    x_ref = np.sum(np.sum(image, axis=(2, 3)) * np.expand_dims(np.arange(x_shape) + 0.5, axis=0), axis=1)
    y_ref = np.sum(np.sum(image, axis=(1, 3)) * np.expand_dims(np.arange(y_shape) + 0.5, axis=0), axis=1)
    z_ref = np.sum(np.sum(image, axis=(1, 2)) * np.expand_dims(np.arange(z_shape) + 0.5, axis=0), axis=1)

    x_ref[indexes] = x_ref[indexes]/sumtot[indexes]
    y_ref[indexes] = y_ref[indexes]/sumtot[indexes]
    z_ref[indexes] = z_ref[indexes]/sumtot[indexes]

    sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z

    x = np.expand_dims(np.arange(x_shape) + 0.5, axis=0)
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(np.arange(y_shape) + 0.5, axis=0)
    y = np.expand_dims(y, axis=2)
    x_mid = np.sum(np.sum(image, axis=2) * x, axis=1)
    y_mid = np.sum(np.sum(image, axis=1) * y, axis=1)
    indexes = np.where(sumz > 0)

    zmask = np.zeros_like(sumz)
    zmask[indexes] = 1
    zunmasked_events = np.sum(zmask, axis=1)

    x_mid[indexes] = x_mid[indexes]/sumz[indexes]
    y_mid[indexes] = y_mid[indexes]/sumz[indexes]
    z = np.arange(z_shape) + 0.5# z indexes
    x_ref = np.expand_dims(x_ref, 1)
    y_ref = np.expand_dims(y_ref, 1)
    z_ref = np.expand_dims(z_ref, 1)

    zproj = np.sqrt((x_mid-x_ref)**2.0  + (z - z_ref)**2.0)
    m = (y_mid-y_ref)/zproj
    z = z * np.ones_like(z_ref)
    indexes = np.where(z<z_ref)
    m[indexes] = -1 * m[indexes]
    ang = (math.pi/2.0) - np.arctan(m)
    ang = ang * zmask

    ang = ang * z # weighted by position
    sumz_tot = z * zmask
    ang = np.sum(ang, axis=1)/np.sum(sumz_tot, axis=1)

    indexes = np.where(amask>0)
    ang[indexes] = 100.
    return ang

if __name__ == "__main__":
   main()
