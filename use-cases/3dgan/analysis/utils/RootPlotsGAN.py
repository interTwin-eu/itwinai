from os import path
import ROOT
import numpy as np
import os
import math
import time
#import numpy.core.umath_tests as umath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance as wass

from utils import ROOTutils as my # common utility functions for root
from utils.GANutils import safe_mkdir, get_sums

##################################### Plots used in detailed analysis ######################################################
def z_test(array1, array2):
   mean1 = np.mean(array1)
   mean2 = np.mean(array2)
   std1=np.std(array1)
   std2=np.std(array2)
   num = array1.shape[0]
   z = (mean1-mean2)/np.sqrt(np.square(std1) + np.square(std2))
   return z


def flip(m, axis):
   if not hasattr(m, 'ndim'):
      m = asarray(m)
   indexer = [slice(None)] * m.ndim
   try:
      indexer[axis] = slice(None, None, -1)
   except IndexError:
      raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                       % (axis, m.ndim))
   return m[tuple(indexer)]

# computes correlation of a set of features and returns Fisher's Transform and names of features
def get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio, ang=0):
   x = sumx.shape[1]
   y = sumy.shape[1]
   z = sumz.shape[1]
   array = np.hstack((sumx[:, 2:x-2], sumy[:, 2:y-2], sumz[:, 2:z-2]))
   names = ['']
   for i in range(2, x-2):
      names = names + ['sumx'  + str(i)]
   for i in range(2, y-2):
      names = names + ['sumy'  + str(i)]
   for i in range(2, z-2):
      names = names + ['sumz'  + str(i)]
   m = momentx.shape[1]
   for i in range(m):
      array = np.hstack((array, momentx[:, i].reshape(-1, 1), momenty[:, i].reshape(-1, 1), momentz[:, i].reshape(-1, 1)))
      names = names + ['momentx' + str(i), 'momenty' + str(i), 'momentz' + str(i)]
   array = np.hstack((array, ecal.reshape(-1, 1), energy.reshape(-1, 1)))
   array = np.hstack((array, hits.reshape(-1, 1), ratio.reshape(-1, 1)))
   if ang:
     array = np.hstack((array, ang.reshape(-1, 1)))
     names = names + ['ecal sum', 'p energy', 'hits', 'ratio1_total', 'theta']
   else:
     names = names + ['ecal sum', 'p energy', 'hits', 'ratio1_total']
   cor= np.corrcoef(array, rowvar=False)
   cor= get_dia(cor)
   fisher= np.arctanh(cor)
   return flip(fisher, axis=0), names

# returns correlation of a set of features and names of features
def get_correlation_small(momentx, momenty, momentz, ecal, energy, hits, ratio1, ratio2, ratio3, ang=None):
   names = ['']
   m=2
   for i in range(m):
      if i==0:
         array = np.hstack((momentx[:, i].reshape(-1, 1), momenty[:, i].reshape(-1, 1), momentz[:, i].reshape(-1, 1)))
      else:
         array = np.hstack((array, momentx[:, i].reshape(-1, 1), momenty[:, i].reshape(-1, 1), momentz[:, i].reshape(-1, 1)))
      names = names + ['Mx' + str(i+1), 'My' + str(i+1), 'Mz' + str(i+1)]
   array = np.hstack((array, ecal.reshape(-1, 1), energy.reshape(-1, 1)))
   array = np.hstack((array, hits.reshape(-1, 1), ratio1.reshape(-1, 1), ratio2.reshape(-1, 1), ratio3.reshape(-1, 1)))
   if ang is not None:
     ang = np.absolute(ang - 1.5708)
     array = np.hstack((array, ang.reshape(-1, 1)))
     names = names + ['$E_{sum}$', '$E_p$', 'H', 'R1', 'R2', 'R3', 'ang']
   else:
     names = names + ['$E_{sum}$', '$E_p$', 'H', 'R1', 'R2', 'R3']
   cor= np.corrcoef(array, rowvar=False)
   cor = get_dia2(cor)
   return flip(cor, axis=0), names
                                                                                             
#Get the lower triangular matrix of given array
def get_dia(array):
   darray = np.zeros_like(array)
   for i in np.arange(array.shape[0]):
      for j in np.arange(i):
         darray[i, j]=array[i, j]
   return darray

#Get the triangular matrix of given array including diagonal
def get_dia2(array):
   darray = np.zeros_like(array)
   for i in np.arange(array.shape[0]):
      for j in np.arange(i+1):
         darray[i, j]=array[i, j]
   return darray
                              

# Compute and plot correlation
def plot_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, gsumx, gsumy, gsumz, gmomentx, gmomenty, gmomentz, gecal, energy, events1, events2, out_file, labels, leg=True):
   ecal = ecal["n_0"]
   hits = my.get_hits(events1)
   actcorr = plot_corr_python(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, my.get_hits(events1), my.ratio1_total(events1), out_file, 'G4', leg=leg)
   for i, key in enumerate(gsumx):
     gcorr = plot_corr_python(gsumx[key], gsumy[key], gsumz[key], gmomentx[key], gmomenty[key], gmomentz[key], 
             gecal[key], energy, my.get_hits(events2[key]), my.ratio1_total(events2[key]), 
                              out_file, 'GAN{}_{}'.format(labels[i], i), compare=True, gprev=actcorr, leg=leg)

# Compute and plot correlation
def plot_correlation_small(momentx, momenty, momentz, ecal, gmomentx, gmomenty, gmomentz, gecal, energy, events1, events2, out_file, labels, leg=True, stest=True, ang=0):
   ecal = ecal["n_0"]
   hits = my.get_hits(events1)
   actcorr = plot_corr_python_small(momentx, momenty, momentz, ecal, energy, my.get_hits(events1), my.ratio1_total(events1), my.ratio2_total(events1), my.ratio3_total(events1), out_file, 'G4', leg=leg, ang=ang)
   for i, key in enumerate(gmomentx):
      gcorr = plot_corr_python_small(gmomentx[key], gmomenty[key], gmomentz[key],
               gecal[key], energy, my.get_hits(events2[key]), my.ratio1_total(events2[key]), my.ratio2_total(events2[key]), my.ratio3_total(events2[key]),
                                     out_file, 'GAN{}'.format(labels[i]), compare=stest, gprev=actcorr, leg=leg, ang=ang)
                       

                                                                              
#Fills a 2D TGraph object
def fill_graph2D(graph, array):
   x = array.shape[0]
   y = array.shape[1]
   N = 0
   for i in range(x):
      for j in range(y):
         graph.SetPoint(N, i, j, array[i, j])
         N+=1

#plot correlation using Python
def plot_corr_python(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio, out_file, label, compare=False, gprev=0, leg=True):
   corr, names = get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio)
   x = np.arange(corr.shape[0]+ 1)
   y = np.arange(corr.shape[1]+ 1)
   X, Y = np.meshgrid(x, y)
   if compare:
     num_squares = corr.shape[0]*(corr.shape[0]-1)/2
     mse_corr = np.sum(np.square(gprev - corr))/num_squares
     print ('mse_corr={}'.format(mse_corr))
     dlabel='{} mse_corr = {:.4f} '.format(label, mse_corr)
   else:
     dlabel=label
   plt.figure()
   plt.pcolor(X, Y, corr, label=dlabel, vmin = -3, vmax = 4)
   plt.xticks(x, names, rotation='vertical', fontsize=4)
   plt.yticks(x, names[::-1], fontsize=4)
   plt.margins(0.1)
   plt.colorbar()
   if leg: plt.legend()
   plt.savefig(out_file + '_python' + label + '.pdf')
   return corr

#plot correlation using Python
def plot_corr_python_small(momentx, momenty, momentz, ecal, energy, hits, ratio1, ratio2, ratio3, out_file, label, compare=False, gprev=0, leg=True, ang=0):
   corr, names = get_correlation_small(momentx, momenty, momentz, ecal, energy, hits, ratio1, ratio2, ratio3, ang=ang)
   x = np.arange(corr.shape[0]+ 1)
   y = np.arange(corr.shape[1]+ 1)
   X, Y = np.meshgrid(x, y)
   if compare:
      num_squares = corr.shape[0]*(corr.shape[0]-1)/2
      error = (gprev - corr)
      mean_error = np.sum(np.absolute(error))/num_squares
      print ('mae={}'.format(mean_error))
      dlabel='{} MAE = {:.4f} '.format(label, mean_error)
   else:
      dlabel=label
   plt.figure()
   plt.pcolor(X, Y, corr, label=dlabel, vmin = -1, vmax = 1)
   plt.xticks(x, names, rotation='vertical', fontsize=8)
   plt.yticks(x, names[::-1], fontsize=8)
   plt.margins(0.1)
   plt.colorbar()
   if leg: plt.legend()
   plt.savefig(out_file + '_python' + label + '.pdf')
   if isinstance(gprev, np.ndarray):
     plt.figure()
     plt.pcolor(X, Y, corr-gprev, label='Diff_' + dlabel, vmin = -1, vmax = 1)
     plt.xticks(x, names, rotation='vertical', fontsize=8)
     plt.yticks(x, names[::-1], fontsize=8)
     plt.margins(0.1)
     plt.colorbar()
     if leg: plt.legend()
     plt.savefig(out_file + '_diff_python' + label + '.pdf')
   return corr
                                                                                              
#plot correlation using root
def plot_corr_root(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio, out_file, label, compare=False, stest=True, gprev=0, leg=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                
   color = 2
   gact, names = get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio)
   Egraph =ROOT.TGraph2D()
   Ggraph =ROOT.TGraph2D()
   fill_graph2D(Egraph, gact)
   Egraph.Draw('colz')
   ylen = len(names)
   c1.Update()
   Egraph.GetYaxis().SetLabelOffset(1)
   Egraph.GetYaxis().SetNdivisions(10* ylen)
   Egraph.GetXaxis().SetLabelOffset(1)
   Egraph.GetXaxis().SetNdivisions(10* ylen)
   ty = ROOT.TText()
   ty.SetTextAlign(32)
   ty.SetTextSize(0.011)
   ty.SetTextFont(72)
   tx = ROOT.TText()
   tx.SetTextAlign(32)
   tx.SetTextSize(0.011)
   tx.SetTextFont(72)
   tx.SetTextAngle(70)
   y = np.arange(ylen)
   for i in y:
      ty.DrawText(-0.42,y[i],names[i])
      tx.DrawText(y[i],-0.42,names[ylen-i-1])
   print(len(names))
   c1.Update()
   legend = ROOT.TLegend(.6, .8, .9, .9)
   if compare:
     num_squares = gact.shape[0]*(gact.shape[0]-1)/2
     mse_corr = np.sum(np.square(gprev - gact))/num_squares
     legend.SetHeader('{}  mse_corr = {:.4f}'.format(label, mse_corr))
   else:
     legend.SetHeader(label)
   if leg: legend.Draw()
   c1.Update()
   c1.Print(out_file + '_' + label + '.pdf')
   return gact

# PLot ecal ratio
def plot_ecal_ratio_profile(ecal1, ecal2, y, labels, out_file, p=[2, 500], ifpdf=True, stest=False, leg=True, grid=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color = 2
   if y.shape[0]> ecal1["n_0"].shape[0]:
      y = y[:ecal1["n_0"].shape[0]]
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep", 100, p[0], p[1])
   Eprof.Sumw2()
   if statbox==False:  Eprof.SetStats(0)
   Eprof.SetTitle("Ratio of Ecal and Ep for {}-{} GeV".format(p[0], p[1]))
   Eprof.GetXaxis().SetTitle("Ep [GeV]")
   # Since the Angle Data has energies multiplied by 50 for ecal depositions
   ratio1=ecal1["n_0"]/y 
   my.fill_profile(Eprof, y, ratio1) 
   Eprof.GetYaxis().SetTitle("Ecal/Ep")
   Eprof.GetYaxis().CenterTitle()
   if p[0] < 100:
      Eprof.GetYaxis().SetRangeUser(0., 0.04)
   else:
      Eprof.GetYaxis().SetRangeUser(0., 0.03)
   Eprof.Draw()
   Eprof.SetLineColor(color)
   color+=1
   legend = ROOT.TLegend(.1, .1, .4, .3)
   legend.AddEntry(Eprof,"G4","l")
   Gprofs=[]
   for i, key in enumerate(ecal2):
      ratio2=ecal2[key]/y
      Gprofs.append(ROOT.TProfile("Gprof" + str(i), "Gprof" + str(i), 100, int(p[0]), int(p[1])))
      Gprof = Gprofs[i]
      Gprof.Sumw2()
      if statbox==False:  Gprof.SetStats(0)
      my.fill_profile(Gprof, y, ratio2)
      error = np.mean(np.abs(ratio1 - ratio2 ))
      color +=1
      if color in [5, 10, 18, 19]:
          color+=1
      Gprof.SetLineColor(color)
      if mono:  Gprof.SetLineStyle(color-2)
      Gprof.Draw('sames')
      c1.Update()
      legend.AddEntry(Gprof, "GAN {}".format(labels[i]), "l")
      if stest:
         ks = Eprof.KolmogorovTest(Gprof, 'UU NORM')
         legend.AddEntry(Gprof, "ks={:.6f}".format(ks))
      if leg: legend.Draw()
      if statbox:  my.stat_pos(Gprof)
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

# PLot ecal ratio
def plot_ecal_relative_profile(ecal1, ecal2, y, labels, out_file, p=[2, 500], ifpdf=True, leg=True, grid=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color = 2
   if y.shape[0]> ecal1["n_0"].shape[0]:
      y = y[:ecal1["n_0"].shape[0]]
   Eprof = ROOT.TProfile("Eprof", "Relative error for Ecal sum vs. Ep", 50, p[0], p[1])
   Eprof.Sumw2()
   if statbox==False:  Eprof.SetStats(0)
   Eprof.SetTitle("Relative Error for sum of  Ecal energies and Ep {}-{} GeV".format(p[0], p[1]))
   my.fill_profile(Eprof, y, (ecal1["n_0"] - ecal1["n_0"])/ ecal1["n_0"])
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("(Ecal_{G4} - Ecal_{GAN})/Ecal_{G4}")
   Eprof.GetYaxis().CenterTitle()
   Eprof.GetYaxis().SetRangeUser(-1, 1)
   Eprof.Draw()
   Eprof.SetLineColor(color)
   color+=1
   legend = ROOT.TLegend(.1, .1, .4, .3)
   legend.AddEntry(Eprof,"G4","l")
   Gprofs=[]
   for i, key in enumerate(ecal2):
      Gprofs.append(ROOT.TProfile("Gprof" + str(i), "Gprof" + str(i), 50, p[0], p[1]))
      Gprof = Gprofs[i]
      Gprof.Sumw2()
      if statbox==False:  Gprof.SetStats(0)
      error = (ecal1["n_0"]- ecal2[key])/ ecal1["n_0"]
      my.fill_profile(Gprof, y, error)
      color +=1
      if color in [5, 10, 18, 19]:
        color+=1
      Gprof.SetLineColor(color)
      if mono: Gprof.SetLineStyle(color-2)
      Gprof.Draw('sames')
      c1.Update()
      legend.AddEntry(Gprof, "GAN {}".format(labels[i]), "l")
      if statbox: my.stat_pos(Gprof)
      if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_aux_relative_profile(aux1, aux2, y, out_file, labels, p=[2, 500], stest=False, ifpdf=True, leg=True, grid=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.1, .1, .4, .3)
   Gprofs=[]
   Eprofs=[]
   for i, key in enumerate(aux1):
     Eprofs.append(ROOT.TProfile("Eprof" + str(i),"Eprof" + str(i), 100, p[0], p[1]))
     Gprofs.append(ROOT.TProfile("Gprof" + str(i),"Gprof" + str(i), 100, p[0], p[1]))
     Eprof= Eprofs[i]
     Gprof= Gprofs[i]
     Eprof.Sumw2()
     Gprof.Sumw2()
     if statbox==False:
       Eprof.SetStats(0)
       Gprof.SetStats(0)
     if i== 0:
       Eprof.SetTitle("Relative Error for Primary Energy for {}-{} GeV".format(p[0], p[1]))
       Eprof.GetXaxis().SetTitle("Ep [GeV]")
       Eprof.GetYaxis().SetTitle("(Ep - Ep_{predicted})/Ep")
       Eprof.GetYaxis().CenterTitle()
       error1=(y - 100 *aux1[key])/y
       if np.mean(np.absolute(error1)) < 0.2:
          Eprof.GetYaxis().SetRangeUser(-0.3, 0.3)
       else:
          Eprof.GetYaxis().SetRangeUser(-1.5, 1.5)
       my.fill_profile(Eprof, y, error1)
       Eprof.SetLineColor(color)
       Eprof.Draw()
       c1.Update()
       mae1= np.mean(np.abs(error1))
       if stest:
          label = "G4 {} MAE={:.4f})".format(labels[i], mae1)
       else:
          label = "G4 {} ".format(labels[i])
       legend.AddEntry(Eprof, label,"l")
       c1.Update()
       color+=2
     else:
       my.fill_profile(Eprof, y, (y - 100 *aux1[key])/y)
       Eprof.Draw('sames')
       legend.AddEntry(Eprof,"G4 " + labels[i],"l")
       color+=1
       if color in [5, 10, 18, 19]:
        color+=1
     error2=(y - 100 *aux2[key])/y
     my.fill_profile(Gprof, y, error2)
     Gprof.SetLineColor(color)
     if mono:  Gprof.SetLineStyle(color-2)
     color+=1
     if color in [5, 10, 18, 19]:
        color+=1
     mae2= np.mean(np.abs(error2))
     Gprof.Draw('sames')
     c1.Update()
     if statbox:my.stat_pos(Gprof)
     if stest:        
        legend.AddEntry(Gprof, "GAN {} (MAE={:.6f})".format(labels[i], mae2), "l")
     else:
        legend.AddEntry(Gprof, "GAN {}".format(labels[i]), "l")
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_hist(ecal1, ecal2, out_file, energy, labels, p=[2, 500], ifpdf=True, stest=True, leg=True, grid=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color=2
   hd = ROOT.TH1F("Geant4", "", 100, 0, 1.5 * p[1]/50)# energies for fixed angle has this rough relation to ecal sum
   hd.Sumw2()
   if statbox==False: hd.SetStats(0)
   hd.GetXaxis().SetTitle("Ecal Sum GeV")
   my.fill_hist(hd, ecal1['n_0'])
   hd = my.normalize(hd, 1)              
   if energy == 0:
      hd.SetTitle("Ecal Sum Histogram for {}-{} GeV".format(p[0], p[1]))
   else:
      hd.SetTitle("Ecal Sum Histogram (Ep ={} GeV)".format(energy) )
   hd.GetYaxis().SetTitle("normalized count")
   hd.GetYaxis().CenterTitle()
   hd.Draw()
   hd.Draw("sames hist")
   hd.SetLineColor(color)
   color+=2
   legend = ROOT.TLegend(.1, .1, .3, .2)
   legend.AddEntry(hd,"G4" ,"l")
   hgs=[]
   pos =0
   for i, key in enumerate(ecal2):
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 100, 0, 1.5 * p[1]/50))
      hg= hgs[i]
      hg.Sumw2()
      if statbox==False: hg.SetStats(0)
      hg.SetLineColor(color)
      if mono: hg.SetLineStyle(color-2)
      color+=1
      if color in [5, 10, 18, 19]:
        color+=1
      c1.Update()
      my.fill_hist(hg, ecal2[key])
      hg =my.normalize(hg, 1)
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      my.Max(hd, hg)
      pos+=1
      c1.Update()
      if statbox: my.stat_pos(hg)
      #if energy == 0:
      if stest:
         ks = hd.KolmogorovTest(hg, 'UU NORM')
         glabel = "GAN {}  K={:.6f}".format(labels[i], ks)
      else:
         glabel = "GAN {}".format(labels[i])
      legend.AddEntry(hg, glabel , "l")
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_flatten_hist(event1, event2, out_file, energy, labels, p=[2, 500], ifpdf=True, log=0, leg=True, grid=True, statbox=True, mono=False, num_events=1000):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color =2
   ROOT.gPad.SetLogx()
   ROOT.gStyle.SetOptStat(111111)
   if log:
      ROOT.gPad.SetLogy()
   hd = ROOT.TH1F("Geant4", "", 100, -12, 1)
   if not statbox: hd.SetStats(0)
   my.BinLogX(hd)
   hd.Sumw2()
   data1= event1[:num_events].flatten()
   my.fill_hist(hd, data1)
   hd =my.normalize(hd,1)
   if energy == 0:
      hd.SetTitle("Cell energies Histogram for {:.2f}-{:.2f} GeV".format(p[0], p[1]))
   else:
      hd.SetTitle("Cell energies Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Cell energy deposition [GeV]")
   hd.Draw()
   hd.Draw('sames hist')
   hd.SetLineColor(color)
   c1.Update()
   if log:
      legend = ROOT.TLegend(.4, .1, .6, .3)
   else:
      legend = ROOT.TLegend(.1, .1, .3, .3)
   legend.AddEntry(hd,"G4","l")
   color+=2
   hgs=[]
   pos = 0
   for i, key in enumerate(event2):
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 100, -12, 1))
      hg = hgs[i]
      my.BinLogX(hg)
      hg.Sumw2()
      data2= event2[key][:num_events].flatten()
      my.fill_hist(hg, data2)
      hg =my.normalize(hg, 1)
      hg.SetLineColor(color)
      if mono: hg.SetLineStyle(color-2)
      color+=1
      if color in [5, 10, 18, 19]:
        color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      if statbox:
         my.stat_pos(hg)
      else:
         hg.SetStats(0)
      c1.Update()
      pos+=1
      c1.Update()
      legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_hits_hist(event1, event2, out_file, energy, labels, p=[2, 500], ifpdf=True, stest=False, thresh=3e-4, leg=True, grid=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color = 2
   hd = ROOT.TH1F("Geant4", "", 50, 0, 2000)
   hd.Sumw2()
   if energy == 0:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for {}-{} GeV Primary Energy".format(thresh, p[0], p[1]))
   else:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for {} GeV Primary Energy".format(thresh, energy) )
            
   my.fill_hist(hd, my.get_hits(event1, thresh))
   hd.GetXaxis().SetTitle("Ecal Hits")
   hd.GetYaxis().SetTitle("normalized count")
   hd.GetYaxis().CenterTitle()
   hd =my.normalize(hd, 1)            
   hd.Draw()
   hd.Draw('sames hist')
   hd.SetLineColor(color)
   if not statbox: hd.SetStats(0)
   color+=2
   hgs=[]
   pos = 0
   if energy==0:
      legend = ROOT.TLegend(.4, .1, .6, .3)
   elif energy > 250:
      legend = ROOT.TLegend(.7, .1, .9, .3)
   else:
      legend = ROOT.TLegend(.1, .1, .3, .3)
   legend.AddEntry(hd,"G4","l")
   for i, key in enumerate(event2):
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 2000))
      hg = hgs[i]
      hg.Sumw2()
      my.fill_hist(hg, my.get_hits(event2[key], thresh))
      hg.SetLineColor(color)
      if mono: hg.SetLineStyle(color-2)
      hg =my.normalize(hg, 1)
      color+=1
      if color in [5, 10, 18, 19]:
        color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
      c1.Update()
      my.Max(hd, hg)
      if statbox:
        my.stat_pos(hg)
      else:
        hg.SetStats(0) 
      pos+=1
      if stest:
         ks = hd.KolmogorovTest(hg, 'UU NORM')
         my.get_hits(event2[key], thresh)
         glabel = "GAN {}  K={:.6f}".format(labels[i], ks)
      else:
         glabel = "GAN {}".format(labels[i])
      legend.AddEntry(hg, glabel , "l")
                             
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_aux_hist(aux1, aux2, out_file, energy, labels, p=[2, 500], ifpdf=True, leg=True, grid=True, stest=False, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color = 2
   if energy > 300:
      legend = ROOT.TLegend(.1, .1, .3, .3)
   else:
      legend = ROOT.TLegend(.7, .1, .9, .3)
   hps=[]
   hgs=[]
   for i, key in enumerate(aux1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 100, 0, 600))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 100, 0, 600))
     hp= hps[i]
     hg= hgs[i]
     hp.Sumw2()
     hg.Sumw2()
     if statbox==False:
          hp.SetStats(0)
          hg.SetStats(0)

     if i== 0:

       hp.SetTitle(" Predicted Primary Energy for {}-{} GeV".format(p[0], p[1]))
       hp.GetXaxis().SetTitle("Ep [GeV]")
       my.fill_hist(hp, 100 *aux1[key])
       hp.Draw()
       hp.Draw('sames hist')
       hp.SetLineColor(color)
       c1.Update()
       color+=2
     else:
       my.fill_hist(hp, 100 *aux1[key])
       hp.SetTitle("Predicted Primary Energy Histogram for {}+/- 5 GeV".format(energy) )
       hp.SetLineColor(color)
       hp.Draw('sames')
       hp.Draw('sames hist')
       c1.Update()
       legend.AddEntry(hp,"G4" + labels[i],"l")
       color+=1
       if color in [5, 10, 18, 19]:
         color+=1
     
     label = "G4 " + labels[i]
     legend.AddEntry(hp, label,"l")
     my.fill_hist(hg, 100 *aux2[key])
     hp =my.normalize(hp, 1)
     hg =my.normalize(hg, 1)
     my.Max(hp, hg)
     hg.SetLineColor(color)
     if mono: hg.SetLineStyle(color-2)
     color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     if statbox:  my.stat_pos(hg)
     c1.Update()
     if stest:
       ks = hp.KolmogorovTest(hg, 'UU NORM')
       label = "GAN {}  K={:.6f}".format(labels[i], ks)
     else:
       label = "GAN " + labels[i]
     legend.AddEntry(hg, label, "l")
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_primary_error_hist(aux1, aux2, y, out_file, energy, labels, p=[2, 500], ifpdf=True, leg=True, grid=True, stest= False, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   if grid: c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.1, .1, .3, .3)
   hps=[]
   hgs=[]
   if y.shape[0]> aux1["n_0"].shape[0]:
      y = y[:aux1["n_0"].shape[0]]
   for i, key in enumerate(aux1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 50, -0.4, 0.4))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, -0.4, 0.4))
     hp= hps[i]
     hg= hgs[i]
     hp.Sumw2()
     hg.Sumw2()
     if i== 0:
       if energy == 0:
          hp.SetTitle("Predicted Energy Relative Error Histogram for {}-{} GeV".format(p[0], p[1]))
       else:
          hp.SetTitle("Aux Energy Relative Error Histogram for {} GeV".format(energy) )
       hp.GetXaxis().SetTitle("Primary GeV")
       hp.GetYaxis().SetTitle("(E_{p} - E_{predicted})/E_{p}")
       hp.GetYaxis().CenterTitle()
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw()
       hp.Draw('sames hist')
       c1.Update()
       hp.SetLineColor(color)
       legend.AddEntry(hp,"G4 " + labels[i],"l")
       c1.Update()
       color+=2
     else:
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw('sames')
       hp.Draw('sames hist')
       legend.AddEntry(hp,"G4 " + labels[i],"l")
       color+=1
       if color in [5, 10, 18, 19]:
        color+=1
     my.fill_hist(hg,  (y - aux2[key]*100)/y)
     hp =my.normalize(hp, 1)
     hg =my.normalize(hg, 1)
     hg.SetLineColor(color)
     if mono: hg.SetLineStyle(color-2)
     color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     if statbox:
        my.stat_pos(hg)
     else:
        hp.SetStats(0)
        hg.SetStats(0)
     c1.Update()
     my.Max(hp, hg)
     c1.Update()
     legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   if leg:legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_realfake_hist(array1, array2, out_file, energy, labels, p=[2, 500], ifpdf=True, leg=True, grid=True, stest= False, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   if grid:  c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hps=[]
   hgs=[]
   for i, key in enumerate(array1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 20, 0, 1.3))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 20, 0, 1.3))
     hp= hps[i]
     hg= hgs[i]
     hp.Sumw2()
     hg.Sumw2()
     if i== 0:
       if energy == 0:
          hp.SetTitle("Real/Fake Histogram for {}-{} GeV".format(p[0], p[1]))
       else:
          hp.SetTitle("Real/Fake Histogram for {} GeV".format(energy) )

       hp.GetXaxis().SetTitle("Real/Fake")
       my.fill_hist(hp, array1[key])
       hp.Draw()
       hp.Draw('sames hist')
       c1.Update()
       hp.GetYaxis().SetTitle('count')
       hp.GetYaxis().CenterTitle()
       hp.SetLineColor(color)
       c1.Update()
       color+=2
     else:
       my.fill_hist(hp, array1[key])
       hp.Draw('sames')
       hp.Draw('sames hist')
       hp.SetLineColor(color)
       c1.Update()
       color+=1
       if color in [5, 10, 18, 19]:
        color+=1
     if stest:
        mean= np.mean(array1[key])
        label = 'G4 {} (mean={})'.format(labels[i], mean)
     else:
        label ='G4 {})'.format(labels[i])
     legend.AddEntry(hp, label,"l")
     c1.Update()
     my.fill_hist(hg,  array2[key])
     hp =my.normalize(hp)
     hg =my.normalize(hg)
     hp.GetYaxis().SetRangeUser(0, 0.5)
     hg.SetLineColor(color)
     if stest:
        mean= np.mean(array2[key])
        label ='GAN {} (mean={})'.format(labels[i], mean)
     else:
        label ='GAN {})'.format(labels[i])
     legend.AddEntry(hg, label,"l")
     if mono: hg.SetLineStyle(color-2)
     color+=1
     if color in [5, 10, 18, 19]:
        color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     if statbox:
        my.stat_pos(hg)
     else:
        hp.SetStats(0)
        hg.SetStats(0)
     c1.Update()
     my.Max(hp, hg)
     c1.Update()
          
     legend.AddEntry(hg, "GAN " + labels[i], "l")
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_max(array1, array2, x, y, z, out_file1, out_file2, out_file3, energy, labels, log=0, p=[2, 500], ifpdf=True, stest=True, leg=True, grid=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetTitle('Weighted Histogram for point of maximum energy deposition along x, y, z axis')
   color = 2
   c1.Divide(2,2)
   h1x = ROOT.TH1F('G4x' + str(energy), '', x, 0, x)
   h1y = ROOT.TH1F('G4y' + str(energy), '', y, 0, y)
   h1z = ROOT.TH1F('G4z' + str(energy), '', z, 0, z)
   h1x.SetLineColor(color)
   h1y.SetLineColor(color)
   h1z.SetLineColor(color)
   h1x.Sumw2()
   h1y.Sumw2()
   h1z.Sumw2()
   if statbox==False:
     h1x.SetStats(0)
     h1y.SetStats(0)
     h1z.SetStats(0)
         
   c1.cd(1)
   if log:
      ROOT.gPad.SetLogy()
   if grid:
      ROOT.gPad.SetGrid()
   my.fill_hist(h1x, array1[:,0])
   h1x=my.normalize(h1x)
   h1x.Draw()
   h1x.Draw('sames hist')
   h1x.GetXaxis().SetTitle("Position of Max Energy (x axis)")
   c1.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   if grid:
      ROOT.gPad.SetGrid()
   my.fill_hist(h1y, array1[:,1])
   h1y=my.normalize(h1y)
   h1y.Draw()
   h1y.Draw('sames hist')
   h1y.GetXaxis().SetTitle("Position of Max Energy (y axis)")
   c1.cd(3)
   if log:
      ROOT.gPad.SetLogy()
   if grid:
      ROOT.gPad.SetGrid()
   my.fill_hist(h1z, array1[:,2])
   h1z=my.normalize(h1z)
   h1z.Draw()
   h1z.Draw('sames hist')
   h1z.GetXaxis().SetTitle("Position of Max Energy (z axis)")
   c1.cd(4)
   c1.Update()
   if ifpdf:
      c1.Print(out_file1 + '.pdf')
   else:
      c1.Print(out_file1 + '.C')
   h2xs=[]
   h2ys=[]
   h2zs=[]
   color+=2
   leg = ROOT.TLegend(0.1,0.4,0.9, 0.9)
   leg.SetTextSize(0.06)
   for i, key in enumerate(array2):
      h2xs.append(ROOT.TH1F('GANx' + str(energy) + labels[i], '', x, 0, x))
      h2ys.append(ROOT.TH1F('GANy' + str(energy) + labels[i], '', y, 0, y))
      h2zs.append(ROOT.TH1F('GANz' + str(energy) + labels[i], '', z, 0, z))
      h2x=h2xs[i]
      h2y=h2ys[i]
      h2z=h2zs[i]
      
      h2x.SetLineColor(color)
      h2y.SetLineColor(color)
      h2z.SetLineColor(color)
      if mono:
         h2x.SetLineStyle(color-2)
         h2y.SetLineStyle(color-2)
         h2z.SetLineStyle(color-2)
                     

      h2x.Sumw2()
      h2y.Sumw2()
      h2z.Sumw2()

      if statbox==False:
         h2x.SetStats(0)
         h2y.SetStats(0)
         h2z.SetStats(0)
            
      c1.cd(1)
      my.fill_hist(h2x, array2[key][:,0])
      h2x=my.normalize(h2x)
      if i==0:
         h2x.Draw()
         h2x.Draw('sames hist')
         h2x.GetXaxis().SetTitle("Position of Max Energy along x axis")
      else:
         h2x.Draw('sames')
         h2x.Draw('sames hist')
      c1.Update()
      if stest:
         ks = h1x.KolmogorovTest(h2x, "WW")
         glabel = "GAN {} X axis K = {:.4f}".format(labels[i], ks)
         leg.AddEntry(h2x, glabel,"l")
      if statbox: my.stat_pos(h2x)
      c1.cd(2)
      my.fill_hist(h2y, array2[key][:,1])
      h2y=my.normalize(h2y)
      if i==0:
         h2y.Draw()
         h2y.Draw('sames hist')
         h2y.GetXaxis().SetTitle("Position of Max Energy along y axis")
      else:
         h2y.Draw('sames')
         h2y.Draw('sames hist')
      c1.Update()
      if stest:
         ks = h1y.KolmogorovTest(h2y, "WW")
         glabel = "GAN {} Y axis K = {:.4f}".format(labels[i], ks)
         leg.AddEntry(h2y, glabel,"l")
      if statbox: my.stat_pos(h2y)         
      c1.cd(3)
      my.fill_hist(h2z, array2[key][:,2])
      h2z=my.normalize(h2z)
      if i==0:
         h2z.Draw()
         h2z.Draw('sames hist')
         h2z.GetXaxis().SetTitle("Position of Max Energy (z axis)")
      else:
         h2z.Draw('sames')
         h2z.Draw('sames hist')
      c1.Update()
      if stest:
         ks = h1z.KolmogorovTest(h2z, "WW")
         glabel = "GAN {} Z axis K = {:.4f}".format(labels[i], ks)
         leg.AddEntry(h2z, glabel,"l")
      color+= 1
      if color in [5, 10, 18, 19]:
        color+=1
      c1.Update()
      if statbox: my.stat_pos(h2z)
      c1.Update()
   c1.cd(4)
   if leg: leg.Draw()
   if ifpdf:
      c1.Print(out_file2 + '.pdf')
   else:
      c1.Print(out_file2 + '.C')

   c1.cd(1)
   h1x.Draw()
   h1x.Draw('sames hist')
   for h2x in h2xs:
     h2x.Draw('sames')
     h2x.Draw('sames hist')
   if not log: my.Max(h1x, h2x)
   c1.Update()
        
   c1.cd(2)
   h1y.Draw()
   h1y.Draw('sames hist')
   for h2y in h2ys:
     h2y.Draw('sames')
     h2y.Draw('sames hist')
   if not log: my.Max(h1y, h2y)
   c1.Update()
   c1.cd(3)
   h1z.Draw()
   h1z.Draw('sames hist')
   for h2z in h2zs:
     h2z.Draw('sames')
     h2z.Draw('sames hist')
   if not log: my.Max(h1z, h2z)
   c1.Update()
   c1.cd(4)
   leg.AddEntry(h1x,"G4","l")
   leg.SetHeader("Max energy deposition along x, y, z axis", "C")
   if not stest:
     for i, h in enumerate(h2xs):
       leg.AddEntry(h, 'GAN ' + labels[i],"l")
   if leg: leg.Draw()
   c1.Update()
   if ifpdf:
      c1.Print(out_file3 + '.pdf')
   else:
      c1.Print(out_file3 + '.C')

def plot_energy_hist_root(array1x, array1y, array1z, array2x, array2y, array2z, x, y, z, out_file1, out_file2, out_file3, energy, labels,
                          log=0, p=[2, 500], ifpdf=True, stest=True, grid=True, leg=True, statbox=True, mono=False, norm=True):
   canvas = ROOT.TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   canvas.SetTitle('Weighted Histogram for energy deposition along x, y, z axis')
   color = 2
   canvas.Divide(2,2)
   h1x = ROOT.TH1F('G4x' + str(energy), '', x, 0, x)
   h1y = ROOT.TH1F('G4y' + str(energy), '', y, 0, y)
   h1z = ROOT.TH1F('G4z' + str(energy), '', z, 0, z)

   h1x.SetLineColor(color)
   h1y.SetLineColor(color)
   h1z.SetLineColor(color)

   h1x.Sumw2()
   h1y.Sumw2()
   h1z.Sumw2()

   if statbox==False:
      h1x.SetStats(0)
      h1y.SetStats(0)
      h1z.SetStats(0)
   color+=2
   if color in [5, 10, 18, 19]:
        color+=1
   canvas.cd(1)
   if log:
      ROOT.gPad.SetLogy()
   if grid:
      ROOT.gPad.SetGrid()
   my.fill_hist_wt(h1x, array1x)
   if norm: h1x=my.normalize(h1x)
   h1x.Draw()
   h1x.Draw('sames hist')
   h1x.GetXaxis().SetTitle("position along x axis")
   h1x.GetYaxis().SetTitle("energy deposition")
   canvas.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   if grid:
      ROOT.gPad.SetGrid()
   my.fill_hist_wt(h1y, array1y)
   if norm: h1y=my.normalize(h1y)
   h1y.Draw()
   h1y.Draw('sames hist')
   h1y.GetXaxis().SetTitle("position along y axis")
   h1y.GetYaxis().SetTitle("energy deposition")
   canvas.cd(3)
   if log:
      ROOT.gPad.SetLogy()
   if grid:
      ROOT.gPad.SetGrid()
   my.fill_hist_wt(h1z, array1z)
   if norm: h1z=my.normalize(h1z)
   h1z.Draw()
   h1z.Draw('sames hist')
   h1z.GetXaxis().SetTitle("position along z axis")
   h1z.GetYaxis().SetTitle("energy deposition")
   canvas.cd(4)
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file1 + '.pdf')
   else:
      canvas.Print(out_file1 + '.C')
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.06)
   h2xs=[]
   h2ys=[]
   h2zs=[]
   for i, key in enumerate(array2x):
      h2xs.append(ROOT.TH1F('GANx' + str(energy)+ labels[i], '', x, 0, x))
      h2ys.append(ROOT.TH1F('GANy' + str(energy)+ labels[i], '', y, 0, y))
      h2zs.append(ROOT.TH1F('GANz' + str(energy)+ labels[i], '', z, 0, z))
      h2x=h2xs[i]
      h2y=h2ys[i]
      h2z=h2zs[i]
      
      h2x.Sumw2()
      h2y.Sumw2()
      h2z.Sumw2()
      
      h2x.SetLineColor(color)
      h2y.SetLineColor(color)
      h2z.SetLineColor(color)

      if mono:
        h2x.SetLineStyle(color-2)
        h2y.SetLineStyle(color-2)
        h2z.SetLineStyle(color-2)
      
      if statbox==False:
        h2x.SetStats(0)
        h2y.SetStats(0)
        h2z.SetStats(0)
      canvas.cd(1)
      my.fill_hist_wt(h2x, array2x[key])
      if norm: h2x=my.normalize(h2x)
      if i==0:
        h2x.Draw()
        h2x.Draw('sames hist')
        h2x.GetXaxis().SetTitle("position along x axis")
        h2x.GetYaxis().SetTitle("energy deposition [GeV]")
      else:
        h2x.Draw('sames')
        h2x.Draw('sames hist')
      canvas.Update()
      if stest:
         res=np.array
         ks= h1x.KolmogorovTest(h2x, 'WW')
         glabel = "GAN {} X axis K= {:.4f}".format(labels[i], ks)
         leg.AddEntry(h2x, glabel,"l")
      canvas.Update()
      if statbox: my.stat_pos(h2x)
      canvas.Update()
      canvas.cd(2)
      my.fill_hist_wt(h2y, array2y[key])
      if norm : h2y=my.normalize(h2y)
      if i==0:
        h2y.Draw()
        h2y.Draw('sames hist')
        h2y.GetXaxis().SetTitle("position along y axis")
        h2y.GetYaxis().SetTitle("energy deposition [GeV]")
      else:
        h2y.Draw('sames')
        h2y.Draw('sames hist')
      canvas.Update()
      if stest:
         ks= h1y.KolmogorovTest(h2y, 'WW')
         glabel = "GAN {} Y axis K= {:.4f}".format(labels[i], ks)
         leg.AddEntry(h2y, glabel,"l")
      canvas.Update()
      if statbox: my.stat_pos(h2y)
      canvas.Update()
      canvas.cd(3)
      my.fill_hist_wt(h2z, array2z[key])
      if norm: h2z=my.normalize(h2z)
      if i==0:
        h2z.Draw()
        h2z.Draw('sames hist')
        h2z.GetXaxis().SetTitle("position along z axis")
        h2z.GetYaxis().SetTitle("energy deposition [GeV]")
      else:
        h2z.Draw('sames')
        h2z.Draw('sames hist')
      canvas.Update()
      canvas.Update()
      if stest:
         ks= h1z.KolmogorovTest(h2z, 'WW')
         glabel = "GAN {} Z axis K= {:.4f}".format(labels[i], ks)
         leg.AddEntry(h2z, glabel,"l")
      canvas.Update()
      if statbox: my.stat_pos(h2z)
      color+=1
      if color in [5, 10, 18, 19]:
        color+=1
      h1x.Sumw2()
      h1y.Sumw2()
      h1z.Sumw2()
                  
      h2x.Sumw2()
      h2y.Sumw2()
      h2z.Sumw2()
                  
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file2 + '.pdf')
   else:
      canvas.Print(out_file2 + '.C')
   canvas.cd(1)
   h1x.Draw()
   h1x.Draw('sames hist')
   for h2x in h2xs:
     h2x.Draw('sames')
     h2x.Draw('sames hist')
   if not log: my.Max(h1x, h2x)
   canvas.Update()
   canvas.cd(2)
   h1y.Draw()
   h1y.Draw('sames hist')
   for h2y in h2ys:
    h2y.Draw('sames')
    h2y.Draw('sames hist')
   if not log: my.Max(h1y, h2y)
   canvas.Update()
   canvas.cd(3)
   h1z.Draw()
   h1z.Draw('sames hist')
   for h2z in h2zs:
     h2z.Draw('sames')
     h2z.Draw('sames hist')
   if not log: my.Max(h1z, h2z)
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(h1x, "G4","l")
   leg.SetHeader("Shower Shapes", "C")
   if not stest:
      for i, h in enumerate(h2xs):
        leg.AddEntry(h, 'GAN ' + labels[i],"l")
      
   if leg: leg.Draw()
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file3 + '.pdf')
   else:
      canvas.Print(out_file3 + '.C')

def plot_moment(array1, array2, out_file, dim, energy, m, labels, p =[2, 500], stest=False, ifpdf=True, grid=True, leg=True, statbox=True, mono=False, ang=1):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   array1= array1[:, m]
   if m==0:
     if dim=='x' or dim=='y':
       bins = 51 if ang else 25
       maxbin = 51 if ang else 25
       minbin= 0
     else:
      bins = 25
      maxbin = 25
      minbin= 0
   else:
     maxbin = np.amax(array1)+ 2
     minbin = min(0, np.amin(array1))
     bins = 50
   if grid: c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hd = ROOT.TH1F("G4"+ dim + str(m)+str(energy), "", bins, minbin, maxbin)
   hd.Sumw2()
   if energy == 0:
      hd.SetTitle("{} {} Moment Histogram for {}-{} GeV".format(m+1, dim, p[0], p[1]))
   else:
      hd.SetTitle("{} {} Moment Histogram for {} GeV".format(m+1, dim, energy) )
      hd.GetXaxis().SetTitle("{} Moment for {} axis".format(m+1, dim))
   my.fill_hist(hd, array1)
   hd =my.normalize(hd, 1)
   hd.Draw()
   hd.Draw('sames hist')
   hd.SetLineColor(color)
   c1.Update()
   if statbox==False:
      hd.SetStats(0)
   legend.AddEntry(hd,"G4","l")
   c1.Update()
   color+=2
   hgs=[]
   for i, key in enumerate(array2):
      hgs.append(ROOT.TH1F("GAN"+ dim + str(m)+str(energy) + str(i), "GAN"+ dim + str(m)+str(i), bins, minbin, maxbin))
      hg= hgs[i]
      hg.Sumw2()
      my.fill_hist(hg, array2[key][:, m])
      hg.SetLineColor(color)
      if mono: hg.SetLineStyle(color-2)
      hg =my.normalize(hg, 1)
      color+=1
      if color in [5, 10, 18, 19]:
        color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      if statbox==False:
         hg.SetStats(0)
      else:
         my.stat_pos(hg)
      if stest:
         ks= hd.KolmogorovTest(hg, 'UU NORM')
         legend.AddEntry(hg,"GAN {} K={:.4f} ".format( str(labels[i]), ks) ,"l")
      else:
         legend.AddEntry(hg,"GAN "+ str(labels[i]),"l")
      if statbox:
         if dim == 'z':
           my.stat_pos(hg)
         else:   
           sb1=hg.GetListOfFunctions().FindObject("stats")
           sb1.SetX1NDC(.4)
           sb1.SetX2NDC(.6)
              
      c1.Update()
      my.Max(hd, hg)
      c1.Update()
      
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_sparsity(events1, events2, out_file, energy, labels, threshmin=-13, threshmax=1, logy=0, min_max=0, ifpdf=True, mono=False, leg=True, grid=True, statbox=True, ang=1):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   title = "Sparsity for electrons with 100-200 GeV primary energy"
   legend = ROOT.TLegend(.8, .8, .9, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   thresh = np.arange(threshmin, threshmax, 1)
   count1 = np.zeros((thresh.shape[0], events1.shape[0]))
   count2 =np.zeros((thresh.shape[0], events1.shape[0]))
   size = np.float64(events1[0].size)
   
   # calculating entries for different threshold applied
   for i in np.arange(thresh.shape[0]):
       t_val = np.power(10.0, thresh[i])
       x_t = np.where(np.squeeze(events1)>t_val, 1, 0)
       count1[i] = np.divide(np.sum(x_t, axis=(1, 2, 3)), size)
       x_gen_t = np.where(np.squeeze(events2['n_0']) > t_val, 1, 0)
       count2[i] = np.divide(np.sum(x_gen_t, axis=(1, 2, 3)), size)                                        
   sparsity1 = ROOT.TGraph()
   sparsity2 = ROOT.TGraph()
   sparsity1b = ROOT.TGraph()
   sparsity2b = ROOT.TGraph()

   mean1=np.mean(count1, axis=1)
   mean2=np.mean(count2, axis=1)
   std1=np.std(count1, axis=1)
   std2=np.std(count2, axis=1)
   min1= np.min(count1, axis=1)
   min2= np.min(count2, axis=1)
   max1= np.max(count1, axis=1)
   max2= np.max(count2, axis=1)

   my.fill_graph(sparsity1, thresh, mean1)
   my.fill_graph(sparsity2, thresh, mean2)
   if min_max:
      area1 = np.concatenate((min1, flip(max1, 0)), axis=0)
      area2 = np.concatenate((min2, flip(max2, 0)), axis=0)
      ymax=max(np.max(max2), np.max(max1))
      ylim = 1.1 * ymax
   else:
      area1 = np.concatenate((mean1+std1, flip(mean1-std1, 0)), axis=0)
      area2 = np.concatenate((mean2+std2, flip(mean2-std2, 0)), axis=0)
      ylim = 0.04 if ang else 0.2
   thresh2=np.concatenate((thresh, flip(thresh, 0)), axis=0)
   my.fill_graph(sparsity1b, thresh2, area1)
   my.fill_graph(sparsity2b, thresh2, area2)
   sparsity1.GetXaxis().SetTitle("log10(threshold[GeV])")
   sparsity1.GetYaxis().SetTitle("Fraction of cells above threshold")
   sparsity1.SetTitle(title)
   sparsity1.SetLineColor(color)
   sparsity1.Draw('APL')
   sparsity1.GetYaxis().SetRangeUser(0, ylim)
   sparsity2.SetLineColor(color+2)
   if mono: sparsity2.SetLineStyle(2)
   sparsity2.Draw('PL')

   sparsity1b.SetFillColorAlpha(color, 0.35)
   sparsity2b.SetFillColorAlpha(color+2, 0.35)

   sparsity1b.Draw('f')
   sparsity2b.Draw('F')

   legend.AddEntry(sparsity1,'G4' ,"l")
   legend.AddEntry(sparsity2,'GAN' + labels[0] ,"l")
   c1.Update()
   if leg:
     legend.Draw()
     c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_sparsity2(events1, events2, out_file, energy, labels, threshmin=-13, threshmax=1, logy=0, min_max=0, ifpdf=True, mono=False, leg=True, grid=True, statbox=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   title = "Sparsity for electrons with 100-200 GeV primary energy"
   legend = ROOT.TLegend(.8, .8, .9, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   thresh = np.arange(threshmin, threshmax, 1)
   count1 = np.zeros((thresh.shape[0], events1.shape[0]))
   size = np.float64(events1[0].size)
   count2=[]
   for index, key in enumerate(events2):
     count2.append(np.zeros((thresh.shape[0], events1.shape[0])))
     # calculating entries for different threshold applied
     for i in np.arange(thresh.shape[0]):
       if index==0:
         t_val = np.power(10.0, thresh[i])
         x_t = np.where(np.squeeze(events1)>t_val, 1, 0)
         count1[i] = np.divide(np.sum(x_t, axis=(1, 2, 3)), size)
       x_gen_t = np.where(np.squeeze(events2[key]) > t_val, 1, 0)
       count2[index] = np.divide(np.sum(x_gen_t, axis=(1, 2, 3)), size)                                        
   sparsity1 = ROOT.TGraph()
   sparsity2 = []
   sparsity1b = ROOT.TGraph()
   sparsity2b = []

   mean1=np.mean(count1, axis=1)
   std1=np.std(count1, axis=1)
   min1= np.min(count1, axis=1)
   max1= np.max(count1, axis=1)

   mean2=[]
   std2 =[]
   min2 =[]
   max2 = []
   for count in count2:
      mean2.append(np.mean(count, axis=1))
      std2.append(np.std(count, axis=1))
      min2.append(np.min(count, axis=1))
      max2.append(np.max(count, axis=1))
   my.fill_graph(sparsity1, thresh, mean1)
   for i, mean in enumerate(mean2):
     sparsity2.append(ROOT.TGraph())
     my.fill_graph(sparsity[i], thresh, mean)
   if min_max:
      area1 = np.concatenate((min1, flip(max1, 0)), axis=0)
      area2 =[]
      for min in min2:
        area2.append(np.concatenate((min2, flip(max2, 0)), axis=0))
      ymax=max(np.amax(max2), np.amax(max1))
      ylim = 1.1 * ymax
   else:
      area1 = np.concatenate((mean1+std1, flip(mean1-std1, 0)), axis=0)
      area2 = []
      for mean, std in zip(mean2, std2):
        area2.append(np.concatenate((mean+std, flip(mean-std, 0)), axis=0))
      ylim = 0.04
   thresh2=np.concatenate((thresh, flip(thresh, 0)), axis=0)
   my.fill_graph(sparsity1b, thresh2, area1)
   for i, s in enumerate(sparsity2b):
     my.fill_graph(s, thresh2, area2[i])
   sparsity1.GetXaxis().SetTitle("log10(threshold[GeV])")
   sparsity1.GetYaxis().SetTitle("Fraction of cells above threshold")
   sparsity1.SetTitle(title)
   sparsity1.SetLineColor(color)
   sparsity1.Draw('APL')
   sparsity1b.SetFillColorAlpha(color, 0.35)
   sparsity1b.Draw('f')
   legend.AddEntry(sparsity1,'G4' ,"l")
   sparsity1.GetYaxis().SetRangeUser(0, ylim)
   color+=2
   for i, s in enumerate(sparsity2b):
     s.SetLineColor(color)
     if mono: s.SetLineStyle(2)
     sparsity2[i].Draw('PL')
     s.SetFillColorAlpha(color+2, 0.35)
     s.Draw('F')
     legend.AddEntry(sparsity2[i],'GAN' + labels[0] ,"l")
     c1.Update()
   if leg:
     legend.Draw()
     c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
       c1.Print(out_file + '.C')

################################### Angle Plots ########################################################################
# Plot histogram of predicted angle
def plot_ang_hist(ang1, ang2, out_file, angle, angtype, labels, p, ifpdf=True, grid=True, leg=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hp=ROOT.TH1F("G4" ,"G4", 50, 0, 3)
   hp.Sumw2()
   hp.SetTitle("Angle Histogram for {} degrees({:.4f} rad) {}".format(angle, np.radians(angle),  angtype) )
   hp.GetXaxis().SetTitle(angtype + ' (radians)')
   hp.GetYaxis().SetTitle('Count')
   hp.GetYaxis().CenterTitle()
   my.fill_hist(hp, ang1['n_0'])
   hp.Draw()
   hp =my.normalize(hp, 1)
   hp.Draw('sames hist')
   c1.Update()
   hp.SetLineColor(color)
   c1.Update()
   legend.AddEntry("G4" ,"G4","l")
   color+=2
   hgs=[]
   for i, key in enumerate(ang2):
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 3))
      hg= hgs[i]
      hg.Sumw2()
      my.fill_hist(hg, ang2[key])
      hg =my.normalize(hg, 1)
      hg.SetLineColor(color)
      if mono:  hg.SetLineStyle(color-2)
      color+=1
      if color in [5, 10, 18, 19]:
        color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      my.Max(hp, hg)
      c1.Update()
      if statbox:
         my.stat_pos(hg)
      else:
         hp.SetStats(0)
         hg.SetStats(0)
      c1.Update()
      legend.AddEntry(hg, "GAN {}".format(labels[i] + ' 1'), "l")
   if leg: legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')
            
def plot_angle_error_hist(ang1, ang2, y, out_file, angle, angtype, labels, p, ifpdf=True, grid=True, leg=True, statbox=True, mono=False):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hps=[]
   hgs=[]
   if y.shape[0]> ang1["n_0"].shape[0]:
      y = y[:ang1["n_0"].shape[0]]
   for i, key in enumerate(ang1):
      hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 50, -1.0, 1.0))
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, -1.0, 1.0))
      hp= hps[i]
      hg= hgs[i]
      hp.Sumw2()
      hg.Sumw2()
      if i== 0:
         hp.SetTitle("Angle Relative Error Histogram for {} Degrees ({:.4f}rad) {}".format(angle, np.radians(angle), angtype) )
         hp.GetXaxis().SetTitle(angtype + " radians")
         hp.GetYaxis().SetTitle("(angle_{act} - angle_{pred})/angle_{act}")
         hp.GetYaxis().CenterTitle()
         my.fill_hist(hp, (y - ang1[key])/y)
         hp.Draw()
         hp.Draw('sames hist')
         c1.Update()
         hp.SetLineColor(color)
         c1.Update()
         legend.AddEntry(hp,"G4 " + labels[i],"l")
         color+=2
      else:
         my.fill_hist(hp, (y - ang1[key])/y)
         hp.Draw('sames')
         hp.Draw('sames hist')
         c1.Update()
         legend.AddEntry(hp,"G4" + labels[i],"l")
         color+=1
         if color in [5, 10, 18, 19]:
            color+=1
      my.fill_hist(hg,  (y - ang2[key])/y)
      hp =my.normalize(hp, 1)
      hg =my.normalize(hg, 1)
      hg.SetLineColor(color)
      if mono: hg.SetLineStyle(color)
      color+=1
      if color in [5, 10, 18, 19]:
        color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      my.Max(hp, hg)
      c1.Update()
      if statbox:
         my.stat_pos(hg)
      else:
         hp.SetStats(0)
         hg.SetStats(0)
      c1.Update()
      legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   if leg:  legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')
                                              
def plot_angle_2Dhist(ang1, ang2, y, out_file, angtype, labels, p, ifpdf=True, grid=True, leg=True, norm=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   if grid: c1.SetGrid()
   hps=[]
   for i, key in enumerate(ang1):
      hps.append(ROOT.TH2F("G4" + labels[i],"G4" + labels[i], 50, 0.75, 2.5, 50, 0.75, 2.5))
      hp= hps[i]
      n = y.shape[0]
      hp.SetStats(0)
      hp.SetTitle("2D Histogram for predicted angles from G4 and GAN images" )
      hp.GetXaxis().SetTitle("3d Angle from G4")
      hp.GetYaxis().SetTitle("3d Angle from GAN")
      for j in np.arange(n):
         hp.Fill(ang1[key][j], ang2[key][j])
      if norm: my.normalize(hp)
      hp.Draw("colz")
      c1.Update()
      
      c1.Modified()
      c1.Update()
      if ifpdf:
        c1.Print(out_file + 'n_'+ str(i) + '.pdf')
      else:
        c1.Print(out_file + 'n_'+ str(i) + '.C')

##################################### Get plots for Fixed angle #####################################################################

def get_plots_multi(var, labels, plots_dir, energies, m, n, ifpdf=True, stest=True, cell=0, corr=0, grid=True, leg=True, statbox=True, mono=0):

    actdir = plots_dir + 'Actual'
    safe_mkdir(actdir)
    discdir = plots_dir + 'disc_outputs'
    safe_mkdir(discdir)
    gendir = plots_dir + 'Generated'
    safe_mkdir(gendir)
    comdir = plots_dir + 'Combined'
    safe_mkdir(comdir)
    mdir = plots_dir + 'Moments'
    safe_mkdir(mdir)
    start = time.time()
    plots = 0
    ang = 0 # there is no angle data                                                                                                                                                                                                                                                      
    for energy in energies:
       x=var["events_act" + str(energy)].shape[1]
       y=var["events_act" + str(energy)].shape[2]
       z=var["events_act" + str(energy)].shape[3]
       maxfile = "Position_of_max_" + str(energy)# + ".pdf"                                                                                                                                                                                                                   
       maxlfile = "Position_of_max_" + str(energy)# + "_log.pdf"                                                                                                                                                                                                                          
       histfile = "hist_" + str(energy)# + ".pdf"                                                                                                                                                                                                                                         
       histlfile = "hist_log" + str(energy)# + ".pdf"                                                                                                                                                                                                                                     
       ecalfile = "ecal_" + str(energy)# + ".pdf"                                                                                                                                                                                                                                         
       energyfile = "energy_" + str(energy)# + ".pdf"                                                                                                                                                                                                                                     
       realfile = "realfake_" + str(energy)# + ".pdf"                                                                                                                                                                                                                                     
       momentfile = "moment" + str(energy)# + ".pdf"                                                                                                                                                                                                                                      
       auxfile = "Auxilliary_"+ str(energy)# + ".pdf"                                                                                                                                                                                                                                     
       ecalerrorfile = "ecal_error" + str(energy)# + ".pdf"                                                                                                                                                                                                                               
       allfile = 'All_energies'#.pdf'                                                                                                                                                                                                                                                     
       allecalfile = 'All_ecal'#.pdf'                                                                                                                                                                                                                                                     
       allecalrelativefile = 'All_ecal_relative'#.pdf'                                                                                                                                                                                                                                    
       allauxrelativefile = 'All_aux_relative'#.pdf'                                                                                                                                                                                                                                      
       allerrorfile = 'All_relative_auxerror'#.pdf'                                                                                                                                                                                                                                       
       correlationfile = 'Corr'
       start = time.time()
       if energy==0:
          plot_ecal_ratio_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], var["energy" + str(energy)], labels, os.path.join(comdir, allecalfile), 
                                  ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
          plots+=1
          plot_aux_relative_profile(var["aux_act" + str(energy)], var["aux_gan"+ str(energy)], var["energy"+ str(energy)], os.path.join(comdir, allauxrelativefile), labels,
                                  ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
          plots+=1
          if corr==1:
             plot_correlation(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)], var["ecal_act" + str(energy)],  var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)], var["energy" + str(energy)], var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, correlationfile), labels)
             plots+=1
          elif corr==2:
             plot_correlation_small(var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)], var["ecal_act" + str(energy)],  var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)], var["energy" + str(energy)], var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, correlationfile+ "small"), labels)
             plots+=1
                                                                              
          if cell:
             plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'flat' + ecalfile), energy, labels,
                                    ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
             plots+=1
             plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'flat_log' + ecalfile), energy, labels, log=1,
                                    ifpdf=ifpdf,  grid=grid, leg=leg, statbox=statbox, mono=mono)
             plots+=1
          plot_sparsity(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'spartsity'), energy, labels,
                       threshmin=-13, threshmax=1, logy=0, min_max=0, ifpdf=ifpdf, mono=mono,
                        leg=leg, grid=grid, statbox=statbox, ang=0)

       plot_ecal_hist(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], os.path.join(discdir, ecalfile), energy, labels, 
                      ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       if cell>1:
          plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'flat' + ecalfile), energy, labels,
                                  ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
          plots+=1
          plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'flat' + ecalfile), energy, labels,
                                  ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
          plots+=1
                    
       plot_ecal_hits_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'hits' + ecalfile), energy, labels, 
                                  ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       plot_aux_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)] , os.path.join(discdir, energyfile), energy, labels,
                     ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)], x, y, z, os.path.join(actdir, maxfile), os.path.join(gendir, maxfile), os.path.join(comdir, maxfile), energy, labels, 
               ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)], x, y, z, os.path.join(actdir, maxlfile), os.path.join(gendir, maxlfile), os.path.join(comdir, 'log' + maxlfile), 
               energy, labels, log=1, ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], 
               var["sumsz_gan"+ str(energy)], x, y, z, os.path.join(actdir, histfile), os.path.join(gendir,histfile), os.path.join(comdir, histfile), energy, labels, 
               ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], 
               var["sumsz_gan"+ str(energy)], x, y, z, os.path.join(actdir, histlfile), os.path.join(gendir, histlfile), os.path.join(comdir, histlfile), energy, labels, 
                             log=1, ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       plot_realfake_hist(var["isreal_act" + str(energy)], var["isreal_gan" + str(energy)], os.path.join(discdir, realfile), energy, labels,
                          ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       plot_primary_error_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)], var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile), 
                               energy, labels, ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
       plots+=1
       for mmt in range(m):
          plot_moment(var["momentX_act" + str(energy)], var["momentX_gan" + str(energy)], os.path.join(mdir, 'x' + str(mmt + 1) + momentfile), 'x', energy, mmt, labels, 
                      ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono, ang=0)
          plots+=1
          plot_moment(var["momentY_act" + str(energy)], var["momentY_gan" + str(energy)], os.path.join(mdir, 'y' + str(mmt + 1) + momentfile), 'y', energy, mmt, labels, 
                      ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono, ang=0)
          plots+=1
          plot_moment(var["momentZ_act" + str(energy)], var["momentZ_gan" + str(energy)], os.path.join(mdir, 'z' + str(mmt + 1) + momentfile), 'z', energy, mmt, labels, 
                      ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono, ang=0)
          plots+=1

    print ('Plots are saved in ', plots_dir)
    plot_time= time.time()- start
    print ('{} Plots are generated in {} seconds'.format(plots, plot_time))


##################################### Get plots for variable angle #####################################################################

def get_plots_angle(var, labels, plots_dir, energies, angles, angtype, m, n, ifpdf=True, stest=True, angloss=1, addloss=0, cell=0, corr=0, grid=True, leg=True, statbox=True, mono=False):
   actdir = plots_dir + 'Actual'
   safe_mkdir(actdir)
   discdir = plots_dir + 'disc_outputs'
   safe_mkdir(discdir)
   gendir = plots_dir + 'Generated'
   safe_mkdir(gendir)
   comdir = plots_dir + 'Combined'
   safe_mkdir(comdir)
   mdir = plots_dir + 'Moments'
   safe_mkdir(mdir)
   start = time.time()
   plots = 0
   ang=1
   for energy in energies:
      x=var["events_act" + str(energy)].shape[1]
      y=var["events_act" + str(energy)].shape[2]
      z=var["events_act" + str(energy)].shape[3]
      maxfile = "Position_of_max_" + str(energy)
      maxlfile = "Position_of_max_" + str(energy)
      histfile = "hist_" + str(energy)
      histlfile = "hist_log" + str(energy)
      ecalfile = "ecal_" + str(energy)
      energyfile = "energy_" + str(energy)
      realfile = "realfake_" + str(energy)
      momentfile = "moment" + str(energy)
      auxfile = "Auxilliary_"+ str(energy)
      ecalerrorfile = "ecal_error" + str(energy)
      angfile = "angle_"+ str(energy)
      aerrorfile = "error_"
      allfile = 'All_energies'
      allecalfile = 'All_ecal'
      allecalrelativefile = 'All_ecal_relative'#.pdf'
      allauxrelativefile = 'All_aux_relative'#.pdf'
      allerrorfile = 'All_relative_auxerror'#.pdf'
      correlationfile = 'Corr'
      if 0 in energies:
         pmin = np.amin(var["energy" + str(energy)])
         pmax = np.amax(var["energy" + str(energy)])
         p = [int(pmin), int(pmax)]
      else:
         p = [100, 200]
                           
      if energy==0:
         plot_ecal_ratio_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                                    var["energy" + str(energy)], labels, os.path.join(comdir, allecalfile),
                                     p, ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         plot_ecal_relative_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                                    var["energy" + str(energy)], labels, os.path.join(comdir, allecalrelativefile),
                                    p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         plot_aux_relative_profile(var["aux_act" + str(energy)], var["aux_gan"+ str(energy)], 
                                   var["energy"+ str(energy)], os.path.join(comdir, allauxrelativefile),
                                   labels, p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono, stest=stest)
         plots+=1
         if corr==1:                                                                                                                
           plot_correlation(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],    
                           var["sumsz_act"+ str(energy)], var["momentX_act" + str(energy)],                                  
                           var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)],                               
                           var["ecal_act" + str(energy)],  var["sumsx_gan"+ str(energy)],                                    
                           var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],                                     
                           var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)],                               
                           var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)],                                  
                           var["energy" + str(energy)], var["events_act" + str(energy)],                                     
                            var["events_gan" + str(energy)], os.path.join(comdir, correlationfile), labels, leg=leg)
         elif corr>1:
           plot_correlation_small(var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)], var["ecal_act" + str(energy)],  var["momentX_gan" + str(energy)],
                                  var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)], var["energy" + str(energy)], var["events_act" + str(energy)],
                                  var["events_gan" + str(energy)], os.path.join(comdir, correlationfile+ "small"), labels, leg=leg, stest=stest, ang= var["angle" + str(energy)] )
           plots+=1
                                         
         if cell:
           plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], 
                                  os.path.join(comdir, 'flat' + 'log' + ecalfile), energy, labels, p=p,
                                  log=1, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
           plots+=1
           plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                  os.path.join(comdir, 'flat' + ecalfile), energy, labels, p=p,
                                  ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
           plots+=1                            
         plot_sparsity(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'spartsity'), energy, labels,
                       threshmin=-13, threshmax=1, logy=0, min_max=0, ifpdf=ifpdf, mono=mono,
                       leg=leg, grid=grid, statbox=statbox)
         plots+=1
      plot_ecal_hist(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                     os.path.join(discdir, ecalfile), energy, labels, p, stest=stest,
                     ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      if cell>1:
         plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], 
                                os.path.join(comdir, 'flat' + 'log' + ecalfile), energy, labels, p=p,
                                log=1, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1     
         plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], 
                                os.path.join(comdir, 'flat' + ecalfile), energy, labels, p=p,
                                ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)  
      plots+=1                                                                                                             
      plot_ecal_hits_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                os.path.join(comdir, 'hits' + ecalfile), energy, labels, p, stest=stest,
                                ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_aux_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)] , 
                    os.path.join(discdir, energyfile), energy, labels, p,
                    ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)],
               x, y, z, os.path.join(actdir, maxfile), os.path.join(gendir, maxfile),
               os.path.join(comdir, maxfile), energy, labels, p=p,
               stest=stest, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)],
               x, y, z, os.path.join(actdir, maxlfile),
               os.path.join(gendir, maxlfile), os.path.join(comdir, 'log' + maxlfile),
               energy, labels, log=1, p=p, stest=stest, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],
                               var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)],
                               var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],
                               x, y, z, os.path.join(actdir, histfile), os.path.join(gendir, histfile),
                               os.path.join(comdir, histfile), energy, labels, p=p, stest=stest,
                               ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],       
                            var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)],
                            var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],
                            x, y, z, os.path.join(actdir, histlfile), os.path.join(gendir, histlfile),
                            os.path.join(comdir, histlfile), energy, labels, log=1, p=p, stest=stest,
                            ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_realfake_hist(var["isreal_act" + str(energy)], var["isreal_gan" + str(energy)],
                         os.path.join(discdir, realfile), energy, labels, p, stest=stest,
                         ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_primary_error_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)],
                         var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile),
                         energy, labels, p, grid=grid, ifpdf=ifpdf, statbox=statbox, mono=mono)
      plots+=1
      plot_angle_2Dhist(var["angle_act" + str(energy)], var["angle_gan" + str(energy)],  var["angle" + str(energy)],
                        os.path.join(discdir, angfile + "ang_2D") , angtype, labels, p,
                        ifpdf=ifpdf, grid=grid, leg=leg)
      plots+=1
      if angloss==2:
         plot_angle_2Dhist(var["angle2_act" + str(energy)], var["angle2_gan" + str(energy)],  var["angle" + str(energy)],
                           os.path.join(discdir, angfile + "ang2_2D") , angtype,
                           labels, p, ifpdf=ifpdf, grid=grid, leg=leg)
         plots+=1
      for mmt in range(m):
         plot_moment(var["momentX_act" + str(energy)], var["momentX_gan" + str(energy)],
                     os.path.join(mdir, 'x' + str(mmt + 1) + momentfile), 'x', energy, mmt,
                     labels, p, ifpdf=ifpdf, grid=grid, leg=leg, mono=mono, stest=stest, statbox=statbox)
         plots+=1
         plot_moment(var["momentY_act" + str(energy)], var["momentY_gan" + str(energy)],
                     os.path.join(mdir, 'y' + str(mmt + 1) + momentfile), 'y', energy, mmt,
                     labels, p, ifpdf=ifpdf, grid=grid, leg=leg, mono=mono, stest=stest, statbox=statbox)
         plots+=1
         plot_moment(var["momentZ_act" + str(energy)], var["momentZ_gan" + str(energy)],
                     os.path.join(mdir, 'z' + str(mmt + 1) + momentfile), 'z', energy, mmt,
                     labels, p, ifpdf=ifpdf, grid=grid, leg=leg, mono=mono, stest=stest, statbox=statbox)
         plots+=1

      ecomdir = os.path.join(comdir, 'energy_' + str(energy))
      safe_mkdir(ecomdir)
      ediscdir = os.path.join(discdir, 'energy_' + str(energy))
      safe_mkdir(ediscdir)
      eactdir = os.path.join(actdir, 'energy_' + str(energy))
      safe_mkdir(eactdir)
      egendir = os.path.join(gendir, 'energy_' + str(energy))
      safe_mkdir(egendir)
      for index, a in enumerate(angles):
         #alabels = ['ang_' + str() for _ in aindexes]
         alabels = ['angle_{} {}'.format(a, _) for _ in labels]
         a2labels = ['angle2_{} {}'.format(a, _) for _ in labels]
         plot_energy_hist_root(var["sumsx_act"+ str(energy) + "ang_" + str(a)], var["sumsy_act"+ str(energy)+ "ang_" + str(a)],
                                  var["sumsz_act"+ str(energy) + "ang_" + str(a)], var["sumsx_gan"+ str(energy)+ "ang_" + str(a)],
                                  var["sumsy_gan"+ str(energy)+ "ang_" + str(a)], var["sumsz_gan"+ str(energy)+ "ang_" + str(a)],
                                  x, y, z, os.path.join(eactdir, histfile + 'ang_' + str(a)), os.path.join(egendir, histfile+ 'ang_' + str(a)),
                                  os.path.join(ecomdir, histfile+ 'ang_' + str(a)), energy, alabels, p=p, stest=stest,
                                  ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         plot_energy_hist_root(var["sumsx_act"+ str(energy) + "ang_" + str(a)], var["sumsy_act"+ str(energy)+ "ang_" + str(a)],
                               var["sumsz_act"+ str(energy) + "ang_" + str(a)], var["sumsx_gan"+ str(energy)+ "ang_" + str(a)],
                               var["sumsy_gan"+ str(energy)+ "ang_" + str(a)], var["sumsz_gan"+ str(energy)+ "ang_" + str(a)],
                               x, y, z, os.path.join(eactdir, histfile + 'ang_' + str(a)), os.path.join(egendir, histfile+ 'ang_' + str(a)),
                               os.path.join(ecomdir, histfile+ 'logang_' + str(a)), energy, alabels, log=1, p=p, stest=stest,
                               ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         plot_ang_hist(var["angle_act" + str(energy) + "ang_" + str(a)], var["angle_gan" + str(energy) + "ang_" + str(a)] ,
                       os.path.join(ediscdir, "ang_" + str(a)), a, angtype, alabels, p=p, ifpdf=ifpdf,
                       grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         plot_angle_error_hist(var["angle_act" + str(energy) + "ang_" + str(a)], var["angle_gan" + str(energy) + "ang_" + str(a)],
                               var["angle" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, aerrorfile + "ang2_" + str(a)),
                               a, angtype, alabels, p=p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1

         if angloss == 2:
            plot_ang_hist(var["angle2_act" + str(energy) + "ang_" + str(a)], var["angle2_gan" + str(energy) + "ang_" + str(a)] ,
                          os.path.join(ediscdir, "ang2_" + str(a)), a, angtype, a2labels, p=p, ifpdf=ifpdf,
                          grid=grid, leg=leg, statbox=statbox, mono=mono)
            plots+=1
            plot_angle_error_hist(var["angle2_act" + str(energy) + "ang_" + str(a)], var["angle2_gan" + str(energy) + "ang_" + str(a)],
                                  var["angle" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, aerrorfile + "ang2_" + str(a)),
                                  a, angtype, a2labels, p=p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
            plots+=1
                                             
         plot_realfake_hist(var["isreal_act" + str(energy) + "ang_" + str(a)], var["isreal_gan" + str(energy)+ "ang_" + str(a)],
                            os.path.join(ediscdir, realfile  + "ang_" + str(a)), a, alabels, p,
                            ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         plot_primary_error_hist(var["aux_act" + str(energy) + "ang_" + str(a)], var["aux_gan" + str(energy) + "ang_" + str(a)],
                      var["energy" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, 'error_' + energyfile + "ang_" + str(a)),
                                 energy, alabels, p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         if cell==3:
            plot_ecal_flatten_hist(var["events_act" + str(energy) + "ang_" + str(a)], var["events_gan" + str(energy) + "ang_" + str(a)],
                                   os.path.join(ecomdir, 'flat' + 'log' + ecalfile + "ang_" + str(a)), energy, labels,
                                   p=p, log=1, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
            plots+=1
            plot_ecal_flatten_hist(var["events_act" + str(energy) + "ang_" + str(a)], var["events_gan" + str(energy) + "ang_" + str(a)],
                                   os.path.join(ecomdir, 'flat' + ecalfile + "ang_" + str(a)), energy, labels,
                                   p=p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
            plots+=1
         if corr==3:
            plot_correlation_small(var["momentX_act" + str(energy)+ "ang_" + str(a)], var["momentY_act" + str(energy)+ "ang_" + str(a)],
                                   var["momentZ_act" + str(energy)+ "ang_" + str(a)], var["ecal_act" + str(energy)+ "ang_" + str(a)],
                                   var["momentX_gan" + str(energy)+ "ang_" + str(a)], var["momentY_gan" + str(energy)+ "ang_" + str(a)],
                                   var["momentZ_gan" + str(energy)+ "ang_" + str(a)], var["ecal_gan" + str(energy)+ "ang_" + str(a)],
                                   var["energy" + str(energy)+ "ang_" + str(a)], var["events_act" + str(energy)+ "ang_" + str(a)],
                                   var["events_gan" + str(energy)+ "ang_" + str(a)], os.path.join(ecomdir, correlationfile+ "small"+ "ang_" + str(a)), labels, leg=leg)
            plots+=1
                       
   print ('Plots are saved in ', plots_dir)
   plot_time= time.time()- start
   print ('{} Plots are generated in {} seconds'.format(plots, plot_time))
                 
################################################# Plots for 2D coloured histograms ########################################################

def PlotEnergyHistGen(events, out_file, energy, thetas, log=0, ifC=False):
   canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,700 ,500) #make
   canvas.SetGrid()
   label = "Weighted Histograms for {} GeV".format(energy)
   canvas.Divide(2,2)
   color = 2
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.05)
   hx=[]
   hy=[]
   hz=[]
   thetas = list(reversed(thetas))
   for i, theta in enumerate(thetas):
      event = events[str(theta)]
      num = event.shape[0]
      sumx, sumy, sumz=get_sums(event)
      x=sumx.shape[1]
      y=sumy.shape[1]
      z=sumz.shape[1]
      hx.append(ROOT.TH1F('GANx{:d}theta_{:d}GeV'.format(theta, energy), '', x, 0, x))
      hy.append(ROOT.TH1F('GANy{:d}theta_{:d}GeV'.format(theta, energy), '', y, 0, y))
      hz.append(ROOT.TH1F('GANz{:d}theta_{:d}GeV'.format(theta, energy), '', z, 0, z))
      hx[i].SetLineColor(color)
      hy[i].SetLineColor(color)
      hz[i].SetLineColor(color)
      hx[i].GetXaxis().SetTitle("X axis")
      hy[i].GetXaxis().SetTitle("Y axis")
      hz[i].GetXaxis().SetTitle("Z axis")
      hx[i].Sumw2()
      hy[i].Sumw2()
      hz[i].Sumw2()
      canvas.cd(1)
      if log:
         gPad.SetLogy()
      my.fill_hist_wt(hx[i], sumx)
      if i ==0:
         hx[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hx[i].DrawNormalized('sames hist')
      canvas.cd(2)
      if log:
         gPad.SetLogy()
      my.fill_hist_wt(hy[i], sumy)
      if i==0:
         hy[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hy[i].DrawNormalized('sames hist')
      canvas.cd(3)
      if log:
         gPad.SetLogy()
      my.fill_hist_wt(hz[i], sumz)
      if i==0:
         hz[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hz[i].DrawNormalized('sames hist')

      canvas.cd(4)
      leg.AddEntry(hx[i], '{}theta {}events'.format(theta, num),"l")
      leg.SetHeader(label, 'C')
      canvas.Update()
      color+= 1
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file + '.pdf')
   if ifC:
      canvas.Print(out_file + '.C')

def MeasPython(image, mod=0):
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
   if mod==0:
      ang = np.sum(ang, axis=1)/zunmasked_events
   if mod==1:
      wang = ang * sumz
      sumz_tot = sumz * zmask
      ang = np.sum(wang, axis=1)/np.sum(sumz_tot, axis=1)
   if mod==2:
      wang = ang * z
      sumz_tot = z * zmask
      ang = np.sum(wang, axis=1)/np.sum(sumz_tot, axis=1)
   indexes = np.where(amask>0)
   ang[indexes] = 100.
   return ang

def PlotEvent(event, energy, theta, out_file, n, opt="", unit='degrees', label=""):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   
   ang1 = MeasPython(np.moveaxis(event, 3, 0))
   ang2 = MeasPython(np.moveaxis(event, 3, 0), mod=2)
   if unit == 'degrees':
      ang1= np.degrees(ang1)
      ang2= np.degrees(ang2)
      theta = np.degrees(theta)
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   leg.SetTextSize(0.05)
   leg.SetHeader("#splitline{Weighted Histograms for energies}{deposited in x, y and z planes}", "C")
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}'.format(energy, theta), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, y, 0, y)
   hx.SetStats(0)
   hy.SetStats(0)
   hz.SetStats(0)
   #ROOT.gPad.SetLogz()
   ROOT.gStyle.SetPalette(1)
   event = np.expand_dims(event, axis=0)
   my.FillHist2D_wt(hx, np.sum(event, axis=1))
   my.FillHist2D_wt(hy, np.sum(event, axis=2))
   my.FillHist2D_wt(hz, np.sum(event, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Single {} event'.format(label),"l")
   leg.AddEntry(hy, 'Energy Input = {:.2f} GeV'.format(energy),"l")
   leg.AddEntry(hz, 'Theta Input  = {:.2f} {}'.format(theta, unit),"l")
   #leg.AddEntry(hz, 'Computed Theta (mean)     = {:.2f} {}'.format(ang1[0], unit),"l")
   #leg.AddEntry(hz, 'Computed Theta (weighted) = {:.2f} {}'.format(ang2[0], unit),"l")
   leg.Draw()
   #my.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotEvent2(aevent, gevent, energy, theta, out_file, n, opt="", unit='degrees', label="", logz=0, ifC=0):
   x = aevent.shape[0]
   print("aevent shape x:", x)
   y = aevent.shape[1]
   print("aevent shape y:", y)
   z = aevent.shape[2]
   print("aevent shape z:", z)
   if x==25:
     canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,800 ,400)
   else:
     canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,900 ,400)
   lsize = 0.04 # axis label size
   tsize = 0.08 # axis title size
   tmargin = 0.02
   bmargin = 0.15
   lmargin = 0.15
   rmargin = 0.17
   if theta:
     #ang1 = MeasPython(np.moveaxis(aevent, 3, 0))
     ang1 = MeasPython(np.moveaxis(aevent, 3, 0), mod=2)
     ang2 = MeasPython(np.moveaxis(gevent, 3, 0), mod=2)
     if unit == 'degrees':
        ang1= np.degrees(ang1)
        ang2= np.degrees(ang2)
        theta = np.degrees(theta)
     
     title = ROOT.TPaveLabel(0.1,0.95,0.9,0.99,"Ep = {:.2f} GeV #theta={:.2f} #circ  meas#theta G4={:.2f} #circ meas#theta GAN={:.2f} #circ ".format(energy, theta, ang1[0], ang2[0]))
   else:
     title = ROOT.TPaveLabel(0.1,0.95,0.9,0.99,"Ep = {:.2f} GeV".format(energy))
   title.SetFillStyle(0)
   title.SetLineColor(0)
   title.SetBorderSize(1)
   title.Draw()
   graphPad = ROOT.TPad("Graphs","Graphs",0.01,0.01,0.95,0.95)
   graphPad.Draw()
   graphPad.cd()
   graphPad.Divide(3,2)
   hx1 = ROOT.TH2F('x1_{:.2f}GeV'.format(energy), '', y, 0, y, z, 0, z)
   hy1 = ROOT.TH2F('y1_{:.2f}GeV'.format(energy), '', x, 0, x, z, 0, z)
   hz1 = ROOT.TH2F('z1_{:.2f}GeV'.format(energy), '', x, 0, x, y, 0, y)
   hx2 = ROOT.TH2F('x2_{:.2f}GeV'.format(energy), '', y, 0, y, z, 0, z)
   hy2 = ROOT.TH2F('y2_{:.2f}GeV'.format(energy), '', x, 0, x, z, 0, z)
   hz2 = ROOT.TH2F('z2_{:.2f}GeV'.format(energy), '', x, 0, x, y, 0, y)
   hx1.SetStats(0)
   hy1.SetStats(0)
   hz1.SetStats(0)
   hx2.SetStats(0)
   hy2.SetStats(0)
   hz2.SetStats(0)

   ROOT.gStyle.SetPalette(1)
   #aevent = np.expand_dims(aevent, axis=-1)
   #print("aevent shape (after expand):", aevent.shape)
   
   hx1_array = np.sum(aevent, axis=0).squeeze()  # Shape should be (51, 25)
   hy1_array = np.sum(aevent, axis=1).squeeze()  # Shape should be (51, 25)
   hz1_array = np.sum(aevent, axis=2).squeeze()  # Shape should be (51, 51)

   print("np.sum(aevent, axis=0) shape before fillhist:", hx1_array.shape)
   print("np.sum(aevent, axis=1) shape before fillhist:", hy1_array.shape)
   print("np.sum(aevent, axis=2) shape before fillhist:", hz1_array.shape)

   my.FillHist2D_wt(hx1, hx1_array)
   my.FillHist2D_wt(hy1, hy1_array)
   my.FillHist2D_wt(hz1, hz1_array)


   # my.FillHist2D_wt(hx1, np.sum(aevent, axis=0))
   # my.FillHist2D_wt(hy1, np.sum(aevent, axis=1))
   # my.FillHist2D_wt(hz1, np.sum(aevent, axis=2))
   #gevent = np.expand_dims(gevent, axis=-1)
   #print("gevent shape (after expand):", gevent.shape)

   hx2_array = np.sum(gevent, axis=0).squeeze()  # Shape should be (51, 25)
   hy2_array = np.sum(gevent, axis=1).squeeze()  # Shape should be (51, 25)
   hz2_array = np.sum(gevent, axis=2).squeeze()  # Shape should be (51, 51)
   
   print("np.sum(gevent, axis=0) shape before fillhist:", hx2_array.shape)
   print("np.sum(gevent, axis=1) shape before fillhist:", hy2_array.shape)
   print("np.sum(gevent, axis=2) shape before fillhist:", hz2_array.shape)

   my.FillHist2D_wt(hx2, hx2_array)
   my.FillHist2D_wt(hy2, hy2_array)
   my.FillHist2D_wt(hz2, hz2_array)

   # my.FillHist2D_wt(hx2, np.sum(gevent, axis=0))
   # my.FillHist2D_wt(hy2, np.sum(gevent, axis=1))
   # my.FillHist2D_wt(hz2, np.sum(gevent, axis=2))
   Min = 1e-4
   Max = 1e-1
   graphPad.cd(1)
   if logz: ROOT.gPad.SetLogz(1)
   hx1.Draw('col')
   hx1.GetXaxis().SetTitle("Y")
   hx1.GetYaxis().SetTitle("Z")
   hx1.GetYaxis().CenterTitle()
   hx1.GetXaxis().SetLabelSize(lsize)
   hx1.GetYaxis().SetLabelSize(lsize)
   hx1.GetXaxis().SetTitleSize(tsize)
   hx1.GetYaxis().SetTitleSize(tsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   if x==51:
     ROOT.gPad.SetRightMargin(0)
   else:
     ROOT.gPad.SetRightMargin(rmargin)
   hx1.SetMinimum(Min)
   hx1.SetMaximum(Max)
   canvas.Update()
   canvas.Update()
   graphPad.cd(2)
   if logz: ROOT.gPad.SetLogz(1)
   hy1.Draw('col')
   hy1.GetXaxis().SetTitle("X")
   hy1.GetYaxis().SetTitle("Z")
   hy1.GetYaxis().CenterTitle()
   hy1.GetXaxis().SetLabelSize(lsize)
   hy1.GetYaxis().SetLabelSize(lsize)
   hy1.GetXaxis().SetTitleSize(tsize)
   hy1.GetYaxis().SetTitleSize(tsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   if x==51:
     ROOT.gPad.SetRightMargin(0)
   else:
     ROOT.gPad.SetRightMargin(rmargin)
   hy1.SetMinimum(Min)
   hy1.SetMaximum(Max)

   canvas.Update()
   canvas.Update()
   graphPad.cd(3)
   if logz: ROOT.gPad.SetLogz(1)
   hz1.Draw(opt)
   hz1.GetXaxis().SetTitle("X")
   hz1.GetYaxis().SetTitle("Y")
   hz1.GetYaxis().CenterTitle()
   hz1.GetXaxis().SetLabelSize(lsize)
   hz1.GetYaxis().SetLabelSize(lsize)
   hz1.GetXaxis().SetTitleSize(tsize)
   hz1.GetYaxis().SetTitleSize(tsize)
   hz1.GetZaxis().SetLabelSize(lsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(rmargin)
   hz1.SetMinimum(Min)
   hz1.SetMaximum(Max)

   canvas.Update()
   graphPad.cd(4)
   if logz: ROOT.gPad.SetLogz(1)
   hx2.Draw('col')
   hx2.GetXaxis().SetTitle("Y")
   hx2.GetYaxis().SetTitle("Z")
   hx2.GetYaxis().CenterTitle()
   hx2.GetXaxis().SetLabelSize(lsize)
   hx2.GetYaxis().SetLabelSize(lsize)
   hx2.GetXaxis().SetTitleSize(tsize)
   hx2.GetYaxis().SetTitleSize(tsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   if x==51:
     ROOT.gPad.SetRightMargin(0)
   else:
     ROOT.gPad.SetRightMargin(rmargin)

   hx2.SetMinimum(Min)
   hx2.SetMaximum(Max)

   canvas.Update()
   graphPad.cd(5)
   if logz: ROOT.gPad.SetLogz(1)
   hy2.Draw('col')
   hy2.GetXaxis().SetTitle("X")
   hy2.GetYaxis().SetTitle("Z")
   hy2.GetYaxis().CenterTitle()
   hy2.GetXaxis().SetLabelSize(lsize)
   hy2.GetYaxis().SetLabelSize(lsize)
   hy2.GetXaxis().SetTitleSize(tsize)
   hy2.GetYaxis().SetTitleSize(tsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   if x==51:
     ROOT.gPad.SetRightMargin(0)
   else:
     ROOT.gPad.SetRightMargin(rmargin)

   hy2.SetMinimum(Min)
   hy2.SetMaximum(Max)

   canvas.Update()
   graphPad.cd(6)
   if logz: ROOT.gPad.SetLogz(1)
   hz2.Draw(opt)
   hz2.GetXaxis().SetTitle("X")
   hz2.GetYaxis().SetTitle("Y")
   hz2.GetYaxis().CenterTitle()
   hz2.GetXaxis().SetLabelSize(lsize)
   hz2.GetYaxis().SetLabelSize(lsize)
   hz2.GetZaxis().SetLabelSize(lsize)
   hz2.GetXaxis().SetTitleSize(tsize)
   hz2.GetYaxis().SetTitleSize(tsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(rmargin)
   hz2.SetMinimum(Min)
   hz2.SetMaximum(Max)
   canvas.Update()

   canvas.Update()
   canvas.Print(out_file + '.pdf')
   if ifC: canvas.Print(out_file + '.C')

def PlotEventFixed(event, energy, out_file, n, opt="", label="", log=0):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   leg.SetTextSize(0.05)
   leg.SetHeader("#splitline{Weighted Histograms for energies}{deposited in x, y and z planes}", "C")
   hx = ROOT.TH2F('x_{:.2f}GeV'.format(energy), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV'.format(energy), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV'.format(energy), '', x, 0, x, y, 0, y)
   hx.SetStats(0)
   hy.SetStats(0)
   hz.SetStats(0)
   if log:
      ROOT.gPad.SetLogz()
   ROOT.gStyle.SetPalette(1)
   event = np.expand_dims(event, axis=0)
   my.FillHist2D_wt(hx, np.sum(event, axis=1))
   my.FillHist2D_wt(hy, np.sum(event, axis=2))
   my.FillHist2D_wt(hz, np.sum(event, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Single {} event'.format(label),"l")
   leg.AddEntry(hy, 'Energy Input = {:.2f} GeV'.format(energy),"l")
   
   leg.Draw()
   #my.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotEventCut(event, energy, theta, out_file, n, opt="", unit='degrees', label=""):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   x2 = event.shape[0]/2
   y2 = event.shape[1]/2
   z2 = event.shape[2]/2
         
   ang1 = MeasPython(np.moveaxis(event, 3, 0))
   ang2 = MeasPython(np.moveaxis(event, 3, 0), mod=2)
   if unit == 'degrees':
     ang1= np.degrees(ang1)
     ang2= np.degrees(ang2)
     theta = np.degrees(theta)
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   leg.SetTextSize(0.05)
   leg.SetHeader("#splitline{Weighted Histograms for energies deposited}{in sections through x, y, z}", 'C')
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}'.format(energy, theta), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, y, 0, y)
   hx.SetStats(0)
   hy.SetStats(0)
   hz.SetStats(0)
   #ROOT.gPad.SetLogz()
   ROOT.gStyle.SetPalette(1)
   event = np.expand_dims(event, axis=0)
   my.FillHist2D_wt(hx, event[:,x2,:,:])
   my.FillHist2D_wt(hy, event[:,:,y2,:])
   my.FillHist2D_wt(hz, event[:,:,:,z2])
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Single {} event at:'.format(label),"l")
   leg.AddEntry(hy, 'Energy Input = {:.2f} GeV'.format(energy),"l")
   leg.AddEntry(hz, 'Theta Input  = {:.2f} {}'.format(theta, unit),"l")
   #leg.AddEntry(hz, 'Computed Theta (mean)     = {:.2f} {}'.format(ang1[0], unit),"l")
   #leg.AddEntry(hz, 'Computed Theta (weighted) = {:.2f} {}'.format(ang2[0], unit),"l")
   leg.Draw()
   #my.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)
                        
def PlotEventCutFixed(event, energy, out_file, n, opt="", label=""):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   x2 = event.shape[0]/2
   y2 = event.shape[1]/2
   z2 = event.shape[2]/2
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   leg.SetTextSize(0.05)
   leg.SetHeader("#splitline{Weighted Histograms for energies deposited}{in sections through x, y, z}", 'C')
   hx = ROOT.TH2F('x_{:.2f}GeV'.format(energy), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV'.format(energy), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV'.format(energy), '', x, 0, x, y, 0, y)
   hx.SetStats(0)
   hy.SetStats(0)
   hz.SetStats(0)
   #ROOT.gPad.SetLogz()
   ROOT.gStyle.SetPalette(1)
   event = np.expand_dims(event, axis=0)
   my.FillHist2D_wt(hx, event[:,x2,:,:])
   my.FillHist2D_wt(hy, event[:,:,y2,:])
   my.FillHist2D_wt(hz, event[:,:,:,z2])
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Single {} event at:'.format(label),"l")
   leg.AddEntry(hy, 'Energy Input = {:.2f} GeV'.format(energy),"l")
   
   leg.Draw()
   #my.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotAngleCut(events, ang, out_file, opt=""):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500)
   canvas.Divide(2,2)
   n = events.shape[0]
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
   ROOT.gStyle.SetPalette(1)
   ROOT.gPad.SetLogz()
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.05)
   hx = ROOT.TH2F('X{} Degree'.format(str(ang)), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('Y{} Degree'.format(str(ang)), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('Z{} Degree'.format(str(ang)), '', x, 0, x, y, 0, y)
   my.FillHist2D_wt(hx, np.sum(events, axis=1))
   my.FillHist2D_wt(hy, np.sum(events, axis=2))
   my.FillHist2D_wt(hz, np.sum(events, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   my.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hy.GetYaxis().CenterTitle()
   canvas.Update()
   my.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hz.GetYaxis().CenterTitle()
   canvas.cd(4)
   leg.SetHeader("#splitline{Weighted Histograms for energies}{deposited in x, y, z planes}", 'C')
   leg.AddEntry(hx, "{} Theta and {} events".format(ang, n), 'l')
   leg.Draw()
   canvas.Update()
   my.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)
                                                                                                                                             
def PlotPosCut(events, xcut, ycut, zcut, energy, out_file, opt='colz'):
   canvas = ROOT.TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   ROOT.gPad.SetLogz()
   ROOT.gStyle.SetPalette(1)
   n = events.shape[0]
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
            
   hx = ROOT.TH2F('x_{}GeV_x={}cut'.format(str(energy), str(xcut)), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{}GeV_y={}cut'.format(str(energy), str(ycut)), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{}GeV_z={}cut'.format(str(energy), str(zcut)), '', x, 0, x, y, 0, y)
   my.FillHist2D_wt(hx, events[:, xcut, :, :])
   my.FillHist2D_wt(hy, events[:, :, ycut, :])
   my.FillHist2D_wt(hz, events[:, :, :, zcut])
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   canvas.Update()
   canvas.Print(out_file)
                                                                                 
