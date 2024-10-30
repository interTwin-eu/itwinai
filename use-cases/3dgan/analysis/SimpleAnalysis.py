import sys
from os import path
import argparse
import h5py
import ROOT
import numpy as np
import time
import keras.backend as K
from utils.GANutils import GetData, GetAngleData, GetDataFiles, generate, safe_mkdir
import utils.ROOTutils as my
import time
sys.path.insert(0,'../')
try:
    import setGPU #if Caltech                                                                                                             
except:
    pass

def get_parser():
   # To parse the input parameters to the script
   parser = argparse.ArgumentParser()
   parser.add_argument('--gweights', type=str, nargs='+', default=['../weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5'], help="Complete PATH to the trained weights to test with hdf5 extension")
   parser.add_argument('--ang', default=1, type=int, help="If using angle vefsion")
   parser.add_argument('--particle', default='Ele', type=str, help="particle type")
   parser.add_argument('--labels', type=str, nargs='+', default=[''], help="labels for different weights")
   parser.add_argument('--xscales',  type=float, nargs='+', help="scaling factor for cell energies")
   parser.add_argument('--datapath', default='full', help='Data to check the output obtained')
   parser.add_argument('--outdir', default= 'results/short_analysis', help='Complete PATH to save the output plot')
   parser.add_argument('--numevents', action='store', type=int, default=10000, help='Max limit for events used for validation')
   parser.add_argument('--latent', action='store', type=int, help='size of latent space to sample')
   parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
   parser.add_argument('--error', type=int, default=0, help='add relative errors to plots')
   parser.add_argument('--stest', type=int, default=0, help='add ktest to plots')
   parser.add_argument('--norm', type=int, default=1, help='normalize shower shapes')
   parser.add_argument('--ifC', type=int, default=0, help='generate .C files')
   parser.add_argument('--leg', type=int, default=1, help='draw legend')
   parser.add_argument('--grid', type=int, default=0, help='draw grid')
   return parser

def main():
   parser = get_parser()
   args = parser.parse_args()
   gweights = args.gweights if isinstance(args.gweights, list) else [args.gweights]
   labels = args.labels if isinstance(args.labels, list) else [args.labels]
   ang = args.ang
   particle = args.particle
   latent = args.latent
   datapath = args.datapath
   outdir = args.outdir
   numevents= args.numevents
   dformat= args.dformat
   error = args.error
   stest = args.stest
   norm = args.norm
   C = args.ifC
   leg = args.leg
   grid = args.grid
   if args.xscales:
     xscales = args.xscales if isinstance(args.xscales, list) else [args.xscales] 
   safe_mkdir(outdir)
   print('{} dir created'.format(outdir))
   K.set_image_data_format(dformat)
   if ang:
     from AngleArch3dGAN import generator
     dscale = 50.
     if not latent:
       latent = 256
     if not args.xscales:
       xscales = [1] * len(gweights) 
     xpower = 0.85
     if datapath=='reduced':
       datapath = "/storage/group/gpu/bigdata/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV                                                        
     elif datapath=='full':
       datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
     else:
       datapath =datapath + "/*scan/*scan_RandomAngle_*.h5"
     datafiles = GetDataFiles(datapath, particle, 1)
     data = datafiles[-1] # use the last file for the plots
     X, Y, angle = GetAngleData(data, angtype='theta', num_events=numevents)
   else:
     from EcalEnergyGan import generator
     dscale = 1.
     if not latent:
       latent = 200
     if not args.xscales:
       xscales = [100.] * len(gweights)
     if datapath=='full':
       datapath = "/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5"
     else:
       datapath =datapath + "/*scan/*scan_*.h5"

     datafiles =GetDataFiles(datapath,particle, 1)    
     data = datafiles[-1] # only use the data of the last file
     X, Y = GetData(data, num_events=numevents)

   X = np.squeeze(X)/dscale # convert data to GeV
   
   #get shape
   x = X.shape[1]
   y = X.shape[2]
   z = X.shape[3]
   
   #select events with significant energy deposition
   xsum = np.sum(X, axis=(1, 2, 3))
   indexes = np.where(xsum > (0.2))
   X=X[indexes]
   Y = Y[indexes]
   if ang: angle = angle[indexes]
   
   num_events = X.shape[0] # number of events can be reduced due to selection
   
   images =[]
   gm=generator(latent_size=latent, dformat = dformat)
   for i, gweight in enumerate(gweights):
      gm.load_weights(gweight)
      if ang:
        angle = angle
        images.append(generate(gm, num_events, [Y/100., angle], latent=latent, concat=2))
        images[i] = np.power(images[i], 1./xpower)
      else:
        images.append(generate(gm, num_events, [Y/100.], latent=latent))
      images[i] = np.squeeze(images[i])/(xscales[i] * dscale)
   plotSF(X, images, Y, labels, out_file=outdir +'/SamplingFraction', error=error, stest=stest, ifC=C, grid=grid, leg=leg)
   plotshapes(X, images, x, y, z, Y, out_file=outdir +'/ShowerShapes',labels=labels, log=0, stest=stest, error=error, norm=norm, ifC=C, grid=grid, leg=leg)
   plotshapes(X, images, x, y, z, Y, out_file=outdir +'/ShowerShapes_log',labels=labels, log=1, stest=stest, error=error, norm=norm, ifC=C, grid=grid, leg=leg)
   print('The plots are saved in {}'.format(outdir))

#Plotting sampling fraction vs. Ep
def plotSF(Data, gan_images, Y, labels, out_file, error=0, stest=0, ifC=0, grid=0, leg=1):
   c=ROOT.TCanvas("c" ,"Sampling Fraction vs. Primary energy" ,200 ,10 ,700 ,500) #make nice
   if grid: c.SetGrid()
   color =2
   ROOT.gStyle.SetOptStat(0)
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 500)
   dsum = np.sum(Data, axis=(1,2, 3))
   dsf = dsum/Y
   for j in np.arange(Y.shape[0]):
     Eprof.Fill( Y[j], dsf[j])
   Eprof.SetTitle("Sampling Fraction (cell energy sum / primary particle energy)")
   Eprof.GetXaxis().SetTitle("Primary particle energy [GeV]")
   Eprof.GetYaxis().SetTitle("Sampling Fraction")
   Eprof.GetYaxis().SetRangeUser(0.01, 0.03)
   Eprof.SetLineColor(color)
   Eprof.Draw()
   if stest:
     legend = ROOT.TLegend(0.6, 0.11, 0.89, 0.4)
   else:
     legend = ROOT.TLegend(0.7, 0.11, 0.89, 0.3)
   legend.AddEntry(Eprof, "Data", "l")
   legend.SetBorderSize(0)
   Gprof = []
   for i, images in enumerate(gan_images):
      Gprof.append( ROOT.TProfile("Gprof" +str(i), "Gprof" + str(i), 100, 0, 500))
      gsum = np.sum(images, axis=(1, 2, 3))
      gsf = gsum/Y
      for j in range(Y.shape[0]):
        Gprof[i].Fill(Y[j], gsf[j])
      color = color + 2
      Gprof[i].SetLineColor(color)
      Gprof[i].Draw('sames')
      c.Modified()
      sf_error = np.absolute((dsf-gsf)/dsf)
      glabel = 'GAN {}'.format(labels[i])
      if error:
         glabel = glabel + ' MRE={:.4f}'.format(np.mean(sf_error))
      legend.AddEntry(Gprof[i], glabel, "l")
      if stest:
         ks = Eprof.KolmogorovTest(Gprof[i], 'WW')
         legend.AddEntry(Gprof[i], 'k={:e}'.format(ks), "l")
      if leg: legend.Draw()
      c.Update()
   c.Print(out_file+'.pdf')
   if ifC:
      c.Print(out_file+'.C')

# plotting shower shapes
def plotshapes(X, generated_images, x, y, z, energy, out_file, labels, log=0, p=[2, 500], norm=0, ifC=0, stest=0, error=0, grid=0, leg=1):
   canvas = ROOT.TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   canvas.SetTitle('Weighted Histogram for energy deposition along x, y, z axis')
   if grid: canvas.SetGrid()
   color = 2
   canvas.Divide(2,2)
   array1x = np.sum(X, axis=(2,3))
   array1y = np.sum(X, axis=(1,3))
   array1z = np.sum(X, axis=(1,2))
   if stest:
     leg = ROOT.TLegend(0.1,0.1,0.9,0.9)
   else:
     leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   #leg.SetTextSize(0.06)
   h1x = ROOT.TH1F('G4x' + str(energy), '', x, 0, x)
   h1y = ROOT.TH1F('G4y' + str(energy), '', y, 0, y)
   h1z = ROOT.TH1F('G4z' + str(energy), '', z, 0, z)
   h1x.Sumw2()
   h1y.Sumw2()
   h1z.Sumw2()
   h1x.SetLineColor(color)
   h1y.SetLineColor(color)
   h1z.SetLineColor(color)
   color+=2
   canvas.cd(1)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1x, array1x)
   if norm: h1x=my.normalize(h1x)
   h1x.Draw()
   h1x.Draw('sames hist')
   h1x.GetXaxis().SetTitle("Energy deposition along x axis")
   leg.AddEntry(h1x, 'G4',"l")
   canvas.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1y, array1y)
   if norm: h1y=my.normalize(h1y)
   h1y.Draw()
   h1y.Draw('sames hist')
   h1y.GetXaxis().SetTitle("Energy deposition along y axis")
   canvas.cd(3)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1z, array1z)
   if norm : h1z=my.normalize(h1z)
   h1z.Draw()
   h1z.Draw('sames hist')
   h1z.GetXaxis().SetTitle("Energy deposition along z axis")
   canvas.cd(4)
   canvas.Update()
   h2xs=[]
   h2ys=[]
   h2zs=[]
   for i, images in enumerate(generated_images):
      array2x = np.sum(images, axis=(2,3))
      array2y = np.sum(images, axis=(1,3))
      array2z = np.sum(images, axis=(1,2))
      errorx = np.divide(np.absolute(array1x-array2x), array1x, out=np.zeros_like(array1x), where=array1x!=0)
      errory = np.divide(np.absolute(array1y-array2y), array1y, out=np.zeros_like(array1y), where=array1y!=0)
      errorz = np.divide(np.absolute(array1z-array2z), array1z, out=np.zeros_like(array1z), where=array1z!=0)

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
      canvas.cd(1)
      my.fill_hist_wt(h2x, array2x)
      if norm: h2x=my.normalize(h2x)
      h2x.Draw('sames')
      h2x.Draw('sames hist')
      canvas.Update()
      #my.stat_pos(h2x)
      if stest:
         res=np.array
         ks= h1x.KolmogorovTest(h2x, 'WW')
         glabel = "GAN {} X axis k= {:e}".format(labels[i], ks)
         leg.AddEntry(h2x, glabel,"l")
      canvas.Update()
      canvas.cd(2)
      my.fill_hist_wt(h2y, array2y)
      if norm: h2y=my.normalize(h2y)
      h2y.Draw('sames')
      h2y.Draw('sames hist')
      canvas.Update()
      #my.stat_pos(h2y)
      if stest:
         ks= h1y.KolmogorovTest(h2y, 'WW')
         glabel = "GAN {} Y axis k= {:e}".format(labels[i], ks)
         leg.AddEntry(h2y, glabel,"l")
      canvas.Update()
      canvas.cd(3)
      my.fill_hist_wt(h2z, array2z)
      if norm: h2z=my.normalize(h2z)
      h2z.Draw('sames')
      h2z.Draw('sames hist')
      canvas.Update()
      #my.stat_pos(h2z)
      canvas.Update()
      if stest:
         ks= h1z.KolmogorovTest(h2z, 'WW')
         glabel = "GAN {} Z axis k= {:e}".format(labels[i], ks)
         leg.AddEntry(h2z, glabel,"l")
      canvas.Update()
      color+=2
   canvas.Update()
   canvas.cd(4)
   leg.SetHeader("Energy deposited along x, y, z axis", "C")
   
   for i, h in enumerate(h2xs):
       glabel = 'GAN ' + labels[i]
       if error:
          tot_error = (np.mean(errorx) + np.mean(errory) + np.mean(errorz))/3.
          glabel = glabel + ' MRE {:.4f}'.format(np.mean(errorz))
       elif not stest:
          leg.AddEntry(h, glabel,"l")     
   if leg: leg.Draw()
   canvas.Update()
   canvas.Print(out_file + '.pdf')
   if ifC:
      canvas.Print(out_file + '.C')

if __name__ == '__main__': 
   main()
