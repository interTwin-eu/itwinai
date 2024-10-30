# To make logrimthic bins for a root histogram                                                                    
import numpy as np
import ROOT
                                                                                                  
# creating log bins
def BinLogX(h):
   axis = h.GetXaxis()
   bins = axis.GetNbins()
   From = axis.GetXmin()
   to = axis.GetXmax()
   width = (to - From) / bins
   new_bins = np.zeros(bins + 1)

   for i in np.arange(bins + 1):
     new_bins[i] = ROOT.TMath.Power(10, From + i * width)
   axis.Set(bins, new_bins)
   new_bins=None

def bmre(h1x, h2x, d=10):
   x1 = h1x.GetArray()
   x1.SetSize(h1x.GetNbinsX())
   x1 = np.array(x1)
   x1 = x1[1:]
   x2 = h2x.GetArray()
   x2.SetSize(h2x.GetNbinsX())
   x2 = np.array(x2)
   x2 =x2[1:]
   print(x1)
   print(x2)
   er = np.abs(np.where(x1>1e-5, (x1-x2)/(x1), 0))
   print(er)
   d = int(er.shape[0]/4)
   print(er.shape[0]/4, d)
   error = np.mean(er[d:-d])
   return (error)



#Fill a histogram from 1D numpy array
def fill_hist(hist, array):
   [hist.Fill(_) for _ in array]

def fill_graph(graph, x, y):
   n = x.shape[0]
   for i in np.arange(n):
      graph.SetPoint(int(i), x[i], y[i])

def fill_hist_2D(h, x, y):
   n = x.shape[0]
   for i in np.arange(n):
      h.Fill(x[i], y[i])

def FillHist3D_wt(h, d):
   n = d.shape[0]
   x = d.shape[1]
   y = d.shape[2]
   z = d.shape[3]
   for i in np.arange(n):
     for j in np.arange(x):
      for k in np.arange(y):
        for l in np.arange(z):
           h.Fill(j, k, l, d[i, j, k, l])

def Filltuple3D_wt(h, d):
   
   x = d.shape[-3]
   y = d.shape[-2]
   z = d.shape[-1]
   #for i in np.arange(n):
   for j in np.arange(x):
      for k in np.arange(y):
        for l in np.arange(z):
           if d[j, k, l]>0:  h.Fill(j, k, l, d[j, k, l])


# normalize in different modes
def normalize(hist, mod=0):
   if mod==0:
      norm = hist.GetEntries()
      hist.Scale(1/norm)
   elif mod==1:   
      if hist.Integral()!=0:
         hist.Scale(1/hist.Integral())
   return hist

#Fill a weighted histogram for event from numpy array
def fill_hist_wt(hist, weight):
   l=weight.shape[1]
   array= np.arange(0, l, 1)
   for i in array:
     for j in np.arange(weight.shape[0]):
        hist.Fill(i, weight[j, i])


#2D weighted histogram        
def FillHist2D_wt(hist, array):
   array= np.squeeze(array, axis=0)
   dim1 = array.shape[0]
   dim2 = array.shape[1]
   dim3 = array.shape[2]
   bin1 = np.arange(dim1)
   bin2 = np.arange(dim2)
   bin3 = np.arange(dim3)
   count = 0
   for j in bin2:
      for k in bin3:
         for i in bin1:
            hist.Fill(j, k, array[i, j, k])
            count+=1
                                                                            
#Hits above a threshold
def get_hits(events, thresh=0.0002):
   hit_array = events>thresh
   hits = np.sum(hit_array, axis=(1, 2, 3))
   return hits

# making ratio of first layer to total
def ratio1_total(events):
   mid= int(events.shape[3]/3)
   array1=np.sum(events[:, :, :, :mid], axis=(1, 2, 3))
   total=np.sum(events[:, :, :, :], axis=(1, 2, 3))
   return array1/total

# making ratio of first layer to total
def ratio2_total(events):
   mid= int(events.shape[3]/3)
   array2=np.sum(events[:, :, :, mid:2 *mid], axis=(1, 2, 3))
   total=np.sum(events[:, :, :, :], axis=(1, 2, 3))
   return array2/total

# making ratio of first layer to total
def ratio3_total(events):
   mid= int(events.shape[3]/3)
   array3=np.sum(events[:, :, :, 2 *mid:], axis=(1, 2, 3))
   total=np.sum(events[:, :, :, :], axis=(1, 2, 3))
   return array3/total


# Position of state box
def stat_pos(a, pos=0):
  # Upper left                                                                                                                                                                                                     
  if pos==0:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.0)
   sb1.SetX2NDC(.2)
  # Upper right                                                                                                                                                                                                    
  if pos==1:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetY1NDC(.3)
   sb1.SetY2NDC(.4)
  # Lower right                                                                                                                                                                                                    
  if pos==2:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.1)
   sb1.SetX2NDC(.3)
   sb1.SetY1NDC(.1)
   sb1.SetY2NDC(.3)
  # Lower left                                                                                                                                                                                                     
  if pos==3:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.7)
   sb1.SetX2NDC(.9)
   sb1.SetY1NDC(.1)
   sb1.SetY2NDC(.3)
  # Upper Center                                                                                                                                                                                                   
  if pos==4:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.4)
   sb1.SetX2NDC(.6)
   sb1.SetY1NDC(.7)
   sb1.SetY2NDC(.9)
  #lower Center                                                                                                                                                                                                    
  if pos==5:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.4)
   sb1.SetX2NDC(.6)
   sb1.SetY1NDC(.1)
   sb1.SetY2NDC(.3)
  return sb1

# Fill root profile
def fill_profile(prof, x, y):
  for i in range(len(y)):
      prof.Fill(x[i], y[i])


# return max
def Max(hist1, hist2):
   max1 = hist1.GetMaximum()
   max2 = hist2.GetBinContent(hist2.GetMaximumBin())
   if max2 > max1:
      hist1.GetYaxis().SetRangeUser(0, 1.2 * max2)
