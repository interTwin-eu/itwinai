##### Common functions #################
import os
import h5py
import numpy as np
import math
import time
import glob
import torch
#import numpy.core.umath_tests as umath

# return a fit for Ecalsum/Ep for Ep
def GetEcalFit(sampled_energies, particle='Ele', mod=0, xscale=1):
    if mod==0:
       return np.multiply(2, sampled_energies)
    elif mod==1:
       if particle == 'Ele':
         root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale
       elif particle == 'Pi0':
         root_fit = [0.0085, -0.094, 2.051]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale

#Divide files in train and test lists     
def DivideFiles(FileSearch="/data/LCD/*/*.h5",
                Fractions=[.5,.5],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
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
    out=[]
    for j in range(len(Fractions)):
        out.append([])
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    print(len(out[0]))
    print(len(out[1]))
    return out

# flips a int array's values with some probability
def BitFlip(x, prob=0.05):
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x
                    

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

# get data for fixed angle
def GetData(datafile, thresh=0, num_events=10000):
   #get data for training
    print( 'Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    y=f.get('target')[:num_events]
    x=np.array(f.get('ECAL')[:num_events])
    y=(np.array(y[:,1]))
    if thresh>0:
       x[x < thresh] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

# sort data by first variable
def sort(data, bins, flag=False, num_events=1000, tolerance=5):
    X = data[0]
    Y = data[1]
    if len(data)>2:
        Z = data[2]
    srt = {}
    for b in bins:
        if b == 0 and flag:
            srt["events" + str(b)] = X[:10000] # More events in random bin
            srt["y" + str(b)] = Y[:10000]
            if len(data)>2:
               srt["z" + str(b)] = Z[:10000]
        else:
            indexes = np.where((Y > b - tolerance ) & ( Y < b + tolerance))
            srt["events" + str(b)] = X[indexes][:num_events]
            srt["y" + str(b)] = Y[indexes][:num_events]
            if len(data)>2:
                srt["z" + str(b)] = Z[indexes][:num_events]
    return srt

# sort data by energy when input = [events, primary energy and angle (for angle version)] 
def sortEnergy(data, ecal, energies, ang=1):
    var={}
    tolerance =5
    for energy in energies:
        if energy==0:
            var["events_act" + str(energy)]=data[0][:5000]
            var["energy" + str(energy)]=data[1][:5000]
            if ang: var["angle_act" + str(energy)]=data[2][:5000]
            var["ecal_act" + str(energy)]=ecal[:5000]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
        else:
            var["indexes" + str(energy)] = np.where((data[1] > (energy - tolerance)/100. ) & ( data[1] < (energy + tolerance)/100.))
            var["events_act" + str(energy)]=data[0][var["indexes" + str(energy)]]
            var["energy" + str(energy)]=data[1][var["indexes" + str(energy)]]
            if ang:  var["angle_act" + str(energy)]=data[2][var["indexes" + str(energy)]]
            var["ecal_act" + str(energy)]=ecal[var["indexes" + str(energy)]]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
    return var

#Optimization metric
def metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1):
    metricp = 0
    metrice = 0
    metrica = 0
    for energy in energies:
        #Relative error on mean moment value for each moment and each axis
        x_act= np.mean(var["momentX_act"+ str(energy)], axis=0)
        x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
        y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
        y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
        z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
        z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
        var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act
        var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
        var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
        #Taking absolute of errors and adding for each axis then scaling by 3
        var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)])+ np.absolute(var["posz_error"+ str(energy)]))/3
        #Summing over moments and dividing for number of moments
        var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
        metricp += var["pos_total"+ str(energy)]
        #Take profile along each axis and find mean along events
        sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis=0), np.mean(var["sumsz_act" + str(energy)], axis=0)
        sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
        var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
        var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
        var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
        #Take absolute of error and mean for all events                                                           
        var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
        #Summing over moments and dividing for number of moments
        var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
        metricp += var["pos_total"+ str(energy)]
        #Take profile along each axis and find mean along events
        sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis= 0), np.mean(var["sumsz_act" + str(energy)], axis=0)
        sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
        var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
        var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
        var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
        #Take absolute of error and mean for all events
        var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)]))/x
        var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)]))/y
        var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/z

        var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
        metrice += var["eprofile_total"+ str(energy)]
        if ang:
            var["angle_error"+ str(energy)] = np.mean(np.absolute((var[angtype + "_act" + str(energy)] - var[angtype + "_gan" + str(energy)])/var[angtype + "_act" + str(energy)]))
            metrica += var["angle_error"+ str(energy)]
    metricp = metricp/len(energies)
    metrice = metrice/len(energies)
    if ang:metrica = metrica/len(energies)
    tot = metricp + metrice
    if ang:tot = tot +metrica
    result = [tot, metricp, metrice]
    if ang: result.append(metrica)
    return result

# Measuring 3D angle from image
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

    #ang = np.sum(ang, axis=1)/zunmasked_events #mean
    ang = ang * z # weighted by position
    sumz_tot = z * zmask
    ang = np.sum(ang, axis=1)/np.sum(sumz_tot, axis=1)

    indexes = np.where(amask>0)
    ang[indexes] = 100.
    return ang

# short version of analysis                                                                                                                      
def OptAnalysisShort(var, generated_images, energies, ang=1):
    m=2
    
    x = generated_images.shape[1]
    y = generated_images.shape[2]
    z = generated_images.shape[3]
    for energy in energies:
      if energy==0:
         var["events_gan" + str(energy)]=generated_images[:5000]
      else:
         var["events_gan" + str(energy)]=generated_images[var["indexes" + str(energy)]]
      var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
      var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
      if ang: var["angle_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])
    return metric(var, energies, m, angtype='angle', x=x, y=y, z=z, ang=ang)
                                                                                                     
def OptAnalysisAngle(var, g, energies, ascale=None, xscale=None, yscale=100, xpower=None, latent=256, concat=1):
    m=2
    for energy in energies:
        var["events_act" + str(energy)] = np.squeeze(var["events_act" + str(energy)])
        if ascale: var["angle_act"+ str(energy)]= var["angle_act"+ str(energy)] * ascale
        if yscale: var["energy" + str(energy)]=var["energy" + str(energy)]/yscale
        num = var["events_act" + str(energy)].shape[0]
        x = var["events_act" + str(energy)].shape[1]
        y = var["events_act" + str(energy)].shape[2]
        z = var["events_act" + str(energy)].shape[3]
        var["events_gan" + str(energy)] = generate(g, num, [var["energy" + str(energy)], (var["angle_act"+ str(energy)])], latent, concat)
        var["events_gan" + str(energy)] = np.squeeze(var["events_gan" + str(energy)])
        if xpower: var["events_gan" + str(energy)] = np.power(var["events_gan" + str(energy)], 1.0/xpower)
        if xscale: var["events_gan" + str(energy)] = var["events_gan" + str(energy)]/xscale
        
        var["ecal_act"+ str(energy)] = np.sum(var["events_act" + str(energy)], axis = (1, 2, 3))
        var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
        var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
        var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
        var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
        var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
        var["angle_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])
    return metric(var, energies, m, angtype='angle', x=x, y=y, z=z, ang=1)
                                                                                   
# Load data from files in arrays
def GetAllDataAngle(datafiles, numevents, thresh=1e-6, angtype='theta'):
    for index, datafile in enumerate(datafiles):
        if index == 0:
            x, y, theta = GetAngleData(datafile, thresh, angtype)
        else:
            while  x.shape[0] < numevents:
                x_temp, y_temp, theta_temp = GetAngleData(datafile)
                x = np.concatenate((x, x_temp), axis=0)
                y = np.concatenate((y, y_temp), axis=0)
                theta = np.concatenate((theta, theta_temp), axis=0)
    return x[:numevents], y[:numevents], theta[:numevents] 
                                                                                   

# sort data for fixed angle
def get_sorted(datafiles, energies, flag=False, num_events1=10000, num_events2=2000, tolerance=5, thresh=0):
    srt = {}
    for index, datafile in enumerate(datafiles):
        data = GetData(datafile, thresh)
        X = data[0]
        sumx = np.sum(np.squeeze(X), axis=(1, 2, 3))
        indexes= np.where(sumx>0)
        X=X[indexes]
        Y = data[1]
        Y=Y[indexes]
        for energy in energies:
            if index== 0:
                if energy == 0:
                    srt["events_act" + str(energy)] = X # More events in random bin
                    srt["energy" + str(energy)] = Y
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                    srt["events_act" + str(energy)] = X[indexes]
                    srt["energy" + str(energy)] = Y[indexes]
            else:
                if energy == 0:
                   if flag:
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X, axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y, axis=0)
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    if srt["events_act" + str(energy)].shape[0] < num_events2:
                        indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                        srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X[indexes], axis=0)
                        srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y[indexes], axis=0)
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events2]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events2]
    return srt


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

# save sorted data
def save_sorted(srt, energies, srtdir, ang=0):
    safe_mkdir(srtdir)
    for energy in energies:
       srtfile = os.path.join(srtdir, "events_{:03d}.h5".format(energy))
       with h5py.File(srtfile ,'w') as outfile:
          outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
          outfile.create_dataset('Target',data=srt["energy" + str(energy)])
          if ang:
             outfile.create_dataset('Angle',data=srt["angle" + str(energy)])
       print ("Sorted data saved to {}".format(srtfile))

# save generated images       
def save_generated(events, cond, energy, gendir):
    safe_mkdir(gendir)
    filename = os.path.join(gendir,"Gen_{:03d}.hdf5".format(energy))
    with h5py.File(filename ,'w') as outfile:
       outfile.create_dataset('ECAL',data=events)
       outfile.create_dataset('Target',data=cond[0])
       if len(cond) > 1:
          outfile.create_dataset('Angle',data=cond[1])
    print ("Generated data saved to ", filename)

# save discriminator results
def save_discriminated(disc, energy, discdir, angloss=1, addloss=0, ang=0):
    safe_mkdir(discdir)
    filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
    with h5py.File(filename ,'w') as outfile:
      outfile.create_dataset('ISREAL_ACT',data=disc["isreal_act" + str(energy)])
      outfile.create_dataset('ISREAL_GAN',data=disc["isreal_gan" + str(energy)])
      outfile.create_dataset('AUX_ACT',data=disc["aux_act" + str(energy)])
      outfile.create_dataset('AUX_GAN',data=disc["aux_gan" + str(energy)])
      outfile.create_dataset('ECAL_ACT',data=disc["ecal_act" + str(energy)])
      outfile.create_dataset('ECAL_GAN',data=disc["ecal_gan" + str(energy)])
      if ang:
          outfile.create_dataset('ANGLE_ACT',data=disc["angle_act" + str(energy)])
          outfile.create_dataset('ANGLE_GAN',data=disc["angle_gan" + str(energy)])
      if angloss == 2:
          outfile.create_dataset('ANGLE2_ACT',data=disc["angle2_act" + str(energy)])
          outfile.create_dataset('ANGLE2_GAN',data=disc["angle2_gan" + str(energy)])
      if addloss:
          outfile.create_dataset('ADDLOSS_ACT',data=disc["addloss_act" + str(energy)])
          outfile.create_dataset('ADDLOSS_GAN',data=disc["addloss_gan" + str(energy)])
    print ("Discriminated data saved to ", filename)

# read D results    
def get_disc(energy, discdir, angloss=1, addloss=0, ang=0):
    filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    isreal_act = np.array(f.get('ISREAL_ACT'))
    isreal_gan = np.array(f.get('ISREAL_GAN'))
    aux_act = np.array(f.get('AUX_ACT'))
    aux_gan = np.array(f.get('AUX_GAN'))
    ecal_act = np.array(f.get('ECAL_ACT'))
    ecal_gan = np.array(f.get('ECAL_GAN'))
    disc_out = [isreal_act, aux_act, ecal_act, isreal_gan, aux_gan, ecal_gan]
    if ang:
       angle_act = np.array(f.get('ANGLE_ACT'))
       angle_gan = np.array(f.get('ANGLE_GAN'))
       disc_out.append(angle_act)
       disc_out.append(angle_gan)
    if angloss == 2:
        angle2_act = np.array(f.get('ANGLE2_ACT'))
        angle2_gan = np.array(f.get('ANGLE2_GAN'))
        disc_out.append(angle2_act)
        disc_out.append(angle2_gan)
    if addloss:
        addloss_act = np.array(f.get('ADDLOSS_ACT'))
        addloss_gan = np.array(f.get('ADDLOSS_GAN'))
        disc_out.append(addloss_act)
        disc_out.append(addloss_gan)
                                        
    print ("Discriminated file ", filename, " is loaded")
    return disc_out

# load sorted data
def load_sorted(sorted_path, energies, ang=0):
    sorted_files = sorted(glob.glob(sorted_path))
    srt = {}
    for f in sorted_files:
       energy = int(filter(str.isdigit, f)[:-1])
       if energy in energies:
          srtfile = h5py.File(f,'r')
          srt["events_act" + str(energy)] = np.array(srtfile.get('ECAL'))
          srt["energy" + str(energy)] = np.array(srtfile.get('Target'))
          if ang:
             srt["angle" + str(energy)] = np.array(srtfile.get('Angle'))
          print( "Loaded from file", f)
    return srt

# load generated data from file
def get_gen(energy, gendir):
    filename = os.path.join(gendir, "Gen_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    print ("Generated file ", filename, " is loaded")
    return generated_images

# generate images
def generate(g, index, cond, latent=256, concat=1, batch_size=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    energy_labels=np.expand_dims(cond[0], axis=1)
    if len(cond)> 1: # that means we also have angle
      angle_labels = cond[1]
      if concat==1:
        noise = np.random.normal(0, 1, (index, latent-1))  
        noise = energy_labels * noise
        gen_in = np.concatenate((angle_labels.reshape(-1, 1), noise), axis=1)
      elif concat==2:
        noise = np.random.normal(0, 1, (index, latent-2))
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
    #generated_images = g.predict(gen_in, verbose=False, batch_size=batch_size)

    gen_in_tensor = torch.tensor(gen_in).float().to(device)
    with torch.no_grad():  # Temporarily set all the requires_grad flags to false
        generated_images = g(gen_in_tensor)
    generated_images_np = generated_images.cpu().numpy()

    return generated_images

# discriminator predict
def discriminate(d, images):
    #disc_out = d.predict(images, verbose=False, batch_size=50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    disc_out_tensor = torch.tensor(disc_out).float().to(device)
    with torch.no_grad():  # Temporarily set all the requires_grad flags to false
        discriminated_images = d(disc_out_tensor)
    disc_out_np = discriminated_images.cpu().numpy()

    return disc_out

# find location of maximum depositions
def get_max(images):
    index = images.shape[0]
    x=images.shape[1]
    y=images.shape[2]
    z=images.shape[3]
    max_pos = np.zeros((index, 3))
    for i in range(index):
       max_p = images[i].argmax()
       max_loc = np.unravel_index(max_p, (x, y, z))
       max_pos[i] = max_loc
    return max_pos

# get sums along different axis
def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

# get moments
def get_moments(sumsx, sumsy, sumsz, totalE, m, x=51, y=51, z=25):
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    totalE = np.squeeze(totalE)
    index = sumsx.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(x), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      #ECAL_momentX = np.divide(umath.inner1d(sumsx, moments) ,totalE)
      ECAL_momentX = np.einsum('ij,ij->i', sumsx, moments) / totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(y), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      #ECAL_momentY = np.divide(umath.inner1d(sumsy, moments), totalE)
      ECAL_momentY = np.einsum('ij,ij->i', sumsy, moments) / totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(z), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      #ECAL_momentZ = np.divide(umath.inner1d(sumsz, moments), totalE)
      ECAL_momentZ = np.einsum('ij,ij->i', sumsz, moments) / totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

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
# scaling of input
def preproc(n, xscale=1):
    return n * xscale

# scaling of output
def postproc(n, xscale=1):
    return n/xscale


def load_pytorch_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))


def perform_calculations_angle(g, d, gweights, dweights, energies, angles, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, xscales, xpowers, angscales, dscale, flags, latent, particle='Ele', Data=GetAngleData, events_per_file=5000, angtype='theta', thresh=1e-6, offset=0.0, angloss=1, addloss=0, concat=1, pre=preproc, post=postproc, tolerance2 = 0.1, num_events1=10000):
    sortedpath = os.path.join(sortdir, 'events_*.h5')
    print( flags)
    # assign values to flags that decide if data is to be read from dataset or pre binned data
    # Also if saved generated and discriminated data is to be used
    
    Test = flags[0]
    save_data = flags[1]
    read_data = flags[2]
    save_gen = flags[3]
    read_gen = flags[4]
    save_disc = flags[5]
    read_disc =  flags[6]
    var= {}
    num_events1= num_events1
    num_events2 = num_events
    ang =1

    # Read from sorted dir with binned data
    if read_data: 
       start = time.time()
       var = load_sorted(sortedpath, energies, ang) # returning a dict with sorted data
       print( "Events were loaded in {} seconds".format(time.time()- start))

    # If reading from unsorted data. The data will be read and sorted in bins   
    else:
       Filesused = int(math.ceil(num_data/events_per_file)) # num_data is number of events to be used from unsorted data/ events in each file
       print(Filesused)
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle]) # get test and train files
       print (Trainfiles)
       print (Testfiles)
       Trainfiles = Trainfiles[: Filesused] # The number of files to read is limited by Fileused
       Testfiles = Testfiles[: Filesused]
       print (Trainfiles)
       print (Testfiles)
       if Test:
          data_files = Testfiles  # Test data will be read in test mode
       else:
          data_files = Trainfiles  # else train data will be used
       start = time.time()
       var = get_sorted_angle(data_files, energies, True, num_events1, num_events2, Data=Data, angtype=angtype, thresh=thresh*dscale, offset=offset) # returning a dict with sorted data. 
       print ("{} events were loaded in {} seconds".format(num_data, time.time() - start))
       
       # If saving the binned data. This will only run if reading from data directly
       if save_data:
          save_sorted(var, energies, sortdir, ang) # saving sorted data in a directory

    total = 0

    print(var.keys())

    # For each energy bin
    for energy in energies:
      # Getting dimensions of ecal images  
      x = var["events_act"+ str(energy)].shape[1]
      y =var["events_act"+ str(energy)].shape[2]
      z =var["events_act"+ str(energy)].shape[3]

      # scaling to GeV
      if not dscale==1: var["events_act"+ str(energy)]= var["events_act"+ str(energy)]/dscale
      #calculations for data events
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0] # number of events in bin
      total += var["index" + str(energy)] # total events 
      ecal =np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))# sum actual events for moment calculations
      var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)]) # get position of maximum deposition
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)]) # get sums along different axis
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)],
                                                                var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], ecal, m, x=x, y=y, z=z) # calculate moments
      for index, a in enumerate(angles):
         indexes = np.where(((var["angle" + str(energy)]) > np.radians(a) - tolerance2) & ((var["angle" + str(energy)]) < np.radians(a) + tolerance2)) # all events with angle within a bin
         # angle bins are added to dict
         var["events_act" + str(energy) + "ang_" + str(a)] = var["events_act" + str(energy)][indexes]
         var["energy" + str(energy) + "ang_" + str(a)] = var["energy" + str(energy)][indexes]
         var["angle" + str(energy) + "ang_" + str(a)] = var["angle" + str(energy)][indexes]
         var["sumsx_act"+ str(energy) + "ang_" + str(a)] = var["sumsx_act"+ str(energy)][indexes]
         var["sumsy_act"+ str(energy) + "ang_" + str(a)] = var["sumsy_act"+ str(energy)][indexes]
         var["sumsz_act"+ str(energy) + "ang_" + str(a)] = var["sumsz_act"+ str(energy)][indexes]
         var["momentX_act"+ str(energy) + "ang_" + str(a)] = var["momentX_act"+ str(energy)][indexes]
         var["momentY_act"+ str(energy) + "ang_" + str(a)] = var["momentY_act"+ str(energy)][indexes]
         var["momentZ_act"+ str(energy) + "ang_" + str(a)] = var["momentZ_act"+ str(energy)][indexes]
                           
         print ('{} for angle bin {} total events were {}'.format(index, a, var["events_act" + str(energy) + "ang_" + str(a)].shape[0]))

    print ("{} events were put in {} bins".format(total, len(energies)))
    #### Generate Data table to screen                                                                                                                                                                             
    print ("Actual Data")
    print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")
    for energy in energies:
       print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1])))

    for energy in energies:
       # creating dicts for all GAN quantities 
       var["events_gan" + str(energy)]={}
       var["isreal_act" + str(energy)]={}
       var["isreal_gan" + str(energy)]={}
       var["aux_act" + str(energy)]={}
       var["aux_gan" + str(energy)]={}
       var["angle_act" + str(energy)]={}
       var["angle_gan" + str(energy)]={}
       if angloss==2:
          var["angle2_act" + str(energy)]={}
          var["angle2_gan" + str(energy)]={}
       if addloss:
          var["addloss_act" + str(energy)]={}
          var["addloss_gan" + str(energy)]={}
                   
       var["ecal_act" + str(energy)]={}
       var["ecal_gan" + str(energy)]={}
       var["max_pos_gan" + str(energy)]={}
       var["sumsx_gan"+ str(energy)]={}
       var["sumsy_gan"+ str(energy)]={}
       var["sumsz_gan"+ str(energy)]={}
       var["momentX_gan" + str(energy)]={}
       var["momentY_gan" + str(energy)]={}
       var["momentZ_gan" + str(energy)]={}
       for index, a in enumerate(angles):
          var["events_gan" + str(energy) + "ang_" + str(a)]={}
          var["isreal_act" + str(energy) + "ang_" + str(a)]={}
          var["isreal_gan" + str(energy) + "ang_" + str(a)]={}
          var["aux_act" + str(energy)+ "ang_" + str(a)]={}
          var["aux_gan" + str(energy)+ "ang_" + str(a)]={}
          var["angle_act" + str(energy)+ "ang_" + str(a)]={}
          var["angle_gan" + str(energy)+ "ang_" + str(a)]={}
          if angloss==2:
            var["angle2_act" + str(energy)+ "ang_" + str(a)]={}
            var["angle2_gan" + str(energy)+ "ang_" + str(a)]={}
          if addloss:
            var["addloss_act" + str(energy)+ "ang_" + str(a)]={}
            var["addloss_gan" + str(energy)+ "ang_" + str(a)]={}
                      
          var["ecal_act" + str(energy)+ "ang_" + str(a)]={}
          var["ecal_gan" + str(energy)+ "ang_" + str(a)]={}
          var["sumsx_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["sumsy_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["sumsz_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["momentX_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["momentY_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["momentZ_gan"+ str(energy)+ "ang_" + str(a)]={}
                              

       for gen_weights, disc_weights, scale, power, ascale, i in zip(gweights, dweights, xscales, xpowers, angscales, np.arange(len(gweights))):
          gendir = gendirs + '/n_' + str(i)
          discdir = discdirs + '/n_' + str(i)
                            
          if read_gen:
             var["events_gan" + str(energy)]['n_'+ str(i)]= get_gen(energy, gendir)
          else:
             #g.load_weights(gen_weights)
             load_pytorch_weights(g, gen_weights)
             start = time.time()
             var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100, (var["angle"+ str(energy)]) * ascale], latent, concat)
             var["events_gan" + str(energy)]['n_'+ str(i)] = post(var["events_gan" + str(energy)]['n_'+ str(i)], scale, power)
             var["events_gan" + str(energy)]['n_'+ str(i)][var["events_gan" + str(energy)]['n_'+ str(i)]< thresh * dscale] = 0

             if save_gen:
                save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], [var["energy" + str(energy)], var["angle"+ str(energy)]], energy, gendir)
             var["events_gan" + str(energy)]['n_'+ str(i)] = pre (var["events_gan" + str(energy)]['n_'+ str(i)], scale, power)
             gen_time = time.time() - start
             print( "Generator took {} seconds to generate {} events".format(gen_time, var["index" +str(energy)]))
          if read_disc:
             disc_out = get_disc(energy, discdir, angloss, addloss, ang)
             var["isreal_act" + str(energy)]['n_'+ str(i)] = disc_out[0]
             var["aux_act" + str(energy)]['n_'+ str(i)] = disc_out[1]
             var["ecal_act"+ str(energy)]['n_'+ str(i)] = disc_out[2]
             var["isreal_gan" + str(energy)]['n_'+ str(i)] = disc_out[3]
             var["aux_gan" + str(energy)]['n_'+ str(i)] = disc_out[4]
             var["ecal_gan"+ str(energy)]['n_'+ str(i)] = disc_out[5]
             var["angle_act"+ str(energy)]['n_'+ str(i)] = disc_out[6]
             var["angle_gan"+ str(energy)]['n_'+ str(i)] = disc_out[7]
             if angloss==2:
                var["angle2_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[8])
                var["angle2_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[9])
             else:
                if addloss:
                    var["addloss_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[8])
                    var["addloss_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[9])
          else:
             #d.load_weights(disc_weights)
             load_pytorch_weights(d, disc_weights)
             start = time.time()
             disc_out_act = discriminate(d, pre(dscale * var["events_act" + str(energy)], scale, power))
             disc_out_gan =discriminate(d, var["events_gan" + str(energy)]['n_'+ str(i)])
             var["isreal_act" + str(energy)]['n_'+ str(i)]= np.array(disc_out_act[0])
             var["isreal_gan" + str(energy)]['n_'+ str(i)]= np.array(disc_out_gan[0])
             var["aux_act" + str(energy)]['n_'+ str(i)] = np.array(disc_out_act[1])
             var["aux_gan" + str(energy)]['n_'+ str(i)]= np.array(disc_out_gan[1])
             var["angle_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[2])
             var["angle_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[2])
             if angloss==2:
                 var["angle2_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[3])
                 var["angle2_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[3])
                 var["ecal_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[4])
                 var["ecal_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[4])
             else:
                 var["ecal_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[3])
                 var["ecal_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[3])
             if addloss:
                 var["addloss_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[4])
                 var["addloss_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[4])
                                
             disc_time = time.time() - start
             print ("Discriminator took {} seconds for {} data and generated events".format(disc_time, var["index" +str(energy)]))

             if save_disc:
               discout = {}
               for key in var:
                  if key in ["isreal_act" + str(energy), "aux_act" + str(energy), "isreal_gan" + str(energy), "aux_gan" + str(energy), "ecal_act"+ str(energy), "ecal_gan"+ str(energy), "angle2_act"+ str(energy), "angle2_gan"+ str(energy), "angle_act"+ str(energy), "angle_gan"+ str(energy), "addloss_act"+ str(energy), "addloss_gan"+ str(energy)]:
                     discout[key]=var[key]['n_'+ str(i)]
               save_discriminated(discout, energy, discdir, angloss, addloss, ang)
          print ('Calculations for ....', energy)
          var["events_gan" + str(energy)]['n_'+ str(i)] = post(var["events_gan" + str(energy)]['n_'+ str(i)], scale, power)/dscale
          #var["events_gan" + str(energy)]['n_'+ str(i)][var["events_gan" + str(energy)]['n_'+ str(i)]< thresh] = 0
          var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["angle_act"+ str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_act" + str(energy)]['n_'+ str(i)]), np.squeeze((var["angle_act"+ str(energy)]['n_'+ str(i)]))/ascale, np.squeeze(var["ecal_act"+ str(energy)]['n_'+ str(i)]/(dscale * scale))
          var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["angle_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["angle_gan"+ str(energy)]['n_'+ str(i)] )/ascale, np.squeeze(var["ecal_gan"+ str(energy)]['n_'+ str(i)]/(dscale * scale))
          if angloss==2:
              var["angle2_act"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["angle2_act"+ str(energy)]['n_'+ str(i)]))/ascale
              var["angle2_gan"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["angle2_gan"+ str(energy)]['n_'+ str(i)]))/ascale
          if addloss:
              var["addloss_act"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["addloss_act"+ str(energy)]['n_'+ str(i)]))
              var["addloss_gan"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["addloss_gan"+ str(energy)]['n_'+ str(i)]))
                            
          var["max_pos_gan" + str(energy)]['n_'+ str(i)] = get_max(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)] = get_sums(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["momentX_gan" + str(energy)]['n_'+ str(i)], var["momentY_gan" + str(energy)]['n_'+ str(i)], var["momentZ_gan" + str(energy)]['n_'+ str(i)] = get_moments(var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)], m, x=x, y=y, z=z)
          for index, a in enumerate(angles):
             indexes = np.where(((var["angle" + str(energy)]) > np.radians(a) - tolerance2) & ((var["angle" + str(energy)]) < np.radians(a) + tolerance2))
             var["events_gan" + str(energy) + "ang_" + str(a)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["sumsx_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["sumsx_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["sumsy_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["sumsy_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["sumsz_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["sumsz_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["isreal_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["isreal_act" + str(energy)]['n_'+ str(i)][indexes]
             var["isreal_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["isreal_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["aux_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["aux_act" + str(energy)]['n_'+ str(i)][indexes]
             var["aux_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["aux_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["ecal_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["ecal_act" + str(energy)]['n_'+ str(i)][indexes]
             var["ecal_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["ecal_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["angle_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle_act" + str(energy)]['n_'+ str(i)][indexes]
             var["angle_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["momentX_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["momentX_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["momentY_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["momentY_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["momentZ_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["momentZ_gan"+ str(energy)]['n_'+ str(i)][indexes]
                                       
             if angloss==2:
               var["angle2_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle2_act" + str(energy)]['n_'+ str(i)][indexes]
               var["angle2_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle2_gan" + str(energy)]['n_'+ str(i)][indexes]
             if addloss:
               var["addloss_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["addloss_act" + str(energy)]['n_'+ str(i)][indexes]
               var["addloss_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["addloss_gan" + str(energy)]['n_'+ str(i)][indexes]
       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))
    for i in np.arange(len(gweights)):
      #### Generate GAN table to screen                                                                                                       
      print( "Generated Data for {}".format(i))
      print( "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")

      for energy in energies:
         print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["max_pos_gan" + str(energy)]['n_'+ str(i)], axis=0), np.mean(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["momentX_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentY_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)]['n_'+ str(i)][:, 1])))
    return vartbe

def perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, scales, thresh, flags, latent, events_per_file=10000, particle='Ele', dformat='channels_last'):
    sortedpath = os.path.join(sortdir, 'events_*.h5')
    Test = flags[0]
    save_data = flags[1]
    read_data = flags[2]
    save_gen = flags[3]
    read_gen = flags[4]
    save_disc = flags[5]
    read_disc =  flags[6]
    var= {}
    num_events1= 10000
    num_events2 = num_events
    if read_data: # Read from sorted dir                                                                                                                                                                           
       start = time.time()
       var = load_sorted(sortedpath, energies)
       sort_time = time.time()- start
       print ("Events were loaded in {} seconds".format(sort_time))
    else:
       # Getting Data                                                                                                                                                                                              
       events_per_file = 10000
       Filesused = int(math.ceil(num_data/events_per_file))
       print(Filesused)
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
       Trainfiles = Trainfiles[: Filesused]
       Testfiles = Testfiles[: Filesused]
       print(len(Trainfiles))
       print(len(Testfiles))
       if Test:
          data_files = Testfiles
       else:
          data_files = Trainfiles
       start = time.time()
       print(data_files)
       var = get_sorted(data_files, energies, True, num_events1, num_events2)
       data_time = time.time() - start
       print ("{} events were loaded in {} seconds".format(num_data, data_time))
       if save_data:
          save_sorted(var, energies, sortdir)
    total = 0
    for energy in energies:
    #calculations for data events
      var["events_act"+ str(energy)]= np.squeeze(var["events_act"+ str(energy)])
      # Getting dimensions of ecal images
      x = var["events_act"+ str(energy)].shape[1]
      y =var["events_act"+ str(energy)].shape[2]
      z =var["events_act"+ str(energy)].shape[3]
                          
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
      total += var["index" + str(energy)]
      var["ecal_act"+ str(energy)]=np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))
      var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)])
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
    data_time = time.time() - start
    print ("{} events were put in {} bins".format(total, len(energies)))
    #### Generate Data table to screen                                                                                                                                                                             
    print ("Actual Data")
    print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")
    for energy in energies:
       print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1])))

    for gen_weights, disc_weights, scale, i in zip(gweights, dweights, scales, np.arange(len(gweights))):
       gendir = gendirs + '/n_' + str(i)
       discdir = discdirs + '/n_' + str(i)
       for energy in energies:
                               
          var["events_gan" + str(energy)]={}
          var["isreal_act" + str(energy)]={}
          var["isreal_gan" + str(energy)]={}
          var["aux_act" + str(energy)]={}
          var["aux_gan" + str(energy)]={}
          var["ecal_act" + str(energy)]={}
          var["ecal_gan" + str(energy)]={}
          var["max_pos_gan" + str(energy)]={}
          var["sumsx_gan"+ str(energy)]={}
          var["sumsy_gan"+ str(energy)]={}
          var["sumsz_gan"+ str(energy)]={}
          var["momentX_gan" + str(energy)]={}
          var["momentY_gan" + str(energy)]={}
          var["momentZ_gan" + str(energy)]={}
          if read_gen:
             var["events_gan" + str(energy)]['n_'+ str(i)]= get_gen(energy, gendir)
          else:
             g.load_weights(gen_weights)
             start = time.time()
             var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100], latent)
             var["events_gan" + str(energy)]['n_'+ str(i)] = np.squeeze(var["events_gan" + str(energy)]['n_'+ str(i)])
             if save_gen:
                save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], var["energy" + str(energy)], energy, gendir)
             gen_time = time.time() - start
            
             print ("Generator took {} seconds to generate {} events".format(gen_time, var["index" +str(energy)]))
          if read_disc:
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)], var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= get_disc(energy, discdir)
          else:
             d.load_weights(disc_weights)
             start = time.time()
             if dformat=='channels_last':
               var["events_act" + str(energy)] = np.expand_dims(var["events_act" + str(energy)], axis=-1)
               var["events_gan" + str(energy)]['n_'+ str(i)] = np.expand_dims(var["events_gan" + str(energy)]['n_'+ str(i)], axis=-1)
             else:
               var["events_act" + str(energy)] = np.expand_dims(var["events_act" + str(energy)], axis=1)
               var["events_gan" + str(energy)]['n_'+ str(i)] = np.expand_dims(var["events_gan" + str(energy)]['n_'+ str(i)], axis=1)
             discout= discriminate(d, var["events_act" + str(energy)] * scale)
             print(len(discout))
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= discriminate(d, var["events_act" + str(energy)] * scale)
             var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= discriminate(d, var["events_gan" + str(energy)]['n_'+ str(i)] )
             disc_time = time.time() - start
             print ("Discriminator took {} seconds for {} data and generated events".format(disc_time, var["index" +str(energy)]))
             var["events_act" + str(energy)]= np.squeeze(var["events_act" + str(energy)])
             var["events_gan" + str(energy)]['n_'+ str(i)]= np.squeeze(var["events_gan" + str(energy)]['n_'+ str(i)])
             if save_disc:
               discout = {}
               for key in var:
                  if key in ["isreal_act" + str(energy), "aux_act" + str(energy), "isreal_gan" + str(energy), "aux_gan" + str(energy), "ecal_act"+ str(energy), "ecal_gan"+ str(energy)]:
                     discout[key]=var[key]['n_'+ str(i)]
               for key in discout:
                   print (key)
               save_discriminated(discout, energy, discdir)
          print ('Calculations for ....', energy)
          var["events_gan" + str(energy)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)]/scale
          var["isreal_act" + str(energy)]['n_'+ str(i)] = np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)])
          var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["ecal_act"+ str(energy)]['n_'+ str(i)]/scale)

          var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["ecal_gan"+ str(energy)]['n_'+ str(i)]/scale)
          var["max_pos_gan" + str(energy)]['n_'+ str(i)] = get_max(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)] = get_sums(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["momentX_gan" + str(energy)]['n_'+ str(i)], var["momentY_gan" + str(energy)]['n_'+ str(i)], var["momentZ_gan" + str(energy)]['n_'+ str(i)] = get_moments(var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)], m, x=x, y=y, z=z)

       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))

       #### Generate GAN table to screen                                                                                                                                                                          
 
       print ("Generated Data")
       print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")

       for energy in energies:
          print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["max_pos_gan" + str(energy)]['n_'+ str(i)], axis=0), np.mean(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["momentX_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentY_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)]['n_'+ str(i)][:, 1])))

    return var
