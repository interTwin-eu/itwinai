import numpy as np
from numpyencoder import NumpyEncoder
import json
import matplotlib.pyplot as plt    

class Payload:
    def __init__(self,freqs:list[float],bandwidths:list[float]=None):
        freqs = [f for f in freqs]        
        self.freqs = freqs
        self.dataframe = []
        self.bandwidths = bandwidths
        self.description = {'Pulsar': None, 'NBRFI': None, 'BBRFI': None}
    
    def plot(self,type:str='img'):
        """Plots the dataframe

        Args:
            type (str, optional): _description_. Defaults to 'img'.

        Returns:
            plt.axes: returns the axes of the plot
        """        
        if type=='img':
            freqs = np.array(self.freqs)
            dataframe = np.array(self.dataframe)
            try:
                rot_phases = np.array(self.rot_phases)
            except:
                print(f'WARNING: Rot phases not assigned')
                rot_phases = np.arange(0,dataframe.shape[1],1)
            plt.figure()
            plt.imshow(dataframe.T,extent=[np.min(rot_phases),np.max(rot_phases),0,len(freqs)])
            #plt.xticks(rot_phases)
            #plt.yticks(freqs)
            plt.xlabel('Phase (degrees)')
            plt.ylabel('freq channel')
            plt.gca().set_aspect('auto')
            return plt.gca()

            
        pass
    
    def add_flux(self,radio_packet:list[list[float]]):
        try:
            freqs_received = radio_packet[1].tolist()            
        except:
            freqs_received = radio_packet[1]            
        try:
            flux_received = radio_packet[0].tolist()
        except AttributeError:
            flux_received = radio_packet[0]

        if np.max(np.abs(np.array(freqs_received)-np.array(self.freqs))) ==0:
        #flux_received = np.array(flux_received)
        #freqs_received = np.array(freqs_received) 
            flux_row = flux_received     
            #flux_row = [flux_received[freqs_received.index(fq)] for fq in self.freqs]        
            self.dataframe.append(flux_row)
        else:
            print(f'ERROR: Freq channels dont match assigned channels')
    
    def add_description(self,description:dict):
        self.description.update(description) 
    
    def assign_bandwidths_to_freqchannels(self,bandwidths:list[float]):
        self.bandwidths = [b for b in bandwidths]

    def assign_rot_phases(self,rot_phases:list[float]):
        self.rot_phases = rot_phases

    def return_payload_shape(self):
        return (np.array(self.dataframe)).shape
    
    def write_payload_to_jsonfile(self,file_name:str):
        with open(file_name, "w") as outfile:
            json_object = json.dumps(self.__dict__,indent=4) 
            outfile.write(json_object)

    @classmethod
    def read_payload_from_jsonfile(cls,filename:str):
        cls_obj = cls(freqs=[])
        f = open(filename)
        payload_json = json.load(f)
        cls_obj.__dict__.update(payload_json)
        return cls_obj
        

        


    

        

    