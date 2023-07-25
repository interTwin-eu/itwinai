import tensorflow as tf
import joblib
import time



def saveparams(file, **kwargs):
    joblib.dump(kwargs, file)

 

def readparams(file):
    return joblib.load(file)



class Timer():

    def __init__(self, timers=['tot_exec_elapsed_time', 'io_elapsed_time', 'training_elapsed_time']):
        # initialize execution times data structure
        self.exec_times = {}
        self.partials = {}
        for t in timers:
            self.exec_times.update({t:0})
            self.partials.update({t:0})

    def start(self, timer):
        # update partial timers to start counting
        self.partials.update({timer:-time.time()})
    
    def stop(self, timer):
        # add ( stop - start ) time to global execution time
        self.exec_times[timer] += self.partials[timer] + time.time()
        # reset partial
        self.partials[timer] = 0

