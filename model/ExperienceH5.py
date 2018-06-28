# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:19:09 2018

@author: DDAngelo
"""

import tables as tb
from traceback import print_exc
import numpy as np
import gc
from model.utils import get_targets

from model.Experience import ExperienceRecorder

class H5ExperienceRecorder(ExperienceRecorder):
    ##interface to HDF5 files to record human play into
    ##data to train initial weights for DFP_Network
    def __init__(self,args):
        super().__init__(args)

        filename = args['mem_location']
        
        try:
            self._kill_any_open_file()
            load_into_mem = args['load_h5_into_mem'] if 'load_h5_into_mem' in args else False
            if load_into_mem:
                print('Loading h5 file into memory...')
                self.h5file = tb.open_file(filename, mode='a', driver="H5FD_CORE")
                print('Done!')
            else:
                self.h5file = tb.open_file(filename, mode='a')
            self.frame = self.h5file.get_node("/","frame")
            self.measurements = self.h5file.get_node("/","measurements")
            self.a_history = self.h5file.get_node("/","a_history")
            self.aidx = self.h5file.get_node("/","aidx")
            self.a_taken_prob = self.h5file.get_node("/","a_taken_prob")
            self.state_value = self.h5file.get_node("/","state_value")
            self.gae = self.h5file.get_node("/","gae")            

        except:
            print_exc()
            try:
                self.h5file.close()
            except:
                pass
            build_file = input('Unable to load H5 File. Would you like to build a new one? This will overwrite any existing file. (y/n): ')
            if build_file=='y':
                self.h5file = tb.open_file(filename, mode='w', title="Doom Replay Data")
                root = self.h5file.root 
                self.frame = self.h5file.create_vlarray(root,'frame',tb.Float32Atom())
                self.measurements = self.h5file.create_vlarray(root,'measurements',tb.Float32Atom())
                self.a_history = self.h5file.create_vlarray(root,'a_history',tb.Float32Atom())
                self.aidx = self.h5file.create_vlarray(root,'aidx',tb.Int32Atom())
                self.a_taken_prob = self.h5file.create_vlarray(root,'a_taken_prob',tb.Float32Atom())
                self.state_value = self.h5file.create_vlarray(root,'state_value',tb.Float32Atom())
                self.gae = self.h5file.create_vlarray(root,'gae',tb.Float32Atom())
                    
            else:
                raise ValueError("No H5 file loaded. Please load valid h5 file or create a new one")      
    
    def _kill_any_open_file(self):
        for obj in gc.get_objects():   # Browse through ALL objects
            if isinstance(obj, tb.file.File):   # Just HDF5 files
                try:
                    print('found open file')
                    obj.close()
                except:
                    pass # Was already closed
         
    def add_data(self,frame,measurments,a_history,aidx,a_taken_prob,state_value,gae):
        #vlarray's only accept 1D array, so must reshape
         
        self.frame.append(frame.reshape(-1)) 
        self.measurements.append(measurments.reshape(-1)) 
        self.a_history.append(a_history.reshape(-1)) 
        self.aidx.append(aidx.reshape(-1)) 
        self.a_taken_prob.append(a_taken_prob.reshape(-1)) 
        self.state_value.append(state_value.reshape(-1))
        self.gae.append(gae)
        
        
            
    def retrieve_batch(self,episode_indecies):  
        frame_batch = [self.frame[i].reshape(-1,self.xdim,self.ydim,3)  for i in episode_indecies]
        measurements_batch = [self.measurements[i].reshape(-1,self.num_measurements) for i in episode_indecies]
        a_history_batch = [self.a_history[i].reshape(-1,self.adim)  for i in episode_indecies]
        aidx_batch = [self.aidx[i].reshape(-1,self.num_groups)  for i in episode_indecies]
        a_taken_prob_batch = [self.a_taken_prob[i] for i in episode_indecies]
        state_value_batch = [self.state_value[i] for i in episode_indecies]
        gae_batch = [self.gae[i] for i in episode_indecies]

        return frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch

    
if __name__=='__main__':
    import time

    #filename = r'C:\Users\djdev\Documents\tensorflow models\doom\h5\human_data.h5'
    filename = r'human_data.h5'
    offsets = [2,4,8,16,32,64]
    recording = H5ExperienceRecorder(filename,offsets)
    
#    for e in range(30):
#        episode_buffer = []
#        print(e)
#        for i in range(np.random.randint(2000,5500)):
#            level_step = int(i)
#            frame = np.ones(shape=[128,128,3],dtype=np.float32)
#            measurements = np.ones(shape=[25],dtype=np.float32)
#            a_history = np.ones(shape=[30,12],dtype=np.int32)
#            a_taken = np.ones(shape=[4],dtype=np.int32)
#            
#            episode_buffer.append([level_step,frame,measurements,a_history,a_taken])
#        
#        recording.append_episode(episode_buffer)
    iters = (recording.n_episodes)//8
    recording.build_offsets(offsets)
    t = time.time()
    for _ in range(iters):
        t2 = time.time()
        batch = recording.get_batch(8)
        print(recording.epoch_progress,'% complete',time.time()-t2,' seconds')
    
    t=time.time()-t
    print(t,'total seconds',t/iters,' s/iter')
        

#        