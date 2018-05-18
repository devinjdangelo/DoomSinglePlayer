# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:19:09 2018

@author: DDAngelo
"""

import numpy as np
from model.utils import get_targets, gen_random_mask
import imageio
import skimage
from skimage import color

class ExperienceRecorder:
    ##Stores experiences in RAM
    def __init__(self,args):
        self.offsets = args['offsets'] if 'offsets' in args else None
        self.sequence_length = args['sequence_length'] if 'sequence_length' in args else None
        self.num_predict_m = args['num_predict_m']
        self.num_observe_m = args['num_observe_m']
        self.num_measurements = args['num_measurements']
        self.xdim,self.ydim = args['framedims']
        self.num_action_splits = args['num_action_splits']
        self.num_buttons = args['num_buttons']
        
        self.max_episodes = args['max_episodes'] if 'max_episodes' in args else None
        
        self.need_to_build_offsets = False
        
        self.frame = []
        self.measurements = []
        self.a_history = []
        self.a_taken = []
        self.valid_indecies = []
        self.lbuffer = []
                   
    def build_offsets(self):
        print('constructing offsets...')
        array = []
        for m in self.measurements:
            m = np.reshape(m,[-1,self.num_measurements])
            array.append(get_targets(m[:,-self.num_predict_m:],self.offsets))
            
        self.targets = array
        self.need_to_build_offsets = False

        print('done!')
            
            
    def append_episode(self,episode_buffer):
               
        l = len(episode_buffer[0])
        
        episode_buffer = np.reshape(np.array(episode_buffer,dtype=object),[-1,l])
        
        frame = np.stack(episode_buffer[:,0])
        measurments = np.stack(episode_buffer[:,1])
        a_history = np.stack(episode_buffer[:,2])
        a_taken = np.stack(episode_buffer[:,3])
        lbuffer = np.stack(episode_buffer[:,4])
        
        self.add_data(frame,measurments,a_history,a_taken,lbuffer)
        self.need_to_build_offsets = True

    def add_data(self,frame,measurments,a_history,a_taken,lbuffer):
        
        if self.max_episodes is not None:
            if self.n_episodes >= self.max_episodes:
                self.frame.pop(0)
                self.measurements.pop(0)
                self.a_history.pop(0)
                self.a_taken.pop(0)
                self.lbuffer.pop(0)
        
        self.frame.append(frame) 
        self.measurements.append(measurments) 
        self.a_history.append(a_history) 
        self.a_taken.append(a_taken) 
        self.lbuffer.append(lbuffer)

        
    def get_batch(self,size,get_lbuff=False):
        #get batch of episodes from recording.
        if self.need_to_build_offsets:
            self.build_offsets()
        
        #if len(self.valid_indecies)<size:
            #self.valid_indecies = list(range(self.n_episodes))
            
        episode_indecies = np.random.choice(self.n_episodes, size, replace=False).tolist()
        #self.valid_indecies = [idx for idx in self.valid_indecies if idx not in episode_indecies]
        
        if not get_lbuff:
            frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch = self.retrieve_batch(episode_indecies)
            return self.process_batch(frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch)
        else:
            frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch,label_batch = self.retrieve_batch(episode_indecies,get_lbuff=True)
            return self.process_batch(frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch,label_batch)

        
    def retrieve_batch(self,episode_indecies,get_lbuff=False):  
        frame_batch = [self.frame[i] for i in episode_indecies]
        measurements_batch = [self.measurements[i] for i in episode_indecies]
        a_history_batch = [self.a_history[i] for i in episode_indecies]
        a_taken_batch = [self.a_taken[i] for i in episode_indecies]
        target_batch = [self.targets[i] for i in episode_indecies]
        if get_lbuff:
            labels_batch = [self.lbuffer[i] for i in episode_indecies]
            return frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch,labels_batch
        else:
            return frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch

        
    def process_batch(self,frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch,label_batch=None):
        episode_lengths = [len(episode) for episode in a_taken_batch]
        masks = [gen_random_mask(l,self.sequence_length,10) for l in episode_lengths]
        frame_batch = np.stack([frame_batch[i][mask] for i,mask in enumerate(masks)])
        measurements_batch = np.stack([measurements_batch[i][mask] for i,mask in enumerate(masks)])
        a_history_batch = np.stack([a_history_batch[i][mask] for i,mask in enumerate(masks)])
        a_taken_batch = np.stack([a_taken_batch[i][mask] for i,mask in enumerate(masks)])
        target_batch = np.stack([target_batch[i][mask] for i,mask in enumerate(masks)])
        
        frame_batch = frame_batch.reshape([-1,self.xdim,self.ydim,3])
        measurements_batch = measurements_batch.reshape([-1, self.num_measurements])
        a_history_batch = a_history_batch.reshape([-1,self.num_buttons])
        a_taken_batch = a_taken_batch.reshape([-1,self.num_action_splits])
        target_batch = target_batch.reshape([-1,len(self.offsets),self.num_predict_m])
        
        if label_batch is not None:
            label_batch = np.stack([label_batch[i][mask] for i,mask in enumerate(masks)])
            label_batch = label_batch.reshape([-1,self.xdim,self.ydim,1])
            return frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch,label_batch
        else:
            return frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch
    
    def masked_episode_to_gif(self):
        episode_indecies = np.random.choice(self.n_episodes, 1, replace=False).tolist()
        frame_batch = [self.frame[i].reshape(-1,self.xdim,self.ydim,3) for i in episode_indecies]
        episode_lengths = [len(episode) for episode in frame_batch]
        masks = [gen_random_mask(l,self.sequence_length,10) for l in episode_lengths]
        frame_batch = np.stack([frame_batch[i][mask] for i,mask in enumerate(masks)])
        frame_batch = frame_batch.reshape([-1,self.xdim,self.ydim,3])
        framergb = np.zeros(frame_batch.shape)
        framergb[:,:,:,0] = frame_batch[:,:,:,0]
        framergb[:,:,:,1] = frame_batch[:,:,:,1]
        framergb[:,:,:,2] = frame_batch[:,:,:,2]
        for i in range(framergb.shape[0]):
            framergb[i,:,:,:] = skimage.color.lab2rgb(framergb[i,:,:,:])
        framergb = framergb
        imageio.mimwrite('./masktest'+'masked.gif',framergb,duration=4.4*4/35)
    
    @property
    def n_episodes(self):
        return len(self.a_taken)
                    
    #@property
    #def epoch_progress(self):
       # return 100 - 100*len(self.valid_indecies)/self.n_episodes
    


#        