# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:19:09 2018

@author: DDAngelo
"""

import numpy as np
from model.utils import GAE
import imageio
import skimage
from skimage import color

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExperienceRecorder:
    ##Stores experiences in RAM
    def __init__(self,args):       
        self.sequence_length = args['sequence_length'] if 'sequence_length' in args else None
        self.reward_weights = args['reward_weights']
        self.lweight = args['lambda']
        self.gweight = args['gamma']
        self.bootstrap = args['bootstrap']
        
        self.keep_every_n_timesteps = args['keep_every_n_steps']
        
        self.num_predict_m = args['num_predict_m']
        self.num_observe_m = args['num_observe_m']
        self.num_measurements = args['num_measurements']
        self.xdim,self.ydim = args['framedims']
        self.adim = args['num_buttons']
        self.num_groups = args['num_action_splits']
        
        self.max_episodes = args['max_episodes'] if 'max_episodes' in args else None
        
        self.need_to_build_offsets = False
        
        self.frame = []
        self.measurements = []
        self.a_history = []
        self.aidx = []
        self.a_taken_prob = [] #pi_old(a,s)
        self.state_value = [] #v(s), critic
        self.gae = []
        self.valid_indecies = []
            
    def append_episode(self,episode_buffer):
               
        l = len(episode_buffer[0])
        
        episode_buffer = np.reshape(np.array(episode_buffer,dtype=object),[-1,l])
        
        frame = np.stack(episode_buffer[:,0]).astype(np.float32,copy=False)
        measurments = np.stack(episode_buffer[:,1]).astype(np.float32,copy=False)
        a_history = np.stack(episode_buffer[:,2]).astype(np.int32,copy=False)
        aidx = np.stack(episode_buffer[:,3]).astype(np.int32,copy=False)
        a_taken_prob = np.stack(episode_buffer[:,4]).astype(np.float32,copy=False)
        state_value = np.stack(episode_buffer[:,5]).astype(np.float32,copy=False)
        gae,rewards = self.get_gae(measurments,state_value)

        if rank==0:
            gather_frames = np.empty([size]+list(frame.shape),dtype=np.float32)
            gather_measurements = np.empty([size]+list(measurments.shape),dtype=np.float32)
            gather_ahist = np.empty([size]+list(a_history.shape),dtype=np.int32)
            gather_aidx = np.empty([size]+list(aidx.shape),dtype=np.int32)
            gather_aprob = np.empty([size]+list(a_taken_prob.shape),dtype=np.float32)
            gather_val = np.empty([size]+list(state_value.shape),dtype=np.float32)
            gather_gae = np.empty([size]+list(gae.shape),dtype=np.float32)
        else:
            gather_frames = None
            gather_measurements = None
            gather_ahist = None
            gather_aidx = None
            gather_aprob = None
            gather_val = None
            gather_gae = None

        comm.Gather(frame,gather_frames,root=0)
        comm.Gather(measurments,gather_measurements,root=0)
        comm.Gather(a_history,gather_ahist,root=0)
        comm.Gather(aidx,gather_aidx,root=0)
        comm.Gather(a_taken_prob,gather_aprob,root=0)
        comm.Gather(state_value,gather_val,root=0)
        comm.Gather(gae,gather_gae,root=0)

        if rank==0:
            for i in range(size): 
                self.add_data(gather_frames[i,:,:,:,:],gather_measurements[i,:,:],
                    gather_ahist[i,:,:],gather_aidx[i,:,:],gather_aprob[i,:],
                    gather_val[i,:],gather_gae[i,:])
        return np.sum(rewards)

    def add_data(self,frame,measurments,a_history,aidx,a_taken_prob,state_value,gae):
        
        if self.max_episodes is not None:
            if self.n_episodes >= self.max_episodes:
                self.frame.pop(0)
                self.measurements.pop(0)
                self.a_history.pop(0)
                self.aidx.pop(0)
                self.a_taken_prob.pop(0)
                self.state_value.pop(0)
                self.gae.pop(0)
        
        self.frame.append(frame.astype(np.float32,copy=False)) 
        self.measurements.append(measurments.astype(np.float32,copy=False)) 
        self.a_history.append(a_history.astype(np.float32,copy=False)) 
        self.aidx.append(aidx.astype(np.float32,copy=False))
        self.a_taken_prob.append(a_taken_prob.astype(np.float32,copy=False)) 
        self.state_value.append(np.array(state_value,dtype=np.float32))
        self.gae.append(np.array(gae,dtype=np.float32))

        
    def get_batch(self,size):
        #get batch of episodes from recording.
        
        if len(self.valid_indecies)<size:
            self.valid_indecies = list(range(self.n_episodes))
        episode_indecies = np.random.choice(self.valid_indecies, size, replace=False).tolist()
        self.valid_indecies = [idx for idx in self.valid_indecies if idx not in episode_indecies]

        
        frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch = self.retrieve_batch(episode_indecies)
        return self.process_batch(frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch)


        
    def retrieve_batch(self,episode_indecies):  
        frame_batch = [self.frame[i] for i in episode_indecies]
        measurements_batch = [self.measurements[i] for i in episode_indecies]
        a_history_batch = [self.a_history[i] for i in episode_indecies]
        aidx_batch = [self.aidx[i] for i in episode_indecies]
        a_taken_prob_batch = [self.a_taken_prob[i] for i in episode_indecies]
        state_value_batch = [self.state_value[i] for i in episode_indecies]
        gae_batch = [self.gae[i] for i in episode_indecies]

        return frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch

        
    def process_batch(self,frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch):
        #clip off last timestep since there is no reward/transition available for that step
        frame_batch = np.stack(frame_batch)
        a_taken_prob_batch = np.stack(a_taken_prob_batch)
        a_history_batch = np.stack(a_history_batch)
        aidx_batch = np.stack(aidx_batch)
        state_value_batch = np.stack(state_value_batch).reshape([-1,self.sequence_length+1])
        gae_batch = np.stack(gae_batch).reshape([-1,self.sequence_length])
        measurements_batch = np.stack(measurements_batch)     
        
        # l = a_taken_batch.shape[1]
        # offset = np.random.randint(4)
        # mask = [False]*self.keep_every_n_timesteps
        # mask[offset] = True
        # mask = mask*self.sequence_length
        # #if keep_every_n_timesteps does not exactly divide episode length
        # #we must chop off the remainder by masking with False
        # if l>len(mask):
        #     mask = mask + [False]*(l-len(mask))
        
        # frame_batch = frame_batch[:,mask,:,:,:]
        # a_history_batch = a_history_batch[:,mask,:]
        # a_taken_batch = a_taken_batch[:,mask,:]
        # measurements_batch = measurements_batch[:,mask,:]
        # target_batch = target_batch[:,mask]

        
        return frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch
        
        
    def get_gae(self,m,v):
        m_diff = np.diff(m[:,-self.num_predict_m:],axis=0)
        #rewards -> seq_len - 1 
        rewards = np.sum(m_diff * self.reward_weights,axis=1)
        gae = GAE(rewards,v,self.gweight,self.lweight,bootstrap=self.bootstrap)
        gae = np.array(gae,dtype=np.float32)
        rewards = np.array(rewards,dtype=np.float32)
        return gae,rewards
    
                
    
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
        return len(self.a_taken_prob)
                    
    #@property
    #def epoch_progress(self):
       # return 100 - 100*len(self.valid_indecies)/self.n_episodes
    


#        