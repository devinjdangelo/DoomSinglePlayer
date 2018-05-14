# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:52:02 2018

@author: DDAngelo
"""
import tensorflow as tf
import numpy as np
from collections import Counter
import itertools as it
import time
import skimage
from skimage import color
import imageio
import csv

from model.Network import DFP_Network
from model.ExperienceH5 import H5ExperienceRecorder
from model.utils import get_targets,gen_random_mask, action_indecies_to_tensor

class Trainer:
    #trains VAE and Future Predictor
    def __init__(self,args):
        
        self._define_action_groups(args['num_action_splits'],args['a_size'],args['group_cond'])
 
        self.num_actions = [len(actions) for actions in self.group_actions]
        args['num_actions'] = self.num_actions
        
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.net = DFP_Network(args)
        self.model_path = args['model_path']
        self.saver = tf.train.Saver(max_to_keep=10,keep_checkpoint_every_n_hours=1)
        if args['load_model']:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            
        if 'mem_location' not in args:
            raise ValueError('When training VAE or future prediction, must provide filepath for ExperienceRecorder. \
                             use key "mem_location')
            
        self.experience_memory = H5ExperienceRecorder(args)
        self.z_offsets = args['z_offsets'] if 'z_offsets' in args else None
        self.zdim = args['z_dim'] if 'z_dim' in args else None
        self.z_num_offsets = len(self.z_offsets)

        self.num_measurements = args['num_measurements']
        self.num_observe_m = args['num_observe_m']
        self.num_predict_m = args['num_predict_m']
        self.num_offsets = args['num_offsets']
        
        self.num_buttons = args['num_buttons'] 
        
        self.levels_normalization_factors = args['levels_normalization']
        self.delta_normalization_factors = args['delta_normalization']
        
        self.sequence_length = args['sequence_length'] if 'sequence_length' in args else None
        
        self.xdim,self.ydim = args['framedims']
        self.num_action_splits = args['num_action_splits']
        
        self.stat_file = args['model_stats'] if 'model_stats' in args else None
        if self.stat_file is not None:
            with open(self.stat_file, 'w') as f:
                    wr = csv.writer(f)
                    wr.writerow(['iter','frame loss','label loss','kl loss','test frame','test label','test kl'])

        
    def _define_action_groups(self,splits,a_sizes,group_cond):
        
        self.group_actions = []
        for i in range(splits):
            group_i = [list(a) for a in it.product([0, 1], repeat=a_sizes[i])]
            group_i = [a for a in group_i if group_cond[i](a)]
            self.group_actions.append(group_i)
           
        self.group_buttons = a_sizes
        
    def train_VAE(self,size,epochs,save_every_n_iters=100):
        iters = int(sum(self.episode_lens)//size * epochs)
        fetch_list = [self.net.reconstruction_loss,
                      self.net.label_loss,
                      self.net.kl_loss,
                      self.net.VAE_apply_grads]
        
        t = time.time()
        losslistl = []
        losslistf = []
        losslistk = []
        for i in range(iters):  
            frame_batch,labels_batch = self.sample_frames2(size)
            labels_prepped = (labels_batch>0).astype(np.float32) #convert to 0 and 1 only
            frame_prepped = np.zeros(frame_batch.shape)
            frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-18.4)/14.5
            frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-3)/8.05
            frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-5.11)/13.30 
            steps = i*size
            feed_dict = {self.net.steps:steps,
                         self.net.observation:frame_prepped,
                         self.net.label:labels_prepped}
            
            floss,lloss,kloss, _ = self.sess.run(fetch_list,feed_dict=feed_dict)
            losslistl.append(lloss)
            losslistf.append(floss)
            losslistk.append(kloss)
            if len(losslistl)>50:
                losslistl.pop(0)
                losslistf.pop(0)
                losslistk.pop(0)
                        
            seconds_per_iter = (time.time()-t)//(i+1)
            minutes_remaining = (seconds_per_iter * (iters-i))//60
            print('iter: ',i,' loss frame',floss, ' Avg loss frame ',np.mean(losslistf),
                  ' loss label ', lloss, ' Avg loss label ', np.mean(losslistl),
                  ' kl loss ', kloss, 'avg kl loss ', np.mean(losslistk),
                  ' seconds per iter:  ',seconds_per_iter,
                  ' time remaining: ' , minutes_remaining,' minutes')
            if i%save_every_n_iters==0:
                self.saver.save(self.sess,self.model_path+str(i)+'.ckpt')
                test_f_loss,test_l_loss,test_klloss = self.test_VAE(i)
                
                if self.stat_file is not None:
                   savelist = [i,np.mean(losslistf),np.mean(losslistl),np.mean(losslistk),test_f_loss,test_l_loss,test_klloss]
                   with open(self.stat_file, 'a') as teststatfile:
                       wr = csv.writer(teststatfile)
                       wr.writerow(['{:.2f}'.format(x) for x in savelist])
                
    def test_VAE(self,step):
        frame,labels = self.sample_frames2(1,testing=True)
        frame_prepped = np.zeros(frame.shape)
        frame_prepped[:,:,:,0] = (frame[:,:,:,0]-18.4)/14.5
        frame_prepped[:,:,:,1] = (frame[:,:,:,1]-3)/8.05
        frame_prepped[:,:,:,2] = (frame[:,:,:,2]-5.11)/13.30 
        
        labels_prepped = (labels>0).astype(np.float32)
        
        fetch_list = [self.net.outframe,self.net.outlabel,
                      self.net.reconstruction_loss,self.net.label_loss,
                      self.net.kl_loss]
        feed_dict = {self.net.steps:0,
                     self.net.observation:frame_prepped,
                     self.net.label:labels_prepped,
                     self.net.conv_training:False}
            
        out_frame,outlabel,floss,lloss,kloss = self.sess.run(fetch_list,feed_dict=feed_dict)
        recon_frame = np.zeros(out_frame.shape)
        recon_frame[:,:,:,0] = out_frame[:,:,:,0]*14.5 + 18.4
        recon_frame[:,:,:,1] = out_frame[:,:,:,1]*8.05 + 3
        recon_frame[:,:,:,2] = out_frame[:,:,:,2]*13.3 + 5.11
               
        framergb = skimage.color.lab2rgb(frame[0,:,:,:].astype(np.float64))
        recon_frame_rgb = skimage.color.lab2rgb(recon_frame[0,:,:,:].astype(np.float64))
        
        outlabel = outlabel[0,:,:,0]
        labels_prepped = labels_prepped[0,:,:,0]
        labels_guess = (outlabel>0.5).astype(np.float32)
        
        imageio.imwrite('./VAE/'+str(step)+'frame.png',framergb)
        imageio.imwrite('./VAE/'+str(step)+'frame_recon.png',recon_frame_rgb)     
        imageio.imwrite('./VAE/'+str(step)+'frame_label_recon.png',outlabel)
        imageio.imwrite('./VAE/'+str(step)+'frame_label_guess.png',labels_guess)
        imageio.imwrite('./VAE/'+str(step)+'frame_label.png',labels_prepped)
        
        return floss,lloss,kloss
        
            
    def sample_frames(self,size):
        #sample frames uniformly at random from all frames in all episodes
        s = sum(self.episode_lens)
        weights = [l/s for l in self.episode_lens]
        episode_indecies = np.random.choice(self.experience_memory.n_episodes, size, replace=True,p=weights).tolist()
        episode_nframe_dict = Counter(episode_indecies) 
        def random_choice_array(array1,array2,size):
            array1 = array1.reshape(-1,self.xdim,self.ydim,3)
            array2 = array2.reshape(-1,self.xdim,self.ydim,1)
            m = len(array1)
            idx = np.random.choice(m,size=size,replace=False)
            return array1[idx,:,:,:],array2[idx,:,:,:]
        
        frames = [random_choice_array(self.experience_memory.frame[key],self.experience_memory.lbuffer[key],size=value) \
                  for key,value in episode_nframe_dict.items()]
        
        observations = np.concatenate([f[0] for f in frames])
        labels = np.concatenate([f[1] for f in frames])
        
        return observations,labels
    
    def sample_frames2(self,size,testing=False):
        #sample frames uniformly at random from all frames in all episodes
        index = np.random.choice(self.experience_memory.n_episodes - 1) if not testing else self.experience_memory.n_episodes - 1

        def random_choice_array(array1,array2,size):
            array1 = array1.reshape(-1,self.xdim,self.ydim,3)
            array2 = array2.reshape(-1,self.xdim,self.ydim,1)
            m = len(array1)
            idx = np.random.choice(m,size=size,replace=False)
            return array1[idx,:,:,:],array2[idx,:,:,:]
        
        observations,labels = random_choice_array(self.experience_memory.frame[index],self.experience_memory.lbuffer[index],size=size)
                
        return observations,labels
            
    def train_latent_future_predictor(self,size,epochs,save_every_n_iters=100):
        self.construct_z_vectors()
        self.construct_z_targets()
        iters = int(self.experience_memory.n_episodes//size * epochs)
        fetch_list = [self.net.z_apply_grads,self.net.latent_z_loss]
        for i in range(iters):
            steps = i*size
            measurements_batch, \
            a_history_batch,a_taken_batch, \
            z_batch,z_target_batch = self.sample_z_sequences(size)
            
            a_tensors = [action_indecies_to_tensor(a_taken_batch[:,i],len(self.z_offsets),self.zdim,
                                               self.num_actions[i]) for i in range(len(self.group_buttons))]
        
            m_in_prepped = self.prep_m(measurements_batch[:,:self.num_observe_m],levels=True,verbose=False)
               
        
            c_state = np.zeros((size, self.net.cell.state_size.c), np.float32)
            h_state = np.zeros((size, self.net.cell.state_size.h), np.float32) 
            
            feed_dict = {self.net.compute_z:False,
                         self.net.feed_z:z_batch,
                         self.net.measurements:m_in_prepped,
                         self.net.action_history:a_history_batch,
                         self.net.z_target:z_target_batch,
                         self.net.steps:steps,
                         self.net.c_in:c_state,
                         self.net.h_in:h_state}
            
            chosen_dict = {i: d for i, d in zip(self.net.z_a_chosen, a_tensors)}
            feed_dict.update(chosen_dict)
            
            _,loss = self.sess.run(fetch_list,feed_dict=feed_dict)
            
            print('i: ', i, ' loss: ', loss)
            if i%save_every_n_iters==0:
                self.saver.save(self.sess,self.model_path+str(i)+'.ckpt')
                
            if i%(iters//epochs) == 0:
                print('recomputing z_vectors')
                self.construct_z_vectors()
                self.construct_z_targets()


            
    def construct_z_vectors(self):
        print('Constructing z vectors...')
        feed_dict = {self.net.steps:0}
        fetch_list = [self.net.latent_z]
        def frames_to_zs(frames):
            feed_dict[self.net.observation] = frames.reshape(-1,self.xdim,self.ydim,3)
            episode_z_vectors = self.sess.run(fetch_list,feed_dict=feed_dict)
            return episode_z_vectors[0]
        
        self.z_vector = [frames_to_zs(frames) for frames in self.experience_memory.frame]
        print('Done!')
        
    def construct_z_targets(self):
        print('constructing z offsets...')
        array = []
        for z in self.z_vector:
            array.append(get_targets(z,self.z_offsets))
        self.z_targets = array
        print('Done!')
        
    def sample_z_sequences(self,size):
                   
        episode_indecies = np.random.choice(self.experience_memory.n_episodes, size, replace=False).tolist()
        
        measurements_batch = [self.experience_memory.measurements[i].reshape(-1,self.num_measurements) for i in episode_indecies]
        a_history_batch = [self.experience_memory.a_history[i].reshape(-1,self.num_buttons) for i in episode_indecies]
        a_taken_batch = [self.experience_memory.a_taken[i].reshape(-1,self.num_action_splits)  for i in episode_indecies]
        z_batch = [self.z_vector[i].reshape(-1,self.zdim) for i in episode_indecies]
        z_target_batch = [self.z_targets[i].reshape(-1,self.z_num_offsets,self.zdim) for i in episode_indecies]

        episode_lengths = [len(episode.reshape(-1,self.num_action_splits)) for episode in a_taken_batch]
        masks = [gen_random_mask(l,self.sequence_length,10) for l in episode_lengths]
        measurements_batch = np.stack([measurements_batch[i][mask] for i,mask in enumerate(masks)])
        a_history_batch = np.stack([a_history_batch[i][mask] for i,mask in enumerate(masks)])
        a_taken_batch = np.stack([a_taken_batch[i][mask] for i,mask in enumerate(masks)])
        z_batch = np.stack([z_batch[i][mask] for i,mask in enumerate(masks)])
        z_target_batch = np.stack([z_target_batch[i][mask] for i,mask in enumerate(masks)])
        
        measurements_batch = measurements_batch.reshape([-1, self.num_measurements])
        a_history_batch = a_history_batch.reshape([-1,self.num_buttons])
        a_taken_batch = a_taken_batch.reshape([-1,self.num_action_splits])
        z_batch = z_batch.reshape([-1,self.zdim])
        z_target_batch = z_target_batch.reshape([-1,len(self.z_offsets),self.zdim])
        
        return measurements_batch,a_history_batch,a_taken_batch,z_batch,z_target_batch
    
        
    def prep_m(self,m,levels=False,verbose=False):
        
        if levels:
            #measurements represent running totals or current value in case of health
            m = np.reshape(m,[-1,self.num_observe_m])
            mout = np.zeros([m.shape[0],self.num_observe_m])
            for i in range(self.num_observe_m):
                a,b = self.levels_normalization_factors[i]
                mout[:,i] = (m[:,i]-a) / b  
           
                if verbose:
                    print("range level ",i,": ", np.amin(mout[:,i])," to ",np.amax(m[:,i]))
        else:
            mout = np.zeros([m.shape[0],self.num_offsets,self.num_predict_m])
            for i in range(self.num_predict_m):
                a,b = self.delta_normalization_factors[i]
                mout[:,:,i] = (m[:,:,i]-a) / b

                if verbose:
                    print("range delta ",i,": ", np.amin(m[:,:,i])," to ",np.amax(m[:,:,i]))

        return mout
        
    
    @property
    def episode_lens(self):
        return [len(episode.reshape(-1,self.num_action_splits)) for episode in self.experience_memory.a_taken]