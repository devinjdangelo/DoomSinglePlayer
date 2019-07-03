import numpy as np
import tensorflow as tf
from vizdoom import *
import imageio
import skimage
from skimage import color
from mpi4py import MPI
import time
from math import ceil

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from model.Network import PPO
from model.utils import *
               
class DoomAgent:
    def __init__(self,args):
           
        self.group_buttons = args['group_buttons']
        self.group_actions = args['group_actions']
        
        self.gif_path = args['gif_path'] if 'gif_path' in args else ''
        
        self.num_predict_m = args['num_predict_m']
        self.num_observe_m = args['num_observe_m']
        self.num_measurements = args['num_measurements']
        self.xdim,self.ydim = args['framedims']
        self.reward_weights = args['reward_weights']
        self.lweight = args['lambda']
        self.gweight = args['gamma']
        
        self.num_action_splits = args['num_action_splits']
        self.num_buttons = args['num_buttons'] 
        self.levels_normalization_factors = args['levels_normalization']
        
        self.colorspace = args['colorspace']
        self.gpu_size = args['gpu_size']
        self.sequence_length = args['episode_length'] 
        
        #self.gpu_time = 0
        #self.sync_time = 0
        self.reset_state()
                        
        if rank==0:            
            tf.reset_default_graph()
            #config = tf.ConfigProto()
            #config.gpu_options.allow_growth=True
            self.sess = tf.Session()
    
            self.net = PPO(args)            
            self.model_path = args['model_path']
            self.saver = tf.train.Saver(max_to_keep=50,keep_checkpoint_every_n_hours=1)
            if args['load_model']:
                self.sess.run(tf.global_variables_initializer())
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                restore_file = ckpt.model_checkpoint_path
                #restore_file = '/home/ddangelo/Documents/Tensorflow/doom-ckpts/21400.ckpt'

                variables_can_be_restored = self.net.intersect_vars(restore_file)
                tempsaver = tf.train.Saver(variables_can_be_restored)
                tempsaver.restore(self.sess,restore_file)

                print(restore_file)
                #self.saver.restore(self.sess,ckpt.model_checkpoint_path)
                #self.saver.restore(self.sess,restore_file)
            else:
                self.sess.run(tf.global_variables_initializer())
                

            
        
        
    def reset_state(self):
        self.c_state = np.zeros(512, dtype=np.float32)
        self.h_state = np.zeros(512, dtype=np.float32)    
        
        self.attack_cooldown = 0
        self.attack_action_in_progress = [0,0]
        
        self.holding_down_use = 0
        self.use_cooldown = 0
        
        #if rank==0:
            #print("GPU Time ", self.gpu_time," seconds")
            #print("Sync Time ",self.sync_time," seconds")
            #self.gpu_time = 0
            #self.sync_time = 0
        
    def save(self,episode):
        self.saver.save(self.sess,self.model_path+str(episode)+'.ckpt')

        
    def update(self,batch_size,batch,steps,lr,clip):

        frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch  = batch
        
        
        frame_batch = frame_batch[:,:-1,:,:,:]
        measurements_batch = measurements_batch[:,:-1,:]
        a_history_batch = a_history_batch[:,:-1,:]
        aidx_batch = aidx_batch[:,:-1,:]
        a_taken_prob_batch = a_taken_prob_batch[:,:-1]
        state_value_batch = state_value_batch[:,:-1]
        gae_batch = gae_batch[:,:]

        frame_batch = frame_batch.reshape([-1,self.xdim,self.ydim,3])
        measurements_batch = measurements_batch.reshape([-1, self.num_measurements])
        a_history_batch = a_history_batch.reshape([-1,self.num_buttons])
        aidx_batch = aidx_batch.reshape([-1,self.num_action_splits])
        a_taken_prob_batch = a_taken_prob_batch.reshape([-1])
        state_value_batch = state_value_batch.reshape([-1])
        gae_batch = gae_batch.reshape([-1])
        
        m_in_prepped = self.prep_m(measurements_batch[:,:self.num_observe_m],verbose=False)
        

        
        frame_prepped = np.zeros(frame_batch.shape,dtype=np.float32)
 
        if self.colorspace == 'RGB':
            frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-50)/25
            frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-50)/25
            frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-50)/25
        elif self.colorspace == 'LAB':
            frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-18.4)/14.5
            frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-3)/8.05
            frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-5.11)/13.3
        else:
            raise ValueError('Colorspace, ',self.colorspace,' is undefined')
        
        gather_frame_prepped = None
        gather_m_in_prepped = None
        gather_a_history_batch = None
        gather_aidx_batch = None
        gather_a_taken_prob_batch = None
        gather_state_value_batch = None
        gather_gae_batch = None
        
        if rank==0:
            gather_frame_prepped = np.empty([size]+list(frame_prepped.shape),dtype=np.float32)
            gather_m_in_prepped = np.empty([size]+list(m_in_prepped.shape),dtype=np.float32)
            gather_a_history_batch = np.empty([size]+list(a_history_batch.shape),dtype=np.float32)
            gather_aidx_batch = np.empty([size]+list(aidx_batch.shape),dtype=np.float32)
            gather_a_taken_prob_batch = np.empty([size]+list(a_taken_prob_batch.shape),dtype=np.float32)
            gather_state_value_batch = np.empty([size]+list(state_value_batch.shape),dtype=np.float32)
            gather_gae_batch = np.empty([size]+list(gae_batch.shape),dtype=np.float32)

        comm.Gather(frame_prepped,gather_frame_prepped,root=0)
        comm.Gather(m_in_prepped,gather_m_in_prepped,root=0)
        comm.Gather(a_history_batch,gather_a_history_batch,root=0)
        comm.Gather(aidx_batch,gather_aidx_batch,root=0)
        comm.Gather(a_taken_prob_batch,gather_a_taken_prob_batch,root=0)
        comm.Gather(state_value_batch,gather_state_value_batch,root=0)
        comm.Gather(gae_batch,gather_gae_batch,root=0)
        
        if rank==0:
            batch_size = batch_size*size
            frame_prepped = gather_frame_prepped
            m_in_prepped = gather_m_in_prepped
            a_history_batch = gather_a_history_batch
            aidx_batch = gather_aidx_batch
            a_taken_prob_batch = gather_a_taken_prob_batch
            state_value_batch = gather_state_value_batch
            gae_batch = gather_gae_batch
            
            frame_prepped = frame_prepped.reshape([-1,self.xdim,self.ydim,3])
            m_in_prepped = m_in_prepped.reshape([-1, self.num_observe_m])
            a_history_batch = a_history_batch.reshape([-1,self.num_buttons])
            aidx_batch = aidx_batch.reshape([-1,self.num_action_splits])
            a_taken_prob_batch = a_taken_prob_batch.reshape([-1])
            state_value_batch = state_value_batch.reshape([-1])
            gae_batch = gae_batch.reshape([-1])
            
            returns_batch = gae_batch + state_value_batch
            gae_batch = (gae_batch - gae_batch.mean())/(gae_batch.std()+1e-8)
            
            c_state = np.zeros((self.gpu_size, self.net.cell.state_size.c), np.float32)
            h_state = np.zeros((self.gpu_size, self.net.cell.state_size.h), np.float32) 
            
            if batch_size >= self.gpu_size:
                n_gpu_feeds = ceil(batch_size / self.gpu_size)
            else:
                raise ValueError('Gpu Batch Size cannot be larger than Batch Size')
                
            plossavg,clossavg,entropyavg = 0,0,0
            self.sess.run(self.net.zero_grads)
            for i in range(n_gpu_feeds):
                if i == n_gpu_feeds - 1:
                    start = (batch_size-self.gpu_size)*self.sequence_length
                    finish = batch_size*self.sequence_length
                else:
                    start = i*self.gpu_size*self.sequence_length
                    finish = (i+1)*self.gpu_size*self.sequence_length
                
                idx = list(range(start,finish))
    
               
                feed_dict = {self.net.observation:frame_prepped[idx,:,:,:],
                    self.net.measurements:m_in_prepped[idx,:],
                    self.net.action_history:a_history_batch[idx,:],
                    self.net.lgprob_a_pi_old:a_taken_prob_batch[idx],
                    self.net.a_taken:aidx_batch[idx,:],
                    self.net.returns:returns_batch[idx],
                    self.net.old_v_pred:state_value_batch[idx],
                    self.net.GAE:gae_batch[idx],
                    self.net.learning_rate:lr,
                    self.net.clip_e:clip,
                    self.net.steps:steps,
                    self.net.c_in:c_state,
                    self.net.h_in:h_state}
                        
                ploss,closs,entropy,_ = self.sess.run([self.net.pg_loss,
                                                self.net.vf_loss,
                                                self.net.entropy,
                                                self.net.accum_grads],feed_dict=feed_dict)
                
                plossavg += ploss
                clossavg += closs
                entropyavg += entropy
                
            plossavg /= n_gpu_feeds
            clossavg /= n_gpu_feeds
            entropyavg /= n_gpu_feeds
            
            g_navg, _ = self.sess.run([self.net.grad_norms,self.net.apply_grads],
                                      feed_dict={self.net.accum_count:n_gpu_feeds,
                                                 self.net.learning_rate:lr})
        
        else:
            plossavg,clossavg,entropyavg,g_navg = 0,0,0,0
            
        return plossavg,clossavg,entropyavg,g_navg


    def choose_action(self,s,m,ahistory,total_steps,testing,selected_weapon):
        
        frame_prepped = np.zeros(s.shape,dtype=np.float32)
        if self.colorspace == 'RGB':
            frame_prepped[:,:,0] = (s[:,:,0]-50)/25
            frame_prepped[:,:,1] = (s[:,:,1]-50)/25
            frame_prepped[:,:,2] = (s[:,:,2]-50)/25
        elif self.colorspace == 'LAB':
            frame_prepped[:,:,0] = (s[:,:,0]-18.4)/14.5
            frame_prepped[:,:,1] = (s[:,:,1]-3)/8.05
            frame_prepped[:,:,2] = (s[:,:,2]-5.11)/13.3
        else:
            raise ValueError('Colorspace, ',self.colorspace,' is undefined')
            
        
        m_in = m[:self.num_observe_m]
        m_prepped = self.prep_m(m_in)
        m_prepped = np.squeeze(m_prepped)
        
        fbatch = None
        m_batch = None
        a_batch = None
        c_in_batch = None
        h_in_batch = None
        
        if rank==0:
            fbatch = np.empty([size]+list(frame_prepped.shape),dtype=np.float32)
            m_batch = np.empty([size]+list(m_prepped.shape),dtype=np.float32)
            a_batch = np.empty([size]+list(ahistory.shape),dtype=np.int32)
            c_in_batch = np.empty([size]+list(self.c_state.shape),dtype=np.float32)
            h_in_batch = np.empty([size]+list(self.h_state.shape),dtype=np.float32)
        
        #comm.Barrier()
        #st = time.time()
        comm.Gather(frame_prepped,fbatch,root=0)
        comm.Gather(m_prepped,m_batch,root=0)
        comm.Gather(ahistory,a_batch,root=0)
        comm.Gather(self.c_state,c_in_batch,root=0)
        comm.Gather(self.h_state,h_in_batch,root=0)
        #self.sync_time += time.time() - st
        
        if rank==0:
            out_tensors = [self.net.lstm_state,self.net.critic_state_value]
            if not testing:
                out_tensors = out_tensors + [self.net.sampled_action,self.net.sampled_action_prob]
            else:
                out_tensors = out_tensors + [self.net.best_action,self.net.best_action_prob]
    
            #gt = time.time()
            lstm_state,sendvalue,sendaction,sendprob = self.sess.run(out_tensors, 
            feed_dict={
            self.net.observation:fbatch,
            self.net.measurements:m_batch,
            self.net.action_history:a_batch,
            self.net.c_in:c_in_batch,
            self.net.h_in:h_in_batch,
            self.net.steps:total_steps,
            self.net.time_steps:1})       
            
            #self.gpu_time += time.time() - gt
    
            c_state, h_state = lstm_state
        else:
            c_state = None
            h_state = None
            sendvalue = None
            sendprob = None
            sendaction = None
            
        #comm.Barrier()
        action = np.empty(5,dtype=np.int32)
        prob = np.empty(1,dtype=np.float32)
        value = np.empty(1,dtype=np.float32)
        #st = time.time()
        comm.Scatter(sendaction,action,root=0)
        comm.Scatter(sendprob,prob,root=0)
        comm.Scatter(sendvalue,value,root=0)
        comm.Scatter(c_state,self.c_state,root=0)
        comm.Scatter(h_state,self.h_state,root=0)
        #self.sync_time += time.time()- st
        
        
        vz_action = np.concatenate([self.group_actions[i][action[i]] for i in range(len(action))])
        vz_action = vz_action.astype(np.int32,copy=False)
        
        #print("Net Raw Action", action)
        #print("vizdoom_action",a)
        

        if self.attack_cooldown>0:
           #vz_action[9:11] = self.attack_action_in_progress
           self.attack_cooldown -= 1
        else:
           self.attack_action_in_progress = vz_action[9:11]

           if vz_action[10]==1:
               self.attack_cooldown = 8  #on the 9th step after pressing switch weapons, the agent will actually fire if fire is pressed
           elif vz_action[9]==1:
               if selected_weapon==1:
                   self.attack_cooldown = 3
               elif selected_weapon==2:
                   self.attack_cooldown = 3
               elif selected_weapon==3:
                   self.attack_cooldown = 7
               elif selected_weapon==4:
                   self.attack_cooldown = 1
               elif selected_weapon==5:
                   self.attack_cooldown = 4
               elif selected_weapon==6:
                   self.attack_cooldown = 1
               elif selected_weapon==7:
                   self.attack_cooldown = 9
               elif selected_weapon==8:
                   self.attack_cooldown = 13
               elif selected_weapon==9:
                   self.attack_cooldown = 0
                   
           else:
               self.attack_cooldown = 0
#                
#        if self.holding_down_use==1:
#            if self.use_cooldown>0:
#                vz_action[8]==0
#                self.use_cooldown -= 1
#            else:
#                self.holding_down_use=0
#                vz_action[8]=0
#        elif vz_action[8]==1:
#             self.holding_down_use=1
#             self.use_cooldown = 6
        #action_array is an action accepted by Vizdoom engine

        attacking = self.attack_action_in_progress[0]==1
        return vz_action,action,prob,value,attacking

    
    def prep_m(self,m,verbose=False):
        
        #measurements represent running totals or current value in case of health
        m = np.reshape(m,[-1,self.num_observe_m])
        mout = np.zeros([m.shape[0],self.num_observe_m],dtype=np.float32)
        for i in range(self.num_observe_m):
            a,b = self.levels_normalization_factors[i]
            mout[:,i] = (m[:,i]-a)/ b  
       
            if verbose:
                print("range level ",i,": ", np.amin(mout[:,i])," to ",np.amax(mout[:,i]))

        return mout


    
