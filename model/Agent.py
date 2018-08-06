import numpy as np
import tensorflow as tf
from vizdoom import *
import imageio
import skimage
from skimage import color


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
        
        tf.reset_default_graph()
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
        self.sess = tf.Session()

        self.net = PPO(args)
        self.reset_state()
        
        self.model_path = args['model_path']
        self.saver = tf.train.Saver(max_to_keep=50,keep_checkpoint_every_n_hours=1)
        if args['load_model']:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            print(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
            #self.saver.restore(self.sess,'/home/djdev/Documents/Tensorflow/Doom/Training/Training8100.ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())
            
        
        
    def reset_state(self):
        self.c_state = np.zeros((1, self.net.cell.state_size.c), np.float32)
        self.h_state = np.zeros((1, self.net.cell.state_size.h), np.float32)    
        
        self.attack_cooldown = 0
        self.attack_action_in_progress = [0,0]
        
        self.holding_down_use = 0
        self.use_cooldown = 0
        
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
        
        returns_batch = gae_batch + state_value_batch
        gae_batch = (gae_batch - gae_batch.mean())/(gae_batch.std()+1e-8)
        
        frame_prepped = np.zeros(frame_batch.shape,dtype=np.float32)
        frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-18.4)/14.5
        frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-3)/8.05
        frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-5.11)/13.30 
        
        c_state = np.zeros((batch_size, self.net.cell.state_size.c), np.float32)
        h_state = np.zeros((batch_size, self.net.cell.state_size.h), np.float32) 
                           
        feed_dict = {self.net.observation:frame_prepped,
            self.net.measurements:m_in_prepped,
            self.net.action_history:a_history_batch,
            self.net.lgprob_a_pi_old:a_taken_prob_batch,
            self.net.a_taken:aidx_batch,
            self.net.returns:returns_batch,
            self.net.old_v_pred:state_value_batch,
            self.net.GAE:gae_batch,
            self.net.learning_rate:lr,
            self.net.clip_e:clip,
            self.net.steps:steps,
            self.net.c_in:c_state,
            self.net.h_in:h_state}
                
        ploss,closs,entropy,g_n,_ = self.sess.run([self.net.pg_loss,
                                        self.net.vf_loss,
                                        self.net.entropy,
                                        self.net.grad_norms,
                                        self.net.apply_grads],feed_dict=feed_dict)
        return ploss,closs,entropy,g_n

    def evaluate_human_batch(self,batch_size,human_batch):
        #takes in batch of human data and evaluates chosen actions, computing pi(a_human), value(state), and GAE(human_episode)
        frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch = human_batch
        
        frame_batch = frame_batch.reshape([-1,self.xdim,self.ydim,3])
        measurements_batch = measurements_batch.reshape([-1, self.num_measurements])
        a_history_batch = a_history_batch.reshape([-1,self.num_buttons])
        aidx_batch = aidx_batch.reshape([-1,self.num_action_splits])
        a_taken_prob_batch = a_taken_prob_batch.reshape([-1])
        state_value_batch = state_value_batch.reshape([-1])
        gae_batch = gae_batch.reshape([-1])       
        
        m_in_prepped = self.prep_m(measurements_batch[:,:self.num_observe_m],verbose=False)
        
        frame_prepped = np.zeros(frame_batch.shape)
        frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-18.4)/14.5
        frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-3)/8.05
        frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-5.11)/13.30 
        
        c_state = np.zeros((batch_size, self.net.cell.state_size.c), np.float32)
        h_state = np.zeros((batch_size, self.net.cell.state_size.h), np.float32) 
        
                           
        feed_dict = {self.net.observation:frame_prepped,
            self.net.measurements:m_in_prepped,
            self.net.action_history:a_history_batch,
            self.net.feed_action:aidx_batch,
            self.net.steps:0,
            self.net.time_steps:513,
            self.net.c_in:c_state,
            self.net.h_in:h_state}
                
        a_taken_prob_batch,state_value_batch = self.sess.run([self.net.feed_action_prob,
                                        self.net.critic_state_value],feed_dict=feed_dict)

        frame_batch= np.reshape(frame_batch,[batch_size,-1,self.xdim,self.ydim,3])
        measurements_batch = np.reshape(measurements_batch,[batch_size,-1,self.num_measurements])
        a_history_batch = np.reshape(a_history_batch,[batch_size,-1,self.num_buttons])
        aidx_batch = np.reshape(aidx_batch,[batch_size,-1,self.num_action_splits])
        a_taken_prob_batch = np.reshape(a_taken_prob_batch,[batch_size,-1])
        state_value_batch = np.reshape(state_value_batch,[batch_size,-1])
                
        m_diff = np.diff(measurements_batch[:,:,-self.num_predict_m:],axis=1)
        #rewards -> seq_len - 1 
        rewards = np.sum(m_diff * self.reward_weights,axis=2)
        gae_batch = np.reshape(gae_batch,[batch_size,-1])

        for i in range(batch_size):
            r = rewards[i,:]
            v = state_value_batch[i,:]
            gae_batch[i,:] = GAE(r,v,self.gweight,self.lweight)
            
        return  frame_batch,measurements_batch,a_history_batch,aidx_batch,a_taken_prob_batch,state_value_batch,gae_batch
        

    def choose_action(self,s,m,ahistory,total_steps,testing,selected_weapon):
                        
        frame_prepped = np.zeros(s.shape)
        frame_prepped[:,:,0] = (s[:,:,0]-18.4)/14.5
        frame_prepped[:,:,1] = (s[:,:,1]-3)/8.05
        frame_prepped[:,:,2] = (s[:,:,2]-5.11)/13.30 
        
        m_in = m[:self.num_observe_m]
        m_prepped = self.prep_m(m_in)
                
        out_tensors = [self.net.lstm_state,self.net.critic_state_value]
        if not testing:
            out_tensors = out_tensors + [self.net.sampled_action,self.net.sampled_action_prob]
        else:
            out_tensors = out_tensors + [self.net.best_action,self.net.best_action_prob]


        lstm_state,value,action,prob = self.sess.run(out_tensors, 
        feed_dict={
        self.net.observation:[frame_prepped],
        self.net.measurements:m_prepped,
        self.net.action_history:[ahistory],
        self.net.c_in:self.c_state,
        self.net.h_in:self.h_state,
        self.net.steps:total_steps,
        self.net.time_steps:1})            

        self.c_state, self.h_state = lstm_state
        # x,y,z,theta,a,e -> x,y,jump,theta,attack,switch,use
        action = action[0]
        prob = prob[0]
        value = value[0]
        vz_action = np.concatenate([self.group_actions[i][action[i]] for i in range(len(action))])

        #print("Net Raw Action", action)
        #print("vizdoom_action",a)
        

        if self.attack_cooldown>0:
            vz_action[9:11] = self.attack_action_in_progress
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
                
        if self.holding_down_use==1:
            if self.use_cooldown>0:
                vz_action[8]==0
                self.use_cooldown -= 1
            else:
                self.holding_down_use=0
                vz_action[8]=0
        elif vz_action[8]==1:
             self.holding_down_use=1
             self.use_cooldown = 6
        #action_array is an action accepted by Vizdoom engine

        return vz_action.tolist(),action,prob,value


    
    def prep_m(self,m,verbose=False):
        
        #measurements represent running totals or current value in case of health
        m = np.reshape(m,[-1,self.num_observe_m])
        mout = np.zeros([m.shape[0],self.num_observe_m])
        for i in range(self.num_observe_m):
            a,b = self.levels_normalization_factors[i]
            mout[:,i] = (m[:,i]-a)/ b  
       
            if verbose:
                print("range level ",i,": ", np.amin(mout[:,i])," to ",np.amax(mout[:,i]))

        return mout


    
