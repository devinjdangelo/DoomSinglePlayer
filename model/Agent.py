import numpy as np
import tensorflow as tf
from vizdoom import *
import imageio
import skimage
from skimage import color


from model.Network import DFP_Network
from model.utils import *
			   
class DoomAgent:
	def __init__(self,args):
		
		self.group_buttons = args['group_buttons']
		self.group_actions = args['group_actions']
		
		self.attack_group_idx = len(self.group_actions) - 1
		attack_action = [0]*self.group_buttons[self.attack_group_idx]
		attack_action[0] = 1
		self.attack_action_idx = self.group_actions[self.attack_group_idx].index(attack_action)

		self.use_group_idx = len(self.group_actions)-2
		use_action = [1]
		self.use_action_idx = self.group_actions[self.use_group_idx].index(use_action)
		
		self.gif_path = args['gif_path'] if 'gif_path' in args else ''
		
		self.num_observe_m = args['num_observe_m']
		self.num_predict_m = args['num_predict_m']
		self.num_offsets = args['num_offsets']
		
		self.exploration = args['exploration'] if 'exploration' in args else 'bayesian'
		
		self.num_actions = [len(actions) for actions in self.group_actions]
		args['num_actions'] = self.num_actions
			   
		self.levels_normalization_factors = args['levels_normalization']
		self.delta_normalization_factors = args['delta_normalization']
		
		self.use_goals_in_net = args['use_goals']
		
		self.goal_vector = args['goal_vector']
		self.goal_weights = [np.tile(self.goal_vector,len(self.group_actions[i])*self.num_offsets).reshape(len(self.group_actions[i]),self.num_offsets,self.num_predict_m) for i in range(len(self.group_actions))]

		self.offset_vector = args['offset_vector']
		self.offset_weights = [np.tile(self.offset_vector,len(self.group_actions[i])*self.num_predict_m).reshape(len(self.group_actions[i]),self.num_offsets,self.num_predict_m) for i in range(len(self.group_actions))]
		
		tf.reset_default_graph()
		self.sess = tf.Session()

		self.net = DFP_Network(args)
		self.reset_state()
		
		self.model_path = args['model_path']
		self.saver = tf.train.Saver(max_to_keep=50,keep_checkpoint_every_n_hours=1)
		vae_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='VAE'))
		if args['load_model']:
			print('Loading Model...')
			if args['load_vae']:
				ckpt = tf.train.get_checkpoint_state(self.model_path+'VAE/')
				self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
				vae_saver.restore(self.sess,ckpt.model_checkpoint_path)
			else:
				ckpt = tf.train.get_checkpoint_state(self.model_path)
				self.saver.restore(self.sess,ckpt.model_checkpoint_path)
		else:
			self.sess.run(tf.global_variables_initializer())
			
		
		
	def reset_state(self):
		self.c_state = np.zeros((1, self.net.cell.state_size.c), np.float32)
		self.h_state = np.zeros((1, self.net.cell.state_size.h), np.float32)	
		
		self.attack_cooldown = 0
		self.attack_action_in_progress = 0 
		
		self.holding_down_use = 0
		self.use_cooldown = 0
		
	def save(self,episode):
		self.saver.save(self.sess,self.model_path+str(episode)+'.ckpt')

		
	def update_c(self,batch_size,batch,steps):
		#Get a batch of experiences from the buffer and use them to update the global network
		#to filter down to the chosen actions by batch.
		frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch = batch

		a_tensors = [action_indecies_to_tensor(a_taken_batch[:,i],self.num_offsets,self.num_predict_m,
											   self.num_actions[i]) for i in range(len(self.group_buttons))]
		
		m_in_prepped = self.prep_m(measurements_batch[:,:self.num_observe_m],levels=True,verbose=False)
		
		target_m_prepped = self.prep_m(target_batch,levels=False,verbose=False)
		
		frame_prepped = np.zeros(frame_batch.shape)
		frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-18.4)/14.5
		frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-3)/8.05
		frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-5.11)/13.30 
		
		c_state = np.zeros((batch_size, self.net.cell.state_size.c), np.float32)
		h_state = np.zeros((batch_size, self.net.cell.state_size.h), np.float32) 
							
		feed_dict = {self.net.observation:frame_prepped,
			self.net.measurements:m_in_prepped,
			self.net.action_history:a_history_batch,
			self.net.target:target_m_prepped,
			self.net.steps:steps,
			self.net.c_in:c_state,
			self.net.h_in:h_state}
		
		chosen_dict = {i: d for i, d in zip(self.net.a_chosen, a_tensors)}
		feed_dict.update(chosen_dict)
		
		if self.use_goals_in_net:
			feed_dict[self.net.goals] = self.goal_vector
		
		loss_mse,loss_classify,g_n,_ = self.sess.run([self.net.loss_mse,
										self.net.loss_classify,
										self.net.grad_norms,
										self.net.apply_grads],feed_dict=feed_dict)
		return loss_mse, loss_classify,g_n
	
	def update_all(self,batch_size,batch,steps):
		frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch,labels_batch = batch

		a_tensors = [action_indecies_to_tensor(a_taken_batch[:,i],self.num_offsets,self.num_predict_m,
											   self.num_actions[i]) for i in range(len(self.group_buttons))]
		
		m_in_prepped = self.prep_m(measurements_batch[:,:self.num_observe_m],levels=True,verbose=False)
		
		target_m_prepped = self.prep_m(target_batch,levels=False,verbose=False)
		
		frame_prepped = np.zeros(frame_batch.shape)
		frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-18.4)/14.5
		frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-3)/8.05
		frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-5.11)/13.30 
		
		c_state = np.zeros((batch_size, self.net.cell.state_size.c), np.float32)
		h_state = np.zeros((batch_size, self.net.cell.state_size.h), np.float32) 
		
		labels_prepped = (labels_batch>0).astype(np.float32) #convert to 0 and 1 only
			   
		
		feed_dict = {self.net.observation:frame_prepped,
					 self.net.label:labels_prepped,
			self.net.measurements:m_in_prepped,
			self.net.action_history:a_history_batch,
			self.net.target:target_m_prepped,
			self.net.steps:steps,
			self.net.c_in:c_state,
			self.net.h_in:h_state}
		
		chosen_dict = {i: d for i, d in zip(self.net.a_chosen, a_tensors)}
		feed_dict.update(chosen_dict)
		
		if self.use_goals_in_net:
			feed_dict[self.net.goals] = self.goal_vector
		
		outlabel,out_frame,loss_mse,loss_classify,loss_recon,loss_lab,loss_kl,g_n,_ = self.sess.run(
									[self.net.outlabel,
									 self.net.outframe,
									self.net.loss_mse,
									self.net.loss_classify,
									self.net.reconstruction_loss,
									self.net.label_loss,
									self.net.kl_loss,
									self.net.grad_norms_all,
									self.net.apply_grads_all],feed_dict=feed_dict)

		if steps%(1025*20)==0:
			self.VAE_gif_out(outlabel,out_frame,frame_batch,labels_prepped,steps)
	
		return loss_mse,loss_classify,loss_recon,loss_lab,loss_kl,g_n
		
	def update_vae(self,batch_size,batch,steps):
		frame_batch,measurements_batch,a_history_batch,a_taken_batch,target_batch,labels_batch = batch

		frame_prepped = np.zeros(frame_batch.shape)
		frame_prepped[:,:,:,0] = (frame_batch[:,:,:,0]-18.4)/14.5
		frame_prepped[:,:,:,1] = (frame_batch[:,:,:,1]-3)/8.05
		frame_prepped[:,:,:,2] = (frame_batch[:,:,:,2]-5.11)/13.30 
		
		labels_prepped = (labels_batch>0).astype(np.float32) #convert to 0 and 1 only
			   
		feed_dict = {self.net.observation:frame_prepped,
					self.net.label:labels_prepped,
					self.net.steps:steps}
	
		outlabel,out_frame,loss_recon,loss_lab,loss_kl,_ = self.sess.run(
									[self.net.outlabel,
									self.net.outframe,
									self.net.reconstruction_loss,
									self.net.label_loss,
									self.net.kl_loss,
									self.net.VAE_apply_grads],feed_dict=feed_dict)
									
		if steps%(1025*20)==0:
			self.VAE_gif_out(outlabel,out_frame,frame_batch,labels_prepped,steps)

		return loss_recon,loss_lab,loss_kl

	def VAE_gif_out(self,outlabel,out_frame,frame_batch,labels_prepped,steps):

		recon_frame = np.zeros(out_frame.shape)
		recon_frame[:,:,:,0] = out_frame[:,:,:,0]*14.5 + 18.4
		recon_frame[:,:,:,1] = out_frame[:,:,:,1]*8.05 + 3
		recon_frame[:,:,:,2] = out_frame[:,:,:,2]*13.3 + 5.11

		recon_frame_rgb = np.stack([skimage.color.lab2rgb(recon_frame[i,:,:,:].astype(np.float64)) for i in range(10)])
		framergb = np.stack([skimage.color.lab2rgb(frame_batch[i,:,:,:].astype(np.float64)) for i in range(10)])	   

		outlabel = outlabel[:10,:,:,0]
		labels_prepped = labels_prepped[:10,:,:,0]

		labels_prepped = (labels_prepped*255).astype(np.uint8)
		outlabel = (outlabel*255).astype(np.uint8)
		recon_frame_rgb = (recon_frame_rgb*255).astype(np.uint8)
		framergb = (framergb*255).astype(np.uint8)

		imageio.mimwrite(self.gif_path+'/'+str(steps)+'labels.gif',labels_prepped,duration=4*4/35)
		imageio.mimwrite(self.gif_path+'/'+str(steps)+'labels_pred.gif',outlabel,duration=4*4/35)
		imageio.mimwrite(self.gif_path+'/'+str(steps)+'frame_pred.gif',recon_frame_rgb,duration=4*4/35)
		imageio.mimwrite(self.gif_path+'/'+str(steps)+'frame.gif',framergb,duration=4*4/35)
	
	def network_pass_to_actions(self,advantages):
		#convert forward pass of network into indecies which indicate
		#which action from each group is most advantageous according to
		#current measurment goal

		a_preds = [advantage[0,:,:,:] * self.goal_weights[i] * self.offset_weights[i] for i,advantage in enumerate(advantages)]
		a_preds = [np.sum(np.sum(pred,axis=2),axis=1) for pred in a_preds]
		a_preds = [np.argmax(pred) for pred in a_preds]

		return a_preds 

	def choose_action(self,s,m,ahistory,total_steps,testing,selected_weapon):
		if self.exploration == 'bayesian':
						
			explore = not testing
			m_in = m[:self.num_observe_m]
			m_prepped = self.prep_m(m_in,levels=True)
			
			out_tensors = [self.net.lstm_state,self.net.advantages2]

			lstm_state,advantages = self.sess.run(out_tensors, 
			feed_dict={
			self.net.observation:[s],
			self.net.measurements:m_prepped,
			self.net.action_history:[ahistory],
			self.net.c_in:self.c_state,
			self.net.h_in:self.h_state,
			self.net.exploring:explore,
			self.net.steps:total_steps,
			self.net.time_steps:1})		 

			self.c_state, self.h_state = lstm_state
									   
			a = self.network_pass_to_actions(advantages)
		 
		else:
			raise ValueError('Exploration policy,',self.exploration,
			', is undefined. Please implement policy in Worker.choose_action')
		

		if self.attack_cooldown>0:
			a[self.attack_group_idx] = self.attack_action_in_progress
			self.attack_cooldown -= 1

		else:
			self.attack_action_in_progress = a[self.attack_group_idx]

			if a[self.attack_group_idx]!=self.attack_action_idx and a[self.attack_group_idx]!=0:
				self.attack_cooldown = 8  #on the 9th step after pressing switch weapons, the agent will actually fire if fire is pressed
			elif a[self.attack_group_idx]==self.attack_action_idx:
				#Need to check the selected weapon numbers and cooldowns are correct
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
					
			elif a[self.attack_group_idx]==0:
				self.attack_cooldown = 0
				
		if self.holding_down_use==1:
			if self.use_cooldown>0:
				a[3]==1
				self.use_cooldown -= 1
			else:
				self.holding_down_use=0
				a[3]=0
		elif a[3]==1:
			 self.holding_down_use=1
			 self.use_cooldown = 8
		#action_array is an action accepted by Vizdoom engine
		a = np.asarray(a)
		action_array = np.concatenate([self.group_actions[i][a[i]] for i in range(len(a))]).tolist()
		return a,action_array

	
	def prep_m(self,m,levels=False,verbose=False):
		
		if levels:
			#measurements represent running totals or current value in case of health
			m = np.reshape(m,[-1,self.num_observe_m])
			mout = np.zeros([m.shape[0],self.num_observe_m])
			for i in range(self.num_observe_m):
				a,b = self.levels_normalization_factors[i]
				mout[:,i] = (m[:,i]-a)/ b  
		   
				if verbose:
					print("range level ",i,": ", np.amin(mout[:,i])," to ",np.amax(mout[:,i]))
		else:
			mout = np.zeros([m.shape[0],self.num_offsets,self.num_predict_m])
			for i in range(self.num_predict_m):
				a,b = self.delta_normalization_factors[i]
				mout[:,:,i] = (m[:,:,i]-a)/b

				if verbose:
					print("range delta ",i,": ", np.amin(mout[:,:,i])," to ",np.amax(mout[:,:,i]))

		return mout
	
		 
				
				

	
