#WARNING, most args should not be changed after data has been recorded or trained on
#if you want to change args, you should start with a fresh, new agent

#currently changing action space definitions or num_measurements will not work with
#previously recorded human data which used different values. 

args = {}

#set to TRUE to load h5 file entirely into memory when the code starts
#there is initial overhead but drawing samples will be much quicker. 
#size of h5 file must be much less than total available RAM to leave room
#for other objects in memory.
args['load_h5_into_mem'] = False

#args['mem_location'] = '/home/ddangelo/Documents/tensorflow models/doom/h5/human_data.h5'
args['framedims'] = (128,128) #note, HUD is rendered and then cropped out and remaining frame is squeezed into these dims

args['a_size'] = [6,1,1,1,2]  #must sum to num_buttons 
args['num_action_splits'] = len(args['a_size'])
args['num_buttons'] = sum(args['a_size'])

#here we define a filter over the actions of each action group
#if cond(a) == True we say action a is allowable and we keep it
#this is used to eliminate actions such as move left + move right
#in reality this action reduces to do nothing
group1_cond = lambda a : a[0]+a[1]<2 and a[2]+a[3]<2 and a[4]+a[5]<2
group2_cond = lambda a : True
group3_cond = lambda a : True
group4_cond = lambda a : True
group5_cond = lambda a : sum(a)<2

#need len to be num_action_splits
args['group_cond'] = [group1_cond,group2_cond,group3_cond,group4_cond,group5_cond]

args['start_lr'] = 1e-4
args['half_lr_every_n_steps'] = 1e6
args['episode_length'] = 1024
args['clip_n_timesteps'] = 256
args['sequence_length'] = 192

args['reward_weights'] = [1/50,1/16,1/60,1/2,1/20,1/100,5,0.5,6,-10,100]
args['reward_discount_rate'] = .98

#dropout args passed to tf.train.piecewise_constant then dropout layer
args['exploration'] = 'MDN' #currently only exploration type implemented. See DoomAgent.choose_action 

#TODO make measurements customizable via args
args['num_measurements'] = 29 #must not be changed without corresponding change to DoomSimulator.process_game_vars
args['num_observe_m'] = 24 #agent observe the first e.g. 24 measurements (normalized by levels_normalization)
args['num_predict_m'] = 11 #agent predicts the last e.g. 9 measurements changes over the future offset steps, normalized by delta_normalization

#normalizations tuples (a,b) correspond to the tranformation (m-a)/b ->z score with mean a and sd b
#where m is the measurement at the same index as the tuple.
#this is used to normalize measurments to rough standard range [-1 to 1] or [0 to 1] as desired.
args['levels_normalization'] = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),
                                (0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),
                                (0,1),(0,1),(50,50),(10,10),(55,55),(2,2),(50,50),(50,50)] 




#currently only static goals are supported, recommended to set args['use_goals'] to False unless dynamic goals are implemented
#goal vector and offset vector can be optimized in the middle of training, does not interact with network or any data in the network
#WARNING changing frame_skip changes the meaning of 'step', other args should be adjusted to account for this, e.g. offsets is in num_steps
args['frame_skip'] = 4 #update DoomAgent.choose_action cooldowns to be dynamic if you want to change to other than 4

args['temp_schedule'] = lambda step : 0.8 * 0.5 ** 2e6


args['use_latent_z'] = True
args['z_dim'] = 256
args['num_mixtures'] = 16

args['gif_path'] = './gifs'



