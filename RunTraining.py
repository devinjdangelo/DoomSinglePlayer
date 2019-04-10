from model.Simulator import DoomSimulator
from DefaultArgs import args


args['mode'] = 'train'
args['model_path'] = '/home/ddangelo/Documents/Tensorflow/doom-ckpts/'
args['load_model'] = True
args['test_stat_file'] = 'agent_test_data2.csv'
args['reset_file'] = True

args['doom_files_path'] = './doomfiles/'
args['doom_engine'] = 'doom2.wad'
args['gif_path'] = './gifs'

#max episodes in RAM
args['max_episodes'] = 32
args['episodes_per_wad'] = 33
args['epochs_per_policy'] = 3
args['clip_e'] = lambda f : f * 0.1
args['learning_rate'] = lambda f: f * 2e-4

args['use_human_data'] = False
args['mem_location'] = '/home/djdev/Documents/Tensorflow/Doom/h5/human_data_1min_d.h5'
args['hsize'] = 0

simulator = DoomSimulator(args)

batch_size = 21
update_c = 32

save_n = 50
test_n = 5000
start_step = 0
#these are the prior average performance metrics
init_explore = 5
init_reward = .7
init_win = 0
init_kills = 0
init_keys = 0


simulator.train(batch_size,update_n=update_c,start_step=start_step,save_n=save_n,test_n=test_n,
    			init_explore=init_explore,init_reward=init_reward,init_win=init_win,init_kills=init_kills,init_keys=init_keys)
