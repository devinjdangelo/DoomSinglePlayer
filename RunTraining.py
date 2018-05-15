from model.Simulator import DoomSimulator
from DefaultArgs import args

args['mode'] = 'train'
args['model_path'] = '/home/djdev/Documents/Tensorflow/Doom/Training/'
args['load_model'] = True
args['load_vae'] = True
#args['mem_location'] = '/home/ddangelo/Documents/tensorflow models/doom/h5/human_data.h5'
args['test_stat_file'] = 'agent_test_data.csv'
args['reset_file'] = True

args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_engine'] = 'doom2.wad'

#max episodes in RAM
args['max_episodes'] = 10

args['start_lr'] = 1e-3
args['half_lr_every_n_steps'] = 1e6

args['use_human_data'] = True
args['mem_location'] = '/media/ddangelo/sandisk/human_train_data.h5'
args['probability_draw_human_data'] = lambda step : 0.8 * 0.5**(step/5e5)


simulator = DoomSimulator(args)

batch_size = 2
update_every_n_episodes = 2
test_every_n = 20
simulator.train(batch_size,update_every_n_episodes,test_every_n_episodes=test_every_n)