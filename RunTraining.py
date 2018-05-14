from model.Simulator import DoomSimulator
from DefaultArgs import args

args['mode'] = 'train'
args['model_path'] = '/home/ddangelo/Documents/tensorflow models/doom/FinalTraining/'
args['load_model'] = False
#args['mem_location'] = '/home/ddangelo/Documents/tensorflow models/doom/h5/human_data.h5'
args['test_stat_file'] = 'agent_test_data.csv'
args['reset_file'] = True

args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_engine'] = 'doom2.wad'

#max episodes in RAM
args['max_episodes'] = 40

args['start_lr'] = 1e-2
args['half_lr_every_n_steps'] = 1e6

simulator = DoomSimulator(args)

batch_size = 8
update_every_n_episodes = 8
test_every_n = 50
simulator.train(batch_size,update_every_n_episodes,test_every_n_episodes=test_every_n)