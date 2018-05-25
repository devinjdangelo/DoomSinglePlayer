from model.Simulator import DoomSimulator
from DefaultArgs import args
from traceback import print_exc


args['mode'] = 'train'
args['model_path'] = '/home/djdev/Documents/Tensorflow/Doom/Training/'
args['load_model'] = True
args['load_vae'] = False
args['test_stat_file'] = 'agent_test_data.csv'
args['reset_file'] = False

args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_engine'] = 'doom2.wad'

#max episodes in RAM
args['max_episodes'] = 12

args['start_lr'] = 3e-3
args['half_lr_every_n_steps'] = 10e6

args['use_human_data'] = True
args['mem_location'] = '/home/djdev/Documents/Tensorflow/Doom/h5/human_data_2min.h5'
args['probability_draw_human_data'] = lambda step : .95 * 0.8**(step/1e6)

args['dropoutrates'] = None
args['dropoutboundaries']= None

simulator = DoomSimulator(args)

batch_size = 4
update_all = 1
update_c = 0
test_n = 25
start_step = 100*1025

try:
    simulator.train(batch_size,update_c_n=update_c,update_all_n=update_all,
    	test_every_n=test_n,start_step=start_step)
except:
    print_exc()
    try:
    	simulator.human_data.h5file.close()
    except:
    	pass