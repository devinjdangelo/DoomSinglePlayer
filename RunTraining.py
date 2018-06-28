from model.Simulator import DoomSimulator
from DefaultArgs import args
from traceback import print_exc


args['mode'] = 'train'
args['model_path'] = '/home/ddangelo/Documents/Tensorflow/Doom/Training/'
args['load_model'] = False
args['test_stat_file'] = 'agent_test_data.csv'
args['reset_file'] = True

args['doom_files_path'] = '/home/ddangelo/Documents/'
args['doom_engine'] = 'doom2.wad'

#max episodes in RAM
args['max_episodes'] = 4
args['epochs_per_policy'] = 3


args['clip_e'] = lambda f : f * 0.1
args['learning_rate'] = lambda f: f * 2.5e-4

args['use_human_data'] = True
args['mem_location'] = '/media/ddangelo/sandisk/human_data_1min_d.h5'
args['hsize'] = 1

simulator = DoomSimulator(args)

batch_size = 2
update_c = 4

save_n = 100
test_n = 200
start_step = 0

try:
    simulator.train(batch_size,update_n=update_c,start_step=start_step,save_n=save_n,test_n=test_n)
except:
    print_exc()
    try:
        simulator.human_data.h5file.close()
    except:
        pass