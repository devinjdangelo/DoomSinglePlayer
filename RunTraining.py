from model.Simulator import DoomSimulator
from DefaultArgs import args
from traceback import print_exc


args['mode'] = 'train'
args['model_path'] = '/home/djdev/Documents/Tensorflow/Doom/Training/'
args['load_model'] = True
args['test_stat_file'] = 'agent_test_data3.csv'
args['reset_file'] = True

#args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_engine'] = 'doom2.wad'

#max episodes in RAM
args['max_episodes'] = 30
args['epochs_per_policy'] = 3


args['clip_e'] = lambda f : f * 0.1
args['learning_rate'] = lambda f: f * 1.0e-4

args['use_human_data'] = False
args['mem_location'] = '/home/djdev/Documents/Tensorflow/Doom/h5/human_data_1min_d.h5'
args['hsize'] = 0

simulator = DoomSimulator(args)

batch_size = 15
update_c = 30

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