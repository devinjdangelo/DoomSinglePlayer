
from model.Simulator import DoomSimulator
from DefaultArgs import args
from traceback import print_exc

#assert hvd.mpi_threads_supported()
from mpi4py import MPI

host,rank = MPI.Get_processor_name(),MPI.COMM_WORLD.Get_rank()
print('Host ',host,' starting rank ',rank)

args['mode'] = 'train'
args['model_path'] = './params/'
args['load_model'] = True
args['reset_file'] = True

#args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_files_path'] = './doomfiles/'
args['doom_engine'] = 'doom2.wad'

#max episodes in RAM
args['max_episodes'] = 48

if rank==0:
    args['test_stat_file'] = '/home/djdev/Dropbox/Deep Learning/doom-ppo-mpi-horovod/agent_test_data_node1.csv'
    args['gif_path'] = '/home/djdev/Dropbox/Deep Learning/doom-ppo-mpi-horovod/gifs'
    args['epochs_per_policy'] = 3
elif rank==8:
    args['test_stat_file'] = '/home/dmid/Dropbox/Deep Learning/doom-ppo-mpi-horovod/agent_test_data_node2.csv'
    args['gif_path'] = '/home/dmid/Dropbox/Deep Learning/doom-ppo-mpi-horovod/gifs'
    args['epochs_per_policy'] = 3
else:
    args['epochs_per_policy'] = None
    args['test_stat_file'] = None


args['clip_e'] = lambda f : f * 0.1
args['learning_rate'] = lambda f: f * 3.0e-4

args['use_human_data'] = False
args['mem_location'] = '/home/djdev/Documents/Tensorflow/Doom/h5/human_data_1min_d.h5'
args['hsize'] = 0

simulator = DoomSimulator(args)

if rank==0:
	batch_size = 15
elif rank==8:
	batch_size = 10
else:
	batch_size = 0

update_c = 48
init_explore = 11
init_reward = 1
init_win = 0.003
init_kills = 0.09

save_n = 200
test_n = 5000
start_step = 0

simulator.train(batch_size,update_n=update_c,start_step=start_step,save_n=save_n,test_n=test_n,
            init_explore=init_explore,init_reward=init_reward,init_win=init_win,init_kills=init_kills)