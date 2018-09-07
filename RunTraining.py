from model.Simulator import DoomSimulator
from DefaultArgs import args
from traceback import print_exc
import horovod.tensorflow as hvd
hvd.init(comm=[0,8])
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
args['max_episodes'] = 16

if rank==0:
    args['test_stat_file'] = 'agent_test_data_node1.csv'
    args['epochs_per_policy'] = 2
elif rank==8:
    args['test_stat_file'] = 'agent_test_data_node2.csv'
    #in the interest of maximizing computational effeciency,
    #we will have to reuse some of the episodes on node 2
    #an extra time to match the iterations of node 1
    #this is because node 1 has 8 workers and maximum batch size of 15
    #node 2 has 4 workers and maximum batch size 10...
    args['epochs_per_policy'] = 2.7 
else:
    args['epochs_per_policy'] = None
    args['test_stat_file'] = None


args['clip_e'] = lambda f : f * 0.1
args['learning_rate'] = lambda f: f * 2.0e-5

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

update_c = 16
init_explore = 38.67
init_reward = 5.342
init_win = 0.0366
init_kills = 0.3791

save_n = 50
test_n = 5000
start_step = 0

simulator.train(batch_size,update_n=update_c,start_step=start_step,save_n=save_n,test_n=test_n,
            init_explore=init_explore,init_reward=init_reward,init_win=init_win,init_kills=init_kills)