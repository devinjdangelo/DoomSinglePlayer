from model.Simulator import DoomSimulator
from DefaultArgs import args

args['mode'] = 'pretrain'
args['model_path'] = '/home/djdev/Documents/Tensorflow/Doom/PreTraining/'
args['load_model'] = False
args['mem_location'] = '/home/djdev/Documents/Tensorflow/Doom/h5/human_data.h5'
args['test_stat_file'] = 'agent_test_data.csv'
args['reset_file'] = True

args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_engine'] = 'doom2.wad'

args['load_h5_into_mem'] = True
args['start_lr'] = 1e-2
args['half_lr_every_n_steps'] = 1e6


simulator = DoomSimulator(args)


batch_size = 4
simulator.pre_train(batch_size,test_every_n_iters=100)

#import numpy as np
#
#n_episodes = simulator.experience_memory.n_episodes
#def ep_to_means_sds(ep):
#    ep = ep.reshape(-1,6,11)
#    mean_std = [(np.mean(ep[:,:,i]),np.std(ep[:,:,i])) for i in range(11)]
#    return mean_std
#
#def ep_to_means_sds_meas(ep):
#    ep = ep.reshape(-1,29)
#    mean_std = [(np.mean(ep[:,i]),np.std(ep[:,i])) for i in range(29)]
#    return mean_std
#
#metrics_levels = [ep_to_means_sds_meas(ep) for ep in simulator.experience_memory.measurements]
#metrics_deltas = [ep_to_means_sds(ep) for ep in simulator.experience_memory.targets]