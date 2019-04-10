from model.Simulator import DoomSimulator
from DefaultArgs import args

args['mode'] = 'record'
args['mem_location'] = '/home/ddangelo/Documents/data.h5'
args['test_stat_file'] = 'human_test_data.csv'
args['reset_file'] = True

#Here you should place doom2.wad and your vizdoom config file
#No spaces allowed in path due to limitation with pyoblige!
args['episodes_per_wad'] = 11
args['doom_files_path'] = '/home/ddangelo/Documents/doomfiles/'
args['doom_engine'] = 'doom2.wad'


simulator = DoomSimulator(args)

simulator.record_episodes() 

