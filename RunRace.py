from model.Simulator import DoomSimulator
from DefaultArgs import args
from traceback import print_exc


args['mode'] = 'Race'
args['model_path'] = '/home/djdev/Documents/Tensorflow/Doom/Training/'
args['load_model'] = True

#args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_files_path'] = '/home/djdev/Documents/'
args['doom_engine'] = 'doom2.wad'





simulator = DoomSimulator(args)

simulator.race()
