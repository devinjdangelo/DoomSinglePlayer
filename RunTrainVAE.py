from model.Trainer import Trainer
from DefaultArgs import args

args['model_path'] = '/home/djdev/Documents/Tensorflow/Doom/VAE/'
args['load_model'] = False
args['mem_location'] = '/home/djdev/Documents/Tensorflow/Doom/h5/human_data_2min.h5'

args['model_stats'] = 'model_stats.csv'

#Must change to False if h5file gets too large!
args['load_h5_into_mem'] = False

args['use_latent_z'] = True
args['start_lr'] = 1e-1
args['half_lr_every_n_steps'] = 1e6

trainer = Trainer(args)
batch_size = 256
epochs = 100
trainer.train_VAE(batch_size,epochs,save_every_n_iters=20)
