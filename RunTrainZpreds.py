from model.Trainer import Trainer
from DefaultArgs import args

args['model_path'] = '/home/djdev/Documents/Tensorflow/Doom/VAE/'
args['load_model'] = False
args['mem_location'] = '/home/djdev/Documents/Tensorflow/Doom/h5/human_data.h5'



trainer = Trainer(args)

batch_size = 8
epochs = 20
trainer.train_latent_future_predictor(batch_size,epochs)