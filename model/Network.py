import tensorflow as tf
from tensorflow.contrib.slim import conv2d, fully_connected, flatten, dropout, conv2d_transpose
from tensorflow.python.keras.initializers import he_normal
from tensorflow.python.keras.layers import LeakyReLU


class DFP_Network():
    def __init__(self,args):
        
        self.a_size = args['num_actions']
        self.num_offsets = args['num_offsets']
        self.num_measurements = [args['num_observe_m'],args['num_predict_m']]
        self.num_classify_m = args['num_classify_m']
        self.framedim = args['framedims']
        self.num_action_splits = args['num_action_splits']
        
        self.start_lr = args['start_lr']
        self.half_lr_every_n_steps = args['half_lr_every_n_steps']
        
        self.dropoutrates = args['dropoutrates'] if 'dropoutrates' in args else None
        self.dropoutboundaries = args['dropoutboundaries'] if 'dropoutrates' in args else None
        if self.dropoutrates is None or self.dropoutboundaries is None:
            self.apply_dropout = False
            print('Bayseian Dropout Exploration not applied, please supply args dropoutrates and dropoutboundaries')
        else:
            self.apply_dropout = True
        
        self.use_goals = args['use_goals']
        self.use_latent_z = args['use_latent_z'] if 'use_latent_z' in args else False
        self.num_buttons = args['num_buttons']
        self.sequence_length = args['sequence_length']     
        self.time_steps = tf.placeholder_with_default(self.sequence_length,shape=())
        self.steps = tf.placeholder(shape=(),dtype=tf.int32)
        
        if self.use_latent_z:
            self.z_dim = args['z_dim'] if 'z_dim' in args else 32
            self.z_offsets = args['z_offsets'] if 'z_offsets' in args else[4,32] 
            self.z_num_offsets = len(self.z_offsets)
        
        self._build_net()
        
    def _build_net(self):
        self._build_inputs()
        self.learning_rate = tf.train.exponential_decay(self.start_lr,self.steps,self.half_lr_every_n_steps,0.5)

        with tf.variable_scope("control/output"):
            self._build_output()
            self._build_loss_train_ops()
        
        if self.use_latent_z:
            #self._build_latent_future_predictor()
            #self._build_latent_z_train_ops()
            self._build_VAE_loss_train_ops()
        
    def _build_inputs(self):
        with tf.variable_scope("control"):
            self._build_measurements()
            self._build_action_history()
            concatlist = [self.in_action3,self.dense_m3]
            
            if self.use_goals:
                self._build_goals()
                concatlist.append(self.dense_g3)
                
        if self.use_latent_z:
            #are we computing Z from frame or feeding it? True->compute it
            with tf.variable_scope("VAE"):
                self.compute_z = tf.placeholder_with_default(True,shape=())
                
                def feed_fn():
                    self.feed_z = tf.placeholder(shape=[None,self.z_dim],dtype=tf.float32)
                    return self.feed_z,tf.zeros([1]),tf.zeros([1])
                
                compute_fn = lambda : self._build_VAE_deconv() #nothing defined within is executed if self.compute_z is False
                
                self.latent_z,self.outframe,self.outlabel = tf.cond(self.compute_z,true_fn=compute_fn,false_fn=feed_fn)
                concatlist.append(self.latent_z)
        else:
            with tf.variable_scope("control"):
                self._build_conv()
                self.latent_z = fully_connected(self.convout,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
                
                concatlist.append(self.latent_z)
        
        with tf.variable_scope("control"):
            self.merged_input1 = tf.concat(concatlist,1,name="InputMerge")
            self.merged_input2 = fully_connected(self.merged_input1,256,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            
            self.cell =  tf.nn.rnn_cell.BasicLSTMCell(256)
            rnn_in = tf.reshape(self.merged_input2,[-1,self.time_steps,256])
            self.c_in = tf.placeholder(shape=[None, self.cell.state_size.c],dtype=tf.float32)
            self.h_in = tf.placeholder(shape=[None, self.cell.state_size.h], dtype=tf.float32)
            state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
             
            self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.cell, rnn_in, initial_state=state_in)
             
            self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, 256])   
            
            self.exploring = tf.placeholder_with_default(False,shape=())
            if self.apply_dropout:
                dropoutrate = tf.train.piecewise_constant(self.steps,self.dropoutboundaries,self.dropoutrates)
            else:
                dropoutrate = 0
                
            self.merged_dropout = dropout(self.lstm_outputs,keep_prob=1-dropoutrate,is_training=self.exploring)
            
    
    def _build_conv(self):
        
        self.conv_training = tf.placeholder_with_default(True,shape=())
        
        self.observation = tf.placeholder(shape=[None,self.framedim[0],self.framedim[1],3],dtype=tf.float32)
        self.conv1 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.observation,num_outputs=32,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        #self.conv1 = tf.layers.batch_normalization(self.conv1,training=self.conv_training)
        self.conv2 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv1,num_outputs=64,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        #self.conv2 = tf.layers.batch_normalization(self.conv2,training=self.conv_training)
        self.conv3 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv2,num_outputs=128,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        #self.conv3 = tf.layers.batch_normalization(self.conv3,training=self.conv_training)
        self.conv4 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv3,num_outputs=256,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        #self.conv4 = tf.layers.batch_normalization(self.conv4,training=self.conv_training)
        self.conv5 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv4,num_outputs=256,
            kernel_size=[4,4],stride=[2,2],padding='VALID')
        #self.conv5 = tf.layers.batch_normalization(self.conv5,training=self.conv_training)
        
        self.convout = flatten(self.conv5)
        
    def _build_VAE_deconv(self):
        
        self._build_conv()
        
        self.dense_mu =  fully_connected(self.convout,self.z_dim,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        
        self.dense_sigma =  fully_connected(self.convout,self.z_dim,
            activation_fn=tf.nn.relu,weights_initializer=he_normal())
        
        #mu=0, sigma=1
        dist = tf.contrib.distributions.Normal(0.0,1.0)
        
        self.z_computed = tf.add(self.dense_mu, tf.multiply(self.dense_sigma,dist.sample(sample_shape=tf.shape(self.dense_sigma))))
        
        dense_in = fully_connected(self.z_computed,1024,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        
        deconv_in = tf.reshape(dense_in,[-1,1,1,1024])
        
        self.deconv_1 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=deconv_in,num_outputs=256,
            kernel_size=[3,3],stride=[1,1],padding='VALID')
        
        #self.deconv_1 = tf.layers.batch_normalization(self.deconv_1,training=self.conv_training)
        
        self.deconv_2 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_1,num_outputs=256,
            kernel_size=[4,4],stride=[2,2],padding='VALID')
        
        #self.deconv_2 = tf.layers.batch_normalization(self.deconv_2,training=self.conv_training)
        
        self.deconv_3 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_2,num_outputs=128,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        
        #self.deconv_3 = tf.layers.batch_normalization(self.deconv_3,training=self.conv_training)
        
        self.deconv_4 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_3,num_outputs=64,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        
        #self.deconv_4 = tf.layers.batch_normalization(self.deconv_4,training=self.conv_training)
        
        self.deconv_5 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_4,num_outputs=32,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        
        #self.deconv_5 = tf.layers.batch_normalization(self.deconv_5,training=self.conv_training)
        
        self.deconv_6 = conv2d_transpose(activation_fn=None,
            weights_initializer=he_normal(),
            inputs=self.deconv_5,num_outputs=3,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        
        self.deconv_6l = conv2d_transpose(activation_fn=None,
            weights_initializer=he_normal(),
            inputs=self.deconv_5,num_outputs=1,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        
        self.label = tf.placeholder(shape=[None,self.framedim[0],self.framedim[1],1],dtype=tf.float32)
        
        return self.z_computed,self.deconv_6,self.deconv_6l
                
    def _build_measurements(self):
        with tf.variable_scope("measurements"):        
            self.measurements = tf.placeholder(shape=[None,self.num_measurements[0]],dtype=tf.float32)
            self.dense_m1 = fully_connected(flatten(self.measurements),128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.dense_m2 = fully_connected(self.dense_m1,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.dense_m3 = fully_connected(self.dense_m2,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
    
    def _build_goals(self):
        with tf.variable_scope("goals"):  
            self.goals = tf.placeholder(shape=[None,self.num_measurements[1]],dtype=tf.float32)
            self.dense_g1 = fully_connected(flatten(self.goals),128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.dense_g2 = fully_connected(self.dense_g1,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.dense_g3 = fully_connected(self.dense_g2,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())  


    def _build_action_history(self):
        with tf.variable_scope("action"):  
            self.action_history = tf.placeholder(shape=[None,self.num_buttons],dtype=tf.float32)
            self.in_action1 = fully_connected(flatten(self.action_history),128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.in_action2 = fully_connected(self.in_action1,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.in_action3 = fully_connected(self.in_action2,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())  
        
    def _build_latent_future_predictor(self):
        
        outputdim = self.z_dim*self.z_num_offsets
        outputdim_as = [outputdim*self.a_size[i] for i in range(self.num_action_splits)]
        
        self.z_expectation1 =  fully_connected(self.merged_dropout,256,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.z_expectation2 =  fully_connected(self.z_expectation1,outputdim,
            activation_fn=None,weights_initializer=he_normal())    
        self.z_expectation2 = tf.reshape(self.z_expectation2,[-1,self.z_num_offsets,self.z_dim])  
        
        layer1 = lambda : fully_connected(self.merged_dropout,256,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        
        
        def advantage_layer2(i):
            layer2 = fully_connected(self.z_advantages1[i],outputdim_as[i],
            activation_fn=None,weights_initializer=he_normal())
            layer2 = tf.reshape(layer2,[-1,self.a_size[i],self.z_num_offsets,self.z_dim])
            layer2 = layer2 - tf.reduce_mean(layer2,axis=1,keep_dims=True)
            return layer2
    
        self.z_advantages1 = [layer1() for _ in range(self.num_action_splits)]
        self.z_advantages2 = [advantage_layer2(i) for i in range(self.num_action_splits)]
        
        self.z_preds_all = tf.concat([tf.add(advantage,self.z_expectation2) for advantage in self.z_advantages2],axis=1)
        self.z_preds_all = flatten(self.z_preds_all)
        
        self.z_a_chosen = [tf.placeholder(shape=[None,self.a_size[i],self.z_num_offsets,self.z_dim],dtype=tf.float32) for i in range(self.num_action_splits)]
        self.z_a_pred = [tf.reduce_sum(tf.multiply(self.z_advantages2[i],self.z_a_chosen[i]),axis=1) for i in range(self.num_action_splits)]

        self.z_a_pred.append(self.z_expectation2)
        self.z_prediction = tf.add_n(self.z_a_pred)
    
    
    def _build_output(self):
        #We calculate separate expectation and advantage streams, then combine then later
        #This technique is described in https://arxiv.org/pdf/1511.06581.pdf
        
        outputdim = self.num_measurements[1]*self.num_offsets
        outputdim_as = [outputdim*self.a_size[i] for i in range(self.num_action_splits)]

        
        #average expectation accross all actions
        self.expectation1 =  fully_connected(self.merged_dropout,256,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.expectation2 =  fully_connected(self.expectation1,outputdim,
            activation_fn=None,weights_initializer=he_normal())    
        self.expectation2 = tf.reshape(self.expectation2,[-1,self.num_offsets,self.num_measurements[1]])    


        #split actions into functionally seperable groups
        #e.g. the expectations of movements depend intimately on
        #combinations of movements (e.g. forward left vs forward right)
        #but the expectations of movements can be seperated from the outcome
        #of switching weapons for example. This separation reduces the
        # number of outputs of the model by an order of magnitude or more
        #when the number of subactions is large while maintaining the ability
        #to choose from a large number of actions.
        
        layer1 = lambda : fully_connected(self.merged_dropout,256,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        
        
        def advantage_layer2(i):
            layer2 = fully_connected(self.advantages1[i],outputdim_as[i],
            activation_fn=None,weights_initializer=he_normal())
            layer2 = tf.reshape(layer2,[-1,self.a_size[i],self.num_offsets,self.num_measurements[1]])
            layer2 = layer2 - tf.reduce_mean(layer2,axis=1,keep_dims=True)
            
            return layer2
        
        self.advantages1 = [layer1() for _ in range(self.num_action_splits)]
        self.advantages2 = [advantage_layer2(i) for i in range(self.num_action_splits)]
        
        self.a_chosen = [tf.placeholder(shape=[None,self.a_size[i],self.num_offsets,self.num_measurements[1]],dtype=tf.float32) for i in range(self.num_action_splits)]
        self.a_pred = [tf.reduce_sum(tf.multiply(self.advantages2[i],self.a_chosen[i]),axis=1) for i in range(self.num_action_splits)]

        self.a_pred.append(self.expectation2)
        self.prediction = tf.add_n(self.a_pred)
                   
        
    def _build_loss_train_ops(self):
        #This is the actual
        self.target = tf.placeholder(shape=[None,self.num_offsets,self.num_measurements[1]],dtype=tf.float32)
        
        self.target_mse = self.target[:,:,:-self.num_classify_m]
        self.target_classify = self.target[:,:,-self.num_classify_m:]
        
        self.prediction_mse = self.prediction[:,:,:-self.num_classify_m]
        self.prediction_classify = self.prediction[:,:,-self.num_classify_m:]
        #Loss function        
        self.loss_mse  = tf.losses.mean_squared_error(self.target_mse,self.prediction_mse)
        self.loss_classify = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.target_classify,logits=self.prediction_classify,pos_weight=50))
        
        self.loss = self.loss_mse + self.loss_classify
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     beta1=0.95,
                                     beta2=0.999,
                                     epsilon = 1e-4)
        
        
        #Get & apply gradients from network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='control')
        self.gradients = tf.gradients(self.loss,global_vars)
        grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,1)
        self.apply_grads = self.trainer.apply_gradients(list(zip(grads,global_vars)))
        
        loss_all = 0.5*self.loss+ 0.5*self.VAE_loss
        
        global_vars_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients_all = tf.gradients(loss_all,global_vars_all)
        grads_all,self.grad_norms_all = tf.clip_by_global_norm(self.gradients_all,1)
        self.apply_grads_all = self.trainer.apply_gradients(list(zip(grads_all,global_vars_all)))
        
    def _build_VAE_loss_train_ops(self):
        
        self.kl_loss =  tf.reduce_mean(1 + self.dense_sigma - tf.square(self.dense_mu) - tf.exp(self.dense_sigma))
        self.reconstruction_loss =  tf.losses.mean_squared_error(self.deconv_6,self.observation)
        #self.label_loss = tf.losses.mean_squared_error(self.deconv_6l,self.label)
        self.label_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label,logits=self.deconv_6l,pos_weight=20))
        
        self.VAE_loss = -0.5 *self.kl_loss + 7*self.reconstruction_loss + 3*self.label_loss
        
        VAE_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     beta1=0.95,
                                     beta2=0.999,
                                     epsilon = 1e-4)
        
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        

        #Get & apply gradients from network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.VAE_loss,global_vars)
        grads,grad_norms = tf.clip_by_global_norm(gradients,1)
        #with tf.control_dependencies(update_ops):
        self.VAE_apply_grads = VAE_trainer.apply_gradients(list(zip(grads,global_vars)))
        
    def _build_latent_z_train_ops(self):
        
        self.z_target = tf.placeholder(shape=[None,self.z_num_offsets,self.z_dim],dtype=tf.float32)
        self.latent_z_loss = tf.losses.mean_squared_error(self.z_target,self.z_prediction)
        
        z_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     beta1=0.95,
                                     beta2=0.999,
                                     epsilon = 1e-4)
        
        #Get & apply gradients from network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.latent_z_loss,global_vars)
        grads,grad_norms = tf.clip_by_global_norm(gradients,1)
        self.z_apply_grads = z_trainer.apply_gradients(list(zip(grads,global_vars)))