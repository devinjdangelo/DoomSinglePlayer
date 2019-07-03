import tensorflow as tf
from tensorflow.contrib.slim import conv2d, fully_connected, flatten, dropout, conv2d_transpose
from tensorflow.python.keras.initializers import he_normal
from tensorflow.python.keras.layers import LeakyReLU

import math


class PPO():
    def __init__(self,args):

        self.num_buttons = args['num_buttons']
        self.a_size = [len(actions) for actions in args['group_actions']]
        self.num_action_splits = args['num_action_splits']
        self.num_measurements = [args['num_observe_m']]
        self.framedim = args['framedims']
        self.sequence_length = args['episode_length']
        self.time_steps = tf.placeholder_with_default(self.sequence_length,shape=())
        self.steps = tf.placeholder(shape=(),dtype=tf.int32)

        self.critic_weight = args['critic_weight']
        self.entropy_weight = args['entropy_weight']

        self._build_net()

    def _build_net(self):
        self._build_inputs()

        with tf.variable_scope("critic"):
            self._build_critic()
        with tf.variable_scope("actor"):
            self._build_actor()

        self._build_loss_train_ops()

    def _build_inputs(self):
        self._build_measurements()
        self._build_action_history()
        concatlist = [self.in_action3,self.dense_m3]

        self._build_conv()
        self.latent_z = fully_connected(self.convout,512,
        activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())

        concatlist.append(self.latent_z)

        self.merged_input1 = tf.concat(concatlist,1,name="InputMerge")
        #self.merged_input2 = fully_connected(self.merged_input1,256,
        #    activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())

        #self.cell =     tf.nn.rnn_cell.BasicLSTMCell(512)
        self.cell =     tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(512)
        rnn_in = tf.reshape(self.merged_input1,[-1,self.time_steps,651])
        self.c_in = tf.placeholder(shape=[None, self.cell.state_size.c],dtype=tf.float32)
        self.h_in = tf.placeholder(shape=[None, self.cell.state_size.h], dtype=tf.float32)
        state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)

        self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.cell, rnn_in, initial_state=state_in,dtype=tf.float32)

        self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, 512])


    def _build_conv(self):

        #self.conv_training = tf.placeholder_with_default(True,shape=())

        self.observation = tf.placeholder(shape=[None,self.framedim[0],self.framedim[1],3],dtype=tf.float32)
        self.conv1 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.observation,num_outputs=16,
            kernel_size=[3,3],stride=[3,3],padding='VALID')
        #self.conv1 = tf.layers.batch_normalization(self.conv1,training=self.conv_training)
        self.conv2 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv1,num_outputs=32,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        #self.conv2 = tf.layers.batch_normalization(self.conv2,training=self.conv_training)
        self.conv3 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv2,num_outputs=64,
            kernel_size=[2,2],stride=[2,2],padding='VALID')
        #self.conv3 = tf.layers.batch_normalization(self.conv3,training=self.conv_training)
        self.conv4 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv3,num_outputs=128,
            kernel_size=[2,2],stride=[1,1],padding='VALID')
        #self.conv4 = tf.layers.batch_normalization(self.conv4,training=self.conv_training)

        #self.conv5 = tf.layers.batch_normalization(self.conv5,training=self.conv_training)

        self.convout = flatten(self.conv4)

    def _build_measurements(self):
        with tf.variable_scope("measurements"):
            self.measurements = tf.placeholder(shape=[None,self.num_measurements[0]],dtype=tf.float32)
            self.dense_m1 = fully_connected(flatten(self.measurements),128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.dense_m2 = fully_connected(self.dense_m1,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.dense_m3 = fully_connected(self.dense_m2,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())

    def _build_action_history(self):
        with tf.variable_scope("action"):
            self.action_history = tf.placeholder(shape=[None,self.num_buttons],dtype=tf.float32)
            #self.in_action1 = fully_connected(flatten(self.action_history),128,
            #    activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            #self.in_action2 = fully_connected(self.in_action1,128,
            #    activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
            self.in_action3 = self.action_history

    def _build_critic(self):
        self.critic_dense = fully_connected(self.lstm_outputs,128,
                activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())

        self.critic_state_value = fully_connected(self.lstm_outputs,1,
                activation_fn=None,weights_initializer=he_normal())

    def _build_actor(self):


        p_layer = lambda size : fully_connected(self.lstm_outputs,size,
                activation_fn=tf.nn.softmax,weights_initializer=he_normal())
        probs = [p_layer(size) for size in self.a_size]

        dist_layer = lambda p : tf.distributions.Categorical(probs=p,dtype=tf.int32)
        self.dists = [dist_layer(p) for p in probs]

        sample_dist = lambda dist : tf.expand_dims(dist.sample(),-1)
        sampled_actions = [sample_dist(dist) for dist in self.dists]

        best_fn = lambda p : tf.expand_dims(tf.argmax(p,axis=1),-1)
        best_actions = [best_fn(p) for p in probs]

        self.sampled_action = tf.concat(sampled_actions,axis=-1)
        self.best_action = tf.concat(best_actions,axis=-1)

        #going to need this frozen for use as pi_thetaold(action)
        self.sampled_action_prob = self.action_prob(self.sampled_action)
        self.best_action_prob = self.action_prob(self.best_action)

        self.feed_action = tf.placeholder(shape=[None,self.num_action_splits],dtype=tf.float32)
        self.feed_action_prob = self.action_prob(self.feed_action)

    def action_prob(self,action):
        lg_probs = [self.dists[i].log_prob(action[:,i]) for i in range(self.num_action_splits)]
        lgprob = tf.add_n(lg_probs)
        return lgprob


    def _build_loss_train_ops(self):
        self.lgprob_a_pi_old = tf.placeholder(shape=[None],dtype=tf.float32)
        self.a_taken = tf.placeholder(shape=[None,self.num_action_splits],dtype=tf.float32)
        #frozen estimate of generalized advantage from old net
        self.GAE = tf.placeholder(shape=[None],dtype=tf.float32)
        #frozen estimate of state value using GAE approach, i.e. GAE+V = Q, E[Q] = V
        self.old_v_pred = tf.placeholder(shape=[None],dtype=tf.float32)
        self.returns = tf.placeholder(shape=[None],dtype=tf.float32)
        self.clip_e = tf.placeholder(shape=(),dtype=tf.float32)

        newlgprob = self.action_prob(self.a_taken)
        #newlgprob = tf.Print(newlgprob,[tf.shape(newlgprob)])
        # e^(lg(new)-lg(old)) = new/old
        #newlgprob = tf.Print(newlgprob,[newlgprob,self.lgprob_a_pi_old],summarize=20)
        rt = tf.exp(newlgprob - self.lgprob_a_pi_old)
        pg_losses1 = -self.GAE * rt
        pg_losses2 = -self.GAE * tf.clip_by_value(rt,1-self.clip_e,1+self.clip_e)
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses1,pg_losses2))

        vpred = tf.squeeze(self.critic_state_value)
        #vpred = tf.Print(vpred,[vpred,self.old_v_pred],summarize=20)
        vpredclipped = self.old_v_pred + tf.clip_by_value(tf.squeeze(self.critic_state_value)-self.old_v_pred,-self.clip_e,self.clip_e)
        vf_losses1 = tf.square(vpred - self.returns)
        vf_losses2 = tf.square(vpredclipped - self.returns)
        self.vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1,vf_losses2))

        #again, by assumption of independence the entropy of overall dist is sum of entropies of component dists
        self.entropy = tf.reduce_mean(tf.add_n([d.entropy() for d in self.dists]))

        c_1 = self.critic_weight
        c_2 = self.entropy_weight
        self.loss = self.pg_loss - c_2*self.entropy + c_1*self.vf_loss

        self.learning_rate = tf.placeholder(shape=(),dtype=tf.float32)


        #Get & apply gradients from network
        self.trainer_c = tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon = 1e-5)
        tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_grads = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        gvs = self.trainer_c.compute_gradients(self.loss, tvs)
        gvs = [(gv[0],gv[1]) if gv[0] is not None else (tf.zeros_like(gv[1]),gv[1]) for gv in gvs]
        self.accum_grads = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

        self.accum_count = tf.placeholder(shape=(),dtype=tf.float32)
        grads = [g/self.accum_count for g in accum_vars]
        grads,self.grad_norms = tf.clip_by_global_norm(grads,0.5)
        self.apply_grads = self.trainer_c.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    def intersect_vars(self,checkpoint_dir):
        globalvars = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        checkpointvars = tf.train.list_variables(checkpoint_dir)
        checkpointvars = [var[0] for var in checkpointvars]
        intersect = [var for var in globalvars if var.name[:-2] in checkpointvars]
        return intersect



