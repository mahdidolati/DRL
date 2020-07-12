from __future__ import division

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

class QnConfigGenerator:
    def __init__(self):
        self.configs = {}

    def build_config(self, per_row_server, vnf_types_len):
        if (per_row_server, vnf_types_len) in self.configs:
            return
        if per_row_server == 10:
            layer_outs = [32, 64, 64, 512]
            kernel_sizes = [[2, 2], [3, 2], [3, 3], [5, 5]]
            strides = [[1, 1], [1, 1], [1, 1], [1, 1]]
        elif per_row_server == 5:
            layer_outs = [32, 64, 64, 512]
            kernel_sizes = [[2, 2], [3, 2], [3, 3], [5, 5]]
            strides = [[1, 1], [1, 1], [1, 1], [1, 1]]
        else:
            'Server no. is not supported!'
        dim_lens = [2 * per_row_server, 2 * per_row_server - 1, vnf_types_len + 2]
        output_size = 4  # left - up - embed - reject
        qc = QnConfig(layer_outs, kernel_sizes, strides, dim_lens, output_size)
        self.configs[(per_row_server, vnf_types_len)] = qc

    def get_config(self, per_row_server, vnf_types_len):
        self.build_config(per_row_server, vnf_types_len)
        return self.configs[(per_row_server, vnf_types_len)]

class QnConfig:
    '''
    layer_outs: number of output neurons in each layer. E.g., for a four layer network: [32, 64, 64, 512].
    kernel_sizes: kernel size for each layer. E.g., for a four layer network: [[2,2], [6,2], [7,2], [8,8]]
    strides: stride of the kernel in each layer. E.g., four a four layer network: [[1,1], [1,1], [1,1], [1,1]]
    dim_lens: dimension of each input. E.g., for a four layer network: [-1, 20, 11, 3]
    output_size: size of the decision layer. E.g., 5
    '''
    def __init__(self, layer_outs, kernel_sizes, strides, dim_lens, output_size):
        n = len(layer_outs)
        self.dims = {}
        for i in range(n):
            self.dims[i] = {}
            self.dims[i]['layer_outs'] = layer_outs[i]
            self.dims[i]['kernel_size'] = kernel_sizes[i]
            self.dims[i]['stride'] = strides[i]
        self.dim_lens = dim_lens
        self.state_len = np.prod(dim_lens)
        self.output_size = output_size


class Qnetwork():
    def __init__(self, qn_config):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, qn_config.state_len], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1] + qn_config.dim_lens)
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=qn_config.dims[0]['layer_outs'],
                                    kernel_size=qn_config.dims[0]['kernel_size'],
                                    stride=qn_config.dims[0]['stride'],
                                    padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1, num_outputs=qn_config.dims[1]['layer_outs'],
                                    kernel_size=qn_config.dims[1]['kernel_size'],
                                    stride=qn_config.dims[1]['stride'],
                                    padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2, num_outputs=qn_config.dims[2]['layer_outs'],
                                    kernel_size=qn_config.dims[2]['kernel_size'],
                                    stride=qn_config.dims[2]['stride'],
                                    padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3, num_outputs=qn_config.dims[3]['layer_outs'],
                                    kernel_size=qn_config.dims[3]['kernel_size'],
                                    stride=qn_config.dims[3]['stride'],
                                    padding='VALID', biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([qn_config.dims[3]['layer_outs']//2, qn_config.output_size]))
        self.VW = tf.Variable(xavier_init([qn_config.dims[3]['layer_outs']//2, 1]))
        self.Value = tf.matmul(self.streamV, self.VW)
        self.Advantage = tf.matmul(self.streamA, self.AW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, qn_config.output_size, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


class Drl:
    def __init__(self, qn_config):
        self.qn_config = qn_config
        self.total_steps = 0
        startE = 0.9  # Starting chance of random action
        annealing_steps = 60000  # How many steps of training to reduce startE to endE.
        # Set the rate of random action decrease.
        self.endE = 0.1  # Final chance of random action
        self.e = startE
        self.stepDrop = (startE - self.endE) / annealing_steps
        self.pre_train_steps = 40000  # How many steps of random actions before training begins.
        #
        tf.reset_default_graph()
        self.mainQN = Qnetwork(self.qn_config)
        self.targetQN = Qnetwork(self.qn_config)
        self.bestQN = Qnetwork(self.qn_config)
        init = tf.global_variables_initializer()
        #
        tau = 0.001  # Rate to update target network toward primary network
        trainables = tf.trainable_variables()
        self.targetOps = self.updateTargetGraph(trainables, tau)
        self.bestOps = self.updateBestGraph(trainables)
        self.myBuffer = experience_buffer()
        self.episodeBuffer = experience_buffer()
        #
        self.sess = tf.Session()
        self.sess.run(init)
        #
        self.pre_train_stage = True
        self.exploration_stage = True
        self.episode_reward = 0.0
        self.max_episode_reward = 0.0
        self.min_episode_reward = np.inf
        print 'Network Constructed'

    def decide(self, s):
        s = self.processState(s)
        # if self.exploration_stage:
        if np.random.rand(1) < self.e or self.total_steps < self.pre_train_steps:
            a = np.random.randint(0, self.qn_config.output_size, 1)[0]
        else:
            a = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s]})[0]
        # else:
        #     a = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s]})[0]
        return a

    def add_experiece(self, s, a, r, s1, d):
        # if not self.exploration_stage:
        #     return
        self.episode_reward += r
        s = self.processState(s)
        s1 = self.processState(s1)
        self.total_steps += 1
        self.episodeBuffer.add(
            np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.
        self.train()
        if d == 1:
            self.episode_end()

    def train(self):
        update_freq = 4  # How often to perform a training step.
        batch_size = 32  # How many experiences to use for each training step.
        y = .99  # Discount factor on the target Q-values
        if self.total_steps > self.pre_train_steps:
            if self.pre_train_stage:
                self.pre_train_stage = False
                print 'Pre-train Stage Completed'

            # print 'do train', self.total_steps
            if self.e > self.endE:
                self.e -= self.stepDrop

            if self.e <= self.endE and self.exploration_stage:
                self.exploration_stage = False
                print 'Exploration Ended'

            if self.total_steps % (update_freq) == 0:
                trainBatch = self.myBuffer.sample(batch_size)  # Get a random batch of experiences.
                # Below we perform the Double-DQN update to the target Q-values
                Q1 = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                Q2 = self.sess.run(self.targetQN.Qout, feed_dict={self.targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(batch_size), Q1]
                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                # Update the network with our target values.
                _ = self.sess.run(self.mainQN.updateModel, \
                             feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), self.mainQN.targetQ: targetQ,
                                        self.mainQN.actions: trainBatch[:, 1]})
                # Update the target network toward the primary network.
                self.updateTarget(self.targetOps, self.sess)

    def episode_end(self):
        self.myBuffer.add(self.episodeBuffer.buffer)
        self.episodeBuffer = experience_buffer()
        if self.episode_reward < self.min_episode_reward:
            self.min_episode_reward = self.episode_reward
        if self.episode_reward > self.max_episode_reward:
            print 'Best Network Updated (e=%f, m=%f): old-best: %f -> new-best: %f' %(self.e, self.min_episode_reward,
                                                                                      self.max_episode_reward,
                                                                                      self.episode_reward)
            self.max_episode_reward = self.episode_reward
            self.episode_reward = 0.0
            self.updateBest(self.bestOps, self.sess)

    def processState(self, states):
        return np.reshape(states,[self.qn_config.state_len])

    # def updateTargetGraph(self, tfVars,tau):
    #     total_vars = len(tfVars)
    #     op_holder = []
    #     for idx,var in enumerate(tfVars[0:total_vars//2]):
    #         op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    #     return op_holder

    def updateTargetGraph(self, tfVars,tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//3]):
            op_holder.append(tfVars[idx+total_vars//3].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//3].value())))
        return op_holder

    def updateBestGraph(self, tfVars):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 3]):
            op_holder.append(tfVars[idx + 2 * total_vars // 3].assign(var.value()))
        return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    def updateBest(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)


def get_dummy_state():
    s = np.zeros((4, 4, 2), dtype=float)
    exp_a = np.random.randint(0, 2)
    if exp_a == 0:
        s[3, 3, 0] = 1.0
        s[3, 3, 1] = 1.0
    else:
        s[0, 0, 0] = 1.0
        s[0, 0, 1] = 1.0
    return s, exp_a

if __name__ == '__main__':
    layer_outs = [32, 64, 64, 512]
    kernel_sizes = [[2,2], [2,2], [2,2], [1,1]]
    strides = [[1,1], [1,1], [1,1], [1,1]]
    dim_lens = [4, 4, 2]
    output_size = 5
    qc = QnConfig(layer_outs, kernel_sizes, strides, dim_lens, output_size)
    drl = Drl(qc)
    s, exp_a = get_dummy_state()
    ep_rew = 0
    ep_len = 50
    for itr in range(30000):
        a = drl.decide(s)
        if ep_len == 0:
            r = 1
            d = 1
            print itr, ep_rew, ep_len, a
            ep_rew = 0
            ep_len = 50
        elif a == exp_a:
            r = 1
            d = 0
            ep_rew += 1
        elif a != exp_a:
            r = -1
            d = 1
            # print ep_rew
            ep_rew = 0
        s1, exp_a = get_dummy_state()
        drl.add_experiece(s, a, r, s1, d)
        s = s1
        ep_len -= 1