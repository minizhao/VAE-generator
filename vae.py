# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from . import embedding


class vae_lstm():
    """
    A variational LSTM autoencoder.
    Reference: Generating Sentences From a Continuous Space (https://arxiv.org/abs/1511.06349)
    """
    def __init__(self, config, is_training=True):
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size  # Number of units in LSTM cell
        self.z_size = config.z_size  # Dimension of z
        self.grad_norm = config.grad_norm  # Gradient clipping
        self.learning_rate = config.learning_rate  # Learning rate
        self.annealing = config.annealing  # Boolean, whether to conduct annealing

        # Encoder input
        self.enc_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='enc_x')
        # Decoder input
        self.dec_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='dec_x')
        # Decoder inputs keep probability
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        # KL divergence annealing rate
        self.kl_rate = tf.placeholder(dtype=tf.float32, shape=[], name='kl_rate')
        # Input z for generating the sentences
        self.z_gen = tf.placeholder(dtype=tf.float32, shape=[None, self.z_size], name='z_gen')
        # embedding vcoabe
        self.vcoabe_list=np.array(embedding.get_emed_list(),dtype=np.int32)
        
        
        if self.annealing:
            self.kl_anneal = tf.Variable(0.0, name='kl_anneal', trainable=False, dtype=tf.float32)
            self.kl_anneal_update = tf.assign(self.kl_anneal, self.kl_rate)

        self.build_graph()
        self.compute_cost()

        if is_training:
            self.train()

    def build_graph(self):
        with tf.device('/cpu:0'), tf.name_scope('enc_embedding'):
            ##embeddin layer
            enc_embedding = tf.Variable(    name='enc_embedding',                                          
                                            initial_value=self.vcoabe_list,
                                            trainable=False,
                                            dtype=tf.float32)
            
            enc_inputs = tf.nn.embedding_lookup(enc_embedding, self.enc_x)

        # Encoder layer
        # =============================================================================

        # self.enc_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
        #                                         forget_bias=1.0,
        #                                         state_is_tuple=False,
        #                                         reuse=tf.get_variable_scope().reuse)
        self.enc_cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                   forget_bias=1.0,
                                                   state_is_tuple=True,
                                                   reuse=tf.get_variable_scope().reuse)
        self.enc_cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                   forget_bias=1.0,
                                                   state_is_tuple=True,
                                                   reuse=tf.get_variable_scope().reuse)

        # self.enc_initial_state = self.enc_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.enc_initial_state_fw = self.enc_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        self.enc_initial_state_bw = self.enc_cell_bw.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope('encoder'):
            # _, state = tf.nn.dynamic_rnn(self.enc_cell,
            #                              inputs=enc_inputs,
            #                              initial_state=self.enc_initial_state)
            _, state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw,
                                                       self.enc_cell_bw,
                                                       inputs=enc_inputs,
                                                       initial_state_fw=self.enc_initial_state_fw,
                                                       initial_state_bw=self.enc_initial_state_bw)

        # self.enc_final_state = state
        state_fw = state[0]
        state_bw = state[1]
        enc_final_state_c = tf.concat((state_fw.c, state_bw.c), 1)
        enc_final_state_h = tf.concat((state_fw.h, state_bw.h), 1)
        self.enc_final_state = tf.concat((enc_final_state_c, enc_final_state_h), 1)

        # Latent layer
        # =============================================================================

        # z_mean
        with tf.name_scope('mean_linear'):
            mean_w = tf.get_variable('mean_w', shape=[self.enc_final_state.shape[1], self.z_size], dtype=tf.float32)
            mean_b = tf.get_variable('mean_b', shape=[self.z_size], dtype=tf.float32)

            self.z_mean = tf.add(tf.matmul(self.enc_final_state, mean_w), mean_b)

        # z_log_var
        with tf.name_scope('log_linear'):
            log_w = tf.get_variable('log_w', shape=[self.enc_final_state.shape[1], self.z_size], dtype=tf.float32)
            log_b = tf.get_variable('log_b', shape=[self.z_size], dtype=tf.float32)

            self.z_log_var = tf.add(tf.matmul(self.enc_final_state, log_w), log_b)

        # Sampling
        epsilon = tf.random_normal(self.z_log_var.shape, dtype=tf.float32)
        self.z = self.z_mean + tf.exp(self.z_log_var / 2) * epsilon

        # Decoder layer
        # =============================================================================
        self.dec_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                forget_bias=1.0,
                                                state_is_tuple=False,
                                                reuse=tf.get_variable_scope().reuse)

        with tf.name_scope('dec_in'):
            dec_in_w = tf.get_variable('dec_in_w', shape=[self.z_size, self.dec_cell.state_size], dtype=tf.float32)
            dec_in_b = tf.get_variable('dec_in_b', shape=[self.dec_cell.state_size], dtype=tf.float32)

            self.dec_initial_state = tf.add(tf.matmul(self.z, dec_in_w), dec_in_b)

        with tf.device('/cpu:0'), tf.name_scope('dec_embedding'):
            dec_embedding = tf.Variable(    name='dec_embedding',                                          
                                            initial_value=self.vcoabe_list,
                                            trainable=False,
                                            dtype=tf.float32)
            dec_inputs = tf.nn.embedding_lookup(dec_embedding, self.dec_x)

        # Decoder inputs dropout
        dec_inputs = tf.nn.dropout(dec_inputs, keep_prob=self.keep_prob)

        with tf.variable_scope('decoder'):
            outputs, state = tf.nn.dynamic_rnn(self.dec_cell,
                                               inputs=dec_inputs,
                                               initial_state=self.dec_initial_state)

        self.final_state = state
        output = tf.reshape(outputs, [-1, self.hidden_size])

        # Softmax output layer
        # =============================================================================

        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.vocab_size])
            softmax_b = tf.get_variable('softmax_b', shape=[self.vocab_size])
            self.logits = tf.add(tf.matmul(output, softmax_w), softmax_b)
            self.preds = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.preds, 1)

    def compute_cost(self):
        """cost = cross entropy + KL divergence"""
        targets = tf.reshape(self.enc_x, [-1])

        with tf.name_scope('loss'):
            with tf.name_scope('xentropy'):
                # Cross entropy loss
                losses = \
                    tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits],
                                                                       [targets],
                                                                       [tf.ones_like(targets, dtype=tf.float32)])
                self.xentropy_loss = tf.reduce_mean(losses)

            with tf.name_scope('kl_divergence'):
                # KL loss
                kl_div = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                              - tf.square(self.z_mean)
                                              - tf.exp(self.z_log_var), 1)
                self.kl_div=tf.reduce_mean(kl_div)
                # KL loss annealing
                if self.annealing:
                    kl_loss = self.kl_anneal_update * kl_div
                else:
                    kl_loss = kl_div
                self.kl_loss = tf.reduce_mean(kl_loss)

            self.cost = tf.reduce_mean(self.xentropy_loss + kl_loss)

    def train(self):
        tvars = tf.trainable_variables()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # Gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
