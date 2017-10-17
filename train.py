# -*- coding: utf-8 -*-
import os
import csv
import time
import json
import datetime
import numpy as np
import pickle as pkl
import tensorflow as tf
import embedding

import data_helper
from vae import vae_lstm
# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Data parameters
tf.flags.DEFINE_string('data_file', 'data/comm1.csv', 'Data file path')
tf.flags.DEFINE_string('language', 'ch', "Language of the data file. You have two choices: ['ch', 'en']")
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_integer('level', 2, '1 for char level, 2 for phrase level')

# Hyperparameters
tf.flags.DEFINE_integer('embedding_size', 100, 'Word embedding size')
tf.flags.DEFINE_integer('hidden_size', 256, 'Number of hidden units in the LSTM cell')
# VAE中隐含变量的维度大小，不要大于100
tf.flags.DEFINE_integer('z_size', 16, 'Latent dimension')
tf.flags.DEFINE_integer('keep_prob', 0.8, 'Dropout keep probability')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
# Gradient clipping中所用参数，不要太大，否则起不到clipping的效果，也不要太小。
tf.flags.DEFINE_float('grad_norm', 5.0, 'Gradient clipping')
# KL退火中所用参数，不要超过1，可设为1e-3
tf.flags.DEFINE_integer('kl_rate', 1e-5, 'KL annealing rate')
# 在多少步之后开始进行KL退火
tf.flags.DEFINE_integer('kl_max_steps', 5000, 'Increase the KL rate after these many steps')
# 是否使用KL退火
tf.flags.DEFINE_boolean('annealing', True, 'Whether to conduct KL annealing')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_integer('evaluate_every_steps', 50, 'Evaluate the model after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 1000, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 5, 'Number of models to store')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
print ('******')


# Load data
# =============================================================================
embedding.word2vec_train(FLAGS.data_file)

pkl_file1 = open('word2vec_file/w_2_idx.pkl', 'rb')
w_2_idx=pkl.load(pkl_file1)

#pkl_file2 = open('word2vec_file/data_10w.pkl', 'rb')
#data=pkl.load(pkl_file2)




data= data_helper.load_data(file_path=FLAGS.data_file,
                                      level=FLAGS.level,
                                      vocab_size=FLAGS.vocab_size,
                                      language=FLAGS.language,
                                      shuffle=True,
                                      vocab_dict=w_2_idx)
                                      

                                   
idx_2_w = dict(zip(w_2_idx.values(), w_2_idx.keys()))

FLAGS.vocab_size = len(w_2_idx)

# iterator
batches = data_helper.batch_iter(data, w_2_idx, FLAGS.batch_size, FLAGS.num_epochs)
FLAGS.max_length = max(map(len, data))

# Save vocabulary to file
vocab_file = open(os.path.join(outdir, 'vocab.pkl'), 'wb')
pkl.dump(w_2_idx, vocab_file)
vocab_file.close()

# Save parameters to file
params = FLAGS.__flags
params_file = open(os.path.join(outdir, 'params.pkl'), 'wb')
pkl.dump(params, params_file)
params_file.close()

# Train
# =============================================================================

with tf.Graph().as_default():
    with tf.Session() as sess:
        print('Start building graph...')
        lm = vae_lstm(FLAGS, is_training=True)
        # Loss Summaries
        loss_summary = tf.summary.scalar('loss', lm.cost)
        xentropy_summary = tf.summary.scalar('xentropy_loss', lm.xentropy_loss)
        kl_summary = tf.summary.scalar('kl_loss', lm.kl_loss)
        kl_div = tf.summary.scalar('kl_div', lm.kl_div)

        # Summary op
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoint)
        print('Graph has been successfully built')

        sess.run(tf.global_variables_initializer())

        def run_step(input_data, kl_anneal_rate):
            """ Run one step of the training process. """
            enc_input, dec_input, sequence_length = input_data
            max_length = max(sequence_length)
            
            dec_sequence_length = np.array(sequence_length)
            enc_sequence_length = dec_sequence_length - 1

            fetches = {'step': lm.global_step,
                       'cost': lm.cost,
                       'xentropy_loss': lm.xentropy_loss,
                       'kl_div':lm.kl_div,
                       'kl_loss':lm.kl_loss,
                       'train_op': lm.train_op,
                       'summaries': train_summary_op,
                       'preds': lm.predictions}
            feed_dict = {lm.enc_x: enc_input,
                         lm.dec_x: dec_input,
                         lm.keep_prob: FLAGS.keep_prob,
                         lm.kl_rate: kl_anneal_rate}

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            xentropy_loss = vars['xentropy_loss']
            kl_div = vars['kl_div']
            kl_loss = vars['kl_loss']
            summaries = vars['summaries']
            preds = vars['preds']
            train_summary_writer.add_summary(summaries, step)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, cross entropy loss: {:g}, kl loss: {:g}, kl_div: {:g}".format(time_str, step, cost,
                                                                                             xentropy_loss,kl_loss,kl_div))

            return cost, preds


        print('Start training ...')
        start = time.time()

        total_cost = 0
        kl_anneal_rate = 0.0
        for batch in batches:
            current_step = tf.train.global_step(sess, lm.global_step)
            # KL annealing
            if current_step > FLAGS.kl_max_steps:
                if kl_anneal_rate < 1.0:
                    kl_anneal_rate += (current_step - FLAGS.kl_max_steps) * FLAGS.kl_rate
                elif kl_anneal_rate >= 1.0:
                    kl_anneal_rate = 1.0
            cost, preds = run_step(batch, kl_anneal_rate)
            current_step = tf.train.global_step(sess, lm.global_step)
            total_cost += cost

            if current_step % FLAGS.evaluate_every_steps == 0:
                aver_cost = total_cost / FLAGS.evaluate_every_steps
                print('\nAverage cost at step {}: {}'.format(current_step, aver_cost))
                total_cost = 0

                print('Input sentence:')
                input_str = "".join([idx_2_w[word] for word in batch[0][0]])
                print(input_str)
                print('Reconstruction result:')
                recon_str = "".join([idx_2_w[word] for word in preds])
                print(recon_str[:len(input_str)])

            if current_step % FLAGS.save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(outdir, 'model/lm'), current_step)
                print('\nModel saved to {}\n'.format(save_path))

        end = time.time()

        print(('\nRun time: {}'.format(end - start)))
        print('\nAll the files have been saved to {}'.format(outdir))
