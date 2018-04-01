import tensorflow as tf

from tensorflow.contrib import rnn

vocab=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]

import numpy as np

embedding_dim = 110

input_x = tf.placeholder(tf.int32,shape=[None,None])

output_y = tf.placeholder(tf.int32,shape=[None,])

word_embedding = tf.get_variable('embed',shape=[len(vocab),110],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

lookup = tf.nn.embedding_lookup(word_embedding,input_x)

without_zero=tf.count_nonzero(input_x,axis=-1)

with tf.variable_scope('encoder') as scope:
    cell=rnn.LSTMCell(num_units=109)
    model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=lookup,sequence_length=without_zero,dtype=tf.float32)


output,(_states_c,states_f)=model

transpo1= tf.transpose(output[0],[1,0,2])
transpo2= tf.transpose(output[1],[1,0,2])

concat = tf.concat((transpo1[-1],transpo2[-1]),axis=-1)

concat2=tf.concat((_states_c.c,states_f.c),axis=-1)

weights_x= tf.get_variable('weight',shape=[2*109,len(vocab)],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

bias_x = tf.get_variable('bias',shape=[len(vocab)],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

result_matmul = tf.matmul ( concat , weights_x) + bias_x

normalizaton = tf.nn.softmax(result_matmul)

max_prob = tf.argmax(normalizaton,axis=-1)

#cross entropy

ce= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_matmul,labels=output_y)
loss = tf.reduce_mean(ce)

#accuracy

acc = tf.reduce_mean(tf.cast((tf.equal(tf.cast(max_prob,tf.int32),output_y)),tf.float32))

train = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mode_run,emd_run,trans1,trans2,conca_run,conca2_run,result_ma_run,max_pro,accc=sess.run([model,lookup,transpo1,transpo2,concat,concat2,result_matmul,max_prob,acc],feed_dict={input_x:np.random.randint(0,10,[23,6]),output_y:np.random.randint(0,14,[23])})

