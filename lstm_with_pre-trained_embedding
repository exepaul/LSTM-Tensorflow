import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

labels=[1,0]


input_x=tf.placeholder(tf.int32,shape=[None,None])

output=tf.placeholder(tf.int32,shape=[None,])

data_x=np.load('word_embedding_lstm.npy')
words_y=np.load('words_list_lstm.npy')

word_embedding=tf.get_variable('W',shape=[400000,100],dtype=tf.float32,initializer=tf.constant_initializer(np.array(data_x)),trainable=False)
embedding_lookup=tf.nn.embedding_lookup(word_embedding,input_x)

sequ_len=tf.count_nonzero(input_x,axis=-1)

with tf.variable_scope('encoder') as scope:
    cell=rnn.LSTMCell(num_units=250)
    model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=embedding_lookup,sequence_length=sequ_len,dtype=tf.float32)

model_output,(fs,fc)=model

concat_out=tf.concat((fs.c,fc.c),axis=-1)

fc_layer= tf.get_variable('weight',shape=[2*250,len(labels)],dtype=tf.float32,initializer=tf.random_normal_initializer(-0.01,0.01))
bias= tf.get_variable('bias',shape=[len(labels)],dtype=tf.float32,initializer=tf.random_normal_initializer(-0.01,0.01))

logi_ts=tf.matmul(concat_out,fc_layer)+bias

#normalization
prob=tf.nn.softmax(logi_ts)
pred=tf.argmax(prob,axis=-1)

#cross_entropy
ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logi_ts,labels=output)
loss=tf.reduce_mean(ce)


#accuracy
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred,tf.int32),output),tf.float32))

#train
train_x=tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a,b,c,d,e,f=sess.run([embedding_lookup,model_output,fs,concat_out,fc_layer,logi_ts],feed_dict={input_x:np.random.randint(0,10,[3,6]),output:np.random.randint(0,10,[3])})
    print("embedding_output",a.shape)  #3x6x100
    print("model_output",b[0].shape)   #3x6x250
    print("each_cell",c.c.shape)         #3x250
    print("concat",d.shape)            #3x500
    print("fc",e.shape)                #500x2
    print("logits_matmul",f.shape)     #3x2
