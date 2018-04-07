import data
from random import shuffle
import resources as rr


dataset = data.create_dataset()
samples = dataset['samples']
shuffle(samples)
trainlen = int(len(samples) * 0.80)
testlen = int(len(samples) * 0.10)
validlen = testlen
    # split
trainset = samples[:trainlen]
testset = samples[trainlen:trainlen + testlen]
validset = samples[trainlen + testlen:]
vocab = dataset['vocab']
batch_size=100
iteration=len(trainset)//batch_size
epoch=10

inder=rr.lang2i
labelss=len(rr.lang2i)


def padding(datar):
    max_w=max([len(i) for i,j in datar])
    final_uy=[]
    labels=[]
    for i,j in datar:
        if len(i)<max_w:
            final_uy.append(i+[0]*(max_w-len(i)))
            labels.append(j)
        else:
            final_uy.append(i)
            labels.append(j)


    return {'seq':final_uy,'label':labels}




import tensorflow as tf



import numpy as np
from tensorflow.contrib import rnn

word_embedding_dim=250

hidden_dim=270



input_x= tf.placeholder(tf.int32,shape=[None,None])
output_y= tf.placeholder(tf.int32,shape=[None,])

word_embedding =tf.get_variable('embedding',shape=[len(vocab),word_embedding_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))


lookup_embedding = tf.nn.embedding_lookup(word_embedding,input_x)


sequence_len=tf.count_nonzero(input_x,axis=-1)

with tf.variable_scope('encoder') as scope:
    cell=rnn.LSTMCell(hidden_dim)
    model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=lookup_embedding,sequence_length=sequence_len,dtype=tf.float32)

model_output,(fs,fc)=model

transpose1=tf.transpose(model_output[0],[1,0,2])

concat=tf.concat((fs.c,fc.c),axis=-1)


weights=tf.get_variable('weight',shape=[2*hidden_dim,labelss],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

bias=tf.get_variable('bias',shape=[labelss],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

final_output=tf.matmul(concat,weights) + bias

#normalizatio
normal_a=tf.nn.softmax(final_output)
pred=tf.argmax(normal_a,axis=-1)


#cross entropy
ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_output,labels=output_y)

loss=tf.reduce_mean(ce)



#evaluate
evalu=tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred,tf.int32),output_y),tf.float32))


#train

train=tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        for j in range(iteration):
            data_batch = padding(trainset[j * batch_size:(j + 1) * batch_size])

            da_batach=data_batch['seq']
            data_laa=data_batch['label']

            a_new, b_new, c_new, d_new, e_new, f_new, g_new_, h_new = sess.run(
            [model, final_output, normal_a, pred, ce, loss, evalu, train],feed_dict={input_x:da_batach,output_y:data_laa})
            print("epoch {} loss {} , iteration {} accuracy {}".format(i,f_new,j,g_new_))

    while True:
        input11 = [int(i.replace(',', '')) for i in input().split()]
        if input11 == 'q':
            break
        else:
            a, b, c = sess.run([model, normal_a, pred], feed_dict={input_x: [input11]})
            tolist = np.array(b[0]).tolist()
            print(sorted([(inder[i], j) for i, j in enumerate(tolist)], key=lambda x: x[1], reverse=True)[:3])
            print(tolist[c[0]])



#interaction



epoch 9 loss 0.3418683707714081 , iteration 156 accuracy 0.8799999952316284
epoch 9 loss 0.5255255699157715 , iteration 157 accuracy 0.8299999833106995
epoch 9 loss 0.4484560787677765 , iteration 158 accuracy 0.8700000047683716
epoch 9 loss 0.41938677430152893 , iteration 159 accuracy 0.8399999737739563
18, 41, 48, 54, 51, 58
[('Arabic', 0.9998551607131958), ('English', 0.00010600956011330709), ('Czech', 2.6400351998745464e-05)]
0.9998551607131958
21, 34, 41, 34, 52
[('Arabic', 0.9930160045623779), ('English', 0.002775779692456126), ('Russian', 0.002372889779508114)]
0.9930160045623779
10, 34, 35, 38, 45, 45, 48
[('Italian', 0.8357179164886475), ('Spanish', 0.12903481721878052), ('Russian', 0.012426197528839111)]
0.8357179164886475
10, 34, 35, 38, 45, 45, 48
[('Italian', 0.8357179164886475), ('Spanish', 0.12903481721878052), ('Russian', 0.012426197528839111)]
0.8357179164886475
10, 34, 35, 38, 45, 45, 48
[('Italian', 0.8357179164886475), ('Spanish', 0.12903481721878052), ('Russian', 0.012426197528839111)]
0.8357179164886475
10, 34, 35, 38, 45, 45, 48
[('Italian', 0.8357179164886475), ('Spanish', 0.12903481721878052), ('Russian', 0.012426197528839111)]
0.8357179164886475
11, 38, 46, 42, 36, 41, 38, 55
[('Russian', 0.9990748167037964), ('German', 0.0004535038024187088), ('Czech', 0.000434660236351192)]
0.9990748167037964
15, 48, 45, 44, 42, 47
[('Russian', 0.9074429869651794), ('English', 0.07504285871982574), ('Czech', 0.0041922456584870815)]
0.9074429869651794
