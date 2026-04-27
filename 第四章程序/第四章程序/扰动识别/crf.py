import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

inputs=tf.random.truncated_normal([2,10,5])
target=tf.convert_to_tensor(np.random.randint(5,size=(2,10)),dtype=tf.int32)
out=tf.keras.layers.Softmax(inputs)

lens=tf.convert_to_tensor([9,6],dtype=tf.int32)
log_likelihood,tran_paras=tfa.text.crf_log_likelihood(inputs, target, lens)
batch_pred_sequence,batch_viterbi_score=tfa.text.crf_decode(inputs,tran_paras,lens)
loss=tf.reduce_sum(-log_likelihood)
print('log_likelihood is :',log_likelihood.numpy())
print('batch_pred_sequence is :',batch_pred_sequence.numpy())
print('loss is :',loss.numpy())
