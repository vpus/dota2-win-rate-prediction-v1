#!/usr/bin/env python

import math
import json
import pickle
import numpy
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, Flatten, Merge, Convolution2D, Reshape, Input, Lambda, merge, RepeatVector, Permute, LSTM, BatchNormalization
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
import theano.tensor as T

import sys

t1 = True
t2 = True
t3 = True
t4 = True

class Advantage(Layer):
    def __init__(self, wt, **kwargs):
        self.W = K.variable(wt)
        super(Advantage, self).__init__(**kwargs)
    def build(self, input_shape=None):
        self.built = True
    def call(self, inputs, mask=None):
        x = inputs[0]
        y = inputs[1]
        t = K.permute_dimensions(self.W[x], [0, 2, 1])
        return t[T.arange(x.shape[0]).reshape((-1, 1)), y]
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 5, 5)

with open('new_combo_2.vp', 'rb') as f:
    t = pickle.Unpickler(f)
    a_rate = t.load()
    a_rate[118] = a_rate[120]
    for i in range(len(a_rate)):
        a_rate[i][118] = a_rate[i][120]

a_rate = numpy.array(a_rate)
# numpy.savetxt("train6_a_rate.csv", a_rate, delimiter=',')
a_rate = (a_rate - a_rate.mean(keepdims=True))/(a_rate.std(keepdims=True))
# numpy.savetxt("adjusted_train6_a_rate.csv", a_rate, delimiter=',')

with open('new_advantage_against.vp', 'rb') as f:
    t = pickle.Unpickler(f)
    b_rate = t.load()
    b_rate[118] = b_rate[120]
    for i in range(len(b_rate)):
        b_rate[i][118] = b_rate[i][120]

b_rate = numpy.array(b_rate)
b_rate = (b_rate - b_rate.mean(keepdims=True))/(b_rate.std(keepdims=True))

# sys.exit()

invalid_keys = [0]

for i in range(len(a_rate)):
    checked = True
    for j in a_rate[i]:
        if not math.isnan(j):
            checked = False
    if checked:
        invalid_keys.append(i)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

dim0 = 3
dim1 = 8
dim2 = 1
dim3 = 10

def my_dot(x, w):
   l = []
   for i in range(5):
       l.append(K.batch_dot(x[:, i], w[:, i, :-1]) + w[:, i, -1])
   return K.concatenate(l, axis=1)

input0 = Input(shape=(5,), dtype='int32', name='team_1')
input1 = Input(shape=(5,), dtype='int32', name='team_2')

adv_metric = Advantage(wt=a_rate, input_shape=(1,), trainable=False)([input0, input0])
reversed_metric = Lambda(lambda x: K.reverse(x, axes=1), output_shape=(5,5))(adv_metric)
forward_lstm = LSTM(5, trainable=t1, input_shape=(5, 5))
reverse_lstm = LSTM(5, trainable=t1, input_shape=(5, 5)) 
output1 = forward_lstm(adv_metric)
output2 = reverse_lstm(reversed_metric)

merged_output = merge([output1, output2], mode='sum', output_shape=(5,))
# merged_output = add([output1, output2])

adv_metric_d = Advantage(wt=a_rate, input_shape=(1,), trainable=False)([input1, input1])
reversed_metric_d = Lambda(lambda x: K.reverse(x, axes=1), output_shape=(5,5))(adv_metric_d)
output1_d = forward_lstm(adv_metric_d)
output2_d = reverse_lstm(reversed_metric_d)

merged_output_d = merge([output1_d, output2_d], mode='sum')
# merged_output_d = add([output1_d,output2_d])

merged = merge([merged_output, merged_output_d], mode=lambda x: x[0] - x[1], output_shape=(5,))

#advangtage
new_adv_metric = Advantage(wt=b_rate, input_shape=(1,), trainable=False)([input0, input1])
new_reversed_metric = Lambda(lambda x: K.reverse(x, axes=1), output_shape=(5,5))(new_adv_metric)
new_output = LSTM(5, input_shape=(5,5), trainable=t2)(new_adv_metric)
new_rev_output = LSTM(5, input_shape=(5,5), trainable=t2)(new_reversed_metric)
new_merged_output = merge([new_output, new_rev_output], mode='sum')


#hero specific data

emb = Embedding(input_dim=125, output_dim=20, input_length=5, trainable=t3)

emb0 = emb(input0)
emb1 = emb(input1)

forward_lstm_emb = LSTM(20, input_shape=(5,20), trainable=t3)
reverse_lstm_emb = LSTM(20, input_shape=(5,20), trainable=t3)

reverse_emb = Lambda(lambda x: K.reverse(x, axes=1), output_shape=(20, 5))

reverse_emb0 = reverse_emb(emb0)
reverse_emb1 = reverse_emb(emb1)

emb0_output0 = forward_lstm_emb(emb0)
emb0_output1 = reverse_lstm_emb(reverse_emb0)

emb1_output0 = forward_lstm_emb(emb1)
emb1_output1 = reverse_lstm_emb(reverse_emb1)

emb0_merged = merge([emb0_output0, emb0_output1], mode='sum')
emb1_merged = merge([emb1_output0, emb1_output1], mode='sum')

emb_merged = merge([emb0_merged, emb1_merged], mode=lambda x: x[0] - x[1], output_shape=(20,))

emb_hid0 = Dense(15, input_dim=20, activation='sigmoid', trainable=t3)(emb_merged)
emb_hid1 = Dense(5, input_dim=15, activation='sigmoid', trainable=t3)(emb_hid0)

# another model (data -> emb_hid1)

_input2 = Input(shape=(5,14), dtype='float32', name='team_1_stat')
_input3 = Input(shape=(5,14), dtype='float32', name='team_2_stat')

input2 = BatchNormalization(axis=2)(_input2)
input3 = BatchNormalization(axis=2)(_input3)

forward_lstm_input = LSTM(14, input_shape=(5,14), trainable=t4)
reverse_lstm_input = LSTM(14, input_shape=(5,14), trainable=t4)

reverse_input = Lambda(lambda x: K.reverse(x, axes=1), output_shape=(14, 5))
reverse_input2 = reverse_input(input2)

reverse_input3 = reverse_input(input3)

input2_output0 = forward_lstm_input(input2)
input2_output1 = reverse_lstm_input(reverse_input2)

input3_output0 = forward_lstm_input(input3)
input3_output1 = reverse_lstm_input(reverse_input3)

input2_merged = merge([input2_output0, input2_output1], mode='sum')
input3_merged = merge([input3_output0, input3_output1], mode='sum')

input_merged = merge([input2_merged, input3_merged], mode=lambda x: x[0] - x[1], output_shape=(14,))

input_hid0 = Dense(10, input_dim=14, activation='sigmoid', trainable=t4)(input_merged)
input_hid1 = Dense(5, input_dim=10, activation='sigmoid', trainable=t4)(input_hid0)

another_output = merge([emb_hid1, input_hid1], mode=lambda x: x[0] - x[1], output_shape=(5,), name='another_output')

new_merged = merge([new_merged_output, merged, emb_hid1], mode='concat',concat_axis=1)

hid_layer = Dense(2, input_dim=15, activation='softmax', name='hid_layer')(new_merged)
model = Model(inputs=[input0, input1, _input2, _input3], outputs=[hid_layer, another_output])

opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.load_weights('train6_v707_new_temp.weights')

another_output1 = merge([merged_output, merged_output_d], mode=lambda x: x[0] - x[1], output_shape=(5,), name='another_output1')

model = Model(inputs=[input0, input1, _input2, _input3], outputs=[hid_layer, another_output1])

model.compile(
    optimizer=opt,
    loss={'hid_layer':'sparse_categorical_crossentropy', 'another_output1': 'mae'},
    loss_weights={'hid_layer': 1, 'another_output1': 0.5},
    metrics=['accuracy'])
history = LossHistory()


rad_squad = [1,2,3,4,5]
dire_squad = [6,7,8,9,120]
# for i in range(1,6):
#     rad_squad.append(int(sys.argv[i]))
# for i in range(6,11):
#     dire_squad.append(int(sys.argv[i]))
# print(rad_squad)
# print(dire_squad)

result0 = []
result1 = []
result2 = []
result3 = []
result4 = []
# passed = False
# for elem in result[0][i]:
#     if elem in invalid_keys:
#         passed = True
# for elem in result[1][i]:
#     if elem in invalid_keys:
#         passed = True
# if passed:
#     continue
# if len(result[2][i]) != 5:
#     continue
# if len(result[3][i]) != 5:
#     continue
rad_squad = [118 if x==120 else x for x in rad_squad]
dire_squad = [118 if x==120 else x for x in dire_squad]
result0.append(rad_squad)
result1.append(dire_squad)
for e in range(5):
    result2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
for e in range(5):
    result3.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
result4.append([0, 0, 0, 0, 0])

result0 = numpy.array(result0)
result1 = numpy.array(result1)
result2 = numpy.array(result2).reshape(-1, 5, 14)
result3 = numpy.array(result3).reshape(-1, 5, 14)
result4 = numpy.array(result4)
# print(result0.shape)
# print('result2.shape')
# print(result2.shape)
# print('result2[0]')
# print(result2[0])
# print('result3.shape')
# print(result3.shape)
# print('result4.shape')
# print(result4.shape)
temp2 = result2.reshape((result2.shape[0]*result2.shape[1], -1))
temp3 = result3.reshape((result3.shape[0]*result3.shape[1], -1))
result2 = (result2 - temp2.mean(axis=0))/temp2.std(axis=0)
result3 = (result3 - temp3.mean(axis=0))/temp3.std(axis=0)

y, y1 = model.predict({'team_1': result0, 'team_2': result1, 'team_1_stat': result2, 'team_2_stat': result3},
          batch_size=32)
#print('y:')
print(y[0])
elem = y[0]

# print(result0[0])
# print(hero_map)
# print([hero_map[i] for i in result0[0]])
# print([hero_map[i] for i in result1[0]])
# print('===============')
# result = {}
# temp = [e/20.0 for e in range(20)]
# for e in temp:
#     result[e] = [0, 0]
# for e in range(len(y)):
#     elem = y[e]
#     elem1 = target[e]
#     if elem1 == 1:
#         p = elem[0] < 0.5
#     else:
#         p = elem[0] > 0.5
#     bracket = math.floor(elem[0] * 20) / 20.0
#     result[bracket][p] = result[bracket][p] + 1
# result1 = {}
# for e in temp:
#     if result[e][1] == 0:
#         continue
#     result1[e] = result[e][1] * 1.0 / (result[e][1] + result[e][0])
# print(result)
# print(result1)

