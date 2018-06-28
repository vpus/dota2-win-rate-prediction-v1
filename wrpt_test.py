#!/usr/bin/env python

import math
import json
import pickle
import numpy
import keras
import wrpt_model
import sys

# Setting weight file name
combo_win_rate = 'new_combo_2.vp'
advantage_rate = 'new_advantage_against.vp'
weight_name = 'train6_v707_new_temp.weights'

# Set input lineups
rad_squad = []
dire_squad = []
for i in range(1,6):
    rad_squad.append(int(sys.argv[i]))
for i in range(6,11):
    dire_squad.append(int(sys.argv[i]))
# For overwrite use
# rad_squad = [1,2,3,4,5]
# dire_squad = [6,7,8,9,120]
print("radiant lineup: ", rad_squad)
print("dire lineup: ", dire_squad)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# read in combo win rate
with open(combo_win_rate, 'rb') as f:
    t = pickle.Unpickler(f)
    a_rate = t.load()
    a_rate[118] = a_rate[120]
    for i in range(len(a_rate)):
        a_rate[i][118] = a_rate[i][120]

a_rate = numpy.array(a_rate)
a_rate = (a_rate - a_rate.mean(keepdims=True))/(a_rate.std(keepdims=True))

# read in advantage rating of one hero against another 
with open(advantage_rate, 'rb') as f:
    t = pickle.Unpickler(f)
    b_rate = t.load()
    b_rate[118] = b_rate[120]
    for i in range(len(b_rate)):
        b_rate[i][118] = b_rate[i][120]

b_rate = numpy.array(b_rate)
b_rate = (b_rate - b_rate.mean(keepdims=True))/(b_rate.std(keepdims=True))

invalid_keys = [0]
for i in range(len(a_rate)):
    checked = True
    for j in a_rate[i]:
        if not math.isnan(j):
            checked = False
    if checked:
        invalid_keys.append(i)

model = wrpt_model.build_model(a_rate, b_rate)
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.load_weights(weight_name)

model.compile(
    optimizer=opt,
    loss={'hid_layer':'sparse_categorical_crossentropy', 'another_output': 'mae'},
    loss_weights={'hid_layer': 1, 'another_output': 0.5},
    metrics=['accuracy'])
history = LossHistory()

result0 = []
result1 = []
result2 = []
result3 = []
result4 = []

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

temp2 = result2.reshape((result2.shape[0]*result2.shape[1], -1))
temp3 = result3.reshape((result3.shape[0]*result3.shape[1], -1))
result2 = (result2 - temp2.mean(axis=0))/temp2.std(axis=0)
result3 = (result3 - temp3.mean(axis=0))/temp3.std(axis=0)

y, y1 = model.predict({'team_1': result0, 'team_2': result1, 'team_1_stat': result2, 'team_2_stat': result3},
          batch_size=32)

print(y[0])
elem = y[0]
