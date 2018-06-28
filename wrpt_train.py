import math
import pickle
import numpy
import keras
import sys
import wrpt_model
from datetime import datetime

# Set training data and support data file names
training_data = '2017-11-07_learning_data_1.vp'
combo_win_rate = 'new_combo_2.vp'
advantage_rate = 'new_advantage_against.vp'
weight_name = 'train6_v707_new_temp.weights'
log_file = "progress_log_6.txt"

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

# if there isn't an exisiting weight to udpate
try:
    model.load_weights(weight_name)
except:
    print("creating new weight")
    pass

model.compile(
    optimizer=opt,
    loss={'hid_layer':'sparse_categorical_crossentropy', 'another_output': 'mae'},
    loss_weights={'hid_layer': 1, 'another_output': 0.5},
    metrics=['accuracy'])
history = LossHistory()

# Read in training data
print("\nLoading training data: {0}\n".format(training_data))
with open(training_data, 'rb') as f:
    t = pickle.Unpickler(f)
    result = t.load()
print('Loading Complete, start training\n')

try:
    target = []
    result0 = []
    result1 = []
    result2 = []
    result3 = []
    result4 = []
    for i in range(len(result[0])):
        passed = False
        for elem in result[0][i]:
            if elem in invalid_keys:
                passed = True
        for elem in result[1][i]:
            if elem in invalid_keys:
                passed = True
        if passed:
            continue
        if len(result[2][i]) != 5:
            continue
        if len(result[3][i]) != 5:
            continue
        result[0][i] = [118 if x==120 else x for x in result[0][i]]
        result[1][i] = [118 if x==120 else x for x in result[1][i]]
        result0.append(result[0][i])
        result1.append(result[1][i])
        for e in result[2][i]:
            result2.append(e)
        for e in result[3][i]:
            result3.append(e)
        result4.append([0, 0, 0, 0, 0])
        target.append(result[4][i][0])

    target = numpy.array(target)
    result0 = numpy.array(result0)
    result1 = numpy.array(result1)
    result2 = numpy.array(result2).reshape(-1, 5, 14)
    result3 = numpy.array(result3).reshape(-1, 5, 14)
    result4 = numpy.array(result4)
    temp2 = result2.reshape((result2.shape[0]*result2.shape[1], -1))
    temp3 = result3.reshape((result3.shape[0]*result3.shape[1], -1))
    result2 = (result2 - temp2.mean(axis=0))/temp2.std(axis=0)
    result3 = (result3 - temp3.mean(axis=0))/temp3.std(axis=0)

    model.fit({'team_1': result0, 'team_2': result1, 'team_1_stat': result2, 'team_2_stat': result3},
                {'hid_layer': target, 'another_output': result4}, 
                batch_size=32, callbacks=[history], validation_split=0.001, epochs=1, verbose=1)

    model.save_weights(weight_name)

    # import matplotlib.pyplot as plt
    history.losses = numpy.array(history.losses)
    history.losses = history.losses[:(int(history.losses.shape[0] / 10000) * 10000)]
    loss_summary = numpy.mean(numpy.array(history.losses).reshape(-1, 10000), axis=1)

    print(loss_summary)

    with open(log_file, "a") as text_file:
        print("{0}: {1}".format(datetime.now().strftime("%Y-%m-%d"), training_data), file=text_file)
    sys.exit()
except Exception as e:
    with open(log_file, "a") as text_file:
        print("Error happened{0}: {1}".format(datetime.now().strftime("%Y-%m-%d"), training_data), file=text_file)
    sys.exit()
    pass
