import funcs_15
import model_selection
import csv
import gc
import os
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras import regularizers
from keras import layers
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt
import pymongo
from tqdm import *
import ydata_profiling as pp
from sklearn.metrics import f1_score as f1score_sklearn
from sklearn.metrics import balanced_accuracy_score

from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss, TomekLinks, RepeatedEditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE, SMOTE

# Script that trains the Future Net given the hyperparameters
# The parameters should be found using a tuner, using the ones in the thesis is also an option

# Adding the characters GO: to allow for the search
def addgo(code) -> str: 'GO:' + str(code)


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def thresholder(y_pred, y_true):
    thresholds = np.arange(0, 1, 0.02)
    scores = [f1score_sklearn(y_true, to_labels(y_pred, t), zero_division=0.0) for t in thresholds]
    ix = np.argmax(scores)
    best_threshold = thresholds[ix]
    return best_threshold


def f1_extraction(X_val, model):
    y_pred_val = model.predict(X_val, verbose=0, batch_size=batch_size)
    y_pred_val = np.reshape(y_pred_val, (1, -1))
    y_pred_val = y_pred_val[0]
    best_threshold = thresholder(y_pred_val, Y_val)
    y_best_pred = (y_pred_val >= best_threshold).astype('int')
    f1_val = f1score_sklearn(Y_val, y_best_pred, zero_division=0.0)
    return f1_val, best_threshold


def loss_wrap(class_weights, apply_balance=True, alpha=0.25, gamma=2.0, focal=False, smoothing=0.0):
    def loss_fn(y_true, y_pred):
        weights = tf.constant([class_weights[0], class_weights[1]])
        applied_weights = tf.gather(weights, y_true)
        if focal:
            loss = tf.keras.losses.BinaryFocalCrossentropy(
                    apply_class_balancing=apply_balance, alpha=alpha, gamma=gamma, from_logits=True, label_smoothing=smoothing,
                    reduction=tf.keras.losses.Reduction.NONE)
            bce = loss(y_true, y_pred)
        else:
            loss = tf.keras.losses.BinaryCrossentropy(
                    from_logits=True, label_smoothing=smoothing, axis=-1, reduction=tf.keras.losses.Reduction.NONE)
            bce = loss(y_true, y_pred)
        tf.cast(applied_weights, tf.float32)
        tf.cast(bce, tf.float32)
        weighted_bce = K.mean(bce * applied_weights)
        return weighted_bce
    return loss_fn

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = future_net(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, future_net.trainable_weights)
    optimizer.apply_gradients(zip(grads, future_net.trainable_weights))
    return loss_value

@tf.function
def val_step(x, y):
    val_logits = future_net(x, training=False)
    val_loss = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)
    return val_loss


start_time = time.time()

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
seed = 42
keras.utils.set_random_seed(seed)

# If using TensorFlow, this will make GPU ops as deterministic as possible, but it will affect the overall performance
# tf.config.experimental.enable_op_determinism()

use_total_score = True

current_datetime = datetime.now() # Get the current date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") # Format the date and time as a string

if use_total_score:
    dir_name = f"future_net_{formatted_datetime}"
else:
    dir_name = f"future_net_{formatted_datetime}"
parent_path = f"/data/gabriele.gradara/reti/future_net_reports"
path = os.path.join(parent_path, dir_name)
os.mkdir(path)

if use_total_score:
    file_name = path + f"/report_future_net_{formatted_datetime}_tsu.txt"  # Use the formatted date and time as a file name
    graph_name = path + f"/graph_future_net_{formatted_datetime}_tsu.png"  # Use the formatted date and time as a file name
    f1_thresh_graph_name = path + f"/f1_thresh_graph_{formatted_datetime}_tsu.png"  # Use the formatted date and time as a file name
else:
    file_name = path + f"/report_future_net_{formatted_datetime}.txt"  # Use the formatted date and time as a file name
    graph_name = path + f"/graph_future_net_{formatted_datetime}.png"  # Use the formatted date and time as a file name
    f1_thresh_graph_name = path + f"/f1_thresh_graph_{formatted_datetime}.png"  # Use the formatted date and time as a file name

gt_path = 'csv/groundtruth_sorted.csv'  #groundtruth path
# flag_path = 'csv/go_flag.csv'
# oa_path = "csv/output_argot_flagged_10000_addedgos.csv"  #output argot path
# oa_path = "csv/output_argot_flagged_1000000_addedgos.csv"  #output argot path
oa_path = "csv/argot_output_flagged_def.csv"
#Establish connection to mongodb database
# client = pymongo.MongoClient('localhost', 27017)
# db = client['ARGOT3']
# simgic_collection = db['simgic_big']

#tf.compat.v1.disable_eager_execution() #Disable eager execution
#tf.enable_eager_execution()+

# x_loss, y_loss, z_loss = funcs_15.init_loss_func_def()
# z_loss = tf.convert_to_tensor(z_loss, dtype=tf.float32)

#Retrieve the content of the groundtruth in list format
gt_sorted = [] #The groundtruth is expected to be previously alphabetically sorted
with (open(gt_path, 'r') as file): # Read the contents of the txt file
    csv_reader=csv.reader(file)
    for row in csv_reader:
        gt_sorted.append(row)
#Retrieve the position of the letters in the sorted ground truth
# alpha_order = funcs_15.alpha_indexer(gt_path)

data = pd.read_csv(oa_path)
# data_flag = pd.read_csv(flag_path)
print(data.head()) # This function returns the first n rows for the object based on position.
print(data.info()) # It is useful for quickly testing if your object has the right type of data in it.

X = data.drop(columns = ['SeqID','GOID','Flag'],inplace=False)
Y = data['Flag']
X['Ontology'] = X['Ontology'].astype(int)
# Get one hot encoding of Ontology column
one_hot = pd.get_dummies(X['Ontology'], dtype=float)
# Drop Ontology column as it is now encoded
X = X.drop('Ontology',axis = 1)
# Join the encoded df
X = X.join(one_hot)
X.rename(
    columns={1: "Cell_comp", 2: "Mol_funcs", 0: "Bio_process"},
    inplace=True,
)

ont_based_prep = False
if ont_based_prep:  # Standard scaling that differentiate between ontologies
    if use_total_score:
        X = X.drop(['Theoretical_TS'], axis=1)  # Drop unneeded data
        X.loc[X["Cell_comp"] == 1.0, ['Total_score', 'Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X.loc[X["Cell_comp"] == 1.0, ['Total_score', 'Inf_content', 'Int_confidence', 'GScore']]))
        X.loc[X["Mol_funcs"] == 1.0, ['Total_score', 'Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(
                X.loc[X["Mol_funcs"] == 1.0, ['Total_score', 'Inf_content', 'Int_confidence', 'GScore']]))
        X.loc[X["Bio_process"] == 1.0, ['Total_score', 'Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(
                X.loc[X["Bio_process"] == 1.0, ['Total_score', 'Inf_content', 'Int_confidence', 'GScore']]))
    else:
        X = X.drop(['Total_score', 'Theoretical_TS'], axis=1)  # Drop unneeded data
        X.loc[X["Cell_comp"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(
                X.loc[X["Cell_comp"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']]))
        X.loc[X["Mol_funcs"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(
                X.loc[X["Mol_funcs"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']]))
        X.loc[X["Bio_process"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(
                X.loc[X["Bio_process"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']]))
else:
    if use_total_score:
        X = X.drop(['Theoretical_TS'], axis=1)  # Drop unneeded data
        X[['Total_score', 'Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(X[['Total_score', 'Inf_content', 'Int_confidence', 'GScore']]))
    else:
        X = X.drop(['Total_score', 'Theoretical_TS'], axis=1) # Drop unneeded data
        X[['Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(X[['Inf_content', 'Int_confidence', 'GScore']]))

X_shape0 = X.shape[0]


print(X.head())
print(Y.head())
#report = pp.ProfileReport(X)
#report.to_file('profile_report.html')

# Data analysis
#pp.ProfileReport(data)
#plt.figure(figsize = (15,15))
#cor_matrix = X.corr()
#sns.heatmap(cor_matrix,annot=True)
#plt.show()

# test_percentage = 15 #% of the data that will be used for testing purposes
# test_quota = round(X.shape[0]*0.01*test_percentage)
# X_test = X[-test_quota:]
# Y_test = Y[-test_quota:]
# X_train = X[:test_quota]
# Y_train = Y[:test_quota]

val_percentage = 15 #% of the data that will be used for validation purposes
test_percentage = 15 #% of the data that will be used for testing purposes
val_quota = round(X.shape[0]*0.01*val_percentage)
train_quota = round(X.shape[0]*0.01*test_percentage)
X_val = X[-val_quota:]
Y_val = Y[-val_quota:]
X_test = X[-val_quota-train_quota:-val_quota-1]
Y_test = Y[-val_quota-train_quota:-val_quota-1]
X_train = X[:-val_quota-train_quota-1]
Y_train = Y[:-val_quota-train_quota-1]

print(X_train.info())
print(X_train.head())
print(Y_train.info())
print(Y_train.head())
del data #Once the data has been divided and preprocessed the original dataframes are not needed anymore

gc.collect() #Free the RAM from the deleted dataframes

batch_size = 100
drop_out = 0.2
usual_use_bias = True
last_use_bias = True  # Force the last regular layer to not use bias to have a better representation of the lat_space
reg_l1 = 0.01
reg_l2 = 0.01
# kernel_initializer = "he_normal"#

leaky_alpha = 0.0
model_ID = 11
initialization_type = 'uniform'
future_net = model_selection.retrieve_model(model_ID, X.shape[1], usual_use_bias, last_use_bias, leaky_alpha, reg_l1, reg_l2,
                                           drop_out,  initialization_type)
keras.utils.plot_model(future_net, path + f"/future_net_structure_{formatted_datetime}.png", show_shapes=True)


# ae_input = Input(shape=(6,))
# act1 = Activation(activations.relu)
#future_net.add(keras.Input(shape=(7,)))
#x = tf.Variable(tf.ones((1, 7)), dtype=tf.float32)
#y = future_net(x)
x = tf.ones([batch_size, X_train.shape[1]], dtype = tf.float32)
y = future_net(x)
future_net.summary() #Summary structure of the model


# Prepare the training dataset.
buffer_quota = 55 #% of the training data that will be stored at a time
buffer_size = round(X_train.shape[0]*0.01*buffer_quota)
print('The buffer size is: '+ str(buffer_size))
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
# train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality()).batch(batch_size, drop_remainder=True)
#Dropping the remainder will allow for a faster overall speed because the program can be tailored to accomodate only
#for one kind of data shape



# # Prepare the test dataset.
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
# test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

# Instantiate an optimizer.
initial_learning_rate = 0.01
# initial_learning_rate * decay_rate ^ (step / decay_steps)
#staircase = True makes (step / decay_steps) an integer division
decay_steps = 250
decay_rate = 0.98
staircase = True
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( # This scheduler provides a decreasing learning rate
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate, # Decreased by 2% every 1000 steps
    staircase=staircase)


opti_type = 'Adam'
if opti_type == 'Adam':
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
elif opti_type == 'SGD_momentum_nest':
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True, momentum=0.9)
elif opti_type == 'SGD_momentum':
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=False, momentum=0.9)
elif opti_type == 'SGD_pure':
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
elif opti_type == 'Adadelta':
    optimizer = keras.optimizers.Adadelta(learning_rate=hp_learning_rate)
elif opti_type == 'Adadelta_fixed':
    optimizer = keras.optimizers.Adadelta(learning_rate=1.0)
else:
    print('Unrecognized optimizer!')

weight = {0: 1.0, 1: 4.0}
apply_balance = True
alpha = 0.
gamma = 2.0
focal = False
smoothing = 0.0
loss_fn = loss_wrap(class_weights=weight, apply_balance=apply_balance, alpha=alpha, gamma=gamma, focal=focal,
                    smoothing=smoothing)
future_net.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=[keras.metrics.BinaryAccuracy()])

# Save the best models
# checkpoint_filepath = '/good_models/checkpoint.model.keras'
# Save the best models
checkpoint_filepath = '/data/gabriele.gradara/reti/good_models/'

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-4,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='loss',
        mode='min',
        save_best_only=True)
]

# Prepare the metrics.
last_time = time.time()-start_time
# TODO: Rivalutare l'uso delle metriche per avere strumenti di valutazione più appropriati
# (guarda come fanno nell'unsupervised learning)
max_epochs = 20
loss_list = []
loss_val_list = []
step_list = []
step_val_list = []

count_1 = np.count_nonzero(Y_train == 1)
count_0 = np.count_nonzero(Y_train == 0)
weight_1 = max(round(count_0/count_1)-1, 1)
print(weight_1)
# class_weight = {0: 1.,
#                 1: 1.}

# Convertion to arrray and then to tensor
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_val = np.asarray(X_val)
Y_val = np.asarray(Y_val)
# Balanced batches to reduce the impact of the minority class
# It may be worth it to reduce the mismatch in size using other means
# Y_train = keras.utils.to_categorical(Y_train, 2)

# sm1 = SMOTE(sampling_strategy=0.5, random_state=seed, k_neighbors=8)
# sm1 = SMOTE(sampling_strategy=0.9, random_state=seed, k_neighbors=3)
# sme = SMOTEENN(random_state=seed, smote=sm1)
sme = SMOTE(sampling_strategy=0.9, random_state=seed, k_neighbors=20)
X_train_res, Y_train_res = sme.fit_resample(X_train, Y_train)
# X_val_res, Y_val_res = sme.fit_resample(X_val, Y_val)


count_1_tr_res = np.count_nonzero(Y_train_res == 1)
count_0_tr_res = np.count_nonzero(Y_train_res == 0)
print('count 1 _tr_res :' + str(count_1_tr_res))
print('count 0 _tr_res :' + str(count_0_tr_res))



train_dataset = tf.data.Dataset.from_tensor_slices((X_train_res, Y_train_res))
train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality())
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)


future_net.compile(
    optimizer=optimizer,  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.BinaryAccuracy()],
)

print(X_train)
print(Y_train)

# weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train_res), y=Y_train_res)
# weights = {i : weights[i] for i in range(len(np.unique(Y_train_res)))}
# weights ={0: 0.7, 1: 4.0}
print('weights')
print(weight)

# history = future_net.fit(training_generator, epochs=max_epochs, validation_data=validation_generator,
#                          callbacks=callbacks, verbose=1, class_weight=weights)
val_acc_metric = keras.metrics.BinaryAccuracy()

checkpoint_filepath = '/data/gabriele.gradara/reti/future_net_models/'
if use_total_score:
    dir_name = f"future_net_models_{formatted_datetime}/"
else:
    dir_name = f"future_net_models_{formatted_datetime}/"
current_checkpoint_path = os.path.join(checkpoint_filepath, dir_name)
os.mkdir(current_checkpoint_path)

max_patience = 10
patience = 0
best_f1_score = 0.0
f1_list = []
threshold_list = []
for epoch in range(1, max_epochs+1):
    epoch_time = time.time()
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
    val_loss = val_step(X_val, Y_val)
    # To avoid overfitting we compute f1 score periodically to stop the training if is not improved
    f1_temp, threshold = f1_extraction(X_val, future_net)
    f1_list.append(f1_temp)
    threshold_list.append(threshold)
    f1_message = f"F1 score {f1_temp} at threshold: {threshold}, at epoch {epoch - 1} in {time.time()-epoch_time}s"
    funcs_15.write_to_file(file_name, f1_message)
    print(f1_message)
    if f1_temp > best_f1_score:
        best_f1_score = f1_temp
        patience = 0
        if use_total_score:
            model_save_name = (current_checkpoint_path + 'future_net_' + str(epoch) + 'of'
                               + str(max_epochs+1) + '_' + formatted_datetime + '_tsu.keras')
        else:
            model_save_name = (current_checkpoint_path + 'future_net_' + str(epoch) + 'of'
                               + str(max_epochs+1) + '_' + formatted_datetime + '.keras')
        future_net.save(model_save_name)
    else:
        patience += 1
        if patience > max_patience:
            print(f'Best model achieved in {epoch - 1} epochs')
            break

val_acc = val_acc_metric.result()
val_acc_metric.reset_states()


plt.figure(figsize=(8, 8))
plt.plot(range(len(f1_list)), f1_list, label="F1 score")
plt.plot(range(len(threshold_list)), threshold_list, label="Threshold")
plt.xlabel("Epochs")
plt.ylabel("")
plt.title("F1 score and threshold plot")
# Save the plot as an image file (e.g., PNG format)
plt.savefig(f1_thresh_graph_name)
plt.close()

# history = future_net.fit(X_train_res, Y_train_res, epochs=max_epochs, validation_data=(X_val_res, Y_val_res),
#                          callbacks=callbacks, verbose=1)

print('count 1 _tr_res :' + str(count_1_tr_res))
print('count 0 _tr_res :' + str(count_0_tr_res))

# history = future_net.fit(X_train, Y_train, batch_size=batch_size, epochs=max_epochs, validation_split=0.3,
#                          callbacks=callbacks, class_weight=class_weight, ra)

test_scores = future_net.evaluate(X_test, Y_test, verbose=1)

print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
print('count 1 :' + str(count_1))
print('count 0 :' + str(count_0))
print('weight 1 :' + str(weight_1))
y_pred = future_net.predict(X_test, batch_size=batch_size)
print(y_pred)

# y_pred = (y_pred >= 0.5).astype(int)

thresholds = np.arange(0, 1, 0.005)
scores = [f1score_sklearn(Y_test, to_labels(y_pred, t), zero_division=0.0) for t in thresholds]
ix = np.argmax(scores)
best_threshold_test = thresholds[ix]
best_f1_test = scores[ix]
test_message = f"Best f1 score test: {best_f1_test}, best threshold {best_threshold_test}"
print(test_message)

y_pred = (y_pred >= best_threshold_test).astype(int)
print(classification_report(Y_test, y_pred))
print('test F1 score: ' + str(f1_score(Y_test, y_pred)))

funcs_15.write_to_file(file_name, "Classification of the test dataset: \n")
funcs_15.write_to_file(file_name, str(classification_report(Y_test, y_pred)))
funcs_15.write_to_file(file_name, '\ntest F1 score: ' + str(f1_score(Y_test, y_pred)))

# Full file evaluation
test_scores_full = future_net.evaluate(X, Y, verbose=1)
print("FULL loss:", test_scores_full[0])
print("FULL accuracy:", test_scores_full[1])
y_pred_full = future_net.predict(X, batch_size=batch_size)
# print(y_pred_full)
np.savetxt(f"preds_future_net/y_pred_pre_full_{formatted_datetime}.txt", y_pred_full)

count_1_full = np.count_nonzero(Y == 1)
count_0_full = np.count_nonzero(Y == 0)
weight_1_full = max(round(count_0_full/count_1_full)-1, 1)
print('FULL count 1 :' + str(count_1_full))
print('FULL count 0 :' + str(count_0_full))
print('FULL weight 1 :' + str(weight_1_full))

thresholds = np.arange(0, 1, 0.005)
scores = [f1score_sklearn(Y, to_labels(y_pred_full, t), zero_division=0.0) for t in thresholds]
ix = np.argmax(scores)
best_threshold = thresholds[ix]
best_f1 = scores[ix]
total_message = f" Best f1 score: {best_f1}, best threshold {best_threshold}"
print(total_message)


y_pred_full = (y_pred_full >= best_threshold).astype(int)

funcs_15.write_to_file(file_name, "Classification of all the dataset used for training: \n")
funcs_15.write_to_file(file_name, str(classification_report(Y, y_pred_full)))
funcs_15.write_to_file(file_name, '\ntest F1 score: ' + str(f1_score(Y, y_pred_full)))
# np.savetxt(f"preds_future_net/y_pred_post_{formatted_datetime}.txt", y_pred)
# print(y_pred_full)
# Print the description of the network in a file to remember and understand better the output
description_file = path + f"/future_net_description_{formatted_datetime}.txt"

network_description = ["Future net", "seed: " + str(seed),
                       "\nmodel ID: " + str(model_ID),
                       "\nDB size:" + str(X_shape0),
                       "\nvalidation percentage: " + str(val_percentage),
                       "\ntest percentage: " + str(test_percentage),
                       "\nbatch size: " + str(batch_size),
                       "\noptimizer: " + opti_type,
                       "\nont_based_prep size: " + str(ont_based_prep),
                       "\ndrop out: " + str(drop_out), "\nleaky alpha: " + str(leaky_alpha),
                       "\nusual use bias: " + str(usual_use_bias), "\nlast use bias: " + str(last_use_bias),
                       "\nl1 regularization: " + str(reg_l1), "\nl2 regularization: " + str(reg_l2),
                       "\nbuffer quota: " + str(buffer_quota),
                       "\nepochs: " + str(max_epochs),
                       "\npatience: " + str(max_patience),
                       "\ninitial learning rate: " + str(initial_learning_rate),
                       "\ndecay steps: " + str(decay_steps), "\ndecay rate: " + str(decay_rate),
                       "\nstaircase: " + str(staircase), "\nused dataset: " + oa_path, "\n"
                        "\nTest scores: " + str(test_scores[0]), "\nTest accuracy:: " + str(test_scores[1]),
                       "\ncount 1 :" + str(count_1), '\nweight 1 :' + str(weight_1),
                       ]

funcs_15.write_to_file(description_file, network_description)
funcs_15.write_to_file(description_file, test_message)
funcs_15.write_to_file(description_file, total_message)
funcs_15.write_to_file(file_name, test_message)
funcs_15.write_to_file(file_name, total_message)


# Plot thresholds, requires saved data

# with open(description_file, 'a+') as f:
#     future_net.summary(print_fn=lambda x: f.write(x + '\n'))
#
# y_true = np.loadtxt("preds_future_net/y_true.txt")
# # y_pred = np.loadtxt("preds_future_net/y_pred_pre_full_2024-02-15_allvars.txt")
# y_pred = np.loadtxt("preds_future_net/y_pred_pre_full_2024-03-18_10-20-59.txt")
# # y_pred = np.loadtxt("preds_future_net/y_pred_pre_full_2024-02-13_15-51-51.txt")
# # y_pred = np.loadtxt("preds_future_net/y_pred_pre_full_standard_InfCon.txt")
#
# print(y_pred)
# print(y_true)
# print('y_pred: ' + str(len(y_pred)) + '   y_true: ' + str(len(y_true)))
#
#
# plt.figure(figsize=(8, 8))
# plt.plot(thresholds, scores, marker='.', label='Thresholding')
# plt.scatter(thresholds[ix], scores[ix], marker='o', color='red', label='Best')
# # axis labels
# plt.xlabel('Thresholds')
# plt.ylabel('Scores')
# plt.legend()
# # show the plot
# plt.show()
# plt.savefig(f'preds_future_net/thresholding_{formatted_datetime}.png')
# plt.close()
# print(f"Elapsed time: {time.time()-start_time}s")

