import funcs_15
import model_selection
import csv
import gc
import json
import tensorflow as tf
import keras
import sys
import re
import os
import math
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras import regularizers
from keras import layers
# from keras import ops
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colormaps
import altair as alt
import pymongo
from tqdm import *

# Script that trains the Main Net given specific hyperparameters, the use of a tuner is recommended to find the best hyperparameters.
# Using the ones in the thesis is also an option.

# Adding the characters GO: to allow for the search
def addgo(code) -> str: 'GO:' + str(code)


def kfold_selection(epoch, train_dataset0, val_dataset0, train_dataset1, val_dataset1, train_dataset2, val_dataset2,
                    train_dataset3, val_dataset3, train_dataset4, val_dataset4, train_dataset5, val_dataset5):
    remainder = epoch % 6
    if remainder == 0:
        return train_dataset0, val_dataset0
    elif remainder == 1:
        return train_dataset1, val_dataset1
    elif remainder == 2:
        return train_dataset2, val_dataset2
    elif remainder == 3:
        return train_dataset3, val_dataset3
    elif remainder == 4:
        return train_dataset4, val_dataset4
    elif remainder == 5:
        return train_dataset5, val_dataset5
    else:
        print('ERROR: Unexpected k-fold selection!!!')
        return train_dataset0, val_dataset0


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

current_datetime = datetime.now()  # Get the current date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")  # Format the date and time as a string

use_total_score = True  # True: use it for training, False: ignore it
# global multiplier
multiplier = 1.0
filtered = re.sub('[.,]', '_', str(multiplier))
mult_sign = 'm' + str(filtered)

if use_total_score:
    dir_name = f"main_net_{formatted_datetime}_{mult_sign}_tsu"
else:
    dir_name = f"main_net_{formatted_datetime}_{mult_sign}"
# Make the directory that contains all the reports
parent_path = f"/data/gabriele.gradara/reti/reports"
path = os.path.join(parent_path, dir_name)
os.mkdir(path)
sources_dir_path = os.path.join(path, "sources")
os.mkdir(sources_dir_path)
misc_graphs_dir_path = os.path.join(path, "misc_graphs")
os.mkdir(misc_graphs_dir_path)

if use_total_score:
    file_name = sources_dir_path + f"/report_{formatted_datetime}_{mult_sign}_tsu.txt"  # Use the formatted date and time as a file name
    abridged_file_name = f"reports/" + dir_name + f"/abridged_report_{formatted_datetime}_{mult_sign}_tsu.txt"  # Use the formatted date and time as a file name
    graph_name = f"reports/" + dir_name + f"/graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
else:
    file_name = sources_dir_path + f"/report_{formatted_datetime}_{mult_sign}.txt"  # Use the formatted date and time as a file name
    abridged_file_name = f"reports/" + dir_name + f"/abridged_report_{formatted_datetime}_{mult_sign}.txt"  # Use the formatted date and time as a file name
    graph_name = f"reports/" + dir_name + f"/graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name

memory_save_mode = True  # Report less info
# Usare la gt specializzata per lo scopo accelera il training
# gt_path = 'csv/groundtruth_sorted.csv'  # groundtruth path
# oa_path = "csv/output_argot_sampled_1000000_addedgos.csv"  # output argot path

# oa_path = "csv/simgics_annotated1000_prediction_quality.csv"  #output argot path
# oa_path = "csv/simgics_annotated1000000_prediction_quality.csv"  #output argot path
oa_path = "csv/simgics_annotated17408776_prediction_quality.csv"  #output argot path

# Establish connection to mongodb database
# client = pymongo.MongoClient('localhost', 27017)
# db = client['ARGOT3']
# simgic_collection = db['simgic_big']

data = pd.read_csv(oa_path)
print(data.head())  # This function returns the first n rows for the object based on position.
print(data.info())  # It is useful for quickly testing if your object has the right type of data in it.
X = data.drop(columns=['SeqID', 'GOID', 'prediction_quality', 'Unnamed: 11'], inplace=False)

ont_based_prep = True
if ont_based_prep:  # Standard scaling that differentiate between ontologies
    if use_total_score:
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
        X = X.drop(['Total_score'], axis=1)  # Drop unneeded data
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
        X[['Total_score', 'Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(X[['Total_score', 'Inf_content', 'Int_confidence', 'GScore']]))
    else:
        X = X.drop(['Total_score'], axis=1)  # Drop unneeded data
        X[['Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(X[['Inf_content', 'Int_confidence', 'GScore']]))

y_prediction_quality = data['prediction_quality'] ## Flag for SMOTE
X_shape0 = X.shape[0]
X_shape1 = X.shape[1]

# big_dataset_threshold = 2000000
# if X_shape0 > big_dataset_threshold:
#     big_dataset_mode = True
# else:
#     big_dataset_mode = False

print(X.head())
# print(Y.head())
len_data = len(data)

# del data
# gc.collect()

# del data #Once the data has been divided and preprocessed the original dataframes are not needed anymore
# report = pp.ProfileReport(X)
# report.to_file('profile_report.html')

# Data analysis
# pp.ProfileReport(data)
# plt.figure(figsize = (15,15))
# cor_matrix = X.corr()
# sns.heatmap(cor_matrix,annot=True)
# plt.show()

batch_size = 50
drop_out = 0.25
leaky_alpha = 0.15
usual_use_bias = True
last_use_bias = False  # Force the last regular layer to not use bias to have a better representation of the lat_space
reg_l1 = 0.005
reg_l2 = 0.005
model_ID = 11
use_MAE = False
initialization_type = 'uniform'

autoencoder = model_selection.retrieve_model(model_ID, X_shape1-1, usual_use_bias, last_use_bias, leaky_alpha, reg_l1,
                                             reg_l2, drop_out,
                                             initialization_type=initialization_type, seed=seed)

keras.utils.plot_model(autoencoder, path + f"/autoencoder_structure_{formatted_datetime}_{mult_sign}.png",
                       show_shapes=True)

# ae_input = Input(shape=(6,))
# act1 = Activation(activations.relu)
# autoencoder.add(keras.Input(shape=(7,)))
# x = tf.Variable(tf.ones((1, 7)), dtype=tf.float32)
# y = autoencoder(x)
x = tf.ones([batch_size, X_shape1-1], dtype=tf.float32)
y = autoencoder(x)
autoencoder.summary()  # Summary structure of the model

# Instantiate an optimizer.
initial_learning_rate = 0.01
# initial_learning_rate * decay_rate ^ (step / decay_steps)
# staircase = True makes (step / decay_steps) an integer division
decay_steps = 5000
decay_rate = 0.98
staircase = True
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  # This scheduler provides a decreasing learning rate
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,  # Decreased by 2% every 1000 steps
    staircase=staircase)

opti_type = 'Adam'
if opti_type == 'Adam':
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# elif opti_type == 'SGD_momentum_nest':
#     optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True, momentum=0.9)
# elif opti_type == 'SGD_momentum':
#     optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=False, momentum=0.9)
# elif opti_type == 'SGD_pure':
#     optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
# elif opti_type == 'Adadelta':
#     optimizer = keras.optimizers.Adadelta(learning_rate=hp_learning_rate)
# elif opti_type == 'Adadelta_fixed':
#     optimizer = keras.optimizers.Adadelta(learning_rate=1.0)
# else:
#     print('Unrecognized optimizer!')

autoencoder.compile(
    optimizer=optimizer,
    loss=funcs_15.loss_MSE_modded,
    metrics=[],
)

n_bins = 20

# Save the best models
checkpoint_filepath = '/good_models/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_best_only=True)

callbacks = tf.keras.callbacks.CallbackList([])
callbacks.append(model_checkpoint_callback)

# Print the description of the network in a file to remember and understand better the output
if use_total_score:  # tsu: Total Score Used
    description_file = path + f"/description_{formatted_datetime}_{mult_sign}_tsu.txt"
    losses_file = sources_dir_path + f"/losses_{formatted_datetime}_{mult_sign}_tsu.txt"
    val_losses_file = sources_dir_path + f"/val_losses_{formatted_datetime}_{mult_sign}_tsu.txt"
    test_losses_file = sources_dir_path + f"/test_losses_{formatted_datetime}_{mult_sign}_tsu.txt"
else:
    description_file = path + f"/description_{formatted_datetime}_{mult_sign}.txt"
    losses_file = sources_dir_path + f"/losses_{formatted_datetime}_{mult_sign}.txt"
    val_losses_file = sources_dir_path + f"/val_losses_{formatted_datetime}_{mult_sign}.txt"
    test_losses_file = sources_dir_path + f"/test_losses_{formatted_datetime}_{mult_sign}.txt"

progress_log_name = f"progress_logs_and_errors/progress_log_{formatted_datetime}_{mult_sign}.txt"

# Save the best models
checkpoint_filepath = '/data/gabriele.gradara/reti/good_models/'
if use_total_score:
    dir_name = f"main_net_models_{formatted_datetime}"
else:
    dir_name = f"main_net_models_{formatted_datetime}"
current_checkpoint_path = os.path.join(checkpoint_filepath, dir_name)
os.mkdir(current_checkpoint_path)

# Store all the information in a variable before splitting the original for testing purposes
test_percentage = 15  # % of the data that will be used for testing purposes
test_quota = round(X_shape0 * 0.01 * test_percentage)

# Prepare the training dataset.
val_percentage = 15  # % of the data that will be used for validation purposes
if X_shape0 > 1100000:
    big_dataset = True
    buffer_quota = 55
else:
    big_dataset = False
    buffer_quota = 100

print('Big dataset: ' + str(big_dataset))
buffer_size = round(X_shape0 * 0.01 * buffer_quota * (1 - (test_percentage / 100)))  # FARE LA SIZE = alle dimensioni della mega batch
print('The buffer size is: ' + str(buffer_size))
# n_folds = 6  # 85/6 = 14.17%
# kfold_val_percentage = (100-test_percentage)/n_folds
# kfold_val_quota = round(X_shape0*0.01*kfold_val_percentage)
val_quota = round(X_shape0 * 0.01 * val_percentage)

# X_test = X[-test_quota:]
# Y_test = Y[-test_quota:]
# X_val = X[-val_quota-test_quota:-test_quota-1]
# Y_val = Y[-val_quota-test_quota:-test_quota-1]
# X_train = X[:-val_quota-test_quota-1]
# Y_train = Y[:-val_quota-test_quota-1]

resample_type = 0  # 0: no o/u, 1 only over, 2 only under, 3 over+under, 4 SMOTE
over_sampling_strat = 0.5
under_sampling_strat = 0.9
smote_sampling_strat = 0.9
smote_k_neighbors = 20

if resample_type == 0:
    resample_description = 'no resampling'
elif resample_type == 1:
    resample_description = 'random oversampler'
elif resample_type == 2:
    resample_description = 'random undersampler'
elif resample_type == 3:
    resample_description = 'random oversampler + random undersampler'
elif resample_type == 4:
    resample_description = 'SMOTENC'




# Prepare the metrics.
last_time = time.time() - start_time
print("Time to start: %.4f" % (last_time))
# TODO: Rivalutare l'uso delle metriche per avere strumenti di valutazione più appropriati
# (guarda come fanno nell'unsupervised learning)
max_epochs = 25
max_patience = 10
loss_list = []
simgics_list = []
simgics_val_list = []
loss_val_list = []
step_list = []
step_val_list = []
best_loss = 1000
epoch_val_list = []
epoch_train_list = []
mean_val_list = []
mean_train_list = []

network_description = ["seed: " + str(seed),
                       "\nmodel ID: " + str(model_ID),
                       "\nDB size:" + str(X_shape0),
                       "\nvalidation percentage: " + str(val_percentage),
                       "\ntest percentage: " + str(test_percentage),
                       "\nbatch size: " + str(batch_size),
                       "\noptimizer: " + opti_type,
                       "\nont_based_prep size: " + str(ont_based_prep),
                       "\nuse total score: " + str(use_total_score),
                       "\ndrop out: " + str(drop_out), "\nleaky alpha: " + str(leaky_alpha),
                       "\nusual use bias: " + str(usual_use_bias), "\nlast use bias: " + str(last_use_bias),
                       "\nl1 regularization: " + str(reg_l1), "\nl2 regularization: " + str(reg_l2),
                       "\nbuffer quota: " + str(buffer_quota),
                       "\nmultiplier: " + str(multiplier),
                       "\nMAE: " + str(use_MAE),
                       "\nresample_type: " + resample_description,
                       "\noversampling strategy: " + str(over_sampling_strat),
                       "\nundersampling strategy: " + str(under_sampling_strat),
                       "\nSmote k_neighbors: " + str(smote_k_neighbors),
                       "\nepochs: " + str(max_epochs),
                       "\npatience: " + str(max_patience),
                       "\ninitial learning rate: " + str(initial_learning_rate),
                       "\ndecay steps: " + str(decay_steps), "\ndecay rate: " + str(decay_rate),
                       "\nstaircase: " + str(staircase), "\nused dataset: " + oa_path, "\n"]
funcs_15.write_to_file(description_file, network_description)

if resample_type != 0:
    X['simgics_eval'] = data['simgics_eval']

    (train_dataset0, val_dataset0, train_dataset1, val_dataset1, train_dataset2, val_dataset2, train_dataset3,
     val_dataset3, train_dataset4, val_dataset4, train_dataset5,
     val_dataset5) = funcs_15.kfold_subdivision_resampled(X, y_prediction_quality, description_file, batch_size,
                                                          test_percentage, resample_type, over_sampling_strat,
                                                          under_sampling_strat, smote_sampling_strat, smote_k_neighbors,
                                                          buffer_size, seed)


    Y = X['simgics_eval']
    X = X.drop(['simgics_eval'], axis=1)
    X_test = X[-test_quota:]
    Y_test = Y[-test_quota:]
else:
    Y = X['simgics_eval']
    X = X.drop(['simgics_eval'], axis=1)
    X_test = X[-test_quota:]
    Y_test = Y[-test_quota:]
    (train_dataset0, val_dataset0, train_dataset1, val_dataset1, train_dataset2, val_dataset2, train_dataset3,
     val_dataset3, train_dataset4, val_dataset4, train_dataset5,
     val_dataset5) = funcs_15.kfold_subdivision(X, Y, batch_size, test_percentage, buffer_size)

# Y = X['simgics_eval']
# X = X.drop(['simgics_eval'], axis=1)
# X_test = X[-test_quota:]
# Y_test = Y[-test_quota:]

n_chart_predictions = 2000
X_chart_pred = X_test.head(n_chart_predictions)  # Points in the chart


del data
del X
del Y
gc.collect()

last_time = time.time() - start_time
print("Time for data preprocessing: %.4f" % (last_time))

with open(description_file, 'a+') as f:
    autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))


funcs_15.write_to_file(abridged_file_name, "Mean validation loss:\n")
with open(progress_log_name, "a+") as f:
    for epoch in tqdm(range(max_epochs), desc="Epochs", unit="epoch", file=f):
        train_loss_this_epoch = []
        funcs_15.write_to_file(file_name, ["", "\nStart of epoch %d" % (epoch,)])
        start_epoch_time = time.time()

        # K-fold cross-validation
        # if big_dataset_mode:
        #     current_train_dataset, current_val_dataset = kfold_selection_light(epoch, train_dataset0, val_dataset0,
        #                                                                  train_dataset1, val_dataset1, train_dataset2,
        #                                                                  val_dataset2, train_dataset3, val_dataset3,
        #                                                                  train_dataset4, val_dataset4, train_dataset5,
        #                                                                  val_dataset5)
        # else:
        current_train_dataset, current_val_dataset = kfold_selection(epoch, train_dataset0, val_dataset0,
                                                                     train_dataset1, val_dataset1, train_dataset2,
                                                                     val_dataset2, train_dataset3, val_dataset3,
                                                                     train_dataset4, val_dataset4, train_dataset5,
                                                                     val_dataset5)
        # TRAINING
        enumerated_train_batches = tqdm(
            enumerate(current_train_dataset),
            desc="Batch",
            position=1,
            leave=False)
        for step, (x_batch_train, y_batch_train) in enumerated_train_batches:
            # print(x_batch_train)
            # print(y_batch_train)
            simgics_list.append(y_batch_train.numpy())
            y_batch_train = tf.cast(y_batch_train, tf.float32)
            if use_MAE:
                loss_batch = funcs_15.training_step_MAE(x_batch_train, y_batch_train, optimizer, autoencoder, multiplier)
            else:
                loss_batch = funcs_15.training_step_MSE(x_batch_train, y_batch_train, optimizer, autoencoder, multiplier)
            float_loss = float(loss_batch.numpy())
            loss_list.append(float_loss)
            train_loss_this_epoch.append(float_loss)
            if not step_list:
                step_list.append(1)
            else:
                last_value = step_list[-1]
                step_list.append(last_value + 1)
            # print("Seen so far: %s samples" % ((step + 1) * batch_size))
            # print("time required for this batch: %.4f" % (time.time()-last_time))
            data = []  # List used to write the report
            data.append("Training loss (for one batch) at step " + str(step) + ": " + str(float_loss))
            data.append(" | Seen so far: " + str((step + 1) * batch_size) + " samples ")
            data.append("| time required for this batch: " + str(time.time() - last_time))
            funcs_15.write_to_file(file_name, data)
            last_time = time.time()
        if not epoch_train_list:
            epoch_train_list.append(1)
        else:
            last_value = epoch_train_list[-1]
            epoch_train_list.append(last_value + 1)
        average_train_loss_this_epoch = sum(train_loss_this_epoch) / len(train_loss_this_epoch)
        mean_train_list.append(float(average_train_loss_this_epoch))

        # VALIDATION
        val_loss_this_epoch = []
        for step, (x_batch_val, y_batch_val) in enumerate(current_val_dataset):
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.

            simgics_val_list.append(y_batch_val.numpy())
            y_batch_val = tf.cast(y_batch_val, tf.float32)
            if use_MAE:
                loss_val = funcs_15.validation_step_MAE(x_batch_val, y_batch_val, autoencoder)
            else:
                loss_val = funcs_15.validation_step_MSE(x_batch_val, y_batch_val, autoencoder)
            float_loss_val = float(loss_val.numpy())
            val_loss_this_epoch.append(float_loss_val)
            loss_val_list.append(float_loss_val)
            if not step_val_list:
                step_val_list.append(1)
            else:
                last_value = step_val_list[-1]
                step_val_list.append(last_value + 1)
            # print("Seen so far: %s samples" % ((step + 1) * batch_size))
            # print("time required for this batch: %.4f" % (time.time()-last_time))
            data_val = []  # List used to write the report
            data_val.append("Validation loss (for one batch) at step " + str(step) + ": " + str(float_loss_val))
            data_val.append(" | Seen so far: " + str((step + 1) * batch_size) + " samples ")
            data_val.append("| time required for this batch: " + str(time.time() - last_time))
            funcs_15.write_to_file(file_name, data_val)
            last_time = time.time()
            # break  # For testing purposes, just one step
        if len(val_loss_this_epoch) != 0:  # If average loss cannot be computed there is no point in doing this
            average_val_loss_this_epoch = sum(val_loss_this_epoch) / len(val_loss_this_epoch)
            abridged_report_message = (f"Average validation loss for epoch {epoch}: {average_val_loss_this_epoch}. "
                                       f"Computed in {float(time.time() - start_epoch_time)}s")
            funcs_15.write_to_file(abridged_file_name, abridged_report_message)
            print(  # print info at every step
                "Average validation loss for epoch %d: %.4f. Computed in %.3fs"
                % (epoch, float(average_val_loss_this_epoch), float(time.time() - start_epoch_time))
            )
            if not epoch_val_list:
                epoch_val_list.append(1)
            else:
                last_value = epoch_val_list[-1]
                epoch_val_list.append(last_value + 1)
            mean_val_list.append(float(average_val_loss_this_epoch))
            if average_val_loss_this_epoch < best_loss:
                patience = 0  # If the model improved reset patience counter
                best_loss = average_val_loss_this_epoch
                # Save the name of the best model
                if use_total_score:
                    model_save_name = (current_checkpoint_path + '/autoencoder_' + str(epoch) + 'of'
                                       + str(max_epochs) + '_' + formatted_datetime + '_tsu.keras')
                else:
                    model_save_name = (current_checkpoint_path + '/autoencoder_' + str(epoch) + 'of'
                                       + str(max_epochs) + '_' + formatted_datetime + '.keras')
                autoencoder.save(model_save_name)
            else:
                patience += 1  # If we teach max patience break from the training and go to the next mega batch
                if patience == max_patience:  # Reduces overfit and improve generalization of the model
                    print('------   Patience exhausted, top model achieved!!!!   ------')
                    break
        else:
            val_length_message = '---  Length of val loss = 0, average not computed  ---'
            funcs_15.write_to_file(file_name, val_length_message)
            print(val_length_message)
            break

graph_loss_list = []
graph_simgics_list = []
for j in range(len(loss_list)):
    graph_loss_list.append(loss_list[j])
    graph_simgics_list.append(max(simgics_list[j]))
    funcs_15.write_to_file(losses_file, str(loss_list[j]) + ", " + str(max(simgics_list[j])))

graph_val_loss_list = []
graph_val_simgics_list = []
for j in range(len(loss_val_list)):
    graph_val_loss_list.append(loss_val_list[j])
    graph_val_simgics_list.append(max(simgics_val_list[j]))
    funcs_15.write_to_file(val_losses_file, str(loss_val_list[j]) + ", " + str(max(simgics_val_list[j])))

if use_total_score:  # tsu: Total Score Used
    train_graph_name = misc_graphs_dir_path + f"/train_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    train_mean_graph_name = misc_graphs_dir_path + f"/train_mean_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    val_graph_name = misc_graphs_dir_path + f"/val_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    val_epoch_name = misc_graphs_dir_path + f"/val_mean_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    simgics_graph_name = misc_graphs_dir_path + f"/simgics_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    # hex_simgics_name = f"reports/" + dir_name + f"/hex_simgics_graph_{formatted_datetime}_{mult_sign}_tsu.png"
    simgics_val_graph_name = misc_graphs_dir_path + f"/simgics_val_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    # hex_simgics_val_name = f"reports/" + dir_name + f"/hex_simgics_val_graph_{formatted_datetime}_{mult_sign}_tsu.png"
else:
    train_graph_name = misc_graphs_dir_path + f"/train_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    train_mean_graph_name = misc_graphs_dir_path + f"/train_mean_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    val_graph_name = misc_graphs_dir_path + f"/val_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    val_epoch_name = misc_graphs_dir_path + f"/val_mean_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    simgics_graph_name = misc_graphs_dir_path + f"/simgics_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    # hex_simgics_name = f"reports/" + dir_name + f"/hex_simgics_graph_{formatted_datetime}_{mult_sign}.png"
    simgics_val_graph_name = misc_graphs_dir_path + f"/simgics_val_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    # hex_simgics_val_name = f"reports/" + dir_name + f"/hex_simgics_val_graph_{formatted_datetime}_{mult_sign}.png"

funcs_15.plot_maker(step_list, loss_list, train_graph_name, xlabel="Steps", ylabel="Train Loss",
                    title="Train Loss Plot")
funcs_15.plot_maker(epoch_train_list, mean_train_list, train_mean_graph_name, xlabel="Epochs",
                    ylabel="Mean training Loss",
                    title="Mean training Loss Plot")
funcs_15.plot_maker(step_val_list, loss_val_list, val_graph_name, xlabel="Steps", ylabel="Validation Loss",
                    title="Validation Loss Plot")
funcs_15.plot_maker(epoch_val_list, mean_val_list, val_epoch_name, xlabel="Epochs", ylabel="Mean validation Loss",
                    title="Mean validation Loss Plot")

# print(len(simgics_list))

plt.figure(figsize=(10, 10))
plt.scatter(graph_loss_list, graph_simgics_list, alpha=0.04)
plt.xlabel("Loss")
plt.ylabel("Simgic")
plt.title("Simgic Plot")
plt.savefig(simgics_graph_name)  # Save the plot as an image file (e.g., PNG format)
plt.close()
# print(graph_loss_list)
# print(len(graph_loss_list))
xmin = min(graph_loss_list)
xmax = max(graph_loss_list)
ymin = min(graph_simgics_list)
ymax = max(graph_simgics_list)

# plt.figure(figsize=(8, 8))
# plt.hexbin(graph_loss_list, graph_simgics_list, bins='log', cmap=plt.cm.Greys, gridsize=50)
# plt.axis([xmin, xmax, ymin, ymax])
# plt.xlabel("Loss")
# plt.ylabel("Simgic")
# plt.title("Simgic Hex Plot")
# cb = plt.colorbar()
# cb.set_label('counts')
# plt.savefig(hex_simgics_name)  # Save the plot as an image file (e.g., PNG format)
# plt.close()

# Same plot but with validation loss
plt.figure(figsize=(10, 10))
plt.scatter(graph_val_loss_list, graph_val_simgics_list, alpha=0.04)
plt.xlabel("Validation Loss")
plt.ylabel("Simgic")
plt.title("Test Simgic Plot")
plt.savefig(simgics_val_graph_name)  # Save the plot as an image file (e.g., PNG format)
plt.close()

xmin = min(graph_val_loss_list)
xmax = max(graph_val_loss_list)
ymin = min(graph_val_simgics_list)
ymax = max(graph_val_simgics_list)

# plt.figure(figsize=(8, 8))
# plt.hexbin(graph_loss_list, graph_simgics_list, bins='log', cmap=plt.cm.Greys, gridsize=50)
# plt.axis([xmin, xmax, ymin, ymax])
# plt.xlabel("Validation Loss")
# plt.ylabel("Simgic")
# plt.title("Val Simgic Hex Plot")
# cb = plt.colorbar()
# cb.set_label('counts')
# plt.savefig(hex_simgics_val_name)  # Save the plot as an image file (e.g., PNG format)
# plt.close()


print("Time taken: %.2fs" % (time.time() - start_time))

# TESTING
best_autoencoder = tf.keras.models.load_model(model_save_name,
                                              custom_objects={'loss_MSE_modded': funcs_15.loss_MSE_modded})
# Show the model architecture
best_autoencoder.summary()

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

test_loss_list = []
simgics_test_list = []
step_test_list = []

funcs_15.write_to_file(abridged_file_name, "\nMean test loss:\n")
start_test_time = time.time()
for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
    # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.

    simgics_test_list.append(y_batch_test.numpy())
    y_batch_test = tf.cast(y_batch_test, tf.float32)
    # loss_test = funcs_15.validation_step_bskc(x_batch_test, simgics, autoencoder, z_loss)
    if use_MAE:
        loss_test = funcs_15.validation_step_MAE(x_batch_test, y_batch_test, autoencoder)
    else:
        loss_test = funcs_15.validation_step_MSE(x_batch_test, y_batch_test, autoencoder)
    float_loss_test = float(loss_test.numpy())
    test_loss_list.append(float_loss_test)
    if not step_test_list:
        step_test_list.append(1)
    else:
        last_value = step_test_list[-1]
        step_test_list.append(last_value + 1)
    # print("Seen so far: %s samples" % ((step + 1) * batch_size))
    # print("time required for this batch: %.4f" % (time.time()-last_time))
    if not memory_save_mode:
        data_test = []  # List used to write the report
        data_test.append("Test loss (for one batch) at step " + str(step) + ": " + str(float_loss_test))
        data_test.append(" | Seen so far: " + str((step + 1) * batch_size) + " samples ")
        data_test.append("| time required for this batch: " + str(time.time() - last_time))
        funcs_15.write_to_file(file_name, data_test)
    last_time = time.time()

if len(test_loss_list) != 0:
    average_test_loss = sum(test_loss_list) / len(test_loss_list)
    funcs_15.write_to_file(file_name, "Average Test loss: " + str(average_test_loss))
    print(  # print info at every step
        "Average test loss: %.4f"
        % (float(average_test_loss))
    )
    abridged_report_message = (f"Average test loss : {average_test_loss}. "
                               f"Computed in {float(time.time() - start_test_time)}s")
    funcs_15.write_to_file(abridged_file_name, abridged_report_message)
    if use_total_score:
        test_graph_name = misc_graphs_dir_path + f"/test_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    else:
        test_graph_name = misc_graphs_dir_path + f"/test_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    funcs_15.plot_maker(step_test_list, test_loss_list, test_graph_name, xlabel="Steps", ylabel="Test Loss",
                        title="Test Loss Plot")
else:
    test_length_message = '---  Length of test loss = 0, average not computed  ---'
    funcs_15.write_to_file(file_name, test_length_message)
    print(test_length_message)

if use_total_score:
    train_val_graph_name = misc_graphs_dir_path + f"/train_val_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    train_mean_val_graph_name = misc_graphs_dir_path + f"/train_mean_val_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    combo_graph_name = misc_graphs_dir_path + f"/combo_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    simgics_test_graph_name = path + f"/simgics_test_graph_{formatted_datetime}_{mult_sign}_tsu.png"  # Use the formatted date and time as a file name
    # hex_simgics_test_name = f"reports/" + dir_name + f"/hex_simgics_test_graph_{formatted_datetime}_{mult_sign}_tsu.png"
    histo_simgic_loss_name = path + f"/histo_simgic_test_loss_{formatted_datetime}_{mult_sign}_tsu.png"
    histo_test_name = path + f"/histo_test_graph_{formatted_datetime}_{mult_sign}_tsu.png"
    histo_error_name = path + f"/histo_error_graph_{formatted_datetime}_{mult_sign}_tsu.png"
    histo_error_name_mass = path + f"/histo_error_graph_mass_{formatted_datetime}_{mult_sign}_tsu.png"
else:
    train_val_graph_name = misc_graphs_dir_path + f"/train_val_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    train_mean_val_graph_name = misc_graphs_dir_path + f"/train_mean_val_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    combo_graph_name = misc_graphs_dir_path + f"/combo_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    simgics_test_graph_name = path + f"/simgics_test_graph_{formatted_datetime}_{mult_sign}.png"  # Use the formatted date and time as a file name
    # hex_simgics_test_name = (f"reports/" + dir_name + f"/hex_simgics_test_graph_{formatted_datetime}_{mult_sign}.png")
    histo_simgic_loss_name = path + f"/histo_simgic_test_loss_{formatted_datetime}_{mult_sign}.png"
    histo_test_name = path + f"/histo_test_graph_{formatted_datetime}_{mult_sign}.png"
    histo_error_name = path + f"/histo_error_graph_{formatted_datetime}_{mult_sign}.png"
    histo_error_name_mass = path + f"/histo_error_graph_mass_{formatted_datetime}_{mult_sign}.png"

plt.figure(figsize=(10, 10))
plt.plot(step_list, loss_list, label="Training loss")
plt.plot(step_val_list, loss_val_list, label="Validation loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Validation and Train Loss plot")
# Save the plot as an image file (e.g., PNG format)
plt.savefig(train_val_graph_name)
plt.close()

plt.figure(figsize=(10, 10))
plt.plot(epoch_train_list, mean_train_list, label="Average mean training loss")
plt.plot(epoch_val_list, mean_val_list, label="Average mean validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Average validation and Train Loss plot")
# Save the plot as an image file (e.g., PNG format)
plt.savefig(train_mean_val_graph_name)
plt.close()

plt.figure(figsize=(10, 10))
plt.plot(step_list, loss_list, label="Training loss")
plt.plot(step_val_list, loss_val_list, label="Validation loss")
plt.plot(step_test_list, test_loss_list, label="Test loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Complete Loss plot")
# Save the plot as an image file (e.g., PNG format)
plt.savefig(combo_graph_name)
plt.close()

## Chart creation
hidden_encoder = Model(inputs=best_autoencoder.input, outputs=best_autoencoder.layers[-2].output)
hidden_encoder.summary()

hidden_chart_predictions = hidden_encoder.predict(X_chart_pred, batch_size=batch_size)
print(hidden_chart_predictions)
# print(hidden_chart_predictions)
pred_grades = best_autoencoder.predict(X_chart_pred, batch_size=batch_size)
X_chart_pred = [sublist[0] for sublist in hidden_chart_predictions]
Y_pred = [sublist[1] for sublist in hidden_chart_predictions]
pred_grades = [sublist[0] for sublist in pred_grades]
# print(pred_grades)
# print(X_chart_pred)
# print(Y_pred)
pred_dict = {'X': X_chart_pred, 'Y': Y_pred, 'Color': pred_grades}
pred_df = pd.DataFrame(pred_dict)
chart = alt.Chart(pred_df).mark_circle().encode(
    x='X',
    y='Y',
    color='Color'
)
if use_total_score:
    pred_name = 'charts/pred_chart_' + formatted_datetime + '_tsu.html'
    info_name = 'charts/pred_info_' + formatted_datetime + '_tsu.txt'
else:
    pred_name = 'charts/pred_chart_' + formatted_datetime + '.html'
    info_name = 'charts/pred_info_' + formatted_datetime + '.txt'

# Optional configuration for rectangle marks
chart = chart.configure_rect(width=200, height=150)
chart.save(pred_name)

funcs_15.write_to_file(info_name, 'pred_grades: ' + str(pred_grades))
funcs_15.write_to_file(info_name, 'X_chart_pred: ' + str(X_chart_pred))
funcs_15.write_to_file(info_name, 'Y_pred: ' + str(Y_pred))
##

graph_test_loss_list = []
graph_simgics_test_list = []
for j in range(len(test_loss_list)):
    graph_test_loss_list.append(test_loss_list[j])
    graph_simgics_test_list.append(max(simgics_test_list[j]))
    funcs_15.write_to_file(test_losses_file, str(test_loss_list[j]) + ", " + str(max(simgics_test_list[j])))

source_analysis = True
print('Run time: ' + str(time.time() - start_time))

funcs_15.confrontation_benchmark(graph_test_loss_list, graph_simgics_test_list, best_autoencoder, path,
                                 formatted_datetime, batch_size, big_dataset, simgic_threshold=0.75,
                                 confrontation_path_start=path,
                                 use_total_score=use_total_score, n_benchmark=10000, save_vars=True,
                                 ontological_analysis=True, ont_based_prep=ont_based_prep, source_analysis=True)

