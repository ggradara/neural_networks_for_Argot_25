import funcs_15
import model_selection
import csv
import gc
import json
import os
import tensorflow as tf
import keras
import keras_tuner
import pymongo
from keras_tuner import HyperParameters
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from keras import layers
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras import regularizers
import pandas as pd
import numpy as np
import time
import keras.backend as K
import tensorflow_probability as tfp
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colormaps
import altair as alt
from tqdm import *
from imblearn.over_sampling import SMOTENC
from tensorflow.keras.callbacks import EarlyStopping

# Tuner hyperband for the Main Net

def determine_value(row):
    if row['Cell_comp'] == 1:
        return 1
    elif row['Mol_funcs'] == 1:
        return 2
    elif row['Bio_process'] == 1:
        return 0
    else:
        print('|||||||||||||||||||||||||||||||||||||||')
        print('ONTOLOGY TYPE NOT FOUND!!!!!!!!!!!!!!!!')
        print('ERROR IN determine_value!!!!!!!!!!!!!!!!')
        print('|||||||||||||||||||||||||||||||||||||||')
        return 0


class HyperAutoencoder(keras_tuner.HyperModel): # Da decidere i nomi per ora teniamo model ne l codice
    def build(self, hp):
        # thorough_search = True
        # tune_model = False # Look for the best models
        # tune_hyper = False # Look for the best hyperparameters
        # input_width = 6
        # output = 1
        # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")  # Format the date and time as a string
        # file_name = f"tuning_reports/tuning_report_{formatted_datetime}.txt"  # Use the formatted date time as a name
        global tune_model
        global tune_hyper
        global tune_init
        global X_shape1
        global trial_number


        if tune_hyper:
            # Tune learning rate for Adam optimizer with values from 0.01, 0.001, or 0.0001
            leaky_alpha = hp.Choice("leaky_alpha", values=[0.0, 0.15, 0.3])
            reg_l1 = hp.Choice("reg_l1", values=[0.001, 0.01, 0.1])
            reg_l2 = hp.Choice("reg_l2", values=[0.001, 0.01, 0.1])
            drop_out = hp.Choice("drop_out", values=[0.1, 0.2, 0.3])
            # Add differentiation for the penultimate layer
            use_bias = hp.Boolean('use_bias', default=True)
            use_last_bias = hp.Boolean('use_last_bias', default=False)
        else:
            leaky_alpha = 0.1
            reg_l1 = 0.005
            reg_l2 = 0.005
            drop_out = 0.1
            use_bias = True
            use_last_bias = False
            hyper_description = ["\n\n\n Code:" + str(trial_number),
                                 "\nleaky alpha: " + str(leaky_alpha),
                                 "\nreg l1: " + str(reg_l1),
                                 "\nreg l2: " + str(reg_l2),
                                 "\ndrop out: " + str(drop_out),
                                 "\nuse bias: " + str(use_bias),
                                 "\nuse last bias: " + str(use_last_bias)]
            funcs_15.write_to_file(description_file, hyper_description)
        if tune_init:
            initializer = hp.Choice("initializer", values=['normal', 'uniform'])
        else:
            initializer = 'uniform'
            init_description = ["initializer: " + initializer]
            funcs_15.write_to_file(description_file, init_description)
        if tune_model:
            # model_ID = hp.Choice("model_ID", values=[0, 1, 2, 3, 4, 5, 6, 7])
            model_ID = hp.Choice("model_ID", values=[7, 9])
            model = model_selection.retrieve_model(model_ID, X_shape1-1, use_bias, use_last_bias, leaky_alpha, reg_l1,
                                                   reg_l2, drop_out, initializer)
        else:
            model_ID = 11
            model = model_selection.retrieve_model(model_ID, X_shape1-1, use_bias, use_last_bias, leaky_alpha, reg_l1,
                                                   reg_l2, drop_out, initializer)
            model_description = ["model ID: " + str(model_ID)]
            funcs_15.write_to_file(description_file, model_description)
        return model


    def fit(self, hp, model, X_train, y_prediction_quality, _val_dataset, _max_epochs, _file_name, _abridged_file_name,
            tune_batch, tune_SMOTE, tune_opti, tune_opti_type, patience, memory_save_mode, buffer_size, *args, **kwargs):

        def loss_MSE_modded(penalty, grades, multiplier):
            # xy_ref_min = [0., 0.]  # Minimum values for both axis
            # xy_ref_max = [20, 1]  # Maximum values for both axis
            # multiplier = 1.0 / multiplier  # I want to enlarge numbers 0<x<1 before using the square root

            penalty_reshaped = tf.reshape(penalty, [-1, 1])  # Reshape simgics
            # mask = tf.equal(penalty_reshaped, 1.)
            mask = tf.greater_equal(penalty_reshaped, 0.75)  # 0.75 is the requirement for the simgic to be considered good
            mask = tf.cast(mask, dtype=tf.float32)
            grades_reshaped = tf.reshape(grades, [-1, 1])  # Reshape grades
            # modded_val_notshaped = tf.stack([penalty_reshaped, grades_reshaped], axis=1)  # Stack the two vectors one aside the other
            # modded_val = tf.reshape(modded_val_notshaped, [-1, 2])
            loss_SE = K.square(penalty_reshaped - grades_reshaped)
            loss_SE_modded = tf.where(tf.cast(mask, dtype=tf.bool), loss_SE * multiplier, loss_SE)
            loss_MSE = K.sqrt(K.mean(loss_SE_modded))
            return loss_MSE

        @tf.function(reduce_retracing=True)
        def training_step_MSE(x_batch_train, simgics_tensor, optimizer, autoencoder, multiplier):
            with tf.GradientTape() as tape:  # Run the forward pass of the layer.
                # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
                grades = autoencoder(x_batch_train, training=True)  # Logits for this minibatch
                # print('distances: ' + str(distances))
                # print('grades: ' + str(grades))
                # simgics_tensor = tf.convert_to_tensor(simgics, dtype=tf.float32)  # Passing tensors to reduce tracing
                grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
                # Compute the loss value for this minibatch.
                loss_value = loss_MSE_modded(simgics_tensor, grades_tensor, multiplier)

            # Use the gradient tape to automatically retrieve the gradients of the trainable
            # variables with respect to the loss.
            # print(loss_value)
            grads = tape.gradient(loss_value, autoencoder.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            # optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
            return loss_value

        @tf.function(reduce_retracing=True)
        def validation_step_MSE(x_batch_val, simgics_tensor, autoencoder):
            grades = autoencoder(x_batch_val, training=False)  # Logits for this minibatch
            # print('distances: ' + str(distances))
            # print('grades: ' + str(grades))
            # simgics_tensor = tf.convert_to_tensor(simgics, dtype=tf.float32)  # Passing tensors to reduce tracing
            grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
            # Compute the loss value for this minibatch.
            loss_value = loss_MSE_modded(simgics_tensor, grades_tensor, 1.0)  # During validation multiplier is always 1
            return loss_value


        # # Function for the actual computation of the loss
        # def loss_bskc(penalty, grades, z_loss):
        #     xy_ref_min = [0.01, 0.01]  # Minimum values for both axis
        #     xy_ref_max = [20, 1]  # Maximum values for both axis
        #
        #     # Reshape inputs for batch processing
        #     penalty_reshaped = tf.reshape(penalty, [-1, 1])  # Reshape simgics
        #     grades_reshaped = tf.reshape(grades, [-1, 1])  # Reshape grades
        #     unknown_val_notshaped = tf.stack([penalty_reshaped, grades_reshaped],
        #                                      axis=1)  # Stack the two vectors one aside the other
        #     unknown_val = tf.reshape(unknown_val_notshaped, [-1, 2])
        #     z_interpolated = tfp.math.batch_interp_regular_nd_grid(unknown_val, xy_ref_min, xy_ref_max, z_loss, axis=-2)
        #     average_loss = K.mean(z_interpolated)
        #     return average_loss
        #
        # @tf.function(reduce_retracing=True)
        # def training_step_bskc(x_batch_train, simgics_tensor, optimizer, autoencoder, z_loss):
        #     with tf.GradientTape() as tape:  # Run the forward pass of the layer.
        #         # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        #         grades = autoencoder(x_batch_train, training=True)  # Logits for this minibatch
        #         grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
        #         linear_penalty = (1 - simgics_tensor) * 20  # Compute a linear penalty from the simgics to feed to the loss
        #         # Compute the loss value for this minibatch.
        #         loss_value = loss_bskc(linear_penalty, grades_tensor, z_loss)
        #
        #     # Use the gradient tape to automatically retrieve the gradients of the trainable
        #     # variables with respect to the loss.
        #     grads = tape.gradient(loss_value, autoencoder.trainable_weights)
        #     # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        #     optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
        #     return loss_value
        #
        # # Function to run the validation step.
        # @tf.function(reduce_retracing=True)
        # def validation_step_bskc(x_batch_val, simgics, model, z_loss_inside):
        #     grades = model(x_batch_val, training=False)  # Logits for this minibatch
        #     # print('distances: ' + str(distances))
        #     # print('grades: ' + str(grades))
        #     simgics_tensor = tf.convert_to_tensor(simgics, dtype=tf.float32)  # Passing tensors to reduce tracing
        #     grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
        #     linear_penalty = tf.abs(simgics_tensor - 1) * 20  # Compute a linear penalty from the simgics
        #     # Compute the loss value for this minibatch.
        #     loss_value = loss_bskc(linear_penalty, grades_tensor, z_loss_inside)
        #     val_loss_metric.update_state(loss_value)
        #     return loss_value

        callbacks = kwargs.pop('callbacks', None)  # Try to get 'callbacks' from kwargs, otherwise set to None
        # Assign the model to the callbacks.
        if callbacks is not None:
            for callback in callbacks:
                callback.model = model

        # # The metric to track validation loss.
        # epoch_loss_metric = keras.metrics.Mean()

        if tune_batch:
            batch_size = hp.Choice("batch_size", values=[100, 150, 200])
        else:
            global default_batch_size
            batch_size = default_batch_size
            batch_size_description = ["batch size: " + str(batch_size)]
            funcs_15.write_to_file(description_file, batch_size_description)

        if tune_multi:
            # hp_multiplier = hp.Choice("multiplier", values=[1.0, 3.0, 5.0, 10.0])
            hp_multiplier = hp.Choice("multiplier", values=[1.0, 5.0, 10.0])
            # hp_multiplier = hp.Choice("multiplier", values=[1.0, 10.0])
        else:
            hp_multiplier = 1.0
            mult_description = ["multiplier: " + str(hp_multiplier)]
            funcs_15.write_to_file(description_file, mult_description)
        if tune_SMOTE:
            # hp_sampling_strat = hp.Choice("sampling_strat", values=[0.3, 0.6, 0.9])
            # hp_k_neighbors = hp.Choice("k_neighbors", values=[2, 5, 10, 20])
            hp_sampling_strat = hp.Choice("sampling_strat", values=[0.3, 0.6])
            hp_k_neighbors = hp.Choice("k_neighbors", values=[2, 20])
        else:
            hp_sampling_strat = 0.9
            hp_k_neighbors = 20
            mult_description = ["\nsampling_strat: " + str(sampling_strat), "\nk_neighbors: " + str(k_neighbors)]
            funcs_15.write_to_file(description_file, mult_description)
        if tune_opti:
            # Tune learning rate for Adam optimizer with values from 0.01, 0.001, or 0.0001
            hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            hp_learning_rate_decay = hp.Choice("learning_rate_decay", values=[0.98, 0.95, 0.9])
            hp_decay_steps = hp.Choice("decay_steps", values=[10000, 5000, 1000])
        else:
            hp_learning_rate = 0.001
            hp_learning_rate_decay = 0.95
            hp_decay_steps = 2000
            opti_description = ["\nlearning rate: " + str(hp_learning_rate),
                                "\nlearning rate decay: " + str(hp_learning_rate_decay),
                                "\ndecay steps: " + str(hp_decay_steps)]
            funcs_15.write_to_file(description_file, opti_description)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            # This scheduler provides a decreasing learning rate
            hp_learning_rate,
            decay_steps=hp_decay_steps,
            decay_rate=hp_learning_rate_decay,  # Decreased by some% every n steps
            staircase=True)

        if tune_opti_type:
            hp_opti_type = hp.Choice("opti_type", values=[0, 2, 3, 4])
        else:
            hp_opti_type = 0
            opti_type_description = ["optimizer type: " + str(hp_opti_type)]
            funcs_15.write_to_file(description_file, opti_type_description)


        if hp_opti_type == 0:
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        elif hp_opti_type == 1:
            optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
        elif hp_opti_type == 2:
            optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True, momentum=0.9)
        elif hp_opti_type == 3:
            optimizer = keras.optimizers.Adadelta(learning_rate=hp_learning_rate)
        elif hp_opti_type == 4:
            optimizer = keras.optimizers.Adadelta(learning_rate=1.0)  # Original Adadelta behaviour
        else:
            optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

        sme = SMOTENC(categorical_features=[X_train.columns.get_loc('Ontology')], sampling_strategy=hp_sampling_strat,
                      random_state=seed, k_neighbors=hp_k_neighbors)
        X_train_res, _ = sme.fit_resample(X_train, y_prediction_quality)
        X_train_res['simgics_eval'] = X_train_res['simgics_eval'].clip(lower=0, upper=1)  # Respect bounds of simgics
        Y_train_res = X_train_res['simgics_eval']  # Simgics is the true Y to use
        X_train_res = funcs_15.one_hot_encode_ontology(X_train_res)  # Reinstate one-hot encoding for training
        X_train_res = X_train_res.drop(['simgics_eval'], axis=1)  # Useless since is in Y

        _train_dataset = tf.data.Dataset.from_tensor_slices((X_train_res, Y_train_res))
        _train_dataset = _train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
        _val_dataset = val_dataset.batch(batch_size, drop_remainder=True)




        loss_list = []
        loss_val_list = []
        step_list = []
        step_val_list = []
        best_loss = 1000
        _last_time = time.time()

        # Presentation
        epochs = trange(
            max_epochs,
            desc="Epoch",
            unit="Epoch")
            # postfix="loss = {loss:.4f}")
        # epochs.set_postfix(loss=0, accuracy=0)

        # The metric to track validation loss.
        val_loss_metric = keras.metrics.Mean()

        not_improved = 0
        impatient = False

        for epoch in epochs:
            # print("\nStart of epoch %d" % (epoch,))
            funcs_15.write_to_file(_file_name, ["", "\nStart of epoch %d" % (epoch,)])
            start_epoch_time = time.time()
            # Iterate over the batches of the train dataset.
            # enumerated_train_batches = tqdm(
            #     enumerate(_train_dataset),
            #     desc="Batch",
            #     position=1,
            #     leave=False)

            # TRAINING
            for step, (x_batch_train, y_batch_train) in enumerate(_train_dataset):
                # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
                y_batch_train = tf.cast(y_batch_train, tf.float32)
                loss = training_step_MSE(x_batch_train, y_batch_train, optimizer, model, hp_multiplier)
                float_loss = float(loss.numpy())
                loss_list.append(float_loss)
                if not step_list:
                    step_list.append(1)
                else:
                    last_value = step_list[-1]
                    step_list.append(last_value + 1)
                # print("Seen so far: %s samples" % ((step + 1) * batch_size))
                # print("time required for this batch: %.4f" % (time.time()-_last_time))
                data = []  # List used to write the report
                if not memory_save_mode:
                    data.append("Training loss (for one batch) at step " + str(step) + ": " + str(loss))
                    data.append(" Seen so far: " + str((step + 1) * batch_size) + " samples ")
                    data.append("time required for this batch: " + str(time.time() - _last_time))
                    funcs_15.write_to_file(_file_name, data)
                _last_time = time.time()
                # callback_checkpoint.on_train_batch_end(logs=logs)
                # Presentation
                # enumerated_batches.set_postfix(
                #     loss=loss)
                # accuracy=float(logs["accuracy"]))
                # break #For testing purposes, just one step

            # VALIDATION
            val_loss_this_epoch = []
            for step, (x_batch_val, y_batch_val) in enumerate(_val_dataset):
                # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
                y_batch_val = tf.cast(y_batch_val, tf.float32)
                loss_val = validation_step_MSE(x_batch_val, y_batch_val, model)
                float_loss_val = float(loss_val.numpy())
                # print('loss_val')
                # print(float_loss_val)
                val_loss_this_epoch.append(float_loss_val)
                loss_val_list.append(float_loss_val)
                if not step_val_list:
                    step_val_list.append(1)
                else:
                    last_value = step_val_list[-1]
                    step_val_list.append(last_value + 1)
                # print("Seen so far: %s samples" % ((step + 1) * batch_size))
                # print("time required for this batch: %.4f" % (time.time()-_last_time))
                data_val = []  # List used to write the report
                data_val.append("Validation loss (for one batch) at step " + str(step) + ": " + str(float_loss_val))
                data_val.append(" Seen so far: " + str((step + 1) * batch_size) + " samples ")
                data_val.append("time required for this batch: " + str(time.time() - _last_time))
                funcs_15.write_to_file(_file_name, data_val)
                _last_time = time.time()
                # break  # For testing purposes, just one step
            average_val_loss_this_epoch = sum(val_loss_this_epoch) / len(val_loss_this_epoch)
            print(  # print info at every step
                "Average validation loss for epoch %d: %.4f. Computed in %.3fs"
                % (epoch, float(average_val_loss_this_epoch), float(time.time() - start_epoch_time))
            )
            abridged_report_message = (f"Average validation loss for epoch {epoch}: {average_val_loss_this_epoch}. "
                                       f"Computed in {float(time.time() - start_epoch_time)}s")
            funcs_15.write_to_file(_abridged_file_name, abridged_report_message)
            if average_val_loss_this_epoch < best_loss:
                best_loss = average_val_loss_this_epoch
                not_improved = 0  # Reset the counter for the patience
            else:  # If the model doesn't improve keep track of it
                not_improved += 1
                if not_improved >= patience:
                    patience_message = f"\nPatience of {patience} exceeded, best model achieved!!\n"
                    print(patience_message)
                    funcs_15.write_to_file(_file_name, patience_message)
                    impatient = True  # Flag to break due to patience met
            # print("Time elapsed this epoch: %.4f" % (time.time() - start_time))

            # Calling the callbacks after epoch.
            val_epoch_loss = float(val_loss_metric.result().numpy())
            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"validation_loss": val_epoch_loss})
            val_loss_metric.reset_states()

            # print(f"Epoch loss: {val_epoch_loss}")
            if impatient:  # At last trigger the break, after everything as been taken care of
                print(f"Best epoch loss: {best_loss}")
                break

        # Return the evaluation metric value.
        funcs_15.write_to_file(_file_name, "Best loss is: " + str(best_loss))
        # Use the formatted date and time as a file name
        train_graph_name = f"tuning_reports/train_graph_{formatted_datetime}.png"
        global trial_number
        curve_label = "Trial " + str(trial_number)
        trial_number += 1
        print("Trial number: " + str(trial_number))
        funcs_15.plot_maker_nofig(step_list, loss_list, train_graph_name, curve_label,
                                  "Steps", "Train Loss", "Train Loss Plot")

        # val_graph_name = f"tuning_reports/val_graph_{formatted_datetime}.png"
        # Use the formatted date and time as a file name

        # funcs_15.plot_maker(step_val_list, loss_val_list, val_graph_name, "Steps", "Validation Loss",
        #                     "Validation Loss Plot")

        return best_loss


# Adding the characters GO: to allow for the search
def addgo(code) -> str: ('GO:' + str(code))


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


########################################################################################
########################################################################################

# OPTIONS:
tune_SMOTE = True
tune_batch = False  # Tune the batch size
tune_opti = False  # Tune the optimizer
tune_hyper = False  # Tune the hyperparameters
tune_model = True  # Tune the structure of the model
tune_opti_type = False
tune_multi = True
tune_init = False
default_batch_size = 100  # Amount used if batch_size is not being tuned
max_epochs = 60  # Max epochs dedicated to each trial
# max_trials = 100  # Max number of trials, if it's not enough it may not exhaust every possibility
n_top_models = 5  # Amount of the best model saved
patience = 20  # Training stops after n consecutive epochs without improvement

########################################################################################
########################################################################################

current_datetime = datetime.now() # Get the current date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") # Format the date and time as a string

use_total_score = True

if use_total_score:
    dir_name = f"hyper_tuning_{formatted_datetime}_tsu"
else:
    dir_name = f"hyper_tuning_{formatted_datetime}"

parent_path = f"/data/gabriele.gradara/reti/tuning_reports"
path = os.path.join(parent_path, dir_name)
os.mkdir(path)


current_datetime = datetime.now() # Get the current date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") # Format the date and time as a string
description_file = path + f"/common_description_{formatted_datetime}.txt"
file_name = path + f"/report_{formatted_datetime}.txt"  # Use the formatted date and time as a file name
abridged_file_name = path + f"/abridged_report_{formatted_datetime}.txt"  # Use the formatted date and time as a file name
graph_name = path + f"/graph_{formatted_datetime}.png"  # Use the formatted date and time as a file name

memory_save_mode = True  # Report less info

# oa_path = "csv/simgics_annotated1000_prediction_quality.csv"  #output argot path
oa_path = "csv/simgics_annotated1000000_prediction_quality.csv"  #output argot path
# oa_path = "csv/simgics_annotated17408776_prediction_quality.csv"  #output argot path


# tf.compat.v1.disable_eager_execution() #Disable eager execution
# tf.enable_eager_execution()+


data = pd.read_csv(oa_path)
X = data.drop(columns=['SeqID', 'GOID', 'prediction_quality', 'Unnamed: 11'], inplace=False)
print(data.head())  # This function returns the first n rows for the object based on position.
print(data.info())  # It is useful for quickly testing if your object has the right type of data in it.

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
        X = X.drop(['Total_score'], axis=1) # Drop unneeded data
        X[['Inf_content', 'Int_confidence', 'GScore']] = (
            StandardScaler().fit_transform(X[['Inf_content', 'Int_confidence', 'GScore']]))


X_shape1 = X.shape[1]
X_shape0 = X.shape[0]

if X_shape0 > 1100000:
    big_dataset = True
else:
    big_dataset = False

basic_description = ["List of all the common parameters shared by all the models:",
                     "\nseed: " + str(seed),
                     "\nDB size:" + str(X_shape0),
                     "\nont_based_prep size: " + str(ont_based_prep)]
funcs_15.write_to_file(description_file, basic_description)



val_percentage = 15  # % of the data that will be used for validation purposes
test_percentage = 15  # % of the data that will be used for testing purposes
val_quota = round(X.shape[0]*0.01*val_percentage)
test_quota = round(X.shape[0]*0.01*test_percentage)

X_val = X[-val_quota:]
Y_val = X_val['simgics_eval']
X_val = X_val.drop(['simgics_eval'], axis=1)

X_test = X[-val_quota-test_quota:-val_quota-1]
Y_test = X_test['simgics_eval']
X_test = X_test.drop(['simgics_eval'], axis=1)

X_train = X[:-val_quota-test_quota-1]
y_prediction_quality = data['prediction_quality']
y_prediction_quality = y_prediction_quality[:-val_quota-test_quota-1]

del data  # Once the data has been divided and preprocessed the original dataframes are not needed anymore
del X
gc.collect()  # Free the RAM from the deleted dataframes

# Initialize HyperParameters for the tuner
hp = HyperParameters()


# Prepare the training dataset.
buffer_quota = 100  # % of the training data that will be stored at a time
buffer_size = round(X_train.shape[0]*0.01*buffer_quota)
print('The buffer size is: '+str(buffer_size))
# Dropping the remainder will allow for a faster overall speed because the program can be tailored to accomodate only
# for one kind of data shape

X_train['Ontology'] = X_train.apply(determine_value, axis=1) # Remove one-hot encoding for SMOTENC
X_train = X_train.drop(columns=["Cell_comp", "Mol_funcs", "Bio_process"], inplace=False)

# There is no need to use the buffer for these smaller datasets
# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
# val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
# test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

# declaration of the global variable
trial_number = 0

# Make a different directory for each run of the tuner
directory = "hyper_autoencoder_" + formatted_datetime
parent_dir = "/data/gabriele.gradara/reti/tuning_reports/"
path = os.path.join(parent_dir, directory)
os.mkdir(path)


# Create the model to tune
hyper_autoencoder = HyperAutoencoder(hp)

tuner = keras_tuner.Hyperband(
    hypermodel=hyper_autoencoder,
    objective=keras_tuner.Objective("validation_loss", "min"),
    max_epochs=max_epochs,
    factor=3,
    hyperband_iterations=1,
    seed=seed,
    hyperparameters=hp,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    overwrite=True,
    directory="tuning_reports",
    project_name=directory,
)



tuner.search(X_train, y_prediction_quality, val_dataset, max_epochs, file_name, abridged_file_name, tune_batch,
             tune_SMOTE, tune_opti, tune_opti_type, patience, memory_save_mode, buffer_size)

# Save the best models
checkpoint_filepath = '/good_models/'

last_time = time.time()-start_time

print("Time taken: %.2fs" % (time.time() - start_time))
# print('Times where it remained inside the dictionary: ' + str(in_db))
# print('Times where it went out of dictionary: ' + str(out_of_db))

# Get the top n models.
top_models = tuner.get_best_models(num_models=n_top_models)
best_model = top_models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(X_shape0,))
best_model.summary()

best_model_test_loss = []
print("# of the top models: " + str(len(top_models)))

# TESTING
# This tests the 5 best models to verify the findings of the tuner


# Write a summary of the best models
summary_name = path + f"/hypertuner_summary_{formatted_datetime}.txt"  # Use the formatted date and time as a file name
summary = ["Results summary", f"\nResults in {tuner.project_dir}", f"\n{tuner.oracle.objective}\n"]
funcs_15.write_to_file(summary_name, summary)

best_trials = tuner.oracle.get_best_trials(n_top_models)
all_step_test_list = []
all_test_loss_list = []

for j in range(len(top_models)):

    top_model_path = os.path.join(path, f"top_model_{j}")
    os.mkdir(top_model_path)

    model_batch = 0  # Initialization of the batch size, for testing purposes
    good_autoencoder = top_models[j]
    # good_autoencoder = tf.keras.models.load_model(model_save_name)
    # Show the model architecture
    good_autoencoder.summary()
    introduction = ["\n" + "Now testing top " + str(j+1) +
                    " out of models " + str(len(top_models)) + "\n"]
    funcs_15.write_to_file(file_name, introduction)
    with open(file_name, 'a+') as f:
        good_autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

    if use_total_score:
        model_save_name = (top_model_path + '/hypertuner_main_net_top' + str(j) + '_of' + str(n_top_models) +
                           formatted_datetime + '_tsu.keras')
    else:
        model_save_name = (top_model_path + '/hypertuner_main_net_top' + str(j) + '_of' + str(n_top_models) +
                           formatted_datetime + '.keras')
    good_autoencoder.save(model_save_name)

    summary = []
    trial = best_trials[j]
    summary.append(f"\n\nTrial {trial.trial_id} summary\n")
    summary.append("\nHyperparameters:")
    if trial.hyperparameters.values:
        for hp, value in trial.hyperparameters.values.items():
            if str(hp) == "batch_size":
                print(value)
                model_batch = value
            summary.append(f"\n{hp}: {value}")
    else:
        summary.append("\ndefault configuration")

    if trial.score is not None:
        summary.append(f"\nScore: {trial.score}")

    if trial.message is not None:
        summary.append("\n" + str(trial.message))
    funcs_15.write_to_file(summary_name, summary)
    with open(summary_name, 'a+') as f:
        top_models[j].summary(print_fn=lambda x: f.write(x + '\n'))

    test_loss_list = []
    step_test_list = []
    simgics_test_list = []
    if model_batch == 0:  # If a custom size is not provided use the default one
        model_batch = default_batch_size
        if tune_batch:
            # If the custom model batch size has not been found and it should've existed throw an error and write it
            # on the reports
            batch_error_message = "\nERROR: BATCH SIZE NOT FOUND!!!\n"
            print(batch_error_message)
            funcs_15.write_to_file(summary_name, batch_error_message)
            funcs_15.write_to_file(file_name, batch_error_message)
    test_dataset_batched = test_dataset.batch(model_batch, drop_remainder=True)
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataset_batched):
        # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
        simgics_test_list.append(y_batch_test.numpy())
        y_batch_test = tf.cast(y_batch_test, tf.float32)
        try:
            # loss_test = funcs_15.validation_step_bskc(x_batch_test, simgics, good_autoencoder, z_loss)
            loss_test = funcs_15.validation_step_MSE(x_batch_test, y_batch_test, good_autoencoder)
            float_loss_test = float(loss_test.numpy())
        except:
            loss_test = 10
            float_loss_test = float(loss_test)
            test_error_message = "Testing step error, loss can't be computed!!!!!!! check screenshot 2024-03-06"
            funcs_15.write_to_file(file_name, test_error_message)
            print(test_error_message)
        test_loss_list.append(float_loss_test)
        if not step_test_list:
            step_test_list.append(1)
        else: 
            last_value = step_test_list[-1]
            step_test_list.append(last_value + 1)
        # print("Seen so far: %s samples" % ((step + 1) * batch_size))
        # print("time required for this batch: %.4f" % (time.time()-last_time))
        if not memory_save_mode:
            data_test = ["Test loss (for one batch) at step " + str(step) + ": " + str(float_loss_test),
                         "| Seen so far: " + str((step + 1) * model_batch) + " samples ",
                         "| time required for this batch: " + str(time.time() - last_time)]  # List used to write the report
            funcs_15.write_to_file(file_name, data_test)
        last_time = time.time()
    average_test_loss = sum(test_loss_list) / len(test_loss_list)
    funcs_15.write_to_file(file_name, "Average Test loss: " + str(average_test_loss))
    print(  # print info at every step
        "Average test loss: %.4f"
        % (float(average_test_loss))
    )

    all_step_test_list.append(step_test_list)
    all_test_loss_list.append(test_loss_list)

    graph_test_loss_list = []
    graph_simgics_test_list = []
    for j in range(len(test_loss_list)):
        graph_test_loss_list.append(test_loss_list[j])
        graph_simgics_test_list.append(max(simgics_test_list[j]))
        # funcs_15.write_to_file(test_losses_file, str(test_loss_list[j]) + ", " + str(max(simgics_test_list[j])))

    funcs_15.confrontation_benchmark(graph_test_loss_list, graph_simgics_test_list, good_autoencoder, top_model_path,
                                     formatted_datetime, model_batch, big_dataset, simgic_threshold=0.75,
                                     confrontation_path_start=top_model_path, use_total_score=use_total_score,
                                     n_benchmark=10000, save_vars=True, ontological_analysis=True,
                                     ont_based_prep=ont_based_prep, source_analysis=True)

# Draw the losses of the top models with different colors
colors = ["b", "g", "r", "c", "m", "y", "k"]
plt.figure()
plt.xlabel("Test steps")
plt.xlabel("Test loss")
plt.title("Best models comparison")
test_graph_name = path + f"/test_best_models_graph_{formatted_datetime}.png"
plt.figure(figsize=(9, 9))
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Test Loss plot for the best " + str(n_top_models) + " models")
for j in range(len(top_models)):
    model_label = "Test loss for the " + str(j + 1) + "° best model"
    plt.plot(all_step_test_list[j], all_test_loss_list[j], label=model_label)
plt.legend()
plt.savefig(test_graph_name)

for j in range(len(best_model_test_loss)):
    funcs_15.write_to_file(file_name, "Average Test loss for the " + str(j+1)
                           + "° best model:" + str(best_model_test_loss[j]))

tuner.results_summary()
