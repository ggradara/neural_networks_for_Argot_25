import funcs_15
import model_selection
import csv
import gc
import os
import tensorflow as tf
# import keras
import keras
import keras_tuner
from keras_tuner import HyperParameters
from keras.layers import Input, Dense, Dropout, LeakyReLU
from sklearn.preprocessing import StandardScaler
from keras import callbacks as callbacks_module
from keras.models import Sequential
from keras import regularizers
import pandas as pd
import numpy as np
import time
import keras.backend as K
import tensorflow_probability as tfp
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt
from tqdm import *
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as f1score_sklearn
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
import itertools

from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss, TomekLinks, RepeatedEditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek 
from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE, SMOTE

# Random tuner for the Future Net, this tuner has a better control on the tested hyperparameters

# Adding the characters GO: to allow for the search
def addgo(code) -> str: ('GO:' + str(code))


def fit(mode, combinations, trial_number, X_train, Y_train, X_val, Y_val, X_test, Y_test, max_epochs, file_name,
        ex_sum_name, X_shape1, current_checkpoint_path):

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
                        from_logits=True, label_smoothing=smoothing, axis=-1, reduction="sum_over_batch_size")
                bce = loss(y_true, y_pred)
            weighted_bce = K.mean(bce * applied_weights)
            return weighted_bce
        return loss_fn

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    @tf.function
    def val_step(x, y):
        val_logits = model(x, training=False)
        val_loss = loss_fn(y, val_logits)
        val_acc_metric.update_state(y, val_logits)
        return val_loss

    trial_time = time.time()


    trial_message = "\nTrial number: " + str(trial_number)
    print(trial_message)
    funcs_15.write_to_file(file_name, trial_message)
    funcs_15.write_to_file(ex_sum_name, trial_message)

    # Default values:
    default_sampling_strat = 0.6
    default_sme_type = 3
    default_k_neighbors = 3
    default_weight = {0: 0.7, 1: 1.3}
    default_hp_learning_rate = 0.0007  # checked
    default_hp_learning_rate_decay = 0.93  # checked
    default_hp_decay_steps = 1000  # checked
    default_reg_l1 = 0.001
    default_reg_l2 = 0.1
    default_drop_out = 0.1
    default_use_last_bias = False
    default_leaky_alpha = 0.0
    default_batch_size = 150  # checked
    default_apply_balance = False
    default_alpha = 2.0
    default_gamma = 0.25
    default_focal = False
    default_smoothing = 0.0
    funcs_15.write_to_file(ex_sum_name, f"\n\nTrial number: {trial_number}")

    if mode == 1:
        sampling_strat = combinations[0]
        sme_type = combinations[1]
        k_neighbors = combinations[2]
        weight = combinations[3]

        hp_learning_rate = default_hp_learning_rate
        hp_learning_rate_decay = default_hp_learning_rate_decay
        hp_decay_steps = default_hp_decay_steps
        reg_l1 = default_reg_l1
        reg_l2 = default_reg_l2
        drop_out = default_drop_out
        use_last_bias = default_use_last_bias
        leaky_alpha = default_leaky_alpha
        batch_size = default_batch_size
        apply_balance = default_apply_balance
        alpha = default_alpha
        gamma = default_gamma
        focal = default_focal
        smoothing = default_smoothing

        mode1_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} --"
                        f" weight: {weight}")
        funcs_15.write_to_file(ex_sum_name, mode1_message)
        funcs_15.write_to_file(file_name, mode1_message)
        print(mode1_message)

        # Selection of optimizer
    elif mode == 2:
        batch_size = combinations[0]
        hp_learning_rate = combinations[1]
        hp_learning_rate_decay = combinations[2]
        hp_decay_steps = combinations[3]

        sampling_strat = default_sampling_strat
        sme_type = default_sme_type
        k_neighbors = default_k_neighbors
        weight = default_weight
        reg_l1 = default_reg_l1
        reg_l2 = default_reg_l2
        drop_out = default_drop_out
        use_last_bias = default_use_last_bias
        leaky_alpha = default_leaky_alpha
        apply_balance = default_apply_balance
        alpha = default_alpha
        gamma = default_gamma
        focal = default_focal
        smoothing = default_smoothing

        mode2_message = (f"batch_size: {batch_size} -- hp_learning_rate: {hp_learning_rate} -- "
                         f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps}")
        funcs_15.write_to_file(ex_sum_name, mode2_message)
        funcs_15.write_to_file(file_name, mode2_message)
        print(mode2_message)

    # Selection of model components [reg_l1, reg_l2, drop_out, use_last_bias, leaky_alpha]
    elif mode == 3:
        reg_l1 = combinations[0]
        reg_l2 = combinations[1]
        drop_out = combinations[2]
        use_last_bias = combinations[3]
        leaky_alpha = combinations[4]

        sampling_strat = default_sampling_strat
        sme_type = default_sme_type
        k_neighbors = default_k_neighbors
        weight = default_weight
        hp_learning_rate = default_hp_learning_rate
        hp_learning_rate_decay = default_hp_learning_rate_decay
        hp_decay_steps = default_hp_decay_steps
        batch_size = default_batch_size
        apply_balance = default_apply_balance
        alpha = default_alpha
        gamma = default_gamma
        focal = default_focal
        smoothing = default_smoothing

        mode3_message = (f"reg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- leaky alpha: {leaky_alpha}"
                         f" -- use_last_bias: {use_last_bias}")
        funcs_15.write_to_file(ex_sum_name, mode3_message)
        funcs_15.write_to_file(file_name, mode3_message)
        print(mode3_message)

    # Selection of loss components
    elif mode == 4:
        apply_balance = combinations[0]
        alpha = combinations[1]
        gamma = combinations[2]
        focal = combinations[3]
        smoothing = combinations[4]

        reg_l1 = default_reg_l1
        reg_l2 = default_reg_l2
        drop_out = default_drop_out
        use_last_bias = default_use_last_bias
        leaky_alpha = default_leaky_alpha
        sampling_strat = default_sampling_strat
        sme_type = default_sme_type
        k_neighbors = default_k_neighbors
        weight = default_weight
        hp_learning_rate = default_hp_learning_rate
        hp_learning_rate_decay = default_hp_learning_rate_decay
        hp_decay_steps = default_hp_decay_steps
        batch_size = default_batch_size


        mode4_message = (f"apply_balance: {apply_balance} -- alpha: {alpha} -- gamma: {gamma} -- focal: {focal} -- "
                         f"smoothing: {smoothing}")
        funcs_15.write_to_file(ex_sum_name, mode4_message)
        funcs_15.write_to_file(file_name, mode4_message)
        print(mode4_message)

        # Used for testing
    elif mode == 0:
        batch_size = combinations[0]

        sampling_strat = default_sampling_strat
        sme_type = default_sme_type
        k_neighbors = default_k_neighbors
        weight = default_weight
        hp_learning_rate = default_hp_learning_rate
        hp_learning_rate_decay = default_hp_learning_rate_decay
        hp_decay_steps = default_hp_decay_steps
        reg_l1 = default_reg_l1
        reg_l2 = default_reg_l2
        drop_out = default_drop_out
        use_last_bias = default_use_last_bias
        leaky_alpha = default_leaky_alpha
        apply_balance = default_apply_balance
        alpha = default_alpha
        gamma = default_gamma
        focal = default_focal
        smoothing = default_smoothing

        mode0_message = (f"batch_size: {batch_size}")
        funcs_15.write_to_file(ex_sum_name, mode0_message)
        funcs_15.write_to_file(file_name, mode0_message)
        print(mode0_message)

    elif mode == 9:
        sampling_strat = combinations[0]
        sme_type = combinations[1]
        k_neighbors = combinations[2]
        weight = combinations[3]
        batch_size = combinations[4]
        hp_learning_rate = combinations[5]
        hp_learning_rate_decay = combinations[6]
        hp_decay_steps = combinations[7]
        reg_l1 = combinations[8]
        reg_l2 = combinations[9]
        drop_out = combinations[10]
        use_last_bias = combinations[11]
        leaky_alpha = combinations[12]

        apply_balance = default_apply_balance
        alpha = default_alpha
        gamma = default_gamma
        focal = default_focal
        smoothing = default_smoothing

        mode1_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} --"
                         f" weight: {weight}")
        funcs_15.write_to_file(ex_sum_name, mode1_message)
        mode2_message = (f"batch_size: {batch_size} -- hp_learning_rate: {hp_learning_rate} -- "
                         f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps}")
        funcs_15.write_to_file(ex_sum_name, mode2_message)
        mode3_message = (f"reg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- leaky alpha: {leaky_alpha}"
                         f" -- use_last_bias: {use_last_bias}")
        funcs_15.write_to_file(ex_sum_name, mode3_message)
        print(mode1_message)
        print(mode2_message)
        print(mode3_message)
        funcs_15.write_to_file(file_name, mode1_message)
        funcs_15.write_to_file(file_name, mode2_message)
        funcs_15.write_to_file(file_name, mode3_message)

    elif mode == 10:
        model_ID = combinations[0]

        sampling_strat = default_sampling_strat
        sme_type = default_sme_type
        k_neighbors = default_k_neighbors
        weight = default_weight
        hp_learning_rate = default_hp_learning_rate
        hp_learning_rate_decay = default_hp_learning_rate_decay
        hp_decay_steps = default_hp_decay_steps
        reg_l1 = default_reg_l1
        reg_l2 = default_reg_l2
        drop_out = default_drop_out
        use_last_bias = default_use_last_bias
        leaky_alpha = default_leaky_alpha
        batch_size = default_batch_size
        apply_balance = default_apply_balance
        alpha = default_alpha
        gamma = default_gamma
        focal = default_focal
        smoothing = default_smoothing

        use_bias = True
        model = model_selection.retrieve_model(model_ID, X_shape1, use_bias, use_last_bias, leaky_alpha, reg_l1, reg_l2,
                                               drop_out)
        mode10_message = (f"Model ID: {combinations}")
        print(mode10_message)
        funcs_15.write_to_file(ex_sum_name, mode10_message)

    if mode != 10:
        # Aggiungere default model e default leaky alpha
        model_ID = 0
        use_bias = True
        model = model_selection.retrieve_model(model_ID, X_shape1, use_bias, use_last_bias, leaky_alpha, reg_l1, reg_l2,
                                               drop_out)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # This scheduler provides a decreasing learning rate
        hp_learning_rate,
        decay_steps=hp_decay_steps,
        decay_rate=hp_learning_rate_decay,  # Decreased by some% every n steps
        staircase=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = loss_wrap(class_weights=weight, apply_balance=apply_balance, alpha=alpha, gamma=gamma, focal=focal,
                                 smoothing=smoothing)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[keras.metrics.BinaryAccuracy()])

    # Python 3.9 is being used, match-case is not available
    if sme_type == 1:
        resample = True
        sme = SMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors)
    elif sme_type == 2:
        resample = True
        sm1 = SMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors)
        sme = SMOTETomek(random_state=seed, smote=sm1)
    elif sme_type == 3:
        resample = True
        sm1 = SMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors)
        sme = SMOTEENN(random_state=seed, smote=sm1)
    elif sme_type == 4:
        resample = True
        sme = SVMSMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors,
                       m_neighbors=k_neighbors + 5)
    elif sme_type == 0:
        resample = False
        # sm1 = SVMSMOTE(sampling_strategy='auto', random_state=seed, k_neighbors=k_neighbors,
        #                m_neighbors=k_neighbors + 5)
        # sme = SMOTEENN(random_state=seed, smote=sm1)
    else:
        sme = SMOTETomek(sampling_strategy=sampling_strat, random_state=seed)
        print('Default case accessed, THIS should NOT have happened!!!')
    if resample:
        X_train_res, Y_train_res = sme.fit_resample(X_train, Y_train)
    else:
        X_train_res = X_train
        Y_train_res = Y_train
    # X_val_res, Y_val_res = sme.fit_resample(X_val, Y_val)

    # X_train = np.asarray(X_train)
    # Y_train = np.asarray(Y_train)
    # X_val = np.asarray(X_val)
    # Y_val = np.asarray(Y_val)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_res, Y_train_res))
    train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality())
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    # val_dataset = tf.data.Dataset.from_tensor_slices((X_val_res, Y_val_res))


    val_acc_metric = keras.metrics.BinaryAccuracy()
    # Weight decision
    # count_1 = np.count_nonzero(Y_train == 1)
    # count_0 = np.count_nonzero(Y_train == 0)
    # weight_1 = max(round(count_0 / count_1) - 1, 1)
    # weight = hp.Int("weight", min_value=1, max_value=weight_1+1, step=1)
    # class_weight = {0: 1.,
    #                 1: weight}
    if weight[0] == 'balanced':
        weight = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train_res), y=Y_train_res)
    weight = from_list_to_dict(weight)

    check_freq = 3
    best_f1_score = 0.0
    for epoch in trange(1, max_epochs+1):
        # To avoid overfitting we compute f1 score periodically to stop the training if is not improved
        if epoch % check_freq == 0:
            f1_temp, threshold = f1_extraction(X_test, model)
            training_message = f"F1 score {f1_temp} at threshold: {threshold}, at epoch {epoch-1}"
            print(training_message)
            funcs_15.write_to_file(file_name, training_message)
            if f1_temp > best_f1_score:
                best_f1_score = f1_temp
            else:
                if use_total_score:
                    model_save_name = (current_checkpoint_path + '/future_net_' + str(trial_number) + '_' +
                                       formatted_datetime + '_tsu.keras')
                else:
                    model_save_name = (current_checkpoint_path + '/future_net_' + str(trial_number) + '_' +
                                       formatted_datetime + '.keras')
                model.save(model_save_name)
                print(f'Best model achieved in {epoch-1} epochs')
                break
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)


    # val_loss = val_step(X_val_res, Y_val_res)
    val_loss = val_step(X_val, Y_val)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    y_pred_test = model.predict(X_test, batch_size=batch_size)
    y_pred_test = np.reshape(y_pred_test, (1, -1))
    y_pred_test = y_pred_test[0]
    best_threshold = thresholder(y_pred_test, Y_test)
    y_best_pred = (y_pred_test >= best_threshold).astype('int')

    f1_test = f1score_sklearn(Y_test, y_best_pred, zero_division=0.0)

    full_message = (f"val loss: {val_loss} -- val acc: {val_acc} -- F1 score: {f1_test} --"
                    f" best threshold: {best_threshold} -- time taken: {time.time() - trial_time}s")
    print(full_message)

    funcs_15.write_to_file(file_name, full_message)
    funcs_15.write_to_file(file_name, str(classification_report(Y_test, y_best_pred)))
    return f1_test


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def thresholder(y_pred, y_true):
    thresholds = np.arange(0, 1, 0.01)
    scores = [f1score_sklearn(y_true, to_labels(y_pred, t), zero_division=0.0) for t in thresholds]
    ix = np.argmax(scores)
    best_threshold = thresholds[ix]
    return best_threshold


def f1_extraction(X_test, model):
    y_pred_test = model.predict(X_test, verbose=0, batch_size=batch_size)
    y_pred_test = np.reshape(y_pred_test, (1, -1))
    y_pred_test = y_pred_test[0]
    best_threshold = thresholder(y_pred_test, Y_test)
    y_best_pred = (y_pred_test >= best_threshold).astype('int')
    f1_test = f1score_sklearn(Y_test, y_best_pred, zero_division=0.0)
    return f1_test, best_threshold


def from_list_to_dict(list1):
    dict1 = {0: list1[0], 1: list1[1]}
    return dict1

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
tf.keras.utils.set_random_seed(seed)

# If using TensorFlow, this will make GPU ops as deterministic as possible, but it will affect the overall performance
# tf.config.experimental.enable_op_determinism()


########################################################################################
########################################################################################

# OPTIONS:
max_trials = 200  # maximum amount of trials allowed
tune_hyper = False  # Tune the hyperparameters
tune_model = True  # Tune the structure of the model
max_depth = 5  # The maximum depth allowed for the exploration of the models
max_epochs = 30  # The maximum amount of epochs allowed
thorough_search = True  # Enhance the scope of the search for the best model
trim = True  # Cut redundant models reducing significantly the amount
alternative_model_source = 2  # 1 std mode, 2 deep mode, 3 mid mode, else bespoke mode
default_batch_size = 100  # Amount used if batch_size is not being tuned
n_top_models = 5  # Amount of the best model saved
use_total_score = True

########################################################################################
########################################################################################


current_datetime = datetime.now() # Get the current date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") # Format the date and time as a string

if use_total_score:
    dir_name = f"manual_tuning_{formatted_datetime}_tsu"
else:
    dir_name = f"manual_tuning_{formatted_datetime}"

parent_path = f"/data/gabriele.gradara/reti/tuning_future_net_reports"
path = os.path.join(parent_path, dir_name)
os.mkdir(path)

if use_total_score:
    file_name = path + f"/manual_future_net_report_{formatted_datetime}_tsu.txt"  # Use the formatted date and time as a file name
    ex_sum_name = path + f"/manual_future_net_extensive_summary_report_{formatted_datetime}_tsu.txt"
    graph_name = path + f"/manual_future_net_graph_{formatted_datetime}_tsu.png"  # Use the formatted date and time as a file name
else:
    file_name = path + f"/manual_future_net_report_{formatted_datetime}.txt"  # Use the formatted date and time as a file name
    ex_sum_name = path + f"/manual_future_net_extensive_summary_report_{formatted_datetime}.txt"
    graph_name = path + f"/manual_future_net_graph_{formatted_datetime}.png"  # Use the formatted date and time as a file name

gt_path = 'csv/groundtruth_sorted.csv'  # ground-truth path
oa_path = "csv/output_argot_flagged_1000000_addedgos.csv"  # output argot path
# oa_path = "csv/argot_output_flagged_def.csv"  # output argot path

# tf.compat.v1.disable_eager_execution() #Disable eager execution
# tf.enable_eager_execution()+



depth = 5
# Retrieve the position of the letters in the sorted ground truth

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

ont_based_prep = True
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

funcs_15.write_to_file(file_name, header)
funcs_15.write_to_file(ex_sum_name, header)
ex_sum_name
print(X.head())
print(Y.head())

X_shape1 = X.shape[1]

val_percentage = 15  # % of the data that will be used for validation purposes
test_percentage = 15  # % of the data that will be used for testing purposes
val_quota = round(X.shape[0]*0.01*val_percentage)
test_quota = round(X.shape[0]*0.01*test_percentage)
X_val = X[-val_quota:]
Y_val = Y[-val_quota:]
X_test = X[-val_quota-test_quota:-val_quota-1]
Y_test = Y[-val_quota-test_quota:-val_quota-1]
X_train = X[:-val_quota-test_quota-1]
Y_train = Y[:-val_quota-test_quota-1]

print(X_train.info())
print(X_train.head())
print(Y_train.info())
print(Y_train.head())
del data  # Once the data has been divided and preprocessed the original dataframes are not needed anymore
gc.collect()  # Free the RAM from the deleted dataframes


# Weight decision
count_1 = np.count_nonzero(Y_train == 1)
count_0 = np.count_nonzero(Y_train == 0)
weight_1 = max(round(count_0 / count_1) - 1, 1)

weights_skl = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)

# Initialize HyperParameters for the tuner
########################################################################################
########################################################################################
# HYPERPARAMETERS:
sampling_strat = [0.6, 0.8, 1]
sme_type = [1, 2, 3, 4]
k_neighbors = [3, 5, 7, 10, 15]
# weight = [{0: 1., 1: 1.},{0: 0.7, 1: 1.3}, {0: 1., 1: weight_1},  {0: 1., 1: 1.5}]  # TODO: Aggiungere i pesi sklearn
# weight = ((1., 1.), (0.7, 1.3), (1., float(weight_1)),  (1., 1.5), tuple(weights_skl), ('balanced'))  # TODO: Aggiungere i pesi sklearn
weight = ((1., 1.), (0.7, 1.3), (1., float(weight_1)),  (1., 1.5), tuple(weights_skl))  # TODO: Aggiungere i pesi sklearn
batch_size = [125, 150, 200, 250]
hp_learning_rate = [3e-4, 5e-4, 7e-4]
hp_learning_rate_decay = [0.98, 0.95, 0.93]
hp_decay_steps = [500, 1000, 1500, 2000]
reg_l1 = [0.005, 0.01, 0.05]
reg_l2 = [0.005, 0.01, 0.05]
drop_out = [0.35, 0.25, 0.3]
use_last_bias = [True, False]
leaky_alpha = [0.0, 0.2, 0.5]
apply_balance = [True, False]
alpha = [0.0, 0.25, 0.5, 0.75]
gamma = [2.0, 1.0, 3.0]
focal = [True, False]
smoothing = [0.0, 1.0]
# Options for bespoke model generation:
input = 6
output = 1
thorough = False
max_depth= 3
trim = False
########################################################################################
########################################################################################

mode = 1
# Selection of under/over sampling strategy
if mode == 1:
    hp_list = [sampling_strat, sme_type, k_neighbors, weight]  # 360
    hp_list_description = ['sampling_strat', 'sme_type', 'k_neighbors', 'weight']
# Selection of optimizer
elif mode == 2:
    hp_list = [batch_size, hp_learning_rate, hp_learning_rate_decay, hp_decay_steps]  # 144
    hp_list_description = ['batch_size', 'hp_learning_rate', 'hp_learning_rate_decay', 'hp_decay_steps']
# Selection of model components
elif mode == 3:
    # hp_list = [reg_l1, reg_l2, drop_out, use_last_bias]  # 162
    # hp_list_description = ['reg_l1', 'reg_l2', 'drop_out', 'use_last_bias']
    hp_list = [reg_l1, reg_l2, drop_out, use_last_bias, leaky_alpha]  # 162
    hp_list_description = ['reg_l1', 'reg_l2', 'drop_out', 'use_last_bias', 'leaky_alpha']
# Selection of loss components
elif mode == 4:
    hp_list = [apply_balance, alpha, gamma, focal, smoothing]  # 96
    hp_list_description = ['apply_balance', 'alpha', 'gamma', 'focal', 'smoothing']
# Used for testing
elif mode == 0:
    hp_list = [batch_size] # 3
    hp_list_description = ['batch_size']
# Full search (unadvisable, very, very long)
elif mode == 9:
    hp_list = [sampling_strat, sme_type, k_neighbors, weight, batch_size, hp_learning_rate, hp_learning_rate_decay,
               hp_decay_steps, reg_l1, reg_l2, drop_out, use_last_bias, leaky_alpha]  #
    hp_list_description = ['sampling_strat', 'sme_type', 'k_neighbors', 'weight', 'batch_size', 'hp_learning_rate',
                           'hp_learning_rate_decay', 'hp_decay_steps', 'reg_l1', 'reg_l2', 'drop_out', 'use_last_bias',
                           'leaky_alpha']
elif mode == 10:
    model_IDs = [0, 1, 2]
    hp_list = [model_IDs]
    hp_list_description = ['model IDs']


# Create all the combinations
combinations = list(itertools.product(*hp_list))
cycles = len(combinations)

checkpoint_filepath = '/data/gabriele.gradara/reti/future_net_models/'
if use_total_score:
    dir_name = f"manual_tuner_future_net_models_{formatted_datetime}"
else:
    dir_name = f"manual_tuner_future_net_models_{formatted_datetime}"
current_checkpoint_path = os.path.join(checkpoint_filepath, dir_name)
os.mkdir(current_checkpoint_path)

hp_list_description.append('f1_score')
scores = []
for trial_number in range(cycles):
    if trial_number > max_trials:
        funcs_15.write_to_file(file_name, f"Max trials reached, tuning stopped !!!!!")
        break
    f1_score = fit(mode, combinations[trial_number], trial_number, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                   max_epochs, file_name, ex_sum_name, X_shape1, current_checkpoint_path)
    scores.append(f1_score)

print(scores)
print(combinations)

print("Time taken: %.2fs" % (time.time() - start_time))

top_models = sorted(zip(scores, range(len(combinations))), reverse=True)[:n_top_models]
print(top_models)

summary_name = path + f"/manual_future_net_summary_{formatted_datetime}.txt"
mode1_message = (f"Top {n_top_models} models:\n Total score used: {use_total_score} \n\n")
funcs_15.write_to_file(summary_name, mode1_message)


for place in range(len(top_models)):

    mode1_message = (f"Top #{place} model\n\n"
                     f"f1_score: {top_models[place][0]}\n"
                     f"Trial: {top_models[place][1]}")
    funcs_15.write_to_file(summary_name, mode1_message)
    if mode == 1:
        sampling_strat = combinations[place][0]
        sme_type = combinations[place][1]
        k_neighbors = combinations[place][2]
        weight = combinations[place][3]
        mode1_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} --"
                         f" weight: {weight}")
        funcs_15.write_to_file(summary_name, mode1_message)
    elif mode == 2:
        batch_size = combinations[place][0]
        hp_learning_rate = combinations[place][1]
        hp_learning_rate_decay = combinations[place][2]
        hp_decay_steps = combinations[place][3]

        mode2_message = (f"batch_size: {batch_size} -- hp_learning_rate: {hp_learning_rate} -- "
                         f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps}")
        funcs_15.write_to_file(summary_name, mode2_message)
    elif mode == 3:
        reg_l1 = combinations[place][0]
        reg_l2 = combinations[place][1]
        drop_out = combinations[place][2]
        use_last_bias = combinations[place][3]
        leaky_alpha = combinations[place][4]

        mode3_message = (f"reg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- leaky alpha: {leaky_alpha}"
                         f" -- use_last_bias: {use_last_bias}")
        funcs_15.write_to_file(summary_name, mode3_message)
    elif mode == 0:
        batch_size = combinations[place][0]
        mode0_message = (f"batch_size: {batch_size}")

        funcs_15.write_to_file(summary_name, mode0_message)
    elif mode == 9:
        sampling_strat = combinations[place][0]
        sme_type = combinations[place][1]
        k_neighbors = combinations[place][2]
        weight = combinations[place][3]
        batch_size = combinations[place][4]
        hp_learning_rate = combinations[place][5]
        hp_learning_rate_decay = combinations[place][6]
        hp_decay_steps = combinations[place][7]
        reg_l1 = combinations[place][8]
        reg_l2 = combinations[place][9]
        drop_out = combinations[place][10]
        use_last_bias = combinations[place][11]
        leaky_alpha = combinations[place][12]
        mode1_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} --"
                         f" weight: {weight}")
        funcs_15.write_to_file(summary_name, mode1_message)
        mode2_message = (f"batch_size: {batch_size} -- hp_learning_rate: {hp_learning_rate} -- "
                         f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps}")
        funcs_15.write_to_file(summary_name, mode2_message)
        mode3_message = (f"reg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- leaky alpha: {leaky_alpha}"
                         f" -- use_last_bias: {use_last_bias}")
        funcs_15.write_to_file(summary_name, mode3_message)

    funcs_15.write_to_file(summary_name, "\n")
