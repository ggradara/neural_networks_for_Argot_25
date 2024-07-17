import funcs_15
import model_selection
import csv
import gc
import os
import random
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, LeakyReLU
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import regularizers
import pandas as pd
import numpy as np
import time
import keras.backend as K
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as f1score_sklearn

from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss, TomekLinks, RepeatedEditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE, SMOTE

# Random tuner for the Future Net

# Adding the characters GO: to allow for the search
def addgo(code) -> str: ('GO:' + str(code))


def fit(combinations, trial_number, X_train, Y_train, X_val, Y_val, X_test, Y_test, max_epochs, file_name, ex_sum_name,
        X_shape1, current_checkpoint_path):

    def loss_wrap(class_weights, apply_balance=True, alpha=0.25, gamma=2.0, focal=False, smoothing=0.0):
        def loss_fn(y_true, y_pred):
            weights = tf.constant([class_weights[0], class_weights[1]])
            applied_weights = tf.gather(weights, y_true)
            if focal:
                loss = tf.keras.losses.BinaryFocalCrossentropy(
                        apply_class_balancing=apply_balance, alpha=alpha, gamma=gamma, from_logits=True,
                        label_smoothing=smoothing, reduction=tf.keras.losses.Reduction.NONE)
                bce = loss(y_true, y_pred)
            else:
                loss = tf.keras.losses.BinaryCrossentropy(
                        # from_logits=True, label_smoothing=smoothing, axis=-1, reduction="sum_over_batch_size")
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

    # Retrieve the data from the list to create the model
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
    model_ID = combinations[13]
    initialization_type = combinations[14]
    opti_type = combinations[15]
    focal = combinations[16]
    smoothing = combinations[17]
    # If Binary focal crossentropy is used, retrieve the relevant paramenters
    if focal:
        apply_balance = combinations[18]
        alpha = combinations[19]
        gamma = combinations[20]

        trial_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} -- "
                         f"weight: {weight} -- batch_size: {batch_size} -- hp_learning_rate: {hp_learning_rate} -- "
                         f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps} -- "
                         f"reg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- use_last_bias: {use_last_bias} --"
                         f"leaky_alpha: {leaky_alpha} -- model ID: {model_ID} -- init type: {initialization_type}"
                         f"opti type: {opti_type} -- focal: {focal} -- smoothing: {smoothing} -- "
                         f"apply_balance: {apply_balance} -- alpha: {alpha} -- gamma: {gamma}")
    else:
        # If the focal variant is not chosen initiate this variables as the default just to pass something
        apply_balance = False
        alpha = 0.25
        gamma = 2.0
        trial_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} -- "
                         f"weight: {weight} -- batch_size: {batch_size} -- hp_learning_rate: {hp_learning_rate} -- "
                         f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps} -- "
                         f"reg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- "
                         f"use_last_bias: {use_last_bias} -- leaky_alpha: {leaky_alpha} -- model ID: {model_ID} -- "
                         f"init type: {initialization_type} -- opti type: {opti_type} -- "
                         f"focal: {focal} --  smoothing: {smoothing} -- ")
    funcs_15.write_to_file(ex_sum_name, "Trial number: " + str(trial_number))
    funcs_15.write_to_file(ex_sum_name, trial_message)
    print("\nTrial number: " + str(trial_number))
    print(trial_message)

        # Selection of U/O sampling strategy
    # Python 3.9 is being used, match-case is not available
    if sme_type == 1:
        sme = SMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors)
    elif sme_type == 2:
        sm1 = SMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors)
        sme = SMOTETomek(random_state=seed, smote=sm1)
    elif sme_type == 3:
        sm1 = SMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors)
        sme = SMOTEENN(random_state=seed, smote=sm1)
    elif sme_type == 4:
        sme = SVMSMOTE(sampling_strategy=sampling_strat, random_state=seed, k_neighbors=k_neighbors,
                       m_neighbors=k_neighbors + 5)
        # sm1 = SVMSMOTE(sampling_strategy='auto', random_state=seed, k_neighbors=k_neighbors,
        #                m_neighbors=k_neighbors + 5)
        # sme = SMOTEENN(random_state=seed, smote=sm1)
    else:
        sme = SMOTETomek(sampling_strategy=sampling_strat, random_state=seed)
        print('Default case accessed, THIS should NOT have happened!!!')

    X_train_res, Y_train_res = sme.fit_resample(X_train, Y_train)
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
    # TODO: Implementazione weighting reattivo
    # Weight decision
    # count_1 = np.count_nonzero(Y_train == 1)
    # count_0 = np.count_nonzero(Y_train == 0)
    # weight_1 = max(round(count_0 / count_1) - 1, 1)
    # weight = hp.Int("weight", min_value=1, max_value=weight_1+1, step=1)
    # class_weight = {0: 1.,
    #                 1: weight}


    # for epoch in range(max_epochs):
    #     # epoch_time = time.time()
    #     # TRAINING
    #     # TODO: Aggiungere check di patience
    #     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    #         loss_value = train_step(x_batch_train, y_batch_train)
    if weight[0] == 'balanced':
        weight = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train_res), y=Y_train_res)
        funcs_15.write_to_file(ex_sum_name, f"True weights: {weights}")

    use_bias = True
    # model_ID = 0
    # Initialize and compile the model
    model = model_selection.retrieve_model(model_ID, X_shape1, use_bias, use_last_bias, leaky_alpha, reg_l1, reg_l2,
                                           drop_out,  initialization_type)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # This scheduler provides a decreasing learning rate
        hp_learning_rate,
        decay_steps=hp_decay_steps,
        decay_rate=hp_learning_rate_decay,  # Decreased by some% every n steps
        staircase=True)

    if opti_type == 0:
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif opti_type == 1:
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    elif opti_type == 2:
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True, momentum=0.3)
    elif opti_type == 3:
        optimizer = keras.optimizers.Adadelta(learning_rate=hp_learning_rate)
    elif opti_type == 4:
        optimizer = keras.optimizers.Adadelta(learning_rate=1.0)  # Original Adadelta behaviour
    else:
        print("Unexpected opti type!!!")
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    loss_fn = loss_wrap(class_weights=weight, apply_balance=apply_balance, alpha=alpha, gamma=gamma, focal=focal,
                        smoothing=smoothing)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[keras.metrics.BinaryAccuracy()])

    check_freq = 5
    best_f1_score = 0.0
    for epoch in range(1, max_epochs+1):
        # To avoid overfitting we compute f1 score periodically to stop the training if is not improved
        if epoch % check_freq == 0:
            f1_temp, threshold = f1_extraction(X_test, model, batch_size)
            print(f"F1 score {f1_temp} at threshold: {threshold}, at epoch {epoch-1}")
            if f1_temp > best_f1_score:
                best_f1_score = f1_temp
            else:
                if use_total_score:
                    model_save_name = (current_checkpoint_path + '/future_net_trial_' + str(trial_number) + '_' +
                                       formatted_datetime + '_tsu.keras')
                else:
                    model_save_name = (current_checkpoint_path + '/future_net_trial_' + str(trial_number) + '_' +
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
    print(y_pred_test)
    y_pred_test = y_pred_test[0]
    best_threshold = thresholder(y_pred_test, Y_test)
    y_best_pred = (y_pred_test >= best_threshold).astype('int')

    f1_test = f1score_sklearn(Y_test, y_best_pred, zero_division=0.0)
    trial_message = "Trial number: " + str(trial_number)
    full_message = (f"val loss: {val_loss} -- val acc: {val_acc} -- F1 score: {f1_test} --"
                    f" best threshold: {best_threshold} -- epochs required: {epoch} -- "
                    f"time taken: {time.time() - trial_time}s")
    print(full_message)
    funcs_15.write_to_file(file_name, trial_message)
    funcs_15.write_to_file(file_name, full_message)
    funcs_15.write_to_file(file_name, str(classification_report(Y_test, y_best_pred)))
    return f1_test


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def thresholder(y_pred, y_true):
    thresholds = np.arange(0, 1, 0.02)
    scores = [f1score_sklearn(y_true, to_labels(y_pred, t), zero_division=0.0) for t in thresholds]
    ix = np.argmax(scores)
    best_threshold = thresholds[ix]
    return best_threshold


def f1_extraction(X_test, model, batch_size):
    y_pred_test = model.predict(X_test, verbose=0, batch_size=batch_size)
    y_pred_test = np.reshape(y_pred_test, (1, -1))
    y_pred_test = y_pred_test[0]
    best_threshold = thresholder(y_pred_test, Y_test)
    y_best_pred = (y_pred_test >= best_threshold).astype('int')
    f1_test = f1score_sklearn(Y_test, y_best_pred, zero_division=0.0)
    return f1_test, best_threshold


def random_gen(trials, sampling_strat, sme_type, k_neighbors, weight, batch_size, hp_learning_rate,
               hp_learning_rate_decay, hp_decay_steps, reg_l1, reg_l2, drop_out, use_last_bias, leaky_alpha, model_ID,
               initialization_type, opti_type, focal, smoothing, apply_balance, alpha, gamma):
    hp_list = []
    previous_combinations = set()
    n_trial = 0
    while n_trial < trials:
        hp_list_temp = (random.choice(sampling_strat), random.choice(sme_type), random.choice(k_neighbors),
                        random.choice(weight), random.choice(batch_size), random.choice(hp_learning_rate),
                        random.choice(hp_learning_rate_decay), random.choice(hp_decay_steps), random.choice(reg_l1),
                        random.choice(reg_l2), random.choice(drop_out), random.choice(use_last_bias),
                        random.choice(leaky_alpha), random.choice(model_ID), random.choice(initialization_type),
                        random.choice(opti_type), random.choice(focal), random.choice(smoothing))
        # If the tuner is using the focal crossentropy I must instantiate the relative parameters
        if hp_list_temp[16]:
            hp_list_temp = list(hp_list_temp)
            hp_list_temp.append(random.choice(apply_balance))
            hp_list_temp.append(random.choice(alpha))
            hp_list_temp.append(random.choice(gamma))
            hp_list_temp = tuple(hp_list_temp)
        if hp_list_temp not in previous_combinations:
            n_trial += 1
            previous_combinations.add(hp_list_temp)
            hp_list.append(hp_list_temp)

    return hp_list    #   , hp_focal


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
trials = 150  # Amount of trials performed
# tune_hyper = True  # Tune the hyperparameters
# tune_model = True  # Tune the structure of the model
# max_depth = 5  # The maximum depth allowed for the exploration of the models
max_epochs = 30  # The maximum amount of epochs allowed
default_batch_size = 50  # Amount used if batch_size is not being tuned
n_top_models = 5  # Amount of the best model saved
use_total_score = False
########################################################################################
########################################################################################


current_datetime = datetime.now() # Get the current date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") # Format the date and time as a string

if use_total_score:
    dir_name = f"random_tuning_{formatted_datetime}_tsu"
else:
    dir_name = f"random_tuning_{formatted_datetime}"

parent_path = f"/data/gabriele.gradara/reti/tuning_future_net_reports"
path = os.path.join(parent_path, dir_name)
os.mkdir(path)

if use_total_score:
    file_name = path + f"/random_future_net_report_{formatted_datetime}_tsu.txt"  # Use the formatted date and time as a file name
    ex_sum_name = path + f"/random_future_net_extensive_summary_report_{formatted_datetime}_tsu.txt"
    graph_name = path + f"/random_future_net_graph_{formatted_datetime}_tsu.png"  # Use the formatted date and time as a file name
else:
    file_name = path + f"/random_future_net_report_{formatted_datetime}.txt"  # Use the formatted date and time as a file name
    ex_sum_name = path + f"/random_future_net_extensive_summary_report_{formatted_datetime}.txt"
    graph_name = path + f"/random_future_net_graph_{formatted_datetime}.png"  # Use the formatted date and time as a file name

oa_path = "csv/output_argot_flagged_1000000_addedgos.csv"  # output argot path

# tf.compat.v1.disable_eager_execution() #Disable eager execution
# tf.enable_eager_execution()+


data = pd.read_csv(oa_path)
# data_flag = pd.read_csv(flag_path)
print(data.head()) # This function returns the first n rows for the object based on position.
print(data.info()) # It is useful for quickly testing if your object has the right type of data in it.

X = data.drop(columns = ['SeqID','GOID','Flag'], inplace=False)
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
print(str(X_test.shape[1]) + '  ' + str(X_test.shape[0]))
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

# Initialize HyperParameters for the tuner
########################################################################################
########################################################################################
# HYPERPARAMETERS:
sampling_strat = (0.5, 0.9)
sme_type = (1, 3)
k_neighbors = (2, 20)
# weight = [{0: 1., 1: 1.},{0: 0.7, 1: 1.3}, {0: 1., 1: weight_1},  {0: 1., 1: 1.5}]  # TODO: Aggiungere i pesi sklearn
# weight = ((1., 1.), (0.7, 1.3), (1., float(weight_1)),  (1., 2.), ("balanced"))  # TODO: Aggiungere i pesi sklearn
weight = ((0.7, 1.3), (1., float(weight_1)), (1., 4.))  # TODO: Aggiungere i pesi sklearn
batch_size = (50, 100, 150)
hp_learning_rate = (1e-2, 1e-3, 1e-4, 1e-5)
hp_learning_rate_decay = (0.98, 0.90, 0.85)
hp_decay_steps = (2000, 1000, 250)
# leaky_alpha = hp.Choice("leaky_alpha", values=[0.05, 0.2, 0.5])
reg_l1 = (0.001, 0.01, 0.1)
reg_l2 = (0.001, 0.01, 0.1)
drop_out = (0.1, 0.2, 0.3)
last_bias = (1, 0)  # TODO: da aggiungere
apply_balance = (True, False)
use_last_bias = (True, False)
leaky_alpha = (0.0, 0.25, 0.5)
alpha = (0.0, 0.75)
gamma = (2.0, 3.0)
focal = (True, False)
smoothing = (0.0, 1.0)
model_ID = (9, 10, 11, 12)
initialization_type = ('uniform', 'normal')
opti_type = (0, 1, 2, 3, 4)
########################################################################################
########################################################################################

# Compute the amximum amount of combinations
max_combs = len(sampling_strat) * len(sme_type) * len(k_neighbors) * len(weight) * len(batch_size) *\
            len(hp_learning_rate) * len(hp_learning_rate_decay) * len(hp_decay_steps) * len(reg_l1) * len(reg_l2) *\
            len(drop_out) * len(use_last_bias) * len(leaky_alpha) * len(smoothing) * len(apply_balance) *\
            len(alpha) * len(gamma) * len(model_ID) * len(initialization_type) * len(opti_type)
print(f"Max combinations: {max_combs}")
if trials > max_combs:
    raise ValueError(f"The amount of trials cannot be higher than the possible combinations: {max_combs}")

# Generate all the different combinations
hp_list_master = []  # List of lists containing all the data
# for trial in range(trials):
hp_list_master = random_gen(trials, sampling_strat, sme_type, k_neighbors, weight, batch_size, hp_learning_rate,
                            hp_learning_rate_decay, hp_decay_steps, reg_l1, reg_l2, drop_out, use_last_bias,
                            leaky_alpha, model_ID, initialization_type, opti_type, focal, smoothing, apply_balance,
                            alpha, gamma)


# print(hp_list_master)
for trial in range(trials):
    hp_list_master[trial] = [i for i in hp_list_master[trial]]
    hp_list_master[trial][3] = from_list_to_dict(hp_list_master[trial][3])
    # hp_list_master.append(hp_list_master[trial])

hp_list_description = ['sampling_strat', 'sme_type', 'k_neighbors', 'weight', 'batch_size', 'hp_learning_rate',
                       'hp_learning_rate_decay', 'hp_decay_steps', 'reg_l1', 'reg_l2', 'drop_out', 'use_last_bias',
                       'leaky_alpha', 'model_ID', 'initialization_type', 'opti_type', 'focal', 'smoothing',
                       'apply_balance', 'alpha', 'gamma', 'f1_score']

# checkpoint_filepath = '/data/gabriele.gradara/reti/future_net_models/'
if use_total_score:
    dir_name = f"random_tuner_future_net_models_{formatted_datetime}"
else:
    dir_name = f"random_tuner_future_net_models_{formatted_datetime}"
current_checkpoint_path = os.path.join(path, dir_name)
os.mkdir(current_checkpoint_path)

scores = []
print('hp_list_master')
print(len(hp_list_master))
print(hp_list_master)
for trial_number in range(len(hp_list_master)):
    f1_score = fit(hp_list_master[trial_number], trial_number, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                   max_epochs, file_name, ex_sum_name, X_shape1, current_checkpoint_path)
    scores.append(f1_score)

print(scores)
print(hp_list_master)

print("Time taken: %.2fs" % (time.time() - start_time))
top_models = sorted(zip(scores, range(len(hp_list_master))), reverse=True)[:n_top_models]
print(top_models)

summary_name = path + f"/random_future_net_summary_{formatted_datetime}.txt"
mode1_message = (f"Top {n_top_models} models:\nTotal score used: {use_total_score} \n\n")
funcs_15.write_to_file(summary_name, mode1_message)
funcs_15.write_to_file(ex_sum_name, "\n" + mode1_message)

# Write the report for the best models
for place in range(len(top_models)):

    mode1_message = (f"\n\nTop #{place} model"
                     f"f1_score: {top_models[place][0]}\n"
                     f"Trial: {top_models[place][1]}")
    funcs_15.write_to_file(summary_name, mode1_message)
    funcs_15.write_to_file(ex_sum_name, mode1_message)

    sampling_strat = hp_list_master[place][0]
    sme_type = hp_list_master[place][1]
    k_neighbors = hp_list_master[place][2]
    weight = hp_list_master[place][3]
    batch_size = hp_list_master[place][4]
    hp_learning_rate = hp_list_master[place][5]
    hp_learning_rate_decay = hp_list_master[place][6]
    hp_decay_steps = hp_list_master[place][7]
    reg_l1 = hp_list_master[place][8]
    reg_l2 = hp_list_master[place][9]
    drop_out = hp_list_master[place][10]
    use_last_bias = hp_list_master[place][11]
    leaky_alpha = hp_list_master[place][12]
    model_ID = hp_list_master[place][13]
    initialization_type = hp_list_master[place][14]
    opti_type = hp_list_master[place][15]
    focal = hp_list_master[place][16]
    smoothing = hp_list_master[place][17]
    if focal:
        apply_balance = hp_list_master[place][18]
        alpha = hp_list_master[place][19]
        gamma = hp_list_master[place][20]
        trial_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} -- "
                     f"weight: {weight} -- batch_size: {batch_size} -- \nhp_learning_rate: {hp_learning_rate} -- "
                     f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps} -- "
                     f"\nreg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- "
                     f"use_last_bias: {use_last_bias} -- \nleaky_alpha: {leaky_alpha} -- model_ID: {model_ID} -- "
                     f"initialization_type: {initialization_type} -- optimization_type: {opti_type} -- "
                     f"\nfocal: {focal} --  smoothing: {smoothing} -- apply_balance: {apply_balance} -- "
                     f"alpha: {alpha} -- gamma: {gamma}  -- "
                     f"\nont based prep: {ont_based_prep} -- total_score_used: {use_total_score}")
    else:
        trial_message = (f"sampling_strat: {sampling_strat} -- sme_type: {sme_type} -- k_neighbors: {k_neighbors} -- "
                         f"weight: {weight} -- batch_size: {batch_size} -- \nhp_learning_rate: {hp_learning_rate} -- "
                         f"hp_learning_rate_decay: {hp_learning_rate_decay} -- hp_decay_steps: {hp_decay_steps} -- "
                         f"\nreg_l1: {reg_l1} -- reg_l2: {reg_l2} -- drop_out: {drop_out} -- "
                         f"use_last_bias: {use_last_bias} -- \nleaky_alpha: {leaky_alpha} -- model_ID: {model_ID} -- "
                         f"initialization_type: {initialization_type} -- optimization_type: {opti_type} -- "
                         f"\nfocal: {focal} --  smoothing: {smoothing} --"
                         f"\nont based prep: {ont_based_prep} -- total_score_used: {use_total_score}")
    funcs_15.write_to_file(ex_sum_name, trial_message)
funcs_15.write_to_file(summary_name, "\n")
