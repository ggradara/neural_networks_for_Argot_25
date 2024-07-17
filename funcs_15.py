import csv
import re
import os
import keras.backend as K
import keras
import tensorflow as tf
import random
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colormaps
import altair as alt
import itertools
from sklearn.metrics import f1_score as f1score_sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm


# Custom function library


def get_choice_list(input: int, output: int, thorough: bool = False, max_depth: int = 3, trim: bool = False):
    # Get a list of list containing all possible useful combinations for depth and width of a NN
    # input: The width of the input
    # output: The width of the output
    # thorough: Adds more elements to the list like nn with depth 1 and depth equal to the output
    # Generate combinations
    combinations = []
    first_list = []
    if thorough:
        for j in range(output + 1, input):
            first_list.append(j)
        combinations.append(first_list)
    if thorough:
        max_depth += 1
    for reps in range(output + 1, max_depth):
        combinations.append(list(itertools.product(range(output, input + 1), repeat=reps)))
    # Print combinations
    to_pop = []
    for check in range(1, max_depth-1):
        for l in range(0, len(combinations[check])):
            for k in range(0, len(combinations[check][0])):
                if k == 0:
                    continue
                # Mark for pop the model that has at least one layer equal to 1
                if combinations[check][l][k] == output:
                    to_pop.append((check, l))
                    break
                # Mark for pop the model that has at least one layer equal to the input
                if combinations[check][l][k] == input:
                    to_pop.append((check, l))
                    break
                # Mark for pop the model does that don't have not increasing layer width
                if combinations[check][l][k] > combinations[check][l][k - 1]:
                    to_pop.append((check, l))
                    break
                # Mark for pop the model does that have the same witdh for 3 layers or more
                if k >= 2:
                    if (combinations[check][l][k] == combinations[check][l][k - 1] and combinations[check][l][k - 1] ==
                            combinations[check][l][k - 2]):
                        to_pop.append((check, l))
                        break
                # Mark for pop the model that does have the same width  as the input for 2 layers or more
                if k >= 1:
                    if (combinations[check][l][k] == combinations[check][l][k - 1] and
                            combinations[check][l][k - 1] == input):
                        to_pop.append((check, l))
                        break

    # Clean the data from undesired combinations
    for j in range(len(to_pop) - 1, -1, -1):
        combinations[to_pop[j][0]].pop(to_pop[j][1])
    clean_combinations = []
    # Create a unique list to store the data
    for j in range(0, len(combinations)):
        for k in range(0, len(combinations[j])):
            clean_combinations.append(combinations[j][k])

    if trim:  # Significantly reduces the amount of combinations produced
        to_trim = []
        previous_trimmed = False
        for j in range(len(clean_combinations) - 1):
            if previous_trimmed:
                previous_trimmed = False
                continue
            if isinstance(clean_combinations[j], int):
                if abs(clean_combinations[j] - clean_combinations[j + 1]) <= 1:
                    previous_trimmed = True
                    to_trim.append(j)
                continue
            if len(clean_combinations[j]) != len(clean_combinations[j + 1]):
                continue
            difference = 0
            for k in range(len(clean_combinations[j])):
                difference = difference + abs(clean_combinations[j][k] - clean_combinations[j + 1][k])
            if difference <= 1:
                previous_trimmed = True
                to_trim.append(j)

        for j in range(len(to_trim) - 1, -1, -1):
            # print('j: ' + str(j) + ' to_pop[j][0]: ' + str(to_pop[j][0]) + ' to_pop[j][1]: ' + str(to_pop[j][1]))
            clean_combinations.pop(to_trim[j])

    # Create a list to store the number of layers for each model
    length_list = []
    for j in range(len(clean_combinations)):
        if isinstance(clean_combinations[j], int):
            length_list.append(1)
        else:
            length_list.append(len(clean_combinations[j]))
    return clean_combinations, length_list


def memory_check(length_data, simgic_dict, diff_time):
    # Prints on screen and on a chosen file info about the simgic dict, memory usage, time required and date
    # length_data: how much data the dict synthesizes
    # simgic_dict: the simgic dict to analyze
    # diff_time: time required to analyze it
    memory_file = 'memory_check.txt'
    current_datetime = datetime.now()  # Get the current date and time
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")  # Format the date and time as a string
    convert_time = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    memory_usage = sys.getsizeof(simgic_dict)
    memory_usage_MB = memory_usage / 1000000
    memory_message = (f"The memory size for a dictionary of SeqID length {length_data} is: {memory_usage} bytes or "
                      f"{memory_usage_MB} MB, computed in {convert_time} h:m:s. Date: {formatted_datetime}")
    print(memory_message)
    write_to_file(memory_file, memory_message)


def alternative_model_selection():
    # Special selection of small and relevant models, used for tuners
    clean_combinations = [5, 3, (5, 2), (4, 2), (3, 2), (5, 4, 2), (6, 4, 2), (5, 4, 3, 2)]
    length_list = [1, 1, 2, 2, 2, 3, 3, 4]
    return clean_combinations, length_list


def alternative_model_selection_mid():
    # Special selection of mid and relevant models, used for tuners
    clean_combinations = [
        (6, 5, 4, 3, 2),
        (6, 6, 4, 4, 2, 2),
        (6, 6, 3, 3, 3, 2),
        (6, 4, 2, 2),
        (6, 8, 4, 2),
        (6, 4, 2),
        (6, 7, 8, 9, 10, 6, 5, 4, 3, 2),
        (6, 6, 5, 5, 4, 4, 3, 3, 2, 2),
        (6, 7, 8, 9, 10, 8, 6, 4, 2),
        (6, 8, 10, 8, 6, 4, 2)
    ]
    length_list = [5, 6, 6, 4, 4, 3, 10, 10, 9, 7]
    return clean_combinations, length_list

def alternative_model_selection_deep():
    # Special selection of big and relevant models, used for tuners
    clean_combinations = [
        (6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2),
        (6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2),
        (6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2),
        (6, 6, 8, 8, 10, 10, 12, 12, 8, 8, 6, 6, 4, 4, 2, 2),
        (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2),
        (6, 6, 8, 8, 12, 12, 14, 14, 10, 10, 8, 8, 6, 6, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2),
        (6, 7, 8, 9, 10, 6, 5, 4, 3, 2),
        (6, 6, 8, 8, 10, 10, 12, 12, 15, 15, 6, 6, 4, 4, 2, 2),
        (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
        (6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12,
         11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2),
        (6, 6, 6, 6, 8, 8, 8, 8, 12, 12, 12, 12, 14, 14, 14, 14, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 6, 6, 4, 4, 4, 4, 2,
         2, 2, 2),
        (6, 5, 4, 3, 2),
    ]
    length_list = [15, 26, 13, 16, 25, 24, 10, 16, 50, 50, 36, 5]
    return clean_combinations, length_list

def write_to_file(file_name, data):
    # Writes on a file or it can create it if it does not already exist
    with open(file_name, "a+") as file:
        for string in data:
            file.write(string)
        file.write("\n")


# Returns a list of the first letter of the SeqID and where to find them
def alpha_indexer(path: str):
    data = []
    # Reads a csv file and stores the data in a list of lists
    # Before it was like that: '../csv/groundtruth_sorted.csv'
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    index_array = {}
    index_array["length"] = len(data)
    first_flag = True
    for j in range(index_array["length"]):
        if first_flag:  # The first character is stored no matter what
            char = data[j][0][0]
            index_array[char] = j
            first_flag = False
        elif char != data[j][0][0]:  # If the new character is different from the last we record the position
            char = data[j][0][0]
            index_array[char] = j
    return index_array


# Returns a list of the first letter of the SeqID and where to find them, depth indicates how many char to use
def alpha_indexer_deep(path: str, depth: int = 2):
    data = []
    # Reads a csv file and stores the data in a list of lists
    # Before it was like that: '../csv/groundtruth_sorted.csv'
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    index_dict = {}
    index_dict["length"] = len(data)
    first_flag = True
    for j in range(index_dict["length"]):
        if first_flag:  # The first character is stored no matter what
            char = data[j][0][0:depth]
            index_dict[char] = j
            first_flag = False
        elif char != data[j][0][0:depth]:  # If the new character is different from the last we record the position
            char = data[j][0][0:depth]
            index_dict[char] = j
    # with open('alpha_indexer_dict.txt', 'w') as file:
    #     # Iterate over the dictionary items and write them to the file
    #     for key, value in index_dict.items():
    #         file.write(f'{key}: {value}\n')
    return index_dict


# Returns a big dictionary with all the possible combinations of paths
def path_retriever(path: str = 'txts/go_complete.txt'):
    # Path: the path where the dictionary is retrieved
    spl = {}  # Initialize shortest path lenght dict
    # Retrieve GO codes
    pattern_full_go = re.compile(r'GO:\d{7}')
    pattern_digit = re.compile(r"'GO:\d{7}':\s'(\d+)")
    # Before: '../txts/go_complete.txt'
    with open(path, 'r') as f:
        line_counter = 1
        for line in f:
            # Every odd cycle we get the master key of the dictionary
            if (line_counter % 2) == 1:
                key = line.strip()
                spl[key] = []
            # Every even cycle we get the combination of paths for that node
            else:
                # List of al GOs and path length
                all_go = re.findall(pattern_full_go, line)  # Splits the GOs in a list
                if all_go[0] == all_go[1]:  # Sometimes the first value is read twice
                    all_go.pop(0)
                all_length = re.findall(pattern_digit,
                                        line)  # Access group 1 (the single digit) and split in a list the hits
                # print('allgo ' +str(len(all_go)))
                # print('all length ' +str(len(all_length)))
                spl_partial = {}  # Dictionary for all GOs and path lengths
                for j in range(len(all_go)):
                    spl_partial[all_go[j]] = []
                    spl_partial[all_go[j]] = all_length[j]  # Association of all GOs to the path length
                # Association to the starting node
                spl[key] = spl_partial
            # Update line counter
            line_counter += 1
    return spl


def path_retriever_slim(path: str = 'txts/go_combinations.txt'):
    # Path: the path where the dictionary is retrieved
    spl = {}  # Initialize shortest path lenght dict
    # Retrieve GO codes
    pattern_full_go = re.compile(r'GO:\d{7}')
    pattern_digit = re.compile(r"GO:\d{7}':\s(\d+)")
    # Before: '../txts/go_complete.txt'
    f = open(path, 'r')
    line_counter = 1
    for line in f:
        # Every odd cycle we get the master key of the dictionary
        if (line_counter % 2) == 1:
            key = line.strip()
            spl[key] = []
        # Every even cycle we get the combination of paths for that node
        else:
            # List of al GOs and path length
            all_go = re.findall(pattern_full_go, line)  # Splits the GOs in a list
            if all_go[0] == all_go[1]:  # Sometimes the first value is read twice
                all_go.pop(0)
            all_length = re.findall(pattern_digit,
                                    line)  # Access group 1 (the single digit) and split in a list the hits
            spl_partial = {}  # Dictionary for all GOs and path lengths
            for j in range(len(all_go)):
                spl_partial[all_go[j]] = []
                spl_partial[all_go[j]] = all_length[j]  # Association of all GOs to the path length
            # Association to the starting node
            spl[key] = spl_partial
        # Update line counter
        line_counter += 1
    return spl


# Prints the progress value if its updated
def progress_str(current_progress, max_progress, past_progress: float = 0, detail: int = 2):
    if round(100 * current_progress / max_progress, detail) != round(100 * past_progress / max_progress,
                                                                     detail):  # if p_p is used, it reduces spam prints
        completion = round(100 * current_progress / max_progress, detail)
        return "-- " + str(completion) + "% completion --"  # returns a progress string to print or write
    else:
        return False


# Basic version of the loss function based on graph distance
# A loss function should accept just y_pred and y_true but for efficency sake the required data is assumed already
# obtained and stored as global variables (computing it may require >30 mins)
def graph_loss_basic(SeqID: str, pred_go: int, spl: dict, gt_sorted, alpha_order: dict):
    """SeqID: The unique identifier of the protein
    pred_go: The predicted GO from argot
    spl: shortest path library, the dictionary that stores the combinations of the shortest paths
    gt_sorted: the groundtruth, alphabetically sorted
    alpha_order: the list of the letters present in the ground truth and where to find them
    """
    # The GO prediction may be an int or a  float
    if type(pred_go) != 'str':
        pred_go = str(pred_go)
        while len(pred_go) < 7:  # Int automatically removes "useless" 0 that are required for the GO codes
            pred_go = '0' + pred_go
        pred_go = 'GO:' + pred_go
    print('pred_go: ' + str(pred_go))
    print('SeqID: ' + str(SeqID))
    print(alpha_order)
    start = alpha_order[SeqID[0]]  # Get the starting row from the first character
    print(SeqID[0])
    stop = alpha_order["length"]  # The total length of the array
    matches = []  # List of gos attributed to the protein
    for j in range(start, stop):
        # For every match found, retrieve the gos
        if SeqID == gt_sorted[j][0]:
            print(gt_sorted[j][0])
            matches.append(gt_sorted[j][1])

    distances = []  # The list of distances of the protein from all the hits
    for j in range(len(matches)):
        distances.append(spl[pred_go][matches[j]])

    distance = min(distances)  # The error assumed is the smallest one (the closest groundtruth go to the prediction)
    # Returns 0 if the GO is correct, otherwise it retuns an integer based on the distance between the nodes
    return distance


def simgic_dict_setup(SeqID: str, pred_go, collection, gt_sorted, alpha_order: dict,  depth: int = 2):
    """ SeqID: The list of unique identifiers of the proteins
        pred_go: The list of predicted GO from argot
        collection: the collection of the db where the simgics are stored
        gt_sorted: the groundtruth, alphabetically sorted
        alpha_order: the list of the letters present in the ground truth and where to find them
        """
    # everything = []
    simgic_dict = {}
    missing_seqs = []
    missing_count = 0
    for k in tqdm(range(len(pred_go))):
        if type(pred_go[k]) == 'int':  # The GO prediction may be an int
            pred_go[k] = str(pred_go[k])
            while len(pred_go[k]) < 7:  # Int automatically removes "useless" 0 that are required for the GO codes
                pred_go[k] = '0' + pred_go[k]
            pred_go[k] = 'GO:' + pred_go[k]
        try:
            start = alpha_order[SeqID[k][0:depth]]  # Get the starting row from the first character
        except KeyError:  # If the protein is missing from the gt don't add them to the dict
            # By default, that means that they will be considered wrong, other actions may be valuated
            missing_count += 1
            missing_ID = SeqID[k][0:depth]
            if missing_ID not in missing_seqs:
                missing_seqs.append(missing_ID)
            continue
        stop = alpha_order["length"]  # The total length of the array
        matches = []  # List of gos attributed to the protein
        for j in range(start, stop):  # Reduce the search space
        # Search for all the GOs in the gt of the protein we are interested in
            if SeqID[k] == gt_sorted[j][0]:
                matches.append(gt_sorted[j][1])
                # Finds all the go gt of the protein and then retrieves all the simgics between the predicted go
                # and then adds the to the dictionary
        best_simgic = 0.
        # print(matches)
        for j in range(len(matches)):
            try:
                try:  # try both combinations (only one works in this DB)
                    current_simgic = query_simgic(collection, pred_go[k], matches[j])
                except:
                    current_simgic = query_simgic(collection, matches[j], pred_go[k])
            except:  # The two gos belong to different branches: lowest score
                current_simgic = 0.
            # everything.append(current_simgic)
            if current_simgic > best_simgic:
                best_simgic = current_simgic
        # everything.append(best_simgic)
        # La selezione del best simgic pare funzionare
        if best_simgic != 0.:
            # print(best_simgic)
            try:
                simgic_dict[SeqID[k]][pred_go[k]] = best_simgic

            except KeyError:
                # If the key can't be declared it means that that particular key is not a dictionary
                simgic_dict[SeqID[k]] = {}
                simgic_dict[SeqID[k]][pred_go[k]] = best_simgic
            # print(simgic_dict.keys())
            # print(simgic_dict[SeqID[k]].keys())
            # print(simgic_dict[SeqID[k]][pred_go[k]])
            # Attribuzione corretta dei valori nel dizionario, funzione tuatto
        # print(everything)
    # print(simgic_dict)
    print(f"There are {missing_count} missing SeqIDs")
    if missing_count > 0:  # If some IDs were missing report them
        missing_report(missing_seqs, missing_count, len(SeqID))
    return simgic_dict


def missing_report(missing_seqs, missing_count, len_SeqID):
    missing_file = 'missing_SeqID_report.txt'
    current_datetime = datetime.now()  # Get the current date and time
    separator = '\n---------------------------------------------------------------------------------------\n'
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")  # Format the date and time as a string
    header_message = f"Missing report for {len_SeqID} IDs. Date: {formatted_datetime}"
    tail_message = f"In total {missing_count} accesses were missed, SeqIDs tend to request access more than once"
    write_to_file(missing_file, separator)
    write_to_file(missing_file, header_message)
    for j in range(len(missing_seqs)):
        write_to_file(missing_file, missing_seqs[j])
    write_to_file(missing_file, tail_message)


def simgic_retriever_fast(SeqID: str, pred_go, simgic_dict):
    simgics = np.zeros(len(SeqID))
    SeqID = [ID.decode('utf-8') for ID in SeqID]
    pred_go = [GO.decode('utf-8') for GO in pred_go]
    for j in range(len(SeqID)):
        try:
            simgics[j] = simgic_dict[SeqID[j]][pred_go[j]]
        except KeyError:
            simgics[j] = 0
    # print(simgics)
    return simgics


def simgic_retriever_fast_str(SeqID: str, pred_go, simgic_dict):
    simgics = np.zeros(len(SeqID))
    for j in range(len(SeqID)):
        try:
            simgics[j] = simgic_dict[SeqID[j]][pred_go[j]]
        except KeyError:
            simgics[j] = 0
    # print(simgics)
    return simgics


# def simgic_retriever_fast(SeqID: str, pred_go, simgic_dict):
#     simgics = []
#     SeqID = [ID.decode('utf-8') for ID in SeqID]
#     pred_go = [GO.decode('utf-8') for GO in pred_go]
#     for j in range(len(SeqID)):
#         try:
#             # print(SeqID[j])
#             # print(pred_go[j])
#             # print(simgic_dict[SeqID[j]][pred_go[j]])
#             simgics.append(simgic_dict[SeqID[j]][pred_go[j]])
#         except KeyError:
#             simgics.append(0.)
#     print(simgics)
#     # Funziona tutto
#     return simgics

def simgic_retriever(SeqID: str, pred_go, collection, gt_sorted, alpha_order: dict, depth: int = 2):

    """SeqID: The unique identifier of the protein
    pred_go: The predicted GO from argot
    collection: the collection of the db where the simgics are stored
    gt_sorted: the groundtruth, alphabetically sorted
    alpha_order: the list of the letters present in the ground truth and where to find them
    batch_size: the training batch_size of this NN
    """

    # pred_go and SeqID are both lists because we are working in batches
    batch_distances = []  # List of distances of that batch
    # pred_go and SeqID are passed as bynary objects
    SeqID = [ID.decode('utf-8') for ID in SeqID]
    pred_go = [GO.decode('utf-8') for GO in pred_go]
    out_of_db = 0  # Sometimes the go are not in the dict, we want to keep an eye on that
    in_db = 0
    batch_simgics = []
    for k in range(len(pred_go)):  # Not batch size because the last batch could be smaller
        if type(pred_go[k]) == 'int':  # The GO prediction may be an int
            pred_go[k] = str(pred_go[k])
            while len(pred_go[k]) < 7:  # Int automatically removes "useless" 0 that are required for the GO codes
                pred_go[k] = '0' + pred_go[k]
            pred_go[k] = 'GO:' + pred_go[k]
        start = alpha_order[SeqID[k][0:depth]]  # Get the starting row from the first character
        stop = alpha_order["length"]  # The total length of the array
        matches = []  # List of gos attributed to the protein
        for j in range(start, stop):
            # For every match found, retrieve the gos
            if SeqID[k] == gt_sorted[j][0]:
                matches.append(gt_sorted[j][1])
        simgics = []  # The list of simgics of the protein from all the hits
        for j in range(len(matches)):
            # print('value of key ' + str(matches[j]) + ' '+ str(pred_go[k]))
            try:
                try:
                    simgics.append(query_simgic(collection, pred_go[k], matches[j]))
                    in_db += 1
                except:
                    simgics.append(query_simgic(collection, matches[j], pred_go[k]))
                    in_db += 1
            except:
                simgics.append(0.)
                out_of_db += 1
        if len(simgics) == 0:
            simgics.append(0.)
        batch_simgics.append(max(element for element in
                                 simgics))  # The error assumed is the smallest one (the closest groundtruth go to the prediction)
        # Returns 0 if the GO is correct, otherwise it retuns an integer based on the distance between the nodes
    return batch_simgics, in_db, out_of_db


def simgic_retriever_eval(SeqID: str, pred_go, collection, gt_sorted, alpha_order: dict, depth: int = 2):

    """ VERSION WITHOUT DECODING TO PROCESS STRINGS DIRECTLY
    SeqID: The unique identifier of the protein
    pred_go: The predicted GO from argot
    collection: the collection of the db where the simgics are stored
    gt_sorted: the groundtruth, alphabetically sorted
    alpha_order: the list of the letters present in the ground truth and where to find them
    """

    # pred_go and SeqID are both lists because we are working in batches
    batch_distances = []  # List of distances of that batch
    # pred_go and SeqID are passed as bynary objects
    batch_simgics = []
    simgics_logger = []
    for k in range(len(pred_go)):  # Not batch size because the last batch could be smaller
        if type(pred_go[k]) == 'int':  # The GO prediction may be an int
            pred_go[k] = str(pred_go[k])
            while len(pred_go[k]) < 7:  # Int automatically removes "useless" 0 that are required for the GO codes
                pred_go[k] = '0' + pred_go[k]
            pred_go[k] = 'GO:' + pred_go[k]
        try:
            start = alpha_order[SeqID[k][0:depth]]  # Get the starting row from the first character
        except KeyError:  # If the protein is missing from the gt don't add them to the dict
            # By default, that means that they will be considered wrong, other actions may be valuated
            continue
        stop = alpha_order["length"]  # The total length of the array
        matches = []  # List of gos attributed to the protein
        for j in range(start, stop):
            # For every match found, retrieve the gos
            if SeqID[k] == gt_sorted[j][0]:
                matches.append(gt_sorted[j][1])
        simgics = []  # The list of simgics of the protein from all the hits
        for j in range(len(matches)):
            # print('value of key ' + str(matches[j]) + ' '+ str(pred_go[k]))
            try:
                try:
                    simgics.append(query_simgic(collection, pred_go[k], matches[j]))
                except:
                    simgics.append(query_simgic(collection, matches[j], pred_go[k]))
            except:
                simgics.append(0.)
        if len(simgics) == 0:
            simgics.append(0.)
        simgics_logger.append(simgics)  # Simgics before the selction for the max value
        batch_simgics.append(max(element for element in
                                 simgics))  # The error assumed is the smallest one (the closest groundtruth go to the prediction)
        # Returns 0 if the GO is correct, otherwise it retuns an integer based on the distance between the nodes
    return batch_simgics, simgics_logger


def distance_retriever(SeqID: str, pred_go, spl: dict, gt_sorted, alpha_order: dict):
    """SeqID: The unique identifier of the protein
    pred_go: The predicted GO from argot
    spl: shortest path library, the dictionary that stores the combinations of the shortest paths
    gt_sorted: the groundtruth, alphabetically sorted
    alpha_order: the list of the letters present in the ground truth and where to find them
    batch_size: the training batch_size of this NN
    """
    # pred_go and SeqID are both lists because we are working in batches
    batch_distances = []  # List of distances of that batch
    # pred_go and SeqID are passed as bynary objects
    SeqID = [ID.decode('utf-8') for ID in SeqID]
    pred_go = [GO.decode('utf-8') for GO in pred_go]
    out_of_dict = 0  # Sometimes the go are not in the dict, we want to keep an eye on that
    in_dict = 0
    for k in range(
            len(pred_go)):  # Not batch size because the last batch could be smaller (eg: buffer = 100, batch = 30)
        if type(pred_go[k]) == 'int':  # The GO prediction may be an int
            pred_go[k] = str(pred_go[k])
            while len(pred_go[k]) < 7:  # Int automatically removes "useless" 0 that are required for the GO codes
                pred_go[k] = '0' + pred_go[k]
            pred_go[k] = 'GO:' + pred_go[k]
        try:
            start = alpha_order[SeqID[k][0]]  # Get the starting row from the first character
        except:  # Sometimes the letter may not be present, or some other problem occurs
            start = 0
        try:
            stop = alpha_order["length"]  # The total length of the array
        except:
            stop = len(gt_sorted)  # Sometimes: stop = alpha_order["length"] #The total length of the array
            # TypeError: 'int' object is not subscriptable, reason unknown

        matches = []  # List of gos attributed to the protein
        for j in range(start, stop):
            # For every match found, retrieve the gos
            if SeqID[k] == gt_sorted[j][0]:
                matches.append(gt_sorted[j][1])

        distances = []  # The list of distances of the protein from all the hits
        for j in range(len(matches)):
            # print('value of key ' + str(matches[j]) + ' '+ str(pred_go[k]))
            try:
                if matches[j] in spl.keys():
                    # print('value of key ' + str(matches[j]) + str(spl[matches[j]]))
                    distances.append(spl[matches[j]][pred_go[k]])
                    in_dict += 1
                else:
                    # print('value of key' + str(pred_go[k]) + str(spl[pred_go[k]]))
                    distances.append(spl[pred_go[k]][matches[j]])
                    in_dict += 1
            except:
                distances.append(20)
                out_of_dict += 1
        if len(distances) == 0:
            distances.append(20)
        batch_distances.append(min(int(element) for element in
                                   distances))  # The error assumed is the smallest one (the closest groundtruth go to the prediction)
        # Returns 0 if the GO is correct, otherwise it retuns an integer based on the distance between the nodes
    return batch_distances, in_dict, out_of_dict


def init_loss_func():
    def equation(x, y, clip_value):
        k = 1
        j = 2.75
        epsilon = 0.1
        x_offset = 0.65  # Fine-tuning factor for x
        x += x_offset
        result = (1 - (k * ((x + 3) + y) + j * (-0.8 * (1 / (x + epsilon)) - 1 * (1 / (y + epsilon)))) ** 5) ** 2
        # Clipping the result
        clipped_result = np.clip(result, 0, clip_value)
        return clipped_result

    density = 1000  # Density of the linspace
    # Create a meshgrid for x and y values
    x_values = np.linspace(0.01, 20, density)  # Avoiding zero for x to prevent division by zero
    y_values = np.linspace(0.01, 1, density)
    x, y = np.meshgrid(x_values, y_values)
    clip_value = 3
    z = equation(x, y, clip_value)

    data_right = []
    pick_offset = 1
    for y_data in range(len(y) - 1, -1, -1):
        for x_data in range(len(x) - 1, -1, -1):
            if z[x_data][y_data] < clip_value:  # Pick the right edge of the cliff for interpolation
                try:
                    data_right.append([x[x_data + pick_offset][y_data], y[x_data + pick_offset][y_data],
                                       z[x_data + pick_offset][y_data]])
                    break
                except:  # If an element out of the list is requested the last element is picked
                    data_right.append([x[len(x) - 1][y_data], y[len(x) - 1][y_data], z[len(x) - 1][y_data]])
                    break

    fixed_point_right = [20, 1, 11]  # Fixed point (20, 2, 10)
    data_right.append(fixed_point_right)
    data_right = np.asarray(data_right, dtype=np.float32)

    # Separate the coordinates into X, Y, Z
    XY_right = data_right[:, 0:2]  # X, Y coordinates
    Z_right = data_right[:, 2]  # Z coordinate

    # Create polynomial features up to degree 2
    poly_right = PolynomialFeatures(degree=2)  # Very distant from the rift, a low degree can keep a steadier slope
    X_poly_right = poly_right.fit_transform(XY_right)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_right, Z_right)

    # Predict Z values using the model
    Z_pred_right = model.predict(X_poly_right)

    # Create a meshgrid for the surface plot
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    xy_mesh = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    xy_poly = poly_right.transform(xy_mesh)
    z_mesh_right = model.predict(xy_poly)
    z_mesh_right = z_mesh_right.reshape(x_mesh.shape)

    ## Left
    data_left = []
    pick_offset_left = 1
    for y_data in range(0, len(y)):
        for x_data in range(0, len(x)):
            if z[x_data][y_data] < clip_value:
                # print(z[x_data][y_data])
                try:
                    data_left.append([x[x_data - pick_offset_left][y_data], y[x_data - pick_offset_left][y_data],
                                      z[x_data - pick_offset_left][y_data]])
                    break
                except:  # If an element out of the list is requested the first element is picked
                    data_right.append([x[0][y_data], y[0][y_data], z[0][y_data]])
                    break

    fixed_point_left = [0, 0, 10]  # Fixed point (20, 2, 10)
    # bottom_right_corner = [20, 0, 0]
    data_left.append(fixed_point_left)
    data_left = np.asarray(data_left, dtype=np.float32)

    # Separate the coordinates into X, Y, Z
    XY_left = data_left[:, 0:2]  # X, Y coordinates
    Z_left = data_left[:, 2]  # Z coordinate

    # Create polynomial features up to degree 2
    poly_left = PolynomialFeatures(degree=3)  # Very close from the rift, a higher degree can converge faster
    X_poly_left = poly_left.fit_transform(XY_left)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_left, Z_left)

    # Predict Z values using the model
    Z_pred_left = model.predict(X_poly_left)

    # Create a meshgrid for the surface plot
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    xy_mesh = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    xy_poly = poly_left.transform(xy_mesh)
    z_mesh_left = model.predict(xy_poly)
    z_mesh_left = z_mesh_left.reshape(x_mesh.shape)
    ##

    z_def = []
    zx_def = []
    left = False  # Flag for the side attribution
    for y_data in range(len(y)):
        for x_data in range(len(x) - 1, -1, -1):
            if z[x_data][y_data] < clip_value:  # If its under the clip value I use the conventional function
                zx_def.append(z[x_data][y_data])
                left = True  # Once the rift is reached, the left side begins
            elif left:  # If is on the left side the left mesh is appended
                zx_def.append(z_mesh_left[x_data][y_data])
            else:  # Otherwise the right one is appended
                zx_def.append(z_mesh_right[x_data][y_data])
        left = False  # Change of row, starts from the right
        z_def.append(zx_def)
        zx_def = []
    # Conversion to numpy array and then to tensor
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    z_def = np.asarray(z_def, dtype=np.float32)

    x_values = tf.constant(x_values)
    y_values = tf.constant(y_values)
    z_def = tf.constant(z_def)

    return x_values, y_values, z_def


def init_loss_func_adv():  # Better version

    def equation_new(x, y, clip_value):
        k = 0.75
        j = 2.4
        epsilon = 0.13
        x_offset = 1.3  # Fine-tuning factor for x
        x += x_offset
        result = (1 - (k * ((x + 3) + y) + j * (-0.8 * (1. / (x + epsilon)) - 1 * (1 / (y + epsilon)))) ** 5) ** 2
        # Clipping the result
        clipped_result = np.clip(result, 0, clip_value)
        return clipped_result

    density = 1000  # Density of the linspace
    # Create a meshgrid for x and y values
    x_values = np.linspace(0.01, 20, density)  # Avoiding zero for x to prevent division by zero
    y_values = np.linspace(0.01, 1, density)
    x, y = np.meshgrid(x_values, y_values)
    clip_value = 0.15  # 0.65

    z = equation_new(x, y, clip_value)

    data_right = []
    pick_offset = 5
    for y_data in range(len(y) - 1, -1, -1):
        for x_data in range(len(x) - 1, -1, -1):
            if z[x_data][y_data] < clip_value:  # Pick the right edge of the cliff for interpolation
                try:
                    data_right.append([x[x_data + pick_offset][y_data], y[x_data + pick_offset][y_data],
                                       z[x_data + pick_offset][y_data]])
                    break
                except:  # If an element out of the list is requested the last element is picked
                    data_right.append([x[len(x) - 1][y_data], y[len(x) - 1][y_data], z[len(x) - 1][y_data]])
                    break

    fixed_point_right = [20, 1, 1]  # Fixed point (20, 2, 10), 20111
    # bottom_right_corner = [20, 0, 0]
    data_right.append(fixed_point_right)
    data_right = np.asarray(data_right, dtype=np.float32)

    # Separate the coordinates into X, Y, Z
    XY_right = data_right[:, 0:2]  # X, Y coordinates
    Z_right = data_right[:, 2]  # Z coordinate

    # Create polynomial features up to degree 2
    poly_right = PolynomialFeatures(degree=2)  # Very distant from the rift, a low degree can keep a steadier slope
    X_poly_right = poly_right.fit_transform(XY_right)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_right, Z_right)

    # Predict Z values using the model
    Z_pred_right = model.predict(X_poly_right)

    # Create a meshgrid for the surface plot
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    xy_mesh = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    xy_poly = poly_right.transform(xy_mesh)
    z_mesh_right = model.predict(xy_poly)
    z_mesh_right = z_mesh_right.reshape(x_mesh.shape)

    ## Left
    data_left = []
    pick_offset_left = 35  # prima era 1 ma 100 va bene????
    for y_data in range(0, len(y)):
        for x_data in range(0, len(x)):
            if z[x_data][y_data] < clip_value:
                # print(z[x_data][y_data])
                try:
                    data_left.append([x[x_data - pick_offset_left][y_data], y[x_data - pick_offset_left][y_data],
                                      z[x_data - pick_offset_left][y_data]])
                    break
                except:  # If an element out of the list is requested the first element is picked
                    data_right.append([x[0][y_data], y[0][y_data], z[0][y_data]])
                    break

    fixed_point_left = [0, 0, 1]  # Fixed point (20, 2, 10)
    perfect_score = [0, 1, 0]
    # bottom_right_corner = [20, 0, 0]
    data_left.append(fixed_point_left)
    data_left.append(perfect_score)
    data_left = np.asarray(data_left, dtype=np.float32)

    # Separate the coordinates into X, Y, Z
    XY_left = data_left[:, 0:2]  # X, Y coordinates
    Z_left = data_left[:, 2]  # Z coordinate

    # Create polynomial features up to degree 2
    poly_left = PolynomialFeatures(degree=2)  # Very close from the rift, a higher degree can converge faster
    X_poly_left = poly_left.fit_transform(XY_left)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_left, Z_left)

    # Predict Z values using the model
    Z_pred_left = model.predict(X_poly_left)

    # Create a meshgrid for the surface plot
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    xy_mesh = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    xy_poly = poly_left.transform(xy_mesh)
    z_mesh_left = model.predict(xy_poly)
    z_mesh_left = z_mesh_left.reshape(x_mesh.shape)
    ##

    z_def = []
    zx_def = []
    left = True  # Flag for the side attribution
    for y_data in range(len(y)):
        for x_data in range(len(x)):
            if z[x_data][y_data] < clip_value:  # If its under the clip value I use the conventional function
                zx_def.append(z[x_data][y_data])
                left = False  # Once the rift is reached, the right side begins
            elif left:  # If is on the left side the left mesh is appended
                zx_def.append(z_mesh_left[x_data][y_data])
            else:  # Otherwise the right one is appended
                zx_def.append(z_mesh_right[x_data][y_data])
        left = True  # Change of row, starts from the left
        z_def.append(zx_def)
        zx_def = []

    z_def = np.asarray(z_def, dtype=np.float32)

    for j in range(density):
        for k in range(density):
            z_def[j][k] = z_def[j][k] * 5
            if z_def[j][k] < 0:
                z_def[j][k] = 0

    # Conversion to numpy array and then to tensor
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    z_def = np.asarray(z_def, dtype=np.float32)

    x_values = tf.constant(x_values)
    y_values = tf.constant(y_values)
    z_def = tf.constant(z_def)

    return x_values, y_values, z_def


def init_loss_func_def2():  # Bestest version

    def equation_new(x, y, clip_value):
        k = 0.75
        j = 2.4
        epsilon = 0.13
        x_offset = 1.3  # Fine-tuning factor for x
        x += x_offset
        result = (1 - (k * ((x + 3) + y) + j * (-0.8 * (1. / (x + epsilon)) - 1 * (1 / (y + epsilon)))) ** 5) ** 2
        # Clipping the result
        clipped_result = np.clip(result, 0, clip_value)
        min_indices = np.argmin(clipped_result, axis=1)
        x -= x_offset
        return clipped_result, min_indices

    density = 1999  # Density of the linspace
    # Create a meshgrid for x and y values
    x_values = np.linspace(0.01, 20, density)  # Avoiding zero for x to prevent division by zero
    y_values = np.linspace(0.01, 1, density)
    x, y = np.meshgrid(x_values, y_values)
    clip_value = 0.15  # 0.65

    # Compute the value of the function (z) and find where the lowest value is found (min_indices)
    z, min_indices = equation_new(x, y, clip_value)

    data_side = []
    pick_offset = 0
    for y_data in range(len(y)):
        for x_data in range(len(x)):
            if x_data > min_indices[y_data]:  # Create a list of the lowest points
                try:
                    data_side.append([x[x_data + pick_offset][y_data], y[x_data + pick_offset][y_data], 0])
                    break
                except:  # If an element out of the list is requested, the last element is picked
                    data_side.append([x[len(x) - 1][y_data], y[len(x) - 1][y_data], 0])
                    break

    data_side_left = data_side
    fixed_point_left = [0, 0, 0.6]  # The starting point for the left function
    # fixed_point_left = [0, 0, 1]  # The starting point for the left function
    data_side_left.append(fixed_point_left)
    # Hard coding of the best and worst scores
    perfect_score = [0, 1, 0]
    worst_score = [20, 0, 0]
    data_side_left.append(perfect_score)
    data_side_left.append(worst_score)

    data_side_left = np.asarray(data_side_left, dtype=np.float32)
    # The left side (concave) and the right side (convex) of the curve will be handled differently
    # The right side use the clip values to draw its part, so it can have a concave shape, better suited for its purpose
    data_right = []
    pick_offset = 0
    for y_data in range(len(y) - 1, -1, -1):
        for x_data in range(len(x) - 1, -1, -1):
            if z[x_data][y_data] < clip_value:  # Pick the right edge of the cliff for interpolation
                try:
                    data_right.append([x[x_data + pick_offset][y_data], y[x_data + pick_offset][y_data],
                                       z[x_data + pick_offset][y_data]])
                    break
                except:  # If an element out of the list is requested the last element is picked
                    data_right.append([x[len(x) - 1][y_data], y[len(x) - 1][y_data], z[len(x) - 1][y_data]])
                    break

    fixed_point_right = [20, 1, 1]  # The starting point for the right function
    data_right.append(fixed_point_right)
    data_right = np.asarray(data_right, dtype=np.float32)

    # Separate the coordinates into X, Y, Z
    XY_right = data_right[:, 0:2]  # X, Y coordinates
    Z_right = data_right[:, 2]  # Z coordinate

    # Create polynomial features up to degree 2
    poly_right = PolynomialFeatures(degree=2,
                                    include_bias=True)  # Very distant from the rift, a low degree can keep a steadier slope
    X_poly_right = poly_right.fit_transform(XY_right)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_right, Z_right)

    # Create a meshgrid for the surface plot
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    xy_mesh = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    xy_poly = poly_right.transform(xy_mesh)
    z_mesh_right = model.predict(xy_poly)
    z_mesh_right = z_mesh_right.reshape(x_mesh.shape)

    # The left side uses a griddata interpolation approach, this ensures the desired shape
    grid_z_left = griddata(data_side_left[:, 0:2], data_side_left[:, 2], (x, y), method='linear')

    z_def = []
    zx_def = []
    for y_data in range(len(y)):
        for x_data in range(len(x)):
            # To prevent cliffs the highest value is consistenly chosen
            if grid_z_left[x_data][y_data] > z_mesh_right[x_data][y_data]:
                zx_def.append(grid_z_left[x_data][y_data])
            else:  # Otherwise the right one is appended
                zx_def.append(z_mesh_right[x_data][y_data])
        z_def.append(zx_def)  # Creation of the definitive table
        zx_def = []
    z_def = np.asarray(z_def, dtype=np.float32)

    min_values = np.min(z_def, axis=0)  # minimum value of each row
    max_values = np.max(z_def, axis=1)  # maximum value of each row

    for y_data in range(density):  # Row selection
        for x_data in range(density):  # Resize, for a more precise function
            z_def[x_data][y_data] = (z_def[x_data][y_data] - min_values[y_data]) * (max_values[y_data] /
                                                                                    (max_values[y_data] - min_values[
                                                                                        y_data]))

    # Conversion to numpy array and then to tensor
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    z_def = np.asarray(z_def, dtype=np.float32)

    for j in range(density):  # Magnification of the function, this allows for steeper derivatives and faster training
        for k in range(density):
            z_def[j][k] = z_def[j][k] * 5
            if z_def[j][k] < 0:  # Negative values are unacceptable for a loss function
                z_def[j][k] = 0

    z_def = gaussian_filter(z_def, sigma=150, order=0, radius=40)  # Smooths out the crinkles, for a better training

    z_def = np.hstack([z_def[:, 0].reshape(-1, 1), z_def])
    z_def = np.vstack([z_def[0], z_def])
    z_def[0, density] = 0
    z_def[density, 0] = 0

    min_values = np.min(z_def, axis=1)
    # Subtract the minimum value of each row from all elements in that row
    z_def = z_def - min_values[:, np.newaxis]

    x_values = tf.constant(x_values)
    y_values = tf.constant(y_values)
    z_def = tf.constant(z_def)

    return x_values, y_values, z_def


def init_loss_func_def():  # Best version

    def equation_new(x, y, clip_value):
        k = 0.75
        j = 2.4
        epsilon = 0.13
        x_offset = 1.3  # Fine-tuning factor for x
        x += x_offset
        result = (1 - (k * ((x + 3) + y) + j * (-0.8 * (1. / (x + epsilon)) - 1 * (1 / (y + epsilon)))) ** 5) ** 2
        # Clipping the result
        clipped_result = np.clip(result, 0, clip_value)
        min_indices = np.argmin(clipped_result, axis=1)
        x -= x_offset
        return clipped_result, min_indices

    density = 1000  # Density of the linspace
    # Create a meshgrid for x and y values
    x_values = np.linspace(0.01, 20, density)  # Avoiding zero for x to prevent division by zero
    y_values = np.linspace(0.01, 1, density)
    x, y = np.meshgrid(x_values, y_values)
    clip_value = 0.15  # 0.65

    # Compute the value of the function (z) and find where the lowest value is found (min_indices)
    z, min_indices = equation_new(x, y, clip_value)

    data_side = []
    pick_offset = 0
    for y_data in range(len(y)):
        for x_data in range(len(x)):
            if x_data > min_indices[y_data]:  # Create a list of the lowest points
                try:
                    data_side.append([x[x_data + pick_offset][y_data], y[x_data + pick_offset][y_data], 0])
                    break
                except:  # If an element out of the list is requested, the last element is picked
                    data_side.append([x[len(x) - 1][y_data], y[len(x) - 1][y_data], 0])
                    break

    data_side_left = data_side
    fixed_point_left = [0, 0, 0.6]  # The starting point for the left function
    data_side_left.append(fixed_point_left)
    # Hard coding of the best and worst scores
    perfect_score = [0, 1, 0]
    worst_score = [20, 0, 0]
    data_side_left.append(perfect_score)
    data_side_left.append(worst_score)

    data_side_left = np.asarray(data_side_left, dtype=np.float32)
    data_side = np.asarray(data_side, dtype=np.float32)
    # The left side (concave) and the right side (convex) of the curve will be handled differently
    # The right side use the clip values to draw its part, so it can have a concave shape, better suited for its purpose
    data_right = []
    pick_offset = 0
    for y_data in range(len(y) - 1, -1, -1):
        for x_data in range(len(x) - 1, -1, -1):
            if z[x_data][y_data] < clip_value:  # Pick the right edge of the cliff for interpolation
                try:
                    data_right.append([x[x_data + pick_offset][y_data], y[x_data + pick_offset][y_data],
                                       z[x_data + pick_offset][y_data]])
                    break
                except:  # If an element out of the list is requested the last element is picked
                    data_right.append([x[len(x) - 1][y_data], y[len(x) - 1][y_data], z[len(x) - 1][y_data]])
                    break

    fixed_point_right = [20, 1, 1]  # The starting point for the right function
    data_right.append(fixed_point_right)
    data_right = np.asarray(data_right, dtype=np.float32)

    # Separate the coordinates into X, Y, Z
    XY_right = data_right[:, 0:2]  # X, Y coordinates
    Z_right = data_right[:, 2]  # Z coordinate

    # Create polynomial features up to degree 2
    poly_right = PolynomialFeatures(degree=2,
                                    include_bias=True)  # Very distant from the rift, a low degree can keep a steadier slope
    X_poly_right = poly_right.fit_transform(XY_right)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_poly_right, Z_right)

    # Predict Z values using the model
    Z_pred_right = model.predict(X_poly_right)

    # Create a meshgrid for the surface plot
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    xy_mesh = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    xy_poly = poly_right.transform(xy_mesh)
    z_mesh_right = model.predict(xy_poly)
    z_mesh_right = z_mesh_right.reshape(x_mesh.shape)

    # The left side uses a griddata interpolation approach, this ensures the desired shape
    grid_z_left = griddata(data_side_left[:, 0:2], data_side_left[:, 2], (x, y), method='linear')

    z_def = []
    zx_def = []
    left = True  # Flag for the side attribution
    for y_data in range(len(y)):
        for x_data in range(len(x)):
            # To prevent cliffs the highest value is consistenly chosen
            if grid_z_left[x_data][y_data] > z_mesh_right[x_data][y_data]:
                zx_def.append(grid_z_left[x_data][y_data])
            else:  # Otherwise the right one is appended
                zx_def.append(z_mesh_right[x_data][y_data])
        z_def.append(zx_def)  # Creation of the definitive table
        zx_def = []
    z_def = np.asarray(z_def, dtype=np.float32)

    min_values = np.min(z_def, axis=0)  # minimum value of each row
    max_values = np.max(z_def, axis=1)  # maximum value of each row


    for y_data in range(density):  # Row selection
        for x_data in range(density):  # Resize, for a more precise function
            z_def[x_data][y_data] = (z_def[x_data][y_data] - min_values[y_data]) * (max_values[y_data] /
                                                                                    (max_values[y_data] - min_values[
                                                                                        y_data]))

    # Conversion to numpy array and then to tensor
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    z_def = np.asarray(z_def, dtype=np.float32)

    for j in range(density):  # Magnification of the function, this allows for steeper derivatives and faster training
        for k in range(density):
            z_def[j][k] = z_def[j][k] * 5
            if z_def[j][k] < 0:  # Negative values are unacceptable for a loss function
                z_def[j][k] = 0

    z_def = gaussian_filter(z_def, sigma=150, order=0, radius=40)  # Smooths out the crinkles, for a better training

    x_values = tf.constant(x_values)
    y_values = tf.constant(y_values)
    z_def = tf.constant(z_def)

    return x_values, y_values, z_def


def loss_b(distances, grades):
    # This type of loss function uses lists so it cannot be used during training but only for debug purposes
    loss = []
    for j in range(len(grades)):
        loss[j] = (1 - (0.4 * (distances[j] + grades[j]) + 1 * (-1 / distances[j] - 1 / grades[j])) ** 2) ** 2
    average_loss = sum(loss) / len(loss)
    return average_loss

def loss_bskc(penalty, grades, z_loss):
    xy_ref_min = [0., 0.]  # Minimum values for both axis
    xy_ref_max = [20, 1]  # Maximum values for both axis

    # Reshape inputs for batch processing
    penalty_reshaped = tf.reshape(penalty, [-1, 1])  # Reshape simgics
    grades_reshaped = tf.reshape(grades, [-1, 1])  # Reshape grades
    unknown_val_notshaped = tf.stack([penalty_reshaped, grades_reshaped], axis=1)  # Stack the two vectors one aside the other
    unknown_val = tf.reshape(unknown_val_notshaped, [-1, 2])
    z_interpolated = tfp.math.batch_interp_regular_nd_grid(unknown_val, xy_ref_min, xy_ref_max, z_loss, axis=-2)
    average_loss = K.mean(z_interpolated)
    # info_name = 'reports/z_interp.txt'
    # info = [  #'\nPenalty before: \n' + str(penalty),
    #          '\nGrades before: \n' + str(grades),
    #          '\nPenalty after: \n' + str(penalty_reshaped),
    #          '\nGrades after: \n' + str(grades_reshaped),
    #          '\nStacked before: \n' + str(unknown_val_notshaped),
    #          '\nStacked after: \n' + str(unknown_val),
    #          '\nZ interpolation: \n' + str(z_interpolated),
    #          '\nAverage loss: \n' + str(average_loss),
    #          ]
    # write_to_file(info_name, info)
    return average_loss


def one_hot_encode_ontology(X):
    X['Ontology'] = X['Ontology'].astype(int)
    # Get one hot encoding of Ontology column
    one_hot = pd.get_dummies(X['Ontology'], dtype=float)
    # Drop Ontology column as it is now encoded
    X = X.drop('Ontology', axis=1)
    # Join the encoded df
    X = X.join(one_hot)
    X.rename(
        columns={1: "Cell_comp", 2: "Mol_funcs", 0: "Bio_process"},
        inplace=True,
    )
    return X

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

# @keras.saving.register_keras_serializable(package="my_package", name="loss_bskc")
# def loss_bskc_modded(penalty, grades, z_loss, multiplier):
#     xy_ref_min = [0., 0.]  # Minimum values for both axis
#     xy_ref_max = [20, 1]  # Maximum values for both axis
#     penalty_reshaped = tf.reshape(penalty, [-1, 1])  # Reshape simgics
#     mask = tf.equal(penalty_reshaped, 20.)
#     mask = tf.cast(mask, dtype=tf.float32)
#     grades_reshaped = tf.reshape(grades, [-1, 1])  # Reshape grades
#     unknown_val_notshaped = tf.stack([penalty_reshaped, grades_reshaped], axis=1)  # Stack the two vectors one aside the other
#     unknown_val = tf.reshape(unknown_val_notshaped, [-1, 2])
#     z_interpolated = tfp.math.batch_interp_regular_nd_grid(unknown_val, xy_ref_min, xy_ref_max, z_loss, axis=-2)
#     z_interpolated_modded = tf.where(tf.cast(mask, dtype=tf.bool), z_interpolated * multiplier, z_interpolated)
#     average_loss = K.mean(z_interpolated_modded)
#     # info_name = 'reports/z_interp500.txt'
#     # info = [  #'\nPenalty before: \n' + str(penalty),
#     #          '\nGrades before: \n' + str(grades),
#     #          '\nPenalty after: \n' + str(penalty_reshaped),
#     #          '\nMask:\n' + str(mask),
#     #          '\nGrades after: \n' + str(grades_reshaped),
#     #          '\nStacked before: \n' + str(unknown_val_notshaped),
#     #          '\nStacked after: \n' + str(unknown_val),
#     #          '\nZ interpolation: \n' + str(z_interpolated),
#     #          '\nZ interpolation modded: \n' + str(z_interpolated_modded),
#     #          '\nAverage loss: \n' + str(average_loss),
#     #          ]
#     # write_to_file(info_name, info)
#     return average_loss


def loss_bskc_modded(penalty, grades, z_loss, multiplier):
    xy_ref_min = [0., 0.]  # Minimum values for both axis
    xy_ref_max = [20, 1]  # Maximum values for both axis
    penalty_reshaped = tf.reshape(penalty, [-1, 1])  # Reshape simgics
    mask = tf.equal(penalty_reshaped, 20.)
    mask = tf.cast(mask, dtype=tf.float32)
    grades_reshaped = tf.reshape(grades, [-1, 1])  # Reshape grades
    unknown_val_notshaped = tf.stack([penalty_reshaped, grades_reshaped], axis=1)  # Stack the two vectors one aside the other
    unknown_val = tf.reshape(unknown_val_notshaped, [-1, 2])
    z_interpolated = tfp.math.batch_interp_regular_nd_grid(unknown_val, xy_ref_min, xy_ref_max, z_loss, axis=-2)
    z_interpolated_modded = tf.where(tf.cast(mask, dtype=tf.bool), z_interpolated * multiplier, z_interpolated)
    average_loss = K.mean(z_interpolated_modded)
    # info_name = 'reports/z_interp500.txt'
    # info = [  #'\nPenalty before: \n' + str(penalty),
    #          '\nGrades before: \n' + str(grades),
    #          '\nPenalty after: \n' + str(penalty_reshaped),
    #          '\nMask:\n' + str(mask),
    #          '\nGrades after: \n' + str(grades_reshaped),
    #          '\nStacked before: \n' + str(unknown_val_notshaped),
    #          '\nStacked after: \n' + str(unknown_val),
    #          '\nZ interpolation: \n' + str(z_interpolated),
    #          '\nZ interpolation modded: \n' + str(z_interpolated_modded),
    #          '\nAverage loss: \n' + str(average_loss),
    #          ]
    # write_to_file(info_name, info)
    return average_loss



def loss_bskc_legacy(penalty, grades, z_loss): # compatible with init_loss_def1
    xy_ref_min = [0.01, 0.01]  # Minimum values for both axis
    xy_ref_max = [20, 1]  # Maximum values for both axis

    # Reshape inputs for batch processing
    penalty_reshaped = tf.reshape(penalty, [-1, 1])  # Reshape simgics
    grades_reshaped = tf.reshape(grades, [-1, 1])  # Reshape grades
    unknown_val_notshaped = tf.stack([penalty_reshaped, grades_reshaped], axis=1)  # Stack the two vectors one aside the other
    unknown_val = tf.reshape(unknown_val_notshaped, [-1, 2])
    z_interpolated = tfp.math.batch_interp_regular_nd_grid(unknown_val, xy_ref_min, xy_ref_max, z_loss, axis=-2)
    average_loss = K.mean(z_interpolated)
    return average_loss


@tf.function(reduce_retracing=True)
def training_step_bskc(x_batch_train, simgics_tensor, optimizer, autoencoder, z_loss, multiplier):
    with tf.GradientTape() as tape:  # Run the forward pass of the layer.
        # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        grades = autoencoder(x_batch_train, training=True)  # Logits for this minibatch

        # print('distances: ' + str(distances))
        # print('grades: ' + str(grades))
        # simgics_tensor = tf.convert_to_tensor(simgics, dtype=tf.float32)  # Passing tensors to reduce tracing
        grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
        linear_penalty = (1 - simgics_tensor) * 20  # Compute a linear penalty from the simgics to feed to the loss

        # Compute the loss value for this minibatch.
        loss_value = loss_bskc_modded(linear_penalty, grades_tensor, z_loss, multiplier)


    # Use the gradient tape to automatically retrieve the gradients of the trainable
    # variables with respect to the loss.
    # print(loss_value)
    grads = tape.gradient(loss_value, autoencoder.trainable_weights)
    # Run one step of gradient descent by updating the value of the variables to minimize the loss.
    # optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
    optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
    return loss_value


@tf.function(reduce_retracing=True)
def validation_step_bskc(x_batch_val, simgics_tensor, autoencoder, z_loss):
    grades = autoencoder(x_batch_val, training=False)  # Logits for this minibatch
    # print('distances: ' + str(distances))
    # print('grades: ' + str(grades))
    # simgics_tensor = tf.convert_to_tensor(simgics, dtype=tf.float32)  # Passing tensors to reduce tracing
    grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
    print(grades_tensor)
    linear_penalty = (1 - simgics_tensor) * 20  # Compute a linear penalty from the simgics to feed to the loss
    print(linear_penalty)
    # Compute the loss value for this minibatch.
    loss_value = loss_bskc(linear_penalty, grades_tensor, z_loss)
    print(loss_value)
    return loss_value



# multiplier = 1/multiplier

def loss_MSE_modded(penalty, grades, multiplier=1.0):
    # xy_ref_min = [0., 0.]  # Minimum values for both axis
    # xy_ref_max = [20, 1]  # Maximum values for both axis
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


def loss_MAE_modded(penalty, grades, multiplier=1.0):
    # xy_ref_min = [0., 0.]  # Minimum values for both axis
    # xy_ref_max = [20, 1]  # Maximum values for both axis
    penalty_reshaped = tf.reshape(penalty, [-1, 1])  # Reshape simgics
    # mask = tf.equal(penalty_reshaped, 1.)
    mask = tf.greater_equal(penalty_reshaped, 0.75)  # 0.75 is the requirement for the simgic to be considered good
    mask = tf.cast(mask, dtype=tf.float32)
    grades_reshaped = tf.reshape(grades, [-1, 1])  # Reshape grades
    # modded_val_notshaped = tf.stack([penalty_reshaped, grades_reshaped], axis=1)  # Stack the two vectors one aside the other
    # modded_val = tf.reshape(modded_val_notshaped, [-1, 2])
    loss_SE = K.abs(penalty_reshaped - grades_reshaped)
    loss_SE_modded = tf.where(tf.cast(mask, dtype=tf.bool), loss_SE * multiplier, loss_SE)
    loss_MSE = K.mean(loss_SE_modded)
    return loss_MSE


def loss_MSE_unmodded(penalty, grades, multiplier=1.0):
    # xy_ref_min = [0., 0.]  # Minimum values for both axis
    # xy_ref_max = [20, 1]  # Maximum values for both axis
    multiplier = 1.0
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
    # loss_MSE = K.mean(loss_SE_modded)
    # info_name = 'reports/z_interp_new.txt'
    # info = [  #'\nPenalty before: \n' + str(penalty),
    #          '\nGrades before: \n' + str(grades),
    #          '\nPenalty after: \n' + str(penalty_reshaped),
    #          '\nMask:\n' + str(mask),
    #          '\nGrades after: \n' + str(grades_reshaped),
    #          '\nStacked before: \n' + str(modded_val_notshaped),
    #          '\nStacked after: \n' + str(modded_val),
    #          '\nSquare error: \n' + str(loss_SE),
    #          '\nSE after mod: \n' + str(loss_SE_modded),
    #          '\nAverage loss: \n' + str(loss_MSE),
    #          ]
    # write_to_file(info_name, info)
    return loss_MSE


@tf.function(reduce_retracing=True)
def training_step_MSE(x_batch_train, simgics_tensor, optimizer, autoencoder, multiplier=1.0):
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
def training_step_MAE(x_batch_train, simgics_tensor, optimizer, autoencoder, multiplier=1.0):
    with tf.GradientTape() as tape:  # Run the forward pass of the layer.
        # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        grades = autoencoder(x_batch_train, training=True)  # Logits for this minibatch
        # print('distances: ' + str(distances))
        # print('grades: ' + str(grades))
        # simgics_tensor = tf.convert_to_tensor(simgics, dtype=tf.float32)  # Passing tensors to reduce tracing
        grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
        # Compute the loss value for this minibatch.
        loss_value = loss_MAE_modded(simgics_tensor, grades_tensor, multiplier)

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


@tf.function(reduce_retracing=True)
def validation_step_MAE(x_batch_val, simgics_tensor, autoencoder):
    grades = autoencoder(x_batch_val, training=False)  # Logits for this minibatch
    # print('distances: ' + str(distances))
    # print('grades: ' + str(grades))
    # simgics_tensor = tf.convert_to_tensor(simgics, dtype=tf.float32)  # Passing tensors to reduce tracing
    grades_tensor = tf.convert_to_tensor(grades, dtype=tf.float32)  # and to optimize the processes
    # Compute the loss value for this minibatch.
    loss_value = loss_MAE_modded(simgics_tensor, grades_tensor, 1.0)  # During validation multiplier is always 1
    return loss_value

# 4654654654



@tf.function(reduce_retracing=True)
def eval_step_bkc(x_batch_val, y_batch_val, autoencoder):
    val_logits = autoencoder(x_batch_val, training=False)
    # Update val metrics
    # val_acc_metric.update_state(y_batch_val, val_logits)


def query_simgic(collection, go1id_value, go2id_value):
    result = collection.find_one({"go1id": go1id_value, "go2id": go2id_value}, {"_id": 0, "simgic": 1})
    try:
        simgic_value = result.get("simgic")
        # print(simgic_value)
        return simgic_value
    except:
        # print('simgic not found!')
        return 0.  # Return 0 if no matching document is found to keep alive the routine


def plot_maker(x, y, name: str, curve_label: str = None,
               xlabel: str =None, ylabel: str = None, title: str = None):
    # x, y: the list containing the coordinates of the points
    # xlabel and ylabel: the label of the axis
    # title: the title of the plot
    # name: the name and path required to save the plot
    plt.figure(figsize=(8, 8))
    if curve_label is not None:
        plt.plot(x, y, label=curve_label)
    else:
        plt.plot(x, y)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if curve_label is not None:
        plt.legend()
    plt.savefig(name)  # Save the plot as an image file (e.g., PNG format)
    plt.close()


def plot_maker_nofig(x, y, name: str, curve_label: str = None,
                     xlabel: str = None, ylabel: str = None, title: str = None):
    # x, y: the list containing the coordinates of the points
    # xlabel and ylabel: the label of the axis
    # title: the title of the plot
    # name: the name and path required to save the plot
    if curve_label is not None:
        plt.plot(x, y, label=curve_label)
    else:
        plt.plot(x, y)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if curve_label is not None:
        plt.legend()
    plt.savefig(name)  # Save the plot as an image file (e.g., PNG format)


def kfold_subdivision(X, Y, batch_size=30, test_percentage=15, buffer_size=100):
    n_folds = 6  # 85/6 = 14.17%
    kfold_val_percentage = (100 - test_percentage) / n_folds
    kfold_val_quota = round(X.shape[0] * 0.01 * kfold_val_percentage)
    test_quota = round(X.shape[0] * 0.01 * test_percentage)

    X = X[:-test_quota]
    Y = Y[:-test_quota]

    # K-fold splitting
    X_val1 = X[:kfold_val_quota]
    Y_val1 = Y[:kfold_val_quota]
    X_train1 = X[kfold_val_quota:]
    Y_train1 = Y[kfold_val_quota:]
    train_dataset1 = tf.data.Dataset.from_tensor_slices((X_train1, Y_train1))
    train_dataset1 = train_dataset1.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset1 = tf.data.Dataset.from_tensor_slices((X_val1, Y_val1))
    val_dataset1 = val_dataset1.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    X_val2 = X[kfold_val_quota:kfold_val_quota * 2]
    Y_val2 = Y[kfold_val_quota:kfold_val_quota * 2]
    X_train2 = np.concatenate((X[:kfold_val_quota], X[kfold_val_quota * 2:]))
    Y_train2 = np.concatenate((Y[:kfold_val_quota], Y[kfold_val_quota * 2:]))
    train_dataset2 = tf.data.Dataset.from_tensor_slices((X_train2, Y_train2))
    train_dataset2 = train_dataset2.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset2 = tf.data.Dataset.from_tensor_slices((X_val2, Y_val2))
    val_dataset2 = val_dataset2.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    X_val3 = X[kfold_val_quota * 2:kfold_val_quota * 3]
    Y_val3 = Y[kfold_val_quota * 2:kfold_val_quota * 3]
    X_train3 = np.concatenate((X[:kfold_val_quota * 2], X[kfold_val_quota * 3:]))
    Y_train3 = np.concatenate((Y[:kfold_val_quota * 2], Y[kfold_val_quota * 3:]))
    train_dataset3 = tf.data.Dataset.from_tensor_slices((X_train3, Y_train3))
    train_dataset3 = train_dataset3.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset3 = tf.data.Dataset.from_tensor_slices((X_val3, Y_val3))
    val_dataset3 = val_dataset3.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    X_val4 = X[kfold_val_quota * 3:kfold_val_quota * 4]
    Y_val4 = Y[kfold_val_quota * 3:kfold_val_quota * 4]
    X_train4 = np.concatenate((X[:kfold_val_quota * 3], X[kfold_val_quota * 4:]))
    Y_train4 = np.concatenate((Y[:kfold_val_quota * 3], Y[kfold_val_quota * 4:]))
    train_dataset4 = tf.data.Dataset.from_tensor_slices((X_train4, Y_train4))
    train_dataset4 = train_dataset4.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset4 = tf.data.Dataset.from_tensor_slices((X_val4, Y_val4))
    val_dataset4 = val_dataset4.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    X_val5 = X[kfold_val_quota * 4:kfold_val_quota * 5]
    Y_val5 = Y[kfold_val_quota * 4:kfold_val_quota * 5]
    X_train5 = np.concatenate((X[:kfold_val_quota * 4], X[kfold_val_quota * 5:]))
    Y_train5 = np.concatenate((Y[:kfold_val_quota * 4], Y[kfold_val_quota * 5:]))
    train_dataset5 = tf.data.Dataset.from_tensor_slices((X_train5, Y_train5))
    train_dataset5 = train_dataset5.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset5 = tf.data.Dataset.from_tensor_slices((X_val5, Y_val5))
    val_dataset5 = val_dataset5.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    X_val0 = X[-kfold_val_quota:]
    Y_val0 = Y[-kfold_val_quota:]
    X_train0 = X[:-kfold_val_quota]
    Y_train0 = Y[:-kfold_val_quota]
    train_dataset0 = tf.data.Dataset.from_tensor_slices((X_train0, Y_train0))
    train_dataset0 = train_dataset0.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset0 = tf.data.Dataset.from_tensor_slices((X_val0, Y_val0))
    val_dataset0 = val_dataset0.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return (train_dataset0, val_dataset0, train_dataset1, val_dataset1, train_dataset2, val_dataset2, train_dataset3,
            val_dataset3, train_dataset4, val_dataset4, train_dataset5, val_dataset5)


def kfold_subdivision_resampled(X, y_prediction_quality, description_file, batch_size, test_percentage, resample_type,
                                over_sampling_strat, under_sampling_strat, smote_sampling_strat, smote_k_neighbors,
                                buffer_size=100, seed=42):

    n_folds = 6  # 85/6 = 14.17%
    kfold_val_percentage = (100 - test_percentage) / n_folds
    kfold_val_quota = round(X.shape[0] * 0.01 * kfold_val_percentage)
    test_quota = round(X.shape[0] * 0.01 * test_percentage)

    X = X[:-test_quota]
    Y = y_prediction_quality[:-test_quota]

    # K-fold splitting
    X_val1 = X[:kfold_val_quota]
    Y_val1 = X_val1['simgics_eval']
    X_val1 = X_val1.drop(['simgics_eval'], axis=1)

    X_train1 = X[kfold_val_quota:]
    Y_train1 = Y[kfold_val_quota:]

    X_train1['Ontology'] = X_train1.apply(determine_value, axis=1) # Remove one-hot encoding for SMOTENC
    X_train1 = X_train1.drop(columns=["Cell_comp", "Mol_funcs", "Bio_process"], inplace=False)

    if resample_type == 1:
        sme = RandomOverSampler(sampling_strategy=over_sampling_strat, random_state=seed)
    elif resample_type == 2:
        sme = RandomUnderSampler(sampling_strategy=under_sampling_strat, random_state=seed)
    elif resample_type == 3:
        sme = RandomOverSampler(sampling_strategy=over_sampling_strat, random_state=seed)
        sme2 = RandomUnderSampler(sampling_strategy=under_sampling_strat, random_state=seed)
    elif resample_type == 4:
        sme = SMOTENC(categorical_features=[X_train1.columns.get_loc('Ontology')], sampling_strategy=smote_sampling_strat,
                      random_state=seed, k_neighbors=smote_k_neighbors)  # The categorical values are where the column ontology is

    X_train1_res, dummy_Y_train1_res = sme.fit_resample(X_train1, Y_train1)
    if resample_type == 3:
        X_train1_res, dummy_Y_train1_res = sme2.fit_resample(X_train1_res, dummy_Y_train1_res)
    elif resample_type == 4:
        X_train1_res['simgics_eval'] = X_train1_res['simgics_eval'].clip(lower=0, upper=1)  # Respect bounds of simgics
    Y_train1_res = X_train1_res['simgics_eval']  # Simgics is the true Y to use
    X_train1_res = one_hot_encode_ontology(X_train1_res)  #Reinstate one-hot encoding for training
    X_train1_res = X_train1_res.drop(['simgics_eval'], axis=1)  #Useless since is in Y

    kfold1_description = ["True length of kfold 1: " + str(X_train1_res.shape[0])]
    write_to_file(description_file, kfold1_description)
    train_dataset1 = tf.data.Dataset.from_tensor_slices((X_train1_res, Y_train1_res))
    train_dataset1 = train_dataset1.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset1 = tf.data.Dataset.from_tensor_slices((X_val1, Y_val1))
    val_dataset1 = val_dataset1.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)


    X_val2 = X[kfold_val_quota:kfold_val_quota * 2]
    Y_val2 = X_val2['simgics_eval']
    X_val2 = X_val2.drop(['simgics_eval'], axis=1)

    X_train2 = pd.concat((X[:kfold_val_quota], X[kfold_val_quota * 2:]))
    Y_train2 = pd.concat((Y[:kfold_val_quota], Y[kfold_val_quota * 2:]))

    X_train2['Ontology'] = X_train2.apply(determine_value, axis=1)  # Remove one-hot encoding for SMOTENC
    X_train2 = X_train2.drop(columns=["Cell_comp", "Mol_funcs", "Bio_process"], inplace=False)
    # sme = SMOTENC(categorical_features=[X_train2.columns.get_loc('Ontology')], sampling_strategy=sampling_strat,
    #               random_state=seed, k_neighbors=k_neighbors)  # The categorical values are where the column ontology is
    X_train2_res, dummy_Y_train2_res = sme.fit_resample(X_train2, Y_train2)
    if resample_type == 3:
        X_train2_res, dummy_Y_train2_res = sme2.fit_resample(X_train2_res, dummy_Y_train2_res)
    elif resample_type == 4:
        X_train2_res['simgics_eval'] = X_train2_res['simgics_eval'].clip(lower=0, upper=1)  # Respect bounds of simgics
    Y_train2_res = X_train2_res['simgics_eval']  # Simgics is the true Y to use
    X_train2_res = one_hot_encode_ontology(X_train2_res)  # Reinstate one-hot encoding for training
    X_train2_res = X_train2_res.drop(['simgics_eval'], axis=1)  # Useless since is in Y

    kfold2_description = ["True length of kfold 2: " + str(X_train2_res.shape[0])]
    write_to_file(description_file, kfold2_description)
    train_dataset2 = tf.data.Dataset.from_tensor_slices((X_train2_res, Y_train2_res))
    train_dataset2 = train_dataset2.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset2 = tf.data.Dataset.from_tensor_slices((X_val2, Y_val2))
    val_dataset2 = val_dataset2.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)


    X_val3 = X[kfold_val_quota * 2:kfold_val_quota * 3]
    Y_val3 = X_val3['simgics_eval']
    X_val3 = X_val3.drop(['simgics_eval'], axis=1)

    X_train3 = pd.concat((X[:kfold_val_quota * 2], X[kfold_val_quota * 3:]))
    Y_train3 = pd.concat((Y[:kfold_val_quota * 2], Y[kfold_val_quota * 3:]))

    X_train3['Ontology'] = X_train3.apply(determine_value, axis=1)  # Remove one-hot encoding for SMOTENC
    X_train3 = X_train3.drop(columns=["Cell_comp", "Mol_funcs", "Bio_process"], inplace=False)
    # sme = SMOTENC(categorical_features=[X_train3.columns.get_loc('Ontology')], sampling_strategy=sampling_strat,
    #               random_state=seed, k_neighbors=k_neighbors)  # The categorical values are where the column ontology is
    X_train3_res, dummy_Y_train3_res = sme.fit_resample(X_train3, Y_train3)
    if resample_type == 3:
        X_train3_res, dummy_Y_train3_res = sme2.fit_resample(X_train3_res, dummy_Y_train3_res)
    elif resample_type == 4:
        X_train3_res['simgics_eval'] = X_train3_res['simgics_eval'].clip(lower=0, upper=1)  # Respect bounds of simgics
    Y_train3_res = X_train3_res['simgics_eval']  # Simgics is the true Y to use
    X_train3_res = one_hot_encode_ontology(X_train3_res)  # Reinstate one-hot encoding for training
    X_train3_res = X_train3_res.drop(['simgics_eval'], axis=1)  # Useless since is in Y

    kfold3_description = ["True length of kfold 3: " + str(X_train3_res.shape[0])]
    write_to_file(description_file, kfold3_description)
    train_dataset3 = tf.data.Dataset.from_tensor_slices((X_train3_res, Y_train3_res))
    train_dataset3 = train_dataset3.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset3 = tf.data.Dataset.from_tensor_slices((X_val3, Y_val3))
    val_dataset3 = val_dataset3.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)


    X_val4 = X[kfold_val_quota * 3:kfold_val_quota * 4]
    Y_val4 = X_val4['simgics_eval']
    X_val4 = X_val4.drop(['simgics_eval'], axis=1)

    X_train4 = pd.concat((X[:kfold_val_quota * 3], X[kfold_val_quota * 4:]))
    Y_train4 = pd.concat((Y[:kfold_val_quota * 3], Y[kfold_val_quota * 4:]))

    X_train4['Ontology'] = X_train4.apply(determine_value, axis=1)  # Remove one-hot encoding for SMOTENC
    X_train4 = X_train4.drop(columns=["Cell_comp", "Mol_funcs", "Bio_process"], inplace=False)
    # sme = SMOTENC(categorical_features=[X_train4.columns.get_loc('Ontology')], sampling_strategy=sampling_strat,
    #               random_state=seed, k_neighbors=k_neighbors)  # The categorical values are where the column ontology is
    X_train4_res, dummy_Y_train4_res = sme.fit_resample(X_train4, Y_train4)
    if resample_type == 3:
        X_train4_res, dummy_Y_train4_res = sme2.fit_resample(X_train4_res, dummy_Y_train4_res)
    elif resample_type == 4:
        X_train4_res['simgics_eval'] = X_train4_res['simgics_eval'].clip(lower=0, upper=1)  # Respect bounds of simgics
    Y_train4_res = X_train4_res['simgics_eval']  # Simgics is the true Y to use
    X_train4_res = one_hot_encode_ontology(X_train4_res)  # Reinstate one-hot encoding for training
    X_train4_res = X_train4_res.drop(['simgics_eval'], axis=1)  # Useless since is in Y

    kfold4_description = ["True length of kfold 4: " + str(X_train4_res.shape[0])]
    write_to_file(description_file, kfold4_description)
    train_dataset4 = tf.data.Dataset.from_tensor_slices((X_train4_res, Y_train4_res))
    train_dataset4 = train_dataset4.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset4 = tf.data.Dataset.from_tensor_slices((X_val4, Y_val4))
    val_dataset4 = val_dataset4.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)


    X_val5 = X[kfold_val_quota * 4:kfold_val_quota * 5]
    Y_val5 = X_val5['simgics_eval']
    X_val5 = X_val5.drop(['simgics_eval'], axis=1)

    X_train5 = pd.concat((X[:kfold_val_quota * 4], X[kfold_val_quota * 5:]))
    Y_train5 = pd.concat((Y[:kfold_val_quota * 4], Y[kfold_val_quota * 5:]))

    X_train5['Ontology'] = X_train5.apply(determine_value, axis=1)  # Remove one-hot encoding for SMOTENC
    X_train5 = X_train5.drop(columns=["Cell_comp", "Mol_funcs", "Bio_process"], inplace=False)
    # sme = SMOTENC(categorical_features=[X_train5.columns.get_loc('Ontology')], sampling_strategy=sampling_strat,
    #               random_state=seed, k_neighbors=k_neighbors)  # The categorical values are where the column ontology is
    X_train5_res, dummy_Y_train5_res = sme.fit_resample(X_train5, Y_train5)
    if resample_type == 3:
        X_train5_res, dummy_Y_train5_res = sme2.fit_resample(X_train5_res, dummy_Y_train5_res)
    elif resample_type == 4:
        X_train5_res['simgics_eval'] = X_train5_res['simgics_eval'].clip(lower=0, upper=1)  # Respect bounds of simgics
    Y_train5_res = X_train5_res['simgics_eval']  # Simgics is the true Y to use
    X_train5_res = one_hot_encode_ontology(X_train5_res)  # Reinstate one-hot encoding for training
    X_train5_res = X_train5_res.drop(['simgics_eval'], axis=1)  # Useless since is in Y

    kfold5_description = ["True length of kfold 5: " + str(X_train5_res.shape[0])]
    write_to_file(description_file, kfold5_description)
    train_dataset5 = tf.data.Dataset.from_tensor_slices((X_train5_res, Y_train5_res))
    train_dataset5 = train_dataset5.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset5 = tf.data.Dataset.from_tensor_slices((X_val5, Y_val5))
    val_dataset5 = val_dataset5.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)


    X_val0 = X[-kfold_val_quota:]
    Y_val0 = X_val0['simgics_eval']
    X_val0 = X_val0.drop(['simgics_eval'], axis=1)

    X_train0 = X[:-kfold_val_quota]
    Y_train0 = Y[:-kfold_val_quota]

    X_train0['Ontology'] = X_train0.apply(determine_value, axis=1)  # Remove one-hot encoding for SMOTENC
    X_train0 = X_train0.drop(columns=["Cell_comp", "Mol_funcs", "Bio_process"], inplace=False)
    # sme = SMOTENC(categorical_features=[X_train0.columns.get_loc('Ontology')], sampling_strategy=sampling_strat,
    #               random_state=seed, k_neighbors=k_neighbors)  # The categorical values are where the column ontology is
    X_train0_res, dummy_Y_train0_res = sme.fit_resample(X_train0, Y_train0)
    if resample_type == 3:
        X_train0_res, dummy_Y_train0_res = sme2.fit_resample(X_train0_res, dummy_Y_train0_res)
    elif resample_type == 4:
        X_train0_res['simgics_eval'] = X_train0_res['simgics_eval'].clip(lower=0, upper=1)  # Respect bounds of simgics
    Y_train0_res = X_train0_res['simgics_eval']  # Simgics is the true Y to use
    X_train0_res = one_hot_encode_ontology(X_train0_res)  # Reinstate one-hot encoding for training
    X_train0_res = X_train0_res.drop(['simgics_eval'], axis=1)  # Useless since is in Y

    kfold0_description = ["True length of kfold 0: " + str(X_train0_res.shape[0])]
    write_to_file(description_file, kfold0_description)
    train_dataset0 = tf.data.Dataset.from_tensor_slices((X_train0_res, Y_train0_res))
    train_dataset0 = train_dataset0.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset0 = tf.data.Dataset.from_tensor_slices((X_val0, Y_val0))
    val_dataset0 = val_dataset0.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return (train_dataset0, val_dataset0, train_dataset1, val_dataset1, train_dataset2, val_dataset2, train_dataset3,
            val_dataset3, train_dataset4, val_dataset4, train_dataset5, val_dataset5)


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


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def thresholder_f1(y_pred, y_true):
    thresholds = np.arange(0, 1, 0.02)
    scores = [f1score_sklearn(y_true, to_labels(y_pred, t), zero_division=0.0) for t in thresholds]
    # scores = [balanced_accuracy_score(y_true, to_labels(y_pred, t)) for t in thresholds]
    ix = np.argmax(scores)
    best_threshold = thresholds[ix]
    return best_threshold


def thresholder_bal_acc(y_pred, y_true):
    thresholds = np.arange(0, 1, 0.02)
    scores = [balanced_accuracy_score(y_true, to_labels(y_pred, t)) for t in thresholds]
    # scores = [balanced_accuracy_score(y_true, to_labels(y_pred, t)) for t in thresholds]
    ix = np.argmax(scores)
    best_threshold = thresholds[ix]
    best_bal_acc = scores[ix]
    return best_threshold, best_bal_acc

#Confrontation data made on unseparated data
def confrontation_benchmark(graph_test_loss_list, graph_simgics_test_list, good_autoencoder, top_dir_path,
                            formatted_datetime, batch_size, big_dataset, simgic_threshold=0.75,
                            confrontation_path_start=False, use_total_score=True, n_benchmark=10000, save_vars=False,
                            ontological_analysis=True, ont_based_prep=False, source_analysis=True):
    """ graph_test_loss_list: List of losses achieved during testing
    graph_simgics_test_list: List of true simgics evaluated during testing
    good_autoencoder: the model that is being evaluated
    top_dir_path: directory in which graphs are saved
    formatted_datetime: datetime to differentiate the data
    confrontation_path_start: the path in which the confrontations with spline are saved, if absent, no computation is made
    use_total_score: does the model uses total scores
    n_benchmark: amount of point plotted in the prediction graphs
    save_vars: Save the computed variables, for debugging purposes
    ont_based_prep: If the rows are scaled indipendently for each ontology"""

    if use_total_score:  # Tags name files if the total score is used during training (tsu: Total Score Used)
        score_tag = '_tsu'
    else:
        score_tag = ''

    # Plot test loss over time
    plt.figure(figsize=(9, 9))
    plt.scatter(graph_test_loss_list, graph_simgics_test_list, alpha=0.2)
    plt.xlabel("Test Loss")
    plt.ylabel("Simgic")
    plt.title("Test Simgic Plot")
    plt.savefig(top_dir_path + f"/simgics_test_graph_{formatted_datetime}{score_tag}.png")
    plt.close()

    # xmin = min(graph_test_loss_list)
    xmax = max(graph_test_loss_list)
    # ymin = min(graph_simgics_test_list)
    # ymax = max(graph_simgics_test_list)

    # Histogram version
    n_bins = 20
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(graph_test_loss_list, graph_simgics_test_list, bins=n_bins,
                                          range=[[0, xmax], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    cmap = colormaps.get_cmap('summer')  # Get desired colormap - you can change this!
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.xlabel("Loss")
    plt.ylabel("Simgic")
    plt.title("3D Histogram simgic/loss")
    plt.savefig(top_dir_path + f"/hist_test_loss_graph_{formatted_datetime}{score_tag}.png")
    plt.close()

    # Default dataset, shared with every computation for confrontability
    if big_dataset:
        dataset_path = 'stats/dataset/simgics_17408776_complete_test_set.csv'
    else:
        dataset_path = 'stats/dataset/simgics_1000000_complete_test_11.csv'
    dataset_data = pd.read_csv(dataset_path)
    simgics_evals = dataset_data['simgics_eval']




    if ont_based_prep:  # Standard scaling that differentiate between ontologies
        if use_total_score:
            X_test_benchmark = dataset_data[['Inf_content', 'Total_score', 'Int_confidence', 'GScore', "Cell_comp",
                                             "Mol_funcs", "Bio_process"]]
            X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, ['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, ['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, ['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, ['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, ['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, ['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
        else:
            X_test_benchmark = dataset_data[['Inf_content', 'Int_confidence', 'GScore', "Cell_comp", "Mol_funcs",
                                             "Bio_process"]]
            X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, ['Inf_content', 'Int_confidence', 'GScore']]))
    else:
        if use_total_score:
            X_test_benchmark = dataset_data[['Inf_content', 'Total_score', 'Int_confidence', 'GScore', "Cell_comp",
                                             "Mol_funcs", "Bio_process"]]
            X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))

        else:
            X_test_benchmark = dataset_data[['Inf_content', 'Int_confidence', 'GScore', "Cell_comp", "Mol_funcs",
                                             "Bio_process"]]
            X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']]))

    print(X_test_benchmark.head())

    pred_amount = min(X_test_benchmark.shape[0], n_benchmark)  # Point for predictions
    X_best_pred = X_test_benchmark.head(pred_amount)  # Very limited amount of data, useful to draw the scatterplot
    y_pred = good_autoencoder.predict(X_best_pred, batch_size=batch_size)
    y_pred = np.squeeze(y_pred)
    simgics_eval_test = simgics_evals[:pred_amount].to_numpy()  # Reduced amount to plot against y_pred
    simgics_eval_test = np.squeeze(simgics_eval_test)  # Squeeze array from [n, 1] dimensions to [n, ] for compatibility

    # Plot model predictions against the ground truth
    plt.figure(figsize=(9, 9))
    plt.scatter(simgics_eval_test, y_pred, alpha=0.04)  # Less data is used to not saturate the plot with blue dots
    plt.xlabel("Simgic")
    plt.ylabel("Evaluation")
    plt.title("Eval Plot")
    plt.savefig(top_dir_path + f"/eval_test_graph_{formatted_datetime}{score_tag}.png")
    plt.close()

    print(type(simgics_eval_test))
    print(simgics_eval_test.shape)

    model_preds = good_autoencoder.predict(X_test_benchmark, batch_size=batch_size)  # Comprehensive test of the model
    model_preds = np.squeeze(model_preds)  # Used to draw the 3D histogram plot
    simgics_evals = simgics_evals.to_numpy()
    simgics_evals = np.squeeze(simgics_evals)

    # Histogram version
    n_bins = 20
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_evals, model_preds, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    cmap = colormaps.get_cmap('summer')  # Get desired colormap - you can change this!
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.xlabel("Simgic")
    plt.ylabel("Evaluation")
    plt.title("3D Histogram simgic/loss")
    plt.savefig(top_dir_path + f"/hist_eval_test_graph_{formatted_datetime}{score_tag}.png")
    plt.close()

    # Compute and plot the error achieved by the model
    pred_error = np.absolute(model_preds - simgics_evals)

    num_bins = 20
    fig, ax = plt.subplots(figsize=(9, 9))
    # the histogram of the data
    weights = np.ones_like(pred_error) / (len(pred_error))
    n, bins, patches = ax.hist(pred_error, num_bins, density=False, weights=weights)
    ax.axvline(pred_error.mean(), linestyle='dashed', linewidth=1)
    ax.set_xlabel('Error')
    ax.set_ylabel('Probability mass')  # Probability density normalizes for the widdth of the bin
    ax.set_title(f'Histogram prediction error test distribution probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path + f"/pred_error_graph_{formatted_datetime}{score_tag}.png")
    plt.close()

    # Compute a performance confrontation with the spline, if a path in which to save it is provided
    if confrontation_path_start is not False:
        confrontation_path = os.path.join(confrontation_path_start, f"spline_confrontation")
        os.mkdir(confrontation_path)

        if big_dataset:
            spline_path = 'stats/spline/original_grades_17408776_complete_test_set.csv'
        else:
            spline_path = "stats/spline/original_grades_25_1000000_11.csv"

        spline_data = pd.read_csv(spline_path)
        spline_grades = spline_data['spline_grades']

        # dataset_path = "stats/dataset/simgics_1000000_complete_test.csv"
        # dataset_data = pd.read_csv(dataset_path)
        # simgics_evals = dataset_data['simgics_eval']


        spline_grades = spline_grades.to_numpy()
        spline_grades = np.squeeze(spline_grades)


        # If required the computed variables can be saved
        if save_vars:
            np.savetxt(confrontation_path + f"/spline_grades{score_tag}.txt", spline_grades, delimiter=',', fmt='%s')
            np.savetxt(confrontation_path + f"/simgics_eval_test{score_tag}.txt", simgics_eval_test, delimiter=',', fmt='%s')
            np.savetxt(confrontation_path + f"/simgics_evals{score_tag}.txt", simgics_evals, delimiter=',', fmt='%s')
            np.savetxt(confrontation_path + f"/model_preds{score_tag}.txt", model_preds, delimiter=',', fmt='%s')
            np.savetxt(confrontation_path + f"/pred_error{score_tag}.txt", pred_error, delimiter=',', fmt='%s')
            np.savetxt(confrontation_path + f"/y_pred{score_tag}.txt", y_pred, delimiter=',', fmt='%s')
            np.savetxt(confrontation_path + f"/X_best_pred{score_tag}.txt", X_best_pred, delimiter=',', fmt='%s')


        # Compute the evaluation error for both the spline and model methods
        spline_error = np.absolute(spline_grades - simgics_evals)
        model_error = np.absolute(model_preds - simgics_evals)

        # av_spline_error = np.sum(spline_error)/len(spline_error)
        # av_model_error = np.sum(model_error)/len(model_error)

        av_spline_error = spline_error.mean()
        av_model_error = model_error.mean()

        error_log_name = top_dir_path + f"/error_log.txt"
        error_message = (f"\nSum of all error for each method\n\n"
                         f"Average error for spline: {av_spline_error}\n"
                         f"Average error for model: {av_model_error}\n")

        write_to_file(error_log_name, error_message)


        # Plot the distribution of the errors for both the model and the spline
        num_bins = 20
        fig, ax = plt.subplots(figsize=(9, 9))
        weights_model_error = np.ones_like(model_error) / (len(model_error))
        weight_spline_error = np.ones_like(spline_error) / (len(spline_error))
        plt.hist([model_error, spline_error], num_bins, label=['model_error', 'spline_error'],
                 weights=[weights_model_error, weight_spline_error], color=['tab:blue', 'tab:orange'])
        plt.axvline(model_error.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
        plt.axvline(spline_error.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
        plt.legend(loc='upper right')
        ax.set_xlabel('Error')
        ax.set_ylabel('Probability mass')  # Probability density normalizes for the widdth of the bin
        ax.set_title(f'Model/Spline error confrontation')
        fig.tight_layout()
        plt.savefig(confrontation_path + f"/model_error_mass_{formatted_datetime}{score_tag}.png")
        plt.close()

        # Plot the distribution of the simgics for the model, the spline and the ground-truth
        num_bins = 20
        fig, ax = plt.subplots(figsize=(9, 9))
        weights_simgic = np.ones_like(simgics_evals) / (len(simgics_evals))
        weights_model = np.ones_like(model_preds) / (len(model_preds))
        weight_spline = np.ones_like(spline_grades) / (len(spline_grades))
        plt.hist([simgics_evals, model_preds, spline_grades], num_bins,
                 label=['simgics_evals', 'model_preds', 'spline_grades'],
                 weights=[weights_simgic, weights_model, weight_spline])
        plt.legend(loc='upper right')
        ax.set_xlabel('Grades')
        ax.set_ylabel('Probability mass')  # Probability density normalizes for the widdth of the bin
        ax.set_title(f'Model/Spline/Simgics confrontation')
        fig.tight_layout()
        plt.savefig(confrontation_path + f"/model_spline_simgics_confrontation_mass_{formatted_datetime}{score_tag}.png")
        plt.close()

        # simgic_threshold = 0.75
        simgic_class = (simgics_evals >= simgic_threshold).astype('int')

        spline_threshold_f1 = thresholder_f1(spline_grades[:len(simgic_class)], simgic_class)
        model_threshold_f1 = thresholder_f1(model_preds[:len(simgic_class)], simgic_class)
        spline_threshold_bal_acc, best_spline_bal_acc = thresholder_bal_acc(spline_grades[:len(simgic_class)], simgic_class)
        model_threshold_bal_acc, best_model_bal_acc = thresholder_bal_acc(model_preds[:len(simgic_class)], simgic_class)

        spline_class_f1 = (spline_grades >= spline_threshold_f1).astype('int')
        model_class_f1 = (model_preds >= model_threshold_f1).astype('int')
        spline_class_bal_acc = (spline_grades >= spline_threshold_bal_acc).astype('int')
        model_class_bal_acc = (model_preds >= model_threshold_bal_acc).astype('int')

        class_name = confrontation_path + f"/classification_report_{formatted_datetime}{score_tag}.txt"

        write_to_file(class_name, "Classification of the spline using f1 score: \n")
        write_to_file(class_name, str(classification_report(simgic_class, spline_class_f1)))
        write_to_file(class_name, f"Best threshold is: {spline_threshold_f1}")
        write_to_file(class_name, f"Classification of the model using f1 score: \n")
        write_to_file(class_name, str(classification_report(simgic_class, model_class_f1)))
        write_to_file(class_name, f"Best threshold is: {model_threshold_f1}")
        write_to_file(class_name, f"\n\n")
        write_to_file(class_name, "Classification of the spline using balanced accuracy: \n")
        write_to_file(class_name, str(classification_report(simgic_class, spline_class_bal_acc)))
        write_to_file(class_name, f"Best balanced accuracy is: {best_spline_bal_acc}")
        write_to_file(class_name, f"Best threshold is: {spline_threshold_bal_acc}")
        write_to_file(class_name, f"Classification of the model using balanced accuracy: \n")
        write_to_file(class_name, str(classification_report(simgic_class, model_class_bal_acc)))
        write_to_file(class_name, f"Best balanced accuracy is: {best_model_bal_acc}")
        write_to_file(class_name, f"Best threshold is: {model_threshold_bal_acc}")

    if ontological_analysis:
        confrontation_benchmark_ontology(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                         error_log_name, batch_size, big_dataset, use_total_score=use_total_score,
                                         n_benchmark=10000, save_vars=False, ont_based_prep=ont_based_prep,
                                         source_analysis = True)


#Confrontation made on data separated by ontology
def confrontation_benchmark_ontology(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                     error_log_name, batch_size, big_dataset, use_total_score=True, n_benchmark=10000,
                                     save_vars=True, ont_based_prep=False, source_analysis=True):
    """good_autoencoder: the model that is being evaluated
    top_dir_path: directory in which graphs are saved
    formatted_datetime: datetime to differentiate the data
    use_total_score: does the model uses total scores
    n_benchmark: amount of point plotted in the prediction graphs
    save_vars: Save the computed variables, for debugging purposes"""

    if use_total_score:  # Tags name files if the total score is used (tsu: Total Score Used)
        score_tag = '_tsu'
    else:
        score_tag = ''

    top_dir_path = os.path.join(top_dir_path, f"ontology_confrontation")
    os.mkdir(top_dir_path)

    if big_dataset:
        dataset_path = 'stats/dataset/simgics_17408776_complete_test_set.csv'
    else:
        dataset_path = 'stats/dataset/simgics_1000000_complete_test_11.csv'
    dataset_data = pd.read_csv(dataset_path)

    # Compute total amount of each ontology
    mol_func = (dataset_data["Mol_funcs"] == 1.0).sum()
    bio_process = (dataset_data["Bio_process"] == 1.0).sum()
    cell_comp = (dataset_data["Cell_comp"] == 1.0).sum()
    # print(f'mol_func: {mol_func}')
    # print(f'bio_process: {bio_process}')
    # print(f'cell_comp: {cell_comp}')

    ontologies = {'cell_comp': cell_comp, 'mol_func': mol_func, 'bio_process': bio_process}
    ont = list(ontologies.keys())
    n_ont = list(ontologies.values())

    # Plot ontolology population with count and probability mass
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')
    ax.set_title(f'Ontology population')
    fig.tight_layout()
    plt.savefig(top_dir_path + f"/ontology_population.png")
    plt.close()

    tot_ont = mol_func + bio_process + cell_comp
    # num_bins = 20
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont / tot_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')  # Probability density normalizes for the width of the bin
    ax.set_title(f'Ontology population mass')
    fig.tight_layout()
    plt.savefig(top_dir_path + f"/ontology_population_mass.png")
    plt.close()


    if ont_based_prep:  # Standard scaling that differentiate between ontologies
        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = dataset_data.loc[dataset_data["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_mol_func = dataset_data.loc[dataset_data["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_bio_process = dataset_data.loc[dataset_data["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        if use_total_score:
            X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
        else:
            X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
    else:
        if use_total_score:   # Standard scaling ontologically agnostic
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))

        else:
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']]))

        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_cell_comp)
        # print(type(X_test_benchmark_cell_comp))
        # print(X_test_benchmark_cell_comp.shape)

        X_test_benchmark_mol_func = X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_mol_func)
        # print(type(X_test_benchmark_mol_func))
        # print(X_test_benchmark_mol_func.shape)

        X_test_benchmark_bio_process = X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_bio_process)
        # print(type(X_test_benchmark_bio_process))
        # print(X_test_benchmark_bio_process.shape)

        if use_total_score:   # Standard scaling ontologically agnostic
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

        else:
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

    print(X_test_benchmark_cell_comp.head())
    print(X_test_benchmark_mol_func.head())
    print(X_test_benchmark_bio_process.head())

    # Retrieve the spline grades from file (already computed)
    if big_dataset:
        spline_path = 'stats/spline/original_grades_17408776_complete_test_set.csv'
    else:
        spline_path = "stats/spline/original_grades_25_1000000_11.csv"
    spline_data = pd.read_csv(spline_path)
    spline_grades = spline_data['spline_grades']

    # Add to the original datasets the respective spline grades
    X_test_benchmark_cell_comp['spline_grades_cc'] = spline_grades.loc[dataset_data["Cell_comp"] == 1.0]
    X_test_benchmark_mol_func['spline_grades_mf'] = spline_grades.loc[dataset_data["Mol_funcs"] == 1.0]
    X_test_benchmark_bio_process['spline_grades_bp'] = spline_grades.loc[dataset_data["Bio_process"] == 1.0]

    # y_pred is a prediction for limited data to draw the scatter plot,
    # model_pred is more comprehensive for the hist plots
    pred_amount_cc = min(X_test_cc.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_cc = X_test_cc.head(pred_amount_cc)
    y_pred_cc = good_autoencoder.predict(X_best_pred_cc, batch_size=batch_size)
    y_pred_cc = np.squeeze(y_pred_cc)
    simgics_eval_cc = X_test_benchmark_cell_comp['simgics_eval'].to_numpy()
    simgics_eval_cc = np.squeeze(simgics_eval_cc)

    pred_amount_mf = min(X_test_mf.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_mf = X_test_mf.head(pred_amount_mf)
    y_pred_mf = good_autoencoder.predict(X_best_pred_mf, batch_size=batch_size)
    y_pred_mf = np.squeeze(y_pred_mf)
    simgics_eval_mf = X_test_benchmark_mol_func['simgics_eval'].to_numpy()
    simgics_eval_mf = np.squeeze(simgics_eval_mf)

    pred_amount_bp = min(X_test_bp.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_bp = X_test_bp.head(pred_amount_bp)
    y_pred_bp = good_autoencoder.predict(X_best_pred_bp, batch_size=batch_size)
    y_pred_bp = np.squeeze(y_pred_bp)
    simgics_eval_bp = X_test_benchmark_bio_process['simgics_eval'].to_numpy()
    simgics_eval_bp = np.squeeze(simgics_eval_bp)

    print(len(simgics_eval_cc))
    print(len(simgics_eval_mf))
    print(len(simgics_eval_bp))
    print(len(y_pred_cc))
    print(len(y_pred_mf))
    print(len(y_pred_bp))

    # Scatter plot prediction/true simgic
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Evaluation plots for each ontology')
    ax1.scatter(simgics_eval_cc[:pred_amount_cc], y_pred_cc, alpha=0.04)
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')
    ax2.scatter(simgics_eval_mf[:pred_amount_mf], y_pred_mf, alpha=0.04)
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')
    ax3.scatter(simgics_eval_bp[:pred_amount_bp], y_pred_bp, alpha=0.04)
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    plt.savefig(top_dir_path + f"/eval_test_graph_{formatted_datetime}{score_tag}_ont.png")
    plt.close()

    # Create the prediction for the full subset of ontological data
    model_preds_cc = good_autoencoder.predict(X_test_cc, batch_size=batch_size)
    model_preds_cc = np.squeeze(model_preds_cc)
    model_preds_mf = good_autoencoder.predict(X_test_mf, batch_size=batch_size)
    model_preds_mf = np.squeeze(model_preds_mf)
    model_preds_bp = good_autoencoder.predict(X_test_bp, batch_size=batch_size)
    model_preds_bp = np.squeeze(model_preds_bp)

    # Histogram version, with more data
    n_bins = 20
    # Set colorbar
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # colorbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Set figure
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle('Evaluation histogram plots for each ontology')
    cmap = colormaps.get_cmap('summer')  # Get desired colormap - you can change this!
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_cc, model_preds_cc, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values

    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_mf, model_preds_mf, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_bp, model_preds_bp, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    # # Apply colorbar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.7])
    # fig.colorbar(colorbar, cax=cbar_ax)
    plt.savefig(top_dir_path + f"/hist_eval_test_graph_{formatted_datetime}{score_tag}_ont.png")
    plt.close()


    # Find the prediction error
    pred_error_cc = np.absolute(simgics_eval_cc - model_preds_cc)
    pred_error_mf = np.absolute(simgics_eval_mf - model_preds_mf)
    pred_error_bp = np.absolute(simgics_eval_bp - model_preds_bp)



    # Histogram plot error distribution for each ontology
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Histogram prediction error distribution per ontology')
    weights_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    n, bins, patches = ax1.hist(pred_error_cc, num_bins, density=False, weights=weights_cc)
    ax1.axvline(pred_error_cc.mean(), linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')
    weights_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    n, bins, patches = ax2.hist(pred_error_mf, num_bins, density=False, weights=weights_mf)
    ax2.axvline(pred_error_mf.mean(), linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')
    weights_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    n, bins, patches = ax3.hist(pred_error_bp, num_bins, density=False, weights=weights_bp)
    ax3.axvline(pred_error_bp.mean(), linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path + f"/pred_error_graph_{formatted_datetime}{score_tag}_ont.png")
    plt.close()

    # Error in prediction from spline
    spline_grades_cc = X_test_benchmark_cell_comp['spline_grades_cc'].to_numpy()
    spline_grades_mf = X_test_benchmark_mol_func['spline_grades_mf'].to_numpy()
    spline_grades_bp = X_test_benchmark_bio_process['spline_grades_bp'].to_numpy()
    spline_grades_cc = np.squeeze(spline_grades_cc)
    spline_grades_mf = np.squeeze(spline_grades_mf)
    spline_grades_bp = np.squeeze(spline_grades_bp)

    spline_error_cc = np.absolute(simgics_eval_cc - spline_grades_cc)
    spline_error_mf = np.absolute(simgics_eval_mf - spline_grades_mf)
    spline_error_bp = np.absolute(simgics_eval_bp - spline_grades_bp)

    # av_spline_error_cc = np.sum(spline_error_cc)/len(spline_error_cc)
    # av_spline_error_mf = np.sum(spline_error_mf)/len(spline_error_mf)
    # av_spline_error_bp = np.sum(spline_error_bp)/len(spline_error_bp)
    # av_pred_error_cc = np.sum(pred_error_cc)/len(pred_error_cc)
    # av_pred_error_mf = np.sum(pred_error_mf)/len(pred_error_mf)
    # av_pred_error_bp = np.sum(pred_error_bp)/len(pred_error_bp)

    av_spline_error_cc = spline_error_cc.mean()
    av_spline_error_mf = spline_error_mf.mean()
    av_spline_error_bp = spline_error_bp.mean()
    av_pred_error_cc = pred_error_cc.mean()
    av_pred_error_mf = pred_error_mf.mean()
    av_pred_error_bp = pred_error_bp.mean()

    # error_log_name = top_dir_path + f"/error_log.txt"
    error_message_ont = (f"\n\nSum of all error for each method for each ontology\n\n"
                         f"Average error for spline in cellular components: {av_spline_error_cc}\n"
                         f"Average error for model in cellular components: {av_pred_error_cc}\n"
                         f"Average error for spline in molecular functions: {av_spline_error_mf}\n"
                         f"Average error for model in molecular functions: {av_pred_error_mf}\n"
                         f"Average error for spline in biological processes: {av_spline_error_bp}\n"
                         f"Average error for model in biological processes: {av_pred_error_bp}\n"
                         )

    write_to_file(error_log_name, error_message_ont)


    # Histogram plot error distribution for each ontology, confrontation between model and spline
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Histogram prediction error distribution model vs spline')

    weights_model_error_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    weight_spline_error_cc = np.ones_like(spline_error_cc) / (len(spline_error_cc))
    ax1.hist([pred_error_cc, spline_error_cc], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_cc, weight_spline_error_cc], color=['tab:blue', 'tab:orange'])
    ax1.axvline(pred_error_cc.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax1.axvline(spline_error_cc.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    weight_spline_error_mf = np.ones_like(spline_error_mf) / (len(spline_error_mf))
    ax2.hist([pred_error_mf, spline_error_mf], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_mf, weight_spline_error_mf], color=['tab:blue', 'tab:orange'])
    ax2.axvline(pred_error_mf.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax2.axvline(spline_error_mf.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    weight_spline_error_bp = np.ones_like(spline_error_bp) / (len(spline_error_bp))
    ax3.hist([pred_error_bp, spline_error_bp], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_bp, weight_spline_error_bp], color=['tab:blue', 'tab:orange'])
    ax3.axvline(pred_error_bp.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax3.axvline(spline_error_bp.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path + f"/pred_error_graph_{formatted_datetime}{score_tag}_spline_model_ont.png")
    plt.close()

    # Histogram plot eval distribution for each ontology, confrontation between model, spline and ground-truth
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Simgic distribution model/spline/dataset')
    weights_simgic_cc = np.ones_like(simgics_eval_cc) / (len(simgics_eval_cc))
    weights_model_cc = np.ones_like(model_preds_cc) / (len(model_preds_cc))
    weight_spline_cc = np.ones_like(spline_grades_cc) / (len(spline_grades_cc))
    ax1.hist([model_preds_cc, spline_grades_cc, simgics_eval_cc], num_bins,
             label=['model_preds', 'spline_grades', 'simgics_eval'],
             weights=[weights_model_cc, weight_spline_cc, weights_simgic_cc],
             color=['tab:blue', 'tab:orange', 'tab:green'])
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_mf = np.ones_like(simgics_eval_mf) / (len(simgics_eval_mf))
    weights_model_mf = np.ones_like(model_preds_mf) / (len(model_preds_mf))
    weight_spline_mf = np.ones_like(spline_grades_mf) / (len(spline_grades_mf))
    ax2.hist([model_preds_mf, spline_grades_mf, simgics_eval_mf], num_bins,
             label=['model_preds', 'spline_grades', 'simgics_eval'],
             weights=[weights_model_mf, weight_spline_mf, weights_simgic_mf],
             color=['tab:blue', 'tab:orange', 'tab:green'])
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_bp = np.ones_like(simgics_eval_bp) / (len(simgics_eval_bp))
    weights_model_bp = np.ones_like(model_preds_bp) / (len(model_preds_bp))
    weight_spline_bp = np.ones_like(spline_grades_bp) / (len(spline_grades_bp))
    ax3.hist([model_preds_bp, spline_grades_bp, simgics_eval_bp], num_bins,
             label=['model_preds', 'spline_grades', 'simgics_eval'],
             weights=[weights_model_bp, weight_spline_bp, weights_simgic_bp],
             color=['tab:blue', 'tab:orange', 'tab:green'])
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path + f"/eval_graph_{formatted_datetime}{score_tag}_spline_model_gt_ont.png")
    plt.close()


    # It's possible to set different thresholds for different ontologies
    # Separation of good and bad simgics through thresholding
    simgic_class_cc = (simgics_eval_cc >= simgic_threshold).astype('int')
    simgic_class_mf = (simgics_eval_mf >= simgic_threshold).astype('int')
    simgic_class_bp = (simgics_eval_bp >= simgic_threshold).astype('int')

    # Find best threshold for each ontology (prioritize f1 score) for spline and model
    spline_threshold_f1_cc = thresholder_f1(spline_grades_cc, simgic_class_cc)
    model_threshold_f1_cc = thresholder_f1(model_preds_cc, simgic_class_cc)
    spline_threshold_f1_mf = thresholder_f1(spline_grades_mf, simgic_class_mf)
    model_threshold_f1_mf = thresholder_f1(model_preds_mf, simgic_class_mf)
    spline_threshold_f1_bp = thresholder_f1(spline_grades_bp, simgic_class_bp)
    model_threshold_f1_bp = thresholder_f1(model_preds_bp, simgic_class_bp)

    # Find best threshold for each ontology (prioritize balanced accuracy) for spline and model
    spline_threshold_bal_acc_cc, best_spline_bal_acc_cc = thresholder_bal_acc(spline_grades_cc, simgic_class_cc)
    model_threshold_bal_acc_cc, best_model_bal_acc_cc = thresholder_bal_acc(model_preds_cc, simgic_class_cc)
    spline_threshold_bal_acc_mf, best_spline_bal_acc_mf = thresholder_bal_acc(spline_grades_mf, simgic_class_mf)
    model_threshold_bal_acc_mf, best_model_bal_acc_mf = thresholder_bal_acc(model_preds_mf, simgic_class_mf)
    spline_threshold_bal_acc_bp, best_spline_bal_acc_bp = thresholder_bal_acc(spline_grades_bp, simgic_class_bp)
    model_threshold_bal_acc_bp, best_model_bal_acc_bp = thresholder_bal_acc(model_preds_bp, simgic_class_bp)

    # Apply the thresholding to discriminate the two categories
    spline_class_f1_cc = (spline_grades_cc >= spline_threshold_f1_cc).astype('int')
    model_class_f1_cc = (model_preds_cc >= model_threshold_f1_cc).astype('int')
    spline_class_f1_mf = (spline_grades_mf >= spline_threshold_f1_mf).astype('int')
    model_class_f1_mf = (model_preds_mf >= model_threshold_f1_mf).astype('int')
    spline_class_f1_bp = (spline_grades_bp >= spline_threshold_f1_bp).astype('int')
    model_class_f1_bp = (model_preds_bp >= model_threshold_f1_bp).astype('int')

    spline_class_bal_acc_cc = (spline_grades_cc >= spline_threshold_bal_acc_cc).astype('int')
    model_class_bal_acc_cc = (model_preds_cc >= model_threshold_bal_acc_cc).astype('int')
    spline_class_bal_acc_mf = (spline_grades_mf >= spline_threshold_bal_acc_mf).astype('int')
    model_class_bal_acc_mf = (model_preds_mf >= model_threshold_bal_acc_mf).astype('int')
    spline_class_bal_acc_bp = (spline_grades_bp >= spline_threshold_bal_acc_bp).astype('int')
    model_class_bal_acc_bp = (model_preds_bp >= model_threshold_bal_acc_bp).astype('int')

    # Write on a file the report
    class_name_ont = top_dir_path + f"/classification_report_{formatted_datetime}{score_tag}.txt"
    # Write the report on the classification, subdivided by ontology
    write_to_file(class_name_ont, "Classification of the spline using f1 score cellular components: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_cc, spline_class_f1_cc)))
    write_to_file(class_name_ont, f"Best threshold is: {spline_threshold_f1_cc}\n")
    write_to_file(class_name_ont, "Classification of the spline using f1 score molecular functions: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_mf, spline_class_f1_mf)))
    write_to_file(class_name_ont, f"Best threshold is: {spline_threshold_f1_mf}\n")
    write_to_file(class_name_ont, "Classification of the spline using f1 score biological processes: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_bp, spline_class_f1_bp)))
    write_to_file(class_name_ont, f"Best threshold is: {spline_threshold_f1_bp}\n")

    write_to_file(class_name_ont, f"\nClassification of the model using f1 score cellular components: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_cc, model_class_f1_cc)))
    write_to_file(class_name_ont, f"Best threshold is: {model_threshold_f1_cc}")
    write_to_file(class_name_ont, f"Classification of the model using f1 score molecular functions: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_mf, model_class_f1_mf)))
    write_to_file(class_name_ont, f"Best threshold is: {model_threshold_f1_mf}")
    write_to_file(class_name_ont, f"Classification of the model using f1 score biological processes: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_bp, model_class_f1_bp)))
    write_to_file(class_name_ont, f"Best threshold is: {model_threshold_f1_bp}")
    write_to_file(class_name_ont, f"\n\n")

    write_to_file(class_name_ont, "Classification of the spline using balanced accuracy cellular components: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_cc, spline_class_bal_acc_cc)))
    write_to_file(class_name_ont, f"Best balanced accuracy is: {best_spline_bal_acc_cc}")
    write_to_file(class_name_ont, f"Best threshold is: {spline_threshold_bal_acc_cc}")
    write_to_file(class_name_ont, "Classification of the spline using balanced accuracy molecular functions: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_mf, spline_class_bal_acc_mf)))
    write_to_file(class_name_ont, f"Best balanced accuracy is: {best_spline_bal_acc_mf}")
    write_to_file(class_name_ont, f"Best threshold is: {spline_threshold_bal_acc_mf}")
    write_to_file(class_name_ont, "Classification of the spline using balanced accuracy biological processes: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_bp, spline_class_bal_acc_bp)))
    write_to_file(class_name_ont, f"Best balanced accuracy is: {best_spline_bal_acc_bp}")
    write_to_file(class_name_ont, f"Best threshold is: {spline_threshold_bal_acc_bp}")

    write_to_file(class_name_ont, f"\nClassification of the model using balanced accuracy cellular components: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_cc, model_class_bal_acc_cc)))
    write_to_file(class_name_ont, f"Best balanced accuracy is: {best_model_bal_acc_cc}")
    write_to_file(class_name_ont, f"Best threshold is: {model_threshold_bal_acc_cc}")
    write_to_file(class_name_ont, f"Classification of the model using balanced accuracy molecular functions: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_mf, model_class_bal_acc_mf)))
    write_to_file(class_name_ont, f"Best balanced accuracy is: {best_model_bal_acc_mf}")
    write_to_file(class_name_ont, f"Best threshold is: {model_threshold_bal_acc_mf}")
    write_to_file(class_name_ont, f"Classification of the model using balanced accuracy biological processes: \n")
    write_to_file(class_name_ont, str(classification_report(simgic_class_bp, model_class_bal_acc_bp)))
    write_to_file(class_name_ont, f"Best balanced accuracy is: {best_model_bal_acc_bp}")
    write_to_file(class_name_ont, f"Best threshold is: {model_threshold_bal_acc_bp}")

    if save_vars:
        # If required the variable used can be saved
        confrontation_path_ont = os.path.join(top_dir_path, f"saved_data_ont")
        os.mkdir(confrontation_path_ont)

        np.savetxt(confrontation_path_ont + f"/spline_grades{score_tag}.txt", spline_grades_cc, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/spline_grades{score_tag}.txt", spline_grades_mf, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/spline_grades{score_tag}.txt", spline_grades_bp, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/simgics_eval_cc{score_tag}.txt", simgics_eval_cc, delimiter=',',fmt='%s')
        np.savetxt(confrontation_path_ont + f"/simgics_eval_mf{score_tag}.txt", simgics_eval_mf, delimiter=',',fmt='%s')
        np.savetxt(confrontation_path_ont + f"/simgics_eval_bp{score_tag}.txt", simgics_eval_bp, delimiter=',',fmt='%s')
        np.savetxt(confrontation_path_ont + f"/model_preds_cc{score_tag}.txt", model_preds_cc, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/model_preds_mf{score_tag}.txt", model_preds_mf, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/model_preds_bp{score_tag}.txt", model_preds_bp, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/pred_error_cc{score_tag}.txt", pred_error_cc, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/pred_error_mf{score_tag}.txt", pred_error_mf, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/pred_error_bp{score_tag}.txt", pred_error_bp, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/X_best_pred_cc{score_tag}.txt", X_best_pred_cc, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/X_best_pred_mf{score_tag}.txt", X_best_pred_mf, delimiter=',', fmt='%s')
        np.savetxt(confrontation_path_ont + f"/X_best_pred_bp{score_tag}.txt", X_best_pred_bp, delimiter=',', fmt='%s')

    if source_analysis:
        confrontation_benchmark_source_only_2(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                             error_log_name, batch_size, use_total_score=True, n_benchmark=10000,
                                             ont_based_prep=False)

#Confrontation benchmark for data from category 2: experimental data
def confrontation_benchmark_source_only_2(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                     error_log_name, batch_size, use_total_score=True, n_benchmark=10000,
                                     ont_based_prep=False):

    """good_autoencoder: the model that is being evaluated
    top_dir_path: directory in which graphs are saved
    formatted_datetime: datetime to differentiate the data
    use_total_score: does the model uses total scores
    n_benchmark: amount of point plotted in the prediction graphs
    save_vars: Save the computed variables, for debugging purposes"""

    if use_total_score:  # Tags name files if the total score is used (tsu: Total Score Used)
        score_tag = '_tsu'
    else:
        score_tag = ''

    top_dir_path_only_2 = os.path.join(top_dir_path, f"source_confrontation_only_2")
    os.mkdir(top_dir_path_only_2)

    dataset_path = 'stats/dataset/simgics_annotated_merged_clean_only_2.csv' #Mettere il dataset quando sarà pronto
    dataset_data = pd.read_csv(dataset_path)

    # Compute total amount of each ontology
    mol_func = (dataset_data["Mol_funcs"] == 1.0).sum()
    bio_process = (dataset_data["Bio_process"] == 1.0).sum()
    cell_comp = (dataset_data["Cell_comp"] == 1.0).sum()
    # print(f'mol_func: {mol_func}')
    # print(f'bio_process: {bio_process}')
    # print(f'cell_comp: {cell_comp}')

    ontologies = {'cell_comp': cell_comp, 'mol_func': mol_func, 'bio_process': bio_process}
    ont = list(ontologies.keys())
    n_ont = list(ontologies.values())

    # Plot ontolology population with count and probability mass
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')
    ax.set_title(f'Ontology population (controlled source)')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_2 + f"/source_population_only_2.png")
    plt.close()

    tot_ont = mol_func + bio_process + cell_comp
    # num_bins = 20
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont / tot_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')  # Probability density normalizes for the width of the bin
    ax.set_title(f'Ontology population mass (controlled source)')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_2 + f"/source_population_mass_only_2.png")
    plt.close()


    if ont_based_prep:  # Standard scaling that differentiate between ontologies
        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = dataset_data.loc[dataset_data["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_mol_func = dataset_data.loc[dataset_data["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_bio_process = dataset_data.loc[dataset_data["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        if use_total_score:
            X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
        else:
            X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
    else:
        if use_total_score:   # Standard scaling ontologically agnostic
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))

        else:
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']]))

        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_cell_comp)
        # print(type(X_test_benchmark_cell_comp))
        # print(X_test_benchmark_cell_comp.shape)

        X_test_benchmark_mol_func = X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_mol_func)
        # print(type(X_test_benchmark_mol_func))
        # print(X_test_benchmark_mol_func.shape)

        X_test_benchmark_bio_process = X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_bio_process)
        # print(type(X_test_benchmark_bio_process))
        # print(X_test_benchmark_bio_process.shape)

        if use_total_score:   # Standard scaling ontologically agnostic
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

        else:
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

    print(X_test_benchmark_cell_comp.head())
    print(X_test_benchmark_mol_func.head())
    print(X_test_benchmark_bio_process.head())

    # Retrieve the spline grades from file (already computed)
    spline_path = "stats/spline/original_grades_only_2.csv"
    spline_data = pd.read_csv(spline_path)
    spline_grades = spline_data['spline_grades']

    # Add to the original datasets the respective spline grades
    X_test_benchmark_cell_comp['spline_grades_cc'] = spline_grades.loc[dataset_data["Cell_comp"] == 1.0]
    X_test_benchmark_mol_func['spline_grades_mf'] = spline_grades.loc[dataset_data["Mol_funcs"] == 1.0]
    X_test_benchmark_bio_process['spline_grades_bp'] = spline_grades.loc[dataset_data["Bio_process"] == 1.0]

    # y_pred is a prediction for limited data to draw the scatter plot,
    # model_pred is more comprehensive for the hist plots
    pred_amount_cc = min(X_test_cc.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_cc = X_test_cc.head(pred_amount_cc)
    y_pred_cc = good_autoencoder.predict(X_best_pred_cc, batch_size=batch_size)
    y_pred_cc = np.squeeze(y_pred_cc)
    simgics_eval_cc = X_test_benchmark_cell_comp['simgics_eval'].to_numpy()
    simgics_eval_cc = np.squeeze(simgics_eval_cc)

    pred_amount_mf = min(X_test_mf.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_mf = X_test_mf.head(pred_amount_mf)
    y_pred_mf = good_autoencoder.predict(X_best_pred_mf, batch_size=batch_size)
    y_pred_mf = np.squeeze(y_pred_mf)
    simgics_eval_mf = X_test_benchmark_mol_func['simgics_eval'].to_numpy()
    simgics_eval_mf = np.squeeze(simgics_eval_mf)

    pred_amount_bp = min(X_test_bp.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_bp = X_test_bp.head(pred_amount_bp)
    y_pred_bp = good_autoencoder.predict(X_best_pred_bp, batch_size=batch_size)
    y_pred_bp = np.squeeze(y_pred_bp)
    simgics_eval_bp = X_test_benchmark_bio_process['simgics_eval'].to_numpy()
    simgics_eval_bp = np.squeeze(simgics_eval_bp)

    print(len(simgics_eval_cc))
    print(len(simgics_eval_mf))
    print(len(simgics_eval_bp))
    print(len(y_pred_cc))
    print(len(y_pred_mf))
    print(len(y_pred_bp))

    # Scatter plot prediction/true simgic
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Evaluation plots for each ontology')
    ax1.scatter(simgics_eval_cc[:pred_amount_cc], y_pred_cc, alpha=0.04)
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')
    ax2.scatter(simgics_eval_mf[:pred_amount_mf], y_pred_mf, alpha=0.04)
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')
    ax3.scatter(simgics_eval_bp[:pred_amount_bp], y_pred_bp, alpha=0.04)
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    plt.savefig(top_dir_path_only_2 + f"/eval_test_graph_{formatted_datetime}{score_tag}_ont_source_only_2.png")
    plt.close()

    # Create the prediction for the full subset of ontological data
    model_preds_cc = good_autoencoder.predict(X_test_cc, batch_size=batch_size)
    model_preds_cc = np.squeeze(model_preds_cc)
    model_preds_mf = good_autoencoder.predict(X_test_mf, batch_size=batch_size)
    model_preds_mf = np.squeeze(model_preds_mf)
    model_preds_bp = good_autoencoder.predict(X_test_bp, batch_size=batch_size)
    model_preds_bp = np.squeeze(model_preds_bp)

    # Histogram version, with more data
    n_bins = 20
    # Set colorbar
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # colorbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Set figure
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle('Evaluation histogram plots for each ontology')
    cmap = colormaps.get_cmap('summer')  # Get desired colormap - you can change this!
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_cc, model_preds_cc, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values

    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_mf, model_preds_mf, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_bp, model_preds_bp, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz)/2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    # # Apply colorbar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.7])
    # fig.colorbar(colorbar, cax=cbar_ax)
    plt.savefig(top_dir_path_only_2 + f"/hist_eval_test_graph_{formatted_datetime}{score_tag}_ont_source_only_2.png")
    plt.close()


    # Find the prediction error
    pred_error_cc = np.absolute(simgics_eval_cc - model_preds_cc)
    pred_error_mf = np.absolute(simgics_eval_mf - model_preds_mf)
    pred_error_bp = np.absolute(simgics_eval_bp - model_preds_bp)



    # Histogram plot error distribution for each ontology
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Histogram prediction error distribution per ontology')
    weights_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    n, bins, patches = ax1.hist(pred_error_cc, num_bins, density=False, weights=weights_cc)
    ax1.axvline(pred_error_cc.mean(), linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')
    weights_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    n, bins, patches = ax2.hist(pred_error_mf, num_bins, density=False, weights=weights_mf)
    ax2.axvline(pred_error_mf.mean(), linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')
    weights_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    n, bins, patches = ax3.hist(pred_error_bp, num_bins, density=False, weights=weights_bp)
    ax3.axvline(pred_error_bp.mean(), linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_2 + f"/pred_error_graph_{formatted_datetime}{score_tag}_ont_source_only_2.png")
    plt.close()

    # Error in prediction from spline
    spline_grades_cc = X_test_benchmark_cell_comp['spline_grades_cc'].to_numpy()
    spline_grades_mf = X_test_benchmark_mol_func['spline_grades_mf'].to_numpy()
    spline_grades_bp = X_test_benchmark_bio_process['spline_grades_bp'].to_numpy()
    spline_grades_cc = np.squeeze(spline_grades_cc)
    spline_grades_mf = np.squeeze(spline_grades_mf)
    spline_grades_bp = np.squeeze(spline_grades_bp)

    spline_error_cc = np.absolute(simgics_eval_cc - spline_grades_cc)
    spline_error_mf = np.absolute(simgics_eval_mf - spline_grades_mf)
    spline_error_bp = np.absolute(simgics_eval_bp - spline_grades_bp)

    # av_spline_error_cc = np.sum(spline_error_cc)/len(spline_error_cc)
    # av_spline_error_mf = np.sum(spline_error_mf)/len(spline_error_mf)
    # av_spline_error_bp = np.sum(spline_error_bp)/len(spline_error_bp)
    # av_pred_error_cc = np.sum(pred_error_cc)/len(pred_error_cc)
    # av_pred_error_mf = np.sum(pred_error_mf)/len(pred_error_mf)
    # av_pred_error_bp = np.sum(pred_error_bp)/len(pred_error_bp)

    av_spline_error_cc = spline_error_cc.mean()
    av_spline_error_mf = spline_error_mf.mean()
    av_spline_error_bp = spline_error_bp.mean()
    av_pred_error_cc = pred_error_cc.mean()
    av_pred_error_mf = pred_error_mf.mean()
    av_pred_error_bp = pred_error_bp.mean()

    # error_log_name = top_dir_path_only_2 + f"/error_log.txt"
    error_message_ont = (f"\n\nSum of all error for each method for each ontology\n\n"
                         f"Average error for spline in cellular components: {av_spline_error_cc}\n"
                         f"Average error for model in cellular components: {av_pred_error_cc}\n"
                         f"Average error for spline in molecular functions: {av_spline_error_mf}\n"
                         f"Average error for model in molecular functions: {av_pred_error_mf}\n"
                         f"Average error for spline in biological processes: {av_spline_error_bp}\n"
                         f"Average error for model in biological processes: {av_pred_error_bp}\n"
                         )

    write_to_file(error_log_name, error_message_ont)


    # Histogram plot error distribution for each ontology, confrontation between model and spline
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Histogram prediction error distribution model vs spline')

    weights_model_error_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    weight_spline_error_cc = np.ones_like(spline_error_cc) / (len(spline_error_cc))
    ax1.hist([pred_error_cc, spline_error_cc], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_cc, weight_spline_error_cc], color=['tab:blue', 'tab:orange'])
    ax1.axvline(pred_error_cc.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax1.axvline(spline_error_cc.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    weight_spline_error_mf = np.ones_like(spline_error_mf) / (len(spline_error_mf))
    ax2.hist([pred_error_mf, spline_error_mf], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_mf, weight_spline_error_mf], color=['tab:blue', 'tab:orange'])
    ax2.axvline(pred_error_mf.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax2.axvline(spline_error_mf.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    weight_spline_error_bp = np.ones_like(spline_error_bp) / (len(spline_error_bp))
    ax3.hist([pred_error_bp, spline_error_bp], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_bp, weight_spline_error_bp], color=['tab:blue', 'tab:orange'])
    ax3.axvline(pred_error_bp.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax3.axvline(spline_error_bp.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_2 + f"/pred_error_graph_{formatted_datetime}{score_tag}_spline_model_ont_source_only_2.png")
    plt.close()

    # Histogram plot eval distribution for each ontology, confrontation between model, spline and ground-truth
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Simgic distribution model/spline/dataset')
    weights_simgic_cc = np.ones_like(simgics_eval_cc) / (len(simgics_eval_cc))
    weights_model_cc = np.ones_like(model_preds_cc) / (len(model_preds_cc))
    weight_spline_cc = np.ones_like(spline_grades_cc) / (len(spline_grades_cc))
    ax1.hist([simgics_eval_cc, model_preds_cc, spline_grades_cc], num_bins,
             label=['simgics_eval_cc', 'model_preds_cc', 'spline_grades_cc'],
             weights=[weights_simgic_cc, weights_model_cc, weight_spline_cc])
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_mf = np.ones_like(simgics_eval_mf) / (len(simgics_eval_mf))
    weights_model_mf = np.ones_like(model_preds_mf) / (len(model_preds_mf))
    weight_spline_mf = np.ones_like(spline_grades_mf) / (len(spline_grades_mf))
    ax2.hist([simgics_eval_mf, model_preds_mf, spline_grades_mf], num_bins,
             label=['simgics_eval_mf', 'model_preds_mf', 'spline_grades_mf'],
             weights=[weights_simgic_mf, weights_model_mf, weight_spline_mf])
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_bp = np.ones_like(simgics_eval_bp) / (len(simgics_eval_bp))
    weights_model_bp = np.ones_like(model_preds_bp) / (len(model_preds_bp))
    weight_spline_bp = np.ones_like(spline_grades_bp) / (len(spline_grades_bp))
    ax3.hist([simgics_eval_bp, model_preds_bp, spline_grades_bp], num_bins,
             label=['simgics_eval_bp', 'model_preds_bp', 'spline_grades_bp'],
             weights=[weights_simgic_bp, weights_model_bp, weight_spline_bp])
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_2 + f"/eval_graph_{formatted_datetime}{score_tag}_spline_model_gt_ont_source_only_2.png")
    plt.close()

    # Since the

    confrontation_benchmark_source_only_1_and_2(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                             error_log_name, batch_size, use_total_score=True, n_benchmark=10000,
                                             ont_based_prep=False)

#Confrontation benchmark for data from category 1 and 2: IEA data with UniprotKB implementation and experimental data
def confrontation_benchmark_source_only_1_and_2(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                     error_log_name, batch_size, use_total_score=True, n_benchmark=10000,
                                     ont_based_prep=False):
    """good_autoencoder: the model that is being evaluated
        top_dir_path: directory in which graphs are saved
        formatted_datetime: datetime to differentiate the data
        use_total_score: does the model uses total scores
        n_benchmark: amount of point plotted in the prediction graphs
        save_vars: Save the computed variables, for debugging purposes"""

    if use_total_score:  # Tags name files if the total score is used (tsu: Total Score Used)
        score_tag = '_tsu'
    else:
        score_tag = ''

    top_dir_path_only_1_and_2 = os.path.join(top_dir_path, f"source_confrontation_only_1_and_2")
    os.mkdir(top_dir_path_only_1_and_2)

    dataset_path = 'stats/dataset/simgics_annotated_merged_clean_only_1_and_2.csv'  # Mettere il dataset quando sarà pronto
    dataset_data = pd.read_csv(dataset_path)

    # Compute total amount of each ontology
    mol_func = (dataset_data["Mol_funcs"] == 1.0).sum()
    bio_process = (dataset_data["Bio_process"] == 1.0).sum()
    cell_comp = (dataset_data["Cell_comp"] == 1.0).sum()
    # print(f'mol_func: {mol_func}')
    # print(f'bio_process: {bio_process}')
    # print(f'cell_comp: {cell_comp}')

    ontologies = {'cell_comp': cell_comp, 'mol_func': mol_func, 'bio_process': bio_process}
    ont = list(ontologies.keys())
    n_ont = list(ontologies.values())

    # Plot ontolology population with count and probability mass
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')
    ax.set_title(f'Ontology population (controlled source)')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_1_and_2 + f"/source_population_only_1_and_2.png")
    plt.close()

    tot_ont = mol_func + bio_process + cell_comp
    # num_bins = 20
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont / tot_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')  # Probability density normalizes for the width of the bin
    ax.set_title(f'Ontology population mass (controlled source)')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_1_and_2 + f"/source_population_mass_only_1_and_2.png")
    plt.close()

    if ont_based_prep:  # Standard scaling that differentiate between ontologies
        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = dataset_data.loc[dataset_data["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_mol_func = dataset_data.loc[dataset_data["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_bio_process = dataset_data.loc[dataset_data["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        if use_total_score:
            X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
        else:
            X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
    else:
        if use_total_score:  # Standard scaling ontologically agnostic
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))

        else:
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']]))

        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_cell_comp)
        # print(type(X_test_benchmark_cell_comp))
        # print(X_test_benchmark_cell_comp.shape)

        X_test_benchmark_mol_func = X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_mol_func)
        # print(type(X_test_benchmark_mol_func))
        # print(X_test_benchmark_mol_func.shape)

        X_test_benchmark_bio_process = X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_bio_process)
        # print(type(X_test_benchmark_bio_process))
        # print(X_test_benchmark_bio_process.shape)

        if use_total_score:  # Standard scaling ontologically agnostic
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

        else:
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

    print(X_test_benchmark_cell_comp.head())
    print(X_test_benchmark_mol_func.head())
    print(X_test_benchmark_bio_process.head())

    # Retrieve the spline grades from file (already computed)
    spline_path = "stats/spline/original_grades_only_1_and_2.csv"
    spline_data = pd.read_csv(spline_path)
    spline_grades = spline_data['spline_grades']

    # Add to the original datasets the respective spline grades
    X_test_benchmark_cell_comp['spline_grades_cc'] = spline_grades.loc[dataset_data["Cell_comp"] == 1.0]
    X_test_benchmark_mol_func['spline_grades_mf'] = spline_grades.loc[dataset_data["Mol_funcs"] == 1.0]
    X_test_benchmark_bio_process['spline_grades_bp'] = spline_grades.loc[dataset_data["Bio_process"] == 1.0]

    # y_pred is a prediction for limited data to draw the scatter plot,
    # model_pred is more comprehensive for the hist plots
    pred_amount_cc = min(X_test_cc.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_cc = X_test_cc.head(pred_amount_cc)
    y_pred_cc = good_autoencoder.predict(X_best_pred_cc, batch_size=batch_size)
    y_pred_cc = np.squeeze(y_pred_cc)
    simgics_eval_cc = X_test_benchmark_cell_comp['simgics_eval'].to_numpy()
    simgics_eval_cc = np.squeeze(simgics_eval_cc)

    pred_amount_mf = min(X_test_mf.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_mf = X_test_mf.head(pred_amount_mf)
    y_pred_mf = good_autoencoder.predict(X_best_pred_mf, batch_size=batch_size)
    y_pred_mf = np.squeeze(y_pred_mf)
    simgics_eval_mf = X_test_benchmark_mol_func['simgics_eval'].to_numpy()
    simgics_eval_mf = np.squeeze(simgics_eval_mf)

    pred_amount_bp = min(X_test_bp.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_bp = X_test_bp.head(pred_amount_bp)
    y_pred_bp = good_autoencoder.predict(X_best_pred_bp, batch_size=batch_size)
    y_pred_bp = np.squeeze(y_pred_bp)
    simgics_eval_bp = X_test_benchmark_bio_process['simgics_eval'].to_numpy()
    simgics_eval_bp = np.squeeze(simgics_eval_bp)

    print(len(simgics_eval_cc))
    print(len(simgics_eval_mf))
    print(len(simgics_eval_bp))
    print(len(y_pred_cc))
    print(len(y_pred_mf))
    print(len(y_pred_bp))

    # Scatter plot prediction/true simgic
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Evaluation plots for each ontology')
    ax1.scatter(simgics_eval_cc[:pred_amount_cc], y_pred_cc, alpha=0.04)
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')
    ax2.scatter(simgics_eval_mf[:pred_amount_mf], y_pred_mf, alpha=0.04)
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')
    ax3.scatter(simgics_eval_bp[:pred_amount_bp], y_pred_bp, alpha=0.04)
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    plt.savefig(top_dir_path_only_1_and_2 + f"/eval_test_graph_{formatted_datetime}{score_tag}_ont_source_only_1_and_2.png")
    plt.close()

    # Create the prediction for the full subset of ontological data
    model_preds_cc = good_autoencoder.predict(X_test_cc, batch_size=batch_size)
    model_preds_cc = np.squeeze(model_preds_cc)
    model_preds_mf = good_autoencoder.predict(X_test_mf, batch_size=batch_size)
    model_preds_mf = np.squeeze(model_preds_mf)
    model_preds_bp = good_autoencoder.predict(X_test_bp, batch_size=batch_size)
    model_preds_bp = np.squeeze(model_preds_bp)

    # Histogram version, with more data
    n_bins = 20
    # Set colorbar
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # colorbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Set figure
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle('Evaluation histogram plots for each ontology')
    cmap = colormaps.get_cmap('summer')  # Get desired colormap - you can change this!
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_cc, model_preds_cc, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz) / 2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values

    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_mf, model_preds_mf, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz) / 2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_bp, model_preds_bp, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz) / 2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    # # Apply colorbar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.7])
    # fig.colorbar(colorbar, cax=cbar_ax)
    plt.savefig(top_dir_path_only_1_and_2 + f"/hist_eval_test_graph_{formatted_datetime}{score_tag}_ont_source_only_1_and_2.png")
    plt.close()

    # Find the prediction error
    pred_error_cc = np.absolute(simgics_eval_cc - model_preds_cc)
    pred_error_mf = np.absolute(simgics_eval_mf - model_preds_mf)
    pred_error_bp = np.absolute(simgics_eval_bp - model_preds_bp)

    # Histogram plot error distribution for each ontology
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Histogram prediction error distribution per ontology')
    weights_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    n, bins, patches = ax1.hist(pred_error_cc, num_bins, density=False, weights=weights_cc)
    ax1.axvline(pred_error_cc.mean(), linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')
    weights_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    n, bins, patches = ax2.hist(pred_error_mf, num_bins, density=False, weights=weights_mf)
    ax2.axvline(pred_error_mf.mean(), linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')
    weights_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    n, bins, patches = ax3.hist(pred_error_bp, num_bins, density=False, weights=weights_bp)
    ax3.axvline(pred_error_bp.mean(), linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_1_and_2 + f"/pred_error_graph_{formatted_datetime}{score_tag}_ont_source_only_1_and_2.png")
    plt.close()

    # Error in prediction from spline
    spline_grades_cc = X_test_benchmark_cell_comp['spline_grades_cc'].to_numpy()
    spline_grades_mf = X_test_benchmark_mol_func['spline_grades_mf'].to_numpy()
    spline_grades_bp = X_test_benchmark_bio_process['spline_grades_bp'].to_numpy()
    spline_grades_cc = np.squeeze(spline_grades_cc)
    spline_grades_mf = np.squeeze(spline_grades_mf)
    spline_grades_bp = np.squeeze(spline_grades_bp)

    spline_error_cc = np.absolute(simgics_eval_cc - spline_grades_cc)
    spline_error_mf = np.absolute(simgics_eval_mf - spline_grades_mf)
    spline_error_bp = np.absolute(simgics_eval_bp - spline_grades_bp)

    # av_spline_error_cc = np.sum(spline_error_cc)/len(spline_error_cc)
    # av_spline_error_mf = np.sum(spline_error_mf)/len(spline_error_mf)
    # av_spline_error_bp = np.sum(spline_error_bp)/len(spline_error_bp)
    # av_pred_error_cc = np.sum(pred_error_cc)/len(pred_error_cc)
    # av_pred_error_mf = np.sum(pred_error_mf)/len(pred_error_mf)
    # av_pred_error_bp = np.sum(pred_error_bp)/len(pred_error_bp)

    av_spline_error_cc = spline_error_cc.mean()
    av_spline_error_mf = spline_error_mf.mean()
    av_spline_error_bp = spline_error_bp.mean()
    av_pred_error_cc = pred_error_cc.mean()
    av_pred_error_mf = pred_error_mf.mean()
    av_pred_error_bp = pred_error_bp.mean()

    # error_log_name = top_dir_path_only_1_and_2 + f"/error_log.txt"
    error_message_ont = (f"\n\nSum of all error for each method for each ontology\n\n"
                         f"Average error for spline in cellular components: {av_spline_error_cc}\n"
                         f"Average error for model in cellular components: {av_pred_error_cc}\n"
                         f"Average error for spline in molecular functions: {av_spline_error_mf}\n"
                         f"Average error for model in molecular functions: {av_pred_error_mf}\n"
                         f"Average error for spline in biological processes: {av_spline_error_bp}\n"
                         f"Average error for model in biological processes: {av_pred_error_bp}\n"
                         )

    write_to_file(error_log_name, error_message_ont)

    # Histogram plot error distribution for each ontology, confrontation between model and spline
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Histogram prediction error distribution model vs spline')

    weights_model_error_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    weight_spline_error_cc = np.ones_like(spline_error_cc) / (len(spline_error_cc))
    ax1.hist([pred_error_cc, spline_error_cc], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_cc, weight_spline_error_cc], color=['tab:blue', 'tab:orange'])
    ax1.axvline(pred_error_cc.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax1.axvline(spline_error_cc.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    weight_spline_error_mf = np.ones_like(spline_error_mf) / (len(spline_error_mf))
    ax2.hist([pred_error_mf, spline_error_mf], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_mf, weight_spline_error_mf], color=['tab:blue', 'tab:orange'])
    ax2.axvline(pred_error_mf.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax2.axvline(spline_error_mf.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    weight_spline_error_bp = np.ones_like(spline_error_bp) / (len(spline_error_bp))
    ax3.hist([pred_error_bp, spline_error_bp], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_bp, weight_spline_error_bp], color=['tab:blue', 'tab:orange'])
    ax3.axvline(pred_error_bp.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax3.axvline(spline_error_bp.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_1_and_2 + f"/pred_error_graph_{formatted_datetime}{score_tag}_spline_model_ont_source_only_1_and_2.png")
    plt.close()

    # Histogram plot eval distribution for each ontology, confrontation between model, spline and ground-truth
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Simgic distribution model/spline/dataset')
    weights_simgic_cc = np.ones_like(simgics_eval_cc) / (len(simgics_eval_cc))
    weights_model_cc = np.ones_like(model_preds_cc) / (len(model_preds_cc))
    weight_spline_cc = np.ones_like(spline_grades_cc) / (len(spline_grades_cc))
    ax1.hist([simgics_eval_cc, model_preds_cc, spline_grades_cc], num_bins,
             label=['simgics_eval_cc', 'model_preds_cc', 'spline_grades_cc'],
             weights=[weights_simgic_cc, weights_model_cc, weight_spline_cc])
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_mf = np.ones_like(simgics_eval_mf) / (len(simgics_eval_mf))
    weights_model_mf = np.ones_like(model_preds_mf) / (len(model_preds_mf))
    weight_spline_mf = np.ones_like(spline_grades_mf) / (len(spline_grades_mf))
    ax2.hist([simgics_eval_mf, model_preds_mf, spline_grades_mf], num_bins,
             label=['simgics_eval_mf', 'model_preds_mf', 'spline_grades_mf'],
             weights=[weights_simgic_mf, weights_model_mf, weight_spline_mf])
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_bp = np.ones_like(simgics_eval_bp) / (len(simgics_eval_bp))
    weights_model_bp = np.ones_like(model_preds_bp) / (len(model_preds_bp))
    weight_spline_bp = np.ones_like(spline_grades_bp) / (len(spline_grades_bp))
    ax3.hist([simgics_eval_bp, model_preds_bp, spline_grades_bp], num_bins,
             label=['simgics_eval_bp', 'model_preds_bp', 'spline_grades_bp'],
             weights=[weights_simgic_bp, weights_model_bp, weight_spline_bp])
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_1_and_2 + f"/eval_graph_{formatted_datetime}{score_tag}_spline_model_gt_ont_source_only_1_and_2.png")
    plt.close()

    # Since the

    confrontation_benchmark_source_only_0(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                                error_log_name, batch_size, use_total_score=True, n_benchmark=10000,
                                                ont_based_prep=False)


#Confrontation benchmark for data from category 0: IEA data with no other property
def confrontation_benchmark_source_only_0(good_autoencoder, top_dir_path, formatted_datetime, simgic_threshold,
                                                error_log_name, batch_size, use_total_score=True, n_benchmark=10000,
                                                ont_based_prep=False):
    """good_autoencoder: the model that is being evaluated
        top_dir_path: directory in which graphs are saved
        formatted_datetime: datetime to differentiate the data
        use_total_score: does the model uses total scores
        n_benchmark: amount of point plotted in the prediction graphs
        save_vars: Save the computed variables, for debugging purposes"""

    if use_total_score:  # Tags name files if the total score is used (tsu: Total Score Used)
        score_tag = '_tsu'
    else:
        score_tag = ''

    top_dir_path_only_0 = os.path.join(top_dir_path, f"source_confrontation_only_0")
    os.mkdir(top_dir_path_only_0)

    dataset_path = 'stats/dataset/simgics_annotated_merged_clean_only_0.csv'  # Mettere il dataset quando sarà pronto
    dataset_data = pd.read_csv(dataset_path)

    # Compute total amount of each ontology
    mol_func = (dataset_data["Mol_funcs"] == 1.0).sum()
    bio_process = (dataset_data["Bio_process"] == 1.0).sum()
    cell_comp = (dataset_data["Cell_comp"] == 1.0).sum()
    # print(f'mol_func: {mol_func}')
    # print(f'bio_process: {bio_process}')
    # print(f'cell_comp: {cell_comp}')

    ontologies = {'cell_comp': cell_comp, 'mol_func': mol_func, 'bio_process': bio_process}
    ont = list(ontologies.keys())
    n_ont = list(ontologies.values())

    # Plot ontolology population with count and probability mass
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')
    ax.set_title(f'Ontology population (controlled source)')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_0 + f"/source_population_only_0.png")
    plt.close()

    tot_ont = mol_func + bio_process + cell_comp
    # num_bins = 20
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(ont, n_ont / tot_ont)
    ax.set_xlabel('Ontology')
    ax.set_ylabel('Count')  # Probability density normalizes for the width of the bin
    ax.set_title(f'Ontology population mass (controlled source)')
    fig.tight_layout()
    plt.savefig(top_dir_path_only_0 + f"/source_population_mass_only_0.png")
    plt.close()

    if ont_based_prep:  # Standard scaling that differentiate between ontologies
        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = dataset_data.loc[dataset_data["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_mol_func = dataset_data.loc[dataset_data["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        X_test_benchmark_bio_process = dataset_data.loc[dataset_data["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]

        if use_total_score:
            X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_cell_comp[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_mol_func[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_bio_process[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
        else:
            X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_cell_comp[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark_mol_func[['Inf_content', 'Int_confidence', 'GScore']]))
            X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark_bio_process[['Inf_content', 'Int_confidence', 'GScore']]))
            # Contains only the data required for the predictions
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
    else:
        if use_total_score:  # Standard scaling ontologically agnostic
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(
                    X_test_benchmark[['Inf_content', 'Total_score', 'Int_confidence', 'GScore']]))

        else:
            X_test_benchmark = dataset_data
            X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']] = (
                StandardScaler().fit_transform(X_test_benchmark[['Inf_content', 'Int_confidence', 'GScore']]))

        # Separate every category in each own dataset
        X_test_benchmark_cell_comp = X_test_benchmark.loc[X_test_benchmark["Cell_comp"] == 1.0, :]
        X_test_benchmark_cell_comp = X_test_benchmark_cell_comp[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_cell_comp)
        # print(type(X_test_benchmark_cell_comp))
        # print(X_test_benchmark_cell_comp.shape)

        X_test_benchmark_mol_func = X_test_benchmark.loc[X_test_benchmark["Mol_funcs"] == 1.0, :]
        X_test_benchmark_mol_func = X_test_benchmark_mol_func[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_mol_func)
        # print(type(X_test_benchmark_mol_func))
        # print(X_test_benchmark_mol_func.shape)

        X_test_benchmark_bio_process = X_test_benchmark.loc[X_test_benchmark["Bio_process"] == 1.0, :]
        X_test_benchmark_bio_process = X_test_benchmark_bio_process[
            ['simgics_eval', 'SeqID', 'GOID', 'Inf_content', 'Total_score',
             'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp',
             'Mol_funcs']]
        # print(X_test_benchmark_bio_process)
        # print(type(X_test_benchmark_bio_process))
        # print(X_test_benchmark_bio_process.shape)

        if use_total_score:  # Standard scaling ontologically agnostic
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Total_score', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

        else:
            X_test_cc = X_test_benchmark_cell_comp[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_mf = X_test_benchmark_mol_func[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]
            X_test_bp = X_test_benchmark_bio_process[
                ['Inf_content', 'Int_confidence', 'GScore', 'Bio_process', 'Cell_comp', 'Mol_funcs']]

    print(X_test_benchmark_cell_comp.head())
    print(X_test_benchmark_mol_func.head())
    print(X_test_benchmark_bio_process.head())

    # Retrieve the spline grades from file (already computed)
    spline_path = "stats/spline/original_grades_only_0.csv"
    spline_data = pd.read_csv(spline_path)
    spline_grades = spline_data['spline_grades']

    # Add to the original datasets the respective spline grades
    X_test_benchmark_cell_comp['spline_grades_cc'] = spline_grades.loc[dataset_data["Cell_comp"] == 1.0]
    X_test_benchmark_mol_func['spline_grades_mf'] = spline_grades.loc[dataset_data["Mol_funcs"] == 1.0]
    X_test_benchmark_bio_process['spline_grades_bp'] = spline_grades.loc[dataset_data["Bio_process"] == 1.0]

    # y_pred is a prediction for limited data to draw the scatter plot,
    # model_pred is more comprehensive for the hist plots
    pred_amount_cc = min(X_test_cc.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_cc = X_test_cc.head(pred_amount_cc)
    y_pred_cc = good_autoencoder.predict(X_best_pred_cc, batch_size=batch_size)
    y_pred_cc = np.squeeze(y_pred_cc)
    simgics_eval_cc = X_test_benchmark_cell_comp['simgics_eval'].to_numpy()
    simgics_eval_cc = np.squeeze(simgics_eval_cc)

    pred_amount_mf = min(X_test_mf.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_mf = X_test_mf.head(pred_amount_mf)
    y_pred_mf = good_autoencoder.predict(X_best_pred_mf, batch_size=batch_size)
    y_pred_mf = np.squeeze(y_pred_mf)
    simgics_eval_mf = X_test_benchmark_mol_func['simgics_eval'].to_numpy()
    simgics_eval_mf = np.squeeze(simgics_eval_mf)

    pred_amount_bp = min(X_test_bp.shape[0], n_benchmark)  # Point for predictions
    X_best_pred_bp = X_test_bp.head(pred_amount_bp)
    y_pred_bp = good_autoencoder.predict(X_best_pred_bp, batch_size=batch_size)
    y_pred_bp = np.squeeze(y_pred_bp)
    simgics_eval_bp = X_test_benchmark_bio_process['simgics_eval'].to_numpy()
    simgics_eval_bp = np.squeeze(simgics_eval_bp)

    print(len(simgics_eval_cc))
    print(len(simgics_eval_mf))
    print(len(simgics_eval_bp))
    print(len(y_pred_cc))
    print(len(y_pred_mf))
    print(len(y_pred_bp))

    # Scatter plot prediction/true simgic
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Evaluation plots for each ontology')
    ax1.scatter(simgics_eval_cc[:pred_amount_cc], y_pred_cc, alpha=0.04)
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')
    ax2.scatter(simgics_eval_mf[:pred_amount_mf], y_pred_mf, alpha=0.04)
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')
    ax3.scatter(simgics_eval_bp[:pred_amount_bp], y_pred_bp, alpha=0.04)
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    plt.savefig(
        top_dir_path_only_0 + f"/eval_test_graph_{formatted_datetime}{score_tag}_ont_source_only_0.png")
    plt.close()

    # Create the prediction for the full subset of ontological data
    model_preds_cc = good_autoencoder.predict(X_test_cc, batch_size=batch_size)
    model_preds_cc = np.squeeze(model_preds_cc)
    model_preds_mf = good_autoencoder.predict(X_test_mf, batch_size=batch_size)
    model_preds_mf = np.squeeze(model_preds_mf)
    model_preds_bp = good_autoencoder.predict(X_test_bp, batch_size=batch_size)
    model_preds_bp = np.squeeze(model_preds_bp)

    # Histogram version, with more data
    n_bins = 20
    # Set colorbar
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # colorbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Set figure
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle('Evaluation histogram plots for each ontology')
    cmap = colormaps.get_cmap('summer')  # Get desired colormap - you can change this!
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_cc, model_preds_cc, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz) / 2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values

    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Evaluation')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_mf, model_preds_mf, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz) / 2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Evaluation')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    hist, xedges, yedges = np.histogram2d(simgics_eval_bp, model_preds_bp, bins=n_bins, range=[[0, 1], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 1 / n_bins * np.ones_like(zpos)
    dz = hist.ravel()
    max_height = np.max(dz) / 2  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Evaluation')
    # # Apply colorbar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.7])
    # fig.colorbar(colorbar, cax=cbar_ax)
    plt.savefig(
        top_dir_path_only_0 + f"/hist_eval_test_graph_{formatted_datetime}{score_tag}_ont_source_only_0.png")
    plt.close()

    # Find the prediction error
    pred_error_cc = np.absolute(simgics_eval_cc - model_preds_cc)
    pred_error_mf = np.absolute(simgics_eval_mf - model_preds_mf)
    pred_error_bp = np.absolute(simgics_eval_bp - model_preds_bp)

    # Histogram plot error distribution for each ontology
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Histogram prediction error distribution per ontology')
    weights_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    n, bins, patches = ax1.hist(pred_error_cc, num_bins, density=False, weights=weights_cc)
    ax1.axvline(pred_error_cc.mean(), linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')
    weights_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    n, bins, patches = ax2.hist(pred_error_mf, num_bins, density=False, weights=weights_mf)
    ax2.axvline(pred_error_mf.mean(), linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')
    weights_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    n, bins, patches = ax3.hist(pred_error_bp, num_bins, density=False, weights=weights_bp)
    ax3.axvline(pred_error_bp.mean(), linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(
        top_dir_path_only_0 + f"/pred_error_graph_{formatted_datetime}{score_tag}_ont_source_only_0.png")
    plt.close()

    # Error in prediction from spline
    spline_grades_cc = X_test_benchmark_cell_comp['spline_grades_cc'].to_numpy()
    spline_grades_mf = X_test_benchmark_mol_func['spline_grades_mf'].to_numpy()
    spline_grades_bp = X_test_benchmark_bio_process['spline_grades_bp'].to_numpy()
    spline_grades_cc = np.squeeze(spline_grades_cc)
    spline_grades_mf = np.squeeze(spline_grades_mf)
    spline_grades_bp = np.squeeze(spline_grades_bp)

    spline_error_cc = np.absolute(simgics_eval_cc - spline_grades_cc)
    spline_error_mf = np.absolute(simgics_eval_mf - spline_grades_mf)
    spline_error_bp = np.absolute(simgics_eval_bp - spline_grades_bp)

    # av_spline_error_cc = np.sum(spline_error_cc)/len(spline_error_cc)
    # av_spline_error_mf = np.sum(spline_error_mf)/len(spline_error_mf)
    # av_spline_error_bp = np.sum(spline_error_bp)/len(spline_error_bp)
    # av_pred_error_cc = np.sum(pred_error_cc)/len(pred_error_cc)
    # av_pred_error_mf = np.sum(pred_error_mf)/len(pred_error_mf)
    # av_pred_error_bp = np.sum(pred_error_bp)/len(pred_error_bp)

    av_spline_error_cc = spline_error_cc.mean()
    av_spline_error_mf = spline_error_mf.mean()
    av_spline_error_bp = spline_error_bp.mean()
    av_pred_error_cc = pred_error_cc.mean()
    av_pred_error_mf = pred_error_mf.mean()
    av_pred_error_bp = pred_error_bp.mean()

    # error_log_name = top_dir_path_only_0 + f"/error_log.txt"
    error_message_ont = (f"\n\nSum of all error for each method for each ontology\n\n"
                         f"Average error for spline in cellular components: {av_spline_error_cc}\n"
                         f"Average error for model in cellular components: {av_pred_error_cc}\n"
                         f"Average error for spline in molecular functions: {av_spline_error_mf}\n"
                         f"Average error for model in molecular functions: {av_pred_error_mf}\n"
                         f"Average error for spline in biological processes: {av_spline_error_bp}\n"
                         f"Average error for model in biological processes: {av_pred_error_bp}\n"
                         )

    write_to_file(error_log_name, error_message_ont)

    # Histogram plot error distribution for each ontology, confrontation between model and spline
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Histogram prediction error distribution model vs spline')

    weights_model_error_cc = np.ones_like(pred_error_cc) / (len(pred_error_cc))
    weight_spline_error_cc = np.ones_like(spline_error_cc) / (len(spline_error_cc))
    ax1.hist([pred_error_cc, spline_error_cc], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_cc, weight_spline_error_cc], color=['tab:blue', 'tab:orange'])
    ax1.axvline(pred_error_cc.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax1.axvline(spline_error_cc.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_mf = np.ones_like(pred_error_mf) / (len(pred_error_mf))
    weight_spline_error_mf = np.ones_like(spline_error_mf) / (len(spline_error_mf))
    ax2.hist([pred_error_mf, spline_error_mf], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_mf, weight_spline_error_mf], color=['tab:blue', 'tab:orange'])
    ax2.axvline(pred_error_mf.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax2.axvline(spline_error_mf.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Error', ylabel='Probability mass')

    weights_model_error_bp = np.ones_like(pred_error_bp) / (len(pred_error_bp))
    weight_spline_error_bp = np.ones_like(spline_error_bp) / (len(spline_error_bp))
    ax3.hist([pred_error_bp, spline_error_bp], num_bins, label=['model_error', 'spline_error'],
             weights=[weights_model_error_bp, weight_spline_error_bp], color=['tab:blue', 'tab:orange'])
    ax3.axvline(pred_error_bp.mean(), color='tab:blue', linestyle='dashed', linewidth=1)
    ax3.axvline(spline_error_bp.mean(), color='tab:orange', linestyle='dashed', linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Error', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(
        top_dir_path_only_0 + f"/pred_error_graph_{formatted_datetime}{score_tag}_spline_model_ont_source_only_0.png")
    plt.close()

    # Histogram plot eval distribution for each ontology, confrontation between model, spline and ground-truth
    num_bins = 20
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Simgic distribution model/spline/dataset')
    weights_simgic_cc = np.ones_like(simgics_eval_cc) / (len(simgics_eval_cc))
    weights_model_cc = np.ones_like(model_preds_cc) / (len(model_preds_cc))
    weight_spline_cc = np.ones_like(spline_grades_cc) / (len(spline_grades_cc))
    ax1.hist([simgics_eval_cc, model_preds_cc, spline_grades_cc], num_bins,
             label=['simgics_eval_cc', 'model_preds_cc', 'spline_grades_cc'],
             weights=[weights_simgic_cc, weights_model_cc, weight_spline_cc])
    ax1.legend(loc='upper right')
    ax1.set_title('Cellular components')
    ax1.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_mf = np.ones_like(simgics_eval_mf) / (len(simgics_eval_mf))
    weights_model_mf = np.ones_like(model_preds_mf) / (len(model_preds_mf))
    weight_spline_mf = np.ones_like(spline_grades_mf) / (len(spline_grades_mf))
    ax2.hist([simgics_eval_mf, model_preds_mf, spline_grades_mf], num_bins,
             label=['simgics_eval_mf', 'model_preds_mf', 'spline_grades_mf'],
             weights=[weights_simgic_mf, weights_model_mf, weight_spline_mf])
    ax2.legend(loc='upper right')
    ax2.set_title('Molecular functions')
    ax2.set(xlabel='Simgic', ylabel='Probability mass')

    weights_simgic_bp = np.ones_like(simgics_eval_bp) / (len(simgics_eval_bp))
    weights_model_bp = np.ones_like(model_preds_bp) / (len(model_preds_bp))
    weight_spline_bp = np.ones_like(spline_grades_bp) / (len(spline_grades_bp))
    ax3.hist([simgics_eval_bp, model_preds_bp, spline_grades_bp], num_bins,
             label=['simgics_eval_bp', 'model_preds_bp', 'spline_grades_bp'],
             weights=[weights_simgic_bp, weights_model_bp, weight_spline_bp])
    ax3.legend(loc='upper right')
    ax3.set_title('Biological processes')
    ax3.set(xlabel='Simgic', ylabel='Probability mass')
    fig.tight_layout()
    plt.savefig(
        top_dir_path_only_0 + f"/eval_graph_{formatted_datetime}{score_tag}_spline_model_gt_ont_source_only_0.png")
    plt.close()



def confrontation_benchmark_future_net():
    return

