"""
Evaluation tools

This script contains functions for evaluating the performance of a model.
This includes creating graphics of these evaluations.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from util_tools.general import to_numpy, get_PIL_image_from_matplotlib_figure

def calculate_accuracy(predictions, labels):
    """
    Args:
        predictions ([int]): List of predictions
        labels ([int]): List of labels/ground truths

    Description:
        Calculates the accuracy given predictions and labels
    """

    predictions = to_numpy(predictions)
    labels = to_numpy(labels)
    
    correct = (predictions == labels).astype(np.int64).sum()
    total = len(labels)

    return correct / total

def calculate_per_class_accuracies(predictions, labels):
    """
    Args:
        predictions ([int]): List of predictions
        labels ([int]): List of labels/ground truths

    Description:
        Calculates the per class accuracies of the given predictions and labels
    """

    predictions = to_numpy(predictions)
    labels = to_numpy(labels)
    
    correct_mask = (predictions == labels)

    accuracies = []
    for lbl in range(labels.max()+1):
        lbl_mask = labels == lbl
        lbl_total = lbl_mask.astype(np.int64).sum()
        lbl_correct = (correct_mask & lbl_mask).astype(np.int64).sum()
        acc = 0
        if lbl_total > 0:
            acc = lbl_correct / lbl_total
        accuracies.append(acc)

    return accuracies

def create_accuracy_column_chart(predictions, labels, lbl_to_name_map=None):
    """
    Args:
        predictions ([int]): List of predictions
        labels ([int]): List of labels/ground truths
        lbl_to_name_map (dict(int=>str)): Map from label to name/class
        kwargs (): Key word arguments for matplotlib.pyplot.bar method

    Description:
        Creates column chart of per class accuracy.
    """

    accuracies = calculate_per_class_accuracies(predictions, labels)

    names = [i if lbl_to_name_map is None else lbl_to_name_map[i] for i in range(len(accuracies))]


    fig = plt.figure()
    plt.title("Per Class Accuracies")
    plt.bar(names, accuracies)
    img = get_PIL_image_from_matplotlib_figure(fig)
    plt.close()

    return img


