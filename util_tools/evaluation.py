"""
Evaluation tools

This script contains functions for evaluating the performance of a model.
This includes creating graphics of these evaluations.
"""

import numpy as np

from util_tools.general import to_numpy

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

