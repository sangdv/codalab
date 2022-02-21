# NOTE: we only support python2.7 on AIHUB.VN at the moment,
# be mindful to check what is supported in sci-kit learns of python2.7 (scikitlearn < 0.22.0)
from numpy import genfromtxt

import os
import sys

from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import pandas as pd
from math import sqrt


def levenshteinDistance(s1, s2):
    # See Stackoverflow
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]*1.0


def cal_cer(ref_text_arr, pred_text_arr):
    """
    TODO: implement your CER funct here
    :param ref_text_arr: reference texts
    :param pred_text_arr: prediction texts
    :return:
    """
    avg_cer = 0.0

    return avg_cer


if __name__ == "__main__":
    [_, input_dir, output_dir] = sys.argv
    submission_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    gt_df = pd.read_csv(os.path.join(truth_dir, 'ground_truth.csv'),
                        delimiter=',')

    gt_df = gt_df.fillna({'anno_texts': 'NULL'})

    submission_df = pd.read_csv(os.path.join(submission_dir, 'results.csv'),
                        delimiter=',')

    submission_df = submission_df.fillna({'anno_texts': 'NULL'})

    gt_quality_score = list(gt_df['anno_image_quality'])

    sub_quality_score = list(submission_df['anno_image_quality'])

    rmse = sqrt(mean_squared_error(gt_quality_score, sub_quality_score))

    # cer
    ref_text_arr = list(gt_df['anno_texts'])
    pred_text_arr = list(submission_df['anno_texts'])
    cer = cal_cer(ref_text_arr, pred_text_arr)

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        output_file.write("RMSE: {:f}\n".format(round(rmse, 6)))
        output_file.write("CER: {:f}".format(round(cer, 6)))
