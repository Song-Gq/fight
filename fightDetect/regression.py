import numpy as np
import plotly.express as px
import plotly
import pandas as pd
import data_process as dp
from sklearn.preprocessing import PolynomialFeatures


src_dir = "test/"
iou_type = 'giou'
score_thre = 2
interp_type = 'previous'
output_suffix = '-' + iou_type + \
    '-interp=' + interp_type + \
    '-score-thre=' + str(score_thre)


if __name__ == '__main__':
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        # drop data with lower scores
        high_score = boxes[boxes['score'] > score_thre]






