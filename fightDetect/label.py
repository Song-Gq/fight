import data_process as dp
import pandas as pd
from datetime import datetime
import os
import re
import plotly.express as px
import plotly
from huang_thresholding import HuangThresholding
import numpy as np


def vid_statis():
    vid_label = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        high_score = dp.get_high_score(boxes, upper_limit=340, min_p_len=valid_min_frame)
        if high_score.shape[0] > 0:
            score_thre = high_score['score_thre'].values[0]
            # high_score = boxes[boxes['score'] > score_thre]
            file_res = high_score.groupby(['idx'])['image_id'].agg(['min', 'max', 'count'])
            file_res = file_res[file_res['count'] > valid_min_frame]
            
            score = high_score.groupby(['idx'])['score'].agg(['min', 'max', 'mean'])
            file_res = pd.merge(file_res, score, how='inner', on=['idx'], suffixes=['frame', 'score'])

            file_res['file'] = re.sub('^(AlphaPose_)|(\.json)$', '', fname)
            file_res['score_thre'] = score_thre
            vid_label = pd.concat([vid_label, file_res], axis=0)
    return vid_label


def draw_score():
    score_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        boxes['file'] = re.sub('^(AlphaPose_)|(\.json)$', '', fname)
        boxes['cat'] = re.sub('^(AlphaPose_)|([0-9]+\.json)$', '', fname)
        score_res = pd.concat([score_res, boxes[['idx', 'score', 'file', 'cat']]])
        # score = raw_df.groupby(['idx'])['score']
    fig = px.histogram(score_res, x='score', color='idx', marginal='box', hover_name='file')
    plotly.offline.plot(fig, filename=output_dir + 'statis-score-hist.html', auto_open=False)


# just for testing
def get_score_thre():
    score_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        boxes['file'] = re.sub('^(AlphaPose_)|(\.json)$', '', fname)
        boxes['cat'] = re.sub('^(AlphaPose_)|([0-9]+\.json)$', '', fname)
        score_res = pd.concat([score_res, boxes[['idx', 'score', 'file', 'cat']]])

    # test_data = np.random.rand(100)
    # test_data = np.load("fightDetect/colonies.npy")

    # data range from [0 to upper_limit/100.0]
    upper_limit = 340
    # histogram bins from [0.00, 0.01) to [3.39, 3.40]
    histogram_data, _ = np.histogram(score_res['score'], bins=[b/100.0 for b in range(0, upper_limit+1)])
    huang_thresholding = HuangThresholding(histogram_data, upper_limit)
    threshold = huang_thresholding.find_threshold()
    score_res = pd.DataFrame()


# args
src_dir = "test-plus/"
score_thre = 2.6
valid_min_frame = 10
output_dir = 'fightDetect/fig/' + src_dir


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    # get_score_thre()

    # draw_score()

    labels = vid_statis()
    labels.to_excel(output_dir + 'labels.' + datetime.now().strftime("%Y-%m-%d.%H-%M-%S") + '.xlsx')
