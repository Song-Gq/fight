import data_process as dp
import pandas as pd
from datetime import datetime
import os
import re
import plotly.express as px
import plotly
from huang_thresholding import HuangThresholding
import numpy as np


def vid_statis(src_dir):
    print('start statistics')
    file_num = 0
    vid_label = pd.DataFrame()
    for _, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        file_num = file_num + 1
        print('processing file: ', file_num)
        # Not Empty Dataframe
        if boxes is not None:
            high_score = dp.get_high_score(boxes, upper_limit=340, 
                                           min_p_len=VALID_MIN_FRAME, 
                                           lower_confi=LOWER_CONFIDENCE)
            if high_score.shape[0] > 0:
                score_thresholds = high_score['score_thre'].values[0]
                # high_score = boxes[boxes['score'] > score_thre]
                file_res = high_score.groupby(['idx'])['image_id'].agg(['min', 'max', 'count'])
                file_res = file_res[file_res['count'] > VALID_MIN_FRAME]
                
                score = high_score.groupby(['idx'])['score'].agg(['min', 'max', 'mean'])
                file_res = pd.merge(file_res, score, how='inner', on=['idx'], suffixes=['frame', 'score'])

                file_res['file'] = re.sub(r'^(AlphaPose_)|(\.json)$', '', fname)
                file_res['score_thre'] = score_thresholds
                vid_label = pd.concat([vid_label, file_res], axis=0)
    return vid_label


def draw_score(src_dir, cat_col='file'):
    score_res = pd.DataFrame()
    for _, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        if boxes is not None:
            boxes['file'] = re.sub(r'^(AlphaPose_)|(\.json)$', '', fname)
            boxes['cat'] = re.sub(r'^(AlphaPose_)|([0-9]+\.json)$', '', fname)
            boxes['dataset'] = src_dir.replace('/', '')
            score_res = pd.concat([score_res, boxes[['idx', 'score', 'file', 'cat', 'dataset']]])
            # score = raw_df.groupby(['idx'])['score']
    fig = px.histogram(score_res, x='score', color=cat_col, marginal='box', hover_name=cat_col)
    plotly.offline.plot(fig, filename=OUTPUT_DIR + 'statis-score-hist.html', auto_open=False)


# compare hist data between two datasets
def draw_diff(src1, src2, out_dir):
    scores = pd.DataFrame()
    for src in [src1, src2]:
        for _, boxes, fname in dp.iter_files('fightDetect/data/' + src):
            if boxes is not None:
                boxes['file'] = re.sub(r'^(AlphaPose_)|(\.json)$', '', fname)
                boxes['cat'] = re.sub(r'^(AlphaPose_)|([0-9]+\.json)$', '', fname)
                boxes['dataset'] = src
                scores = pd.concat([scores, boxes[['idx', 'score', 'file', 'cat', 'dataset']]])
    fig = px.histogram(scores, x='score', color='dataset', marginal='box', hover_name='dataset')
    plotly.offline.plot(fig, filename=out_dir + 
                        src1.replace('/', '') + '.vs.' + 
                        src2.replace('/', '') +
                        '-statis-score-hist-diff.html', 
                        auto_open=False)


# just for testing
def get_score_thre(src_dir):
    score_res = pd.DataFrame()
    for _, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        boxes['file'] = re.sub(r'^(AlphaPose_)|(\.json)$', '', fname)
        boxes['cat'] = re.sub(r'^(AlphaPose_)|([0-9]+\.json)$', '', fname)
        score_res = pd.concat([score_res, boxes[['idx', 'score', 'file', 'cat']]])

    # test_data = np.random.rand(100)
    # test_data = np.load("fightDetect/colonies.npy")

    # data range from [0 to upper_limit/100.0]
    upper_limit = 340
    # histogram bins from [0.00, 0.01) to [3.39, 3.40]
    histogram_data, _ = np.histogram(score_res['score'], bins=[b/100.0 for b in range(0, upper_limit+1)])
    huang_thresholding = HuangThresholding(histogram_data, upper_limit)
    return huang_thresholding.find_threshold()


# args
SRC_DIR = "test-plus/"
# score_thre = 2.6
VALID_MIN_FRAME = 5
LOWER_CONFIDENCE = False


OUTPUT_SUFFIX = '-valid_minf=' + str(VALID_MIN_FRAME)
OUTPUT_DIR = 'fightDetect/fig/' + SRC_DIR +  \
    datetime.now().strftime(r"%Y-%m-%d.%H-%M-%S") + \
    OUTPUT_SUFFIX + '/'


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # get_score_thre()

    draw_score(SRC_DIR, cat_col='dataset')

    # labels = vid_statis(SRC_DIR)
    # labels.to_excel(OUTPUT_DIR + 'labels.' + datetime.now().strftime(r"%Y-%m-%d.%H-%M-%S") + '.xlsx')

    # diff_out = 'fightDetect/fig/diff/'
    # os.makedirs(diff_out, exist_ok=True)
    # draw_diff("fight-sur/", "test-plus/", diff_out)

