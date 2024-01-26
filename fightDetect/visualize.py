import numpy as np
import plotly.express as px
import plotly
import pandas as pd
import data_process as dp
from datetime import datetime
import os


def draw_fig(src_dir, key_df, json_name):
    # 9. left hand: 27-x, 28-y, 29-confidence
    fig = px.scatter_3d(key_df, x='27', y='28', z='image_id', color='29', symbol='idx')
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-left-hand.html', auto_open=False)

    # 15. left ankle: 45-x, 46-y, 47-confidence
    fig = px.scatter_3d(key_df, x='45', y='46', z='image_id', color='47', symbol='idx')
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-left-ankle.html', auto_open=False)

    # calculate the speed
    speed = dp.key_speed(key_df)

    # speed.fillna(0.)
    fig = px.scatter_3d(speed[speed['29'] > 0.5], x='27', y='28', z='image_id', color='29_speed', symbol='idx')
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-left-hand-speed.html', auto_open=False)

    fig = px.scatter_3d(speed[speed['47'] > 0.5], x='45', y='46', z='image_id', color='47_speed', symbol='idx')
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-left-ankle-speed.html', auto_open=False)


def draw_iou(src_dir, merged_df, json_name):
    fig = px.line(merged_df, x='image_id', y='iou', color='comb', markers=False,
                  labels={'image_id': '帧', 'iou': 'GIoU', 'comb': '行人A+B'})
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-iou.html', auto_open=False)
    # fig.show()


def draw_fft(src_dir, merged_df, json_name):
    fig = px.line(merged_df, x='image_id', y='y', color='comb', markers=False)
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-fft.html', auto_open=False)


def draw_xy(src_dir, box_df, json_name):
    # box ouput format: x, y, w, h
    fig = px.scatter_3d(box_df, x='0', y='1', z='image_id', color='score', symbol='idx')
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-xy.html', auto_open=False)


def draw_xy_fft(src_dir, fft_df, json_name):
    x_fft = fft_df[['image_id', 'x_fft', 'idx']].copy(deep=True)
    x_fft.columns = ['image_id', 'fft', 'idx']
    x_fft['axis'] = 'x'
    y_fft = fft_df[['image_id', 'y_fft', 'idx']].copy(deep=True)
    y_fft.columns = ['image_id', 'fft', 'idx']
    y_fft['axis'] = 'y'
    fft_xy = pd.concat([x_fft, y_fft], axis=0)

    fig = px.line(fft_xy, x='image_id', y='fft', color='idx', facet_col='axis')
    fig.update_layout(font=dict(size=18))
    plotly.offline.plot(fig, filename=OUTPUT_DIR + json_name + '-xy-fft.html', auto_open=False)


# args
SRC_DIR = "test-single/"
VALID_MIN_FRAME = 30
# for x, y location segmentation and regression
# and for calculating iou
INTERP_METHOD = 'previous'
IOU_TYPE = 'giou'
LOWER_CONFIDENCE = False

OUTPUT_SUFFIX = '-valid_minf=' + str(VALID_MIN_FRAME)
OUTPUT_DIR = 'fightDetect/fig/' + SRC_DIR +  \
    datetime.now().strftime(r"%Y-%m-%d.%H-%M-%S") + \
    OUTPUT_SUFFIX + '/'

# src_dir = "test-single/"
# iou_type = 'giou'
# # score_thre = 2
# interp_type = 'previous'
# output_suffix = '-' + iou_type + \
#     '-interp=' + interp_type + \
#     '-score-thre=' + str(score_thre)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + SRC_DIR):
        # drop data with lower scores
        # high_score = boxes[boxes['score'] > score_thre]
        high_score = dp.get_high_score(boxes, upper_limit=340, 
                                min_p_len=VALID_MIN_FRAME, 
                                lower_confi=LOWER_CONFIDENCE)
        if high_score.shape[0] > 0:
            
            # cal iou and do fft
            iou_df, fft_df = dp.comb_iou_fft(high_score, iou_type=IOU_TYPE, 
                                             interp_type=INTERP_METHOD)
            if iou_df.shape[0] > 0:
                draw_iou(SRC_DIR, iou_df, fname)
                draw_fft(SRC_DIR, fft_df, fname)
                
                # draw x, y trajectoreis of box centers
                draw_xy(SRC_DIR, high_score, fname)

                xy_fft_df = dp.do_xy_fft(high_score, interp_type=INTERP_METHOD)
                draw_xy_fft(SRC_DIR, xy_fft_df, fname)
            
            print()

