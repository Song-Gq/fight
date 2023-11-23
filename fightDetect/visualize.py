import numpy as np
import plotly.express as px
import plotly
import pandas as pd
import data_process as dp


def draw_fig(key_df, json_name):
    # 9. left hand: 27-x, 28-y, 29-confidence
    fig = px.scatter_3d(key_df, x='27', y='28', z='image_id', color='29', symbol='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-left-hand.html')

    # 15. left ankle: 45-x, 46-y, 47-confidence
    fig = px.scatter_3d(key_df, x='45', y='46', z='image_id', color='47', symbol='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-left-ankle.html')

    # calculate the speed
    speed = key_df.sort_values(by=['idx', 'image_id'])
    diff = speed.groupby('idx').diff().fillna(0.)
    confi_cols = list(range(2, 78, 3))
    for c in confi_cols:
        diff[str(c)] = np.sqrt(diff[str(c-1)]**2 + diff[str(c-2)]**2) / diff['image_id']
    diff = diff[[str(c) for c in range(2, 78, 3)]]
    diff = diff.add_suffix('_speed')
    speed = pd.concat([speed, diff], axis=1)

    # speed.fillna(0.)
    fig = px.scatter_3d(speed[speed['29'] > 0.5], x='27', y='28', z='image_id', color='29_speed', symbol='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-left-hand-speed.html')

    fig = px.scatter_3d(speed[speed['47'] > 0.5], x='45', y='46', z='image_id', color='47_speed', symbol='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-left-ankle-speed.html')


def draw_iou(merged_df, json_name):
    fig = px.line(merged_df, x='image_id', y='iou', color='comb', markers=True)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + output_suffix + '.html')
    # fig.show()


def draw_fft(merged_df, json_name):
    fig = px.line(merged_df, x='image_id', y='y', color='comb', markers=True)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + output_suffix + '-fft.html')


def draw_xy(box_df, json_name):
    # box ouput format: x, y, w, h
    fig = px.scatter_3d(box_df, x='0', y='1', z='image_id', color='score', symbol='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-xy.html')


def draw_xy_fft(fft_df, json_name):
    x_fft = fft_df[['image_id', 'x_fft', 'idx']].copy(deep=True)
    x_fft.columns = ['image_id', 'fft', 'idx']
    x_fft['axis'] = 'x'
    y_fft = fft_df[['image_id', 'y_fft', 'idx']].copy(deep=True)
    y_fft.columns = ['image_id', 'fft', 'idx']
    y_fft['axis'] = 'y'
    fft_xy = pd.concat([x_fft, y_fft], axis=0)

    fig = px.line(fft_xy, x='image_id', y='fft', color='idx', facet_col='axis')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-xy-fft.html')


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

        # cal iou and do fft
        # iou_df, fft_df = dp.comb_iou_fft(high_score, iou_type=iou_type, 
        #                                  interp_type=interp_type)
        # draw_iou(iou_df, fname)
        # draw_fft(fft_df, fname)
        
        # draw x, y trajectoreis of box centers
        # draw_xy(high_score, fname)

        xy_fft_df = dp.do_xy_fft(high_score, interp_type=interp_type)
        draw_xy_fft(xy_fft_df, fname)
        
        print()

