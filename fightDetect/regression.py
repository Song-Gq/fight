import numpy as np
import plotly.express as px
import plotly
import pandas as pd
import data_process as dp


def draw_3d_reg(reg_df, json_name):
    # 3d scatter
    fig = px.scatter_3d(reg_df, x='0reg', y='1reg', z='image_id', symbol='idx', color='score')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-xy-reg.html')

    fig = px.scatter_3d(reg_df, x='0', y='1', z='image_id', symbol='idx', color='score')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-xy.html')


def draw_2d_reg(reg_df, json_name):
    # 2d line
    fig = px.line(reg_df, x='image_id', y='0reg', facet_col='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-x-reg.html')

    fig = px.line(reg_df, x='image_id', y='0', facet_col='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-x.html')

    fig = px.line(reg_df, x='image_id', y='1reg', facet_col='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-y-reg.html')

    fig = px.line(reg_df, x='image_id', y='1', facet_col='idx')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-y.html')


src_dir = "test/"
# iou_type = 'giou'
score_thre = 2.6
# interp_type = 'previous'
linear = True
output_suffix = '-reg=' + str(linear) + \
    '-score-thre=' + str(score_thre)


if __name__ == '__main__':
    statis_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        # drop data with lower scores
        # using higher threshold here
        high_score = boxes[boxes['score'] > score_thre]

        x_reg = dp.poly_regress(high_score, '0', linear=linear)
        y_reg = dp.poly_regress(high_score, '1', linear=linear)

        xy_reg = pd.merge(x_reg, y_reg, on=['image_id', 'idx'], how='inner')
        # drop data where the length < 10 (either x or y) from one person
        valid_p = xy_reg['idx'].unique()
        valid_df = high_score[high_score['idx'].isin(valid_p)]
        reg_res = pd.merge(xy_reg, valid_df, on=['image_id', 'idx'], how='outer')

        # draw_3d_reg(reg_res, fname)
        # draw_2d_reg(reg_res, fname)

        # cal the diff between reg and raw data
        # only the values of points in 'high_score' is calculated 
        # which means missing points is not considered
        diff = pd.merge(xy_reg, valid_df, on=['image_id', 'idx'], how='inner')
        diff['x_diff'] = diff['0'] - diff['0reg']
        # diff['x_diff'] = np.abs(diff['0'] - diff['0reg'])
        diff['y_diff'] = diff['1'] - diff['1reg']

        # mean diff
        statis = diff.groupby(['idx'])['x_diff'].agg(np.mean).reset_index()
        statis['file'] = fname
        statis_res = pd.concat([statis_res, statis], axis=0)

        # draw fig
        # fig = px.scatter_3d(diff, x='x_diff', y='y_diff', z='image_id', symbol='idx', color='score')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-xy-reg-diff.html')
        
        

    print()

        # # 3d scatter
        # fig = px.scatter_3d(xy_reg, x='0reg', y='1reg', z='image_id', symbol='idx', color='score')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-xy-reg.html')

        # fig = px.scatter_3d(xy_reg, x='0', y='1', z='image_id', symbol='idx', color='score')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-xy.html')

        # # 2d line
        # fig = px.line(xy_reg, x='image_id', y='0reg', facet_col='idx')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-x-reg.html')

        # fig = px.line(xy_reg, x='image_id', y='0', facet_col='idx')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-x.html')

        # fig = px.line(xy_reg, x='image_id', y='1reg', facet_col='idx')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-y-reg.html')

        # fig = px.line(xy_reg, x='image_id', y='1', facet_col='idx')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-y.html')


