import numpy as np
import plotly.express as px
import plotly
import pandas as pd
import data_process as dp
import xlwt


def draw_3d_reg(reg_df, json_name, segmented=False):
    color_col = 'idx' if segmented else 'score'
    symb_col = 'seg_comb' if segmented else 'idx'
    # 3d scatter
    fig = px.scatter_3d(reg_df, x='0reg', y='1reg', z='image_id', symbol=symb_col, color=color_col)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-xy-reg.html')

    fig = px.scatter_3d(reg_df, x='0', y='1', z='image_id', symbol=symb_col, color=color_col)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-xy.html')


def draw_2d_reg(reg_df, json_name, segmented=False):
    subplot_row_x = 'segx' if segmented else None
    subplot_row_y = 'segy' if segmented else None
    reg_df.sort_values(by=['idx', 'image_id'], axis=0, inplace=True)

    # 2d line
    fig = px.line(reg_df, x='image_id', y='0reg', facet_col='idx', facet_row=subplot_row_x)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-x-reg.html')

    fig = px.line(reg_df, x='image_id', y='0', facet_col='idx', facet_row=subplot_row_x)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-x.html')

    fig = px.line(reg_df, x='image_id', y='1reg', facet_col='idx', facet_row=subplot_row_y)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-y-reg.html')

    fig = px.line(reg_df, x='image_id', y='1', facet_col='idx', facet_row=subplot_row_y)
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + json_name + '-y.html')


def abs_mean(x):
    return abs(x).mean()


def do_poly_reg(box_df, lin_reg=False, min_len=10):
    x_reg_df = dp.poly_regress(box_df, '0', linear=lin_reg, min_len=min_len)
    y_reg_df = dp.poly_regress(box_df, '1', linear=lin_reg, min_len=min_len)
    return pd.merge(x_reg_df, y_reg_df, on=['image_id', 'idx'], how='inner')


def do_tree_reg(box_df, min_len=10):
    x_reg_df = dp.tree_reg(box_df, '0', min_len=min_len)
    y_reg_df = dp.tree_reg(box_df, '1', min_len=min_len)
    return pd.merge(x_reg_df, y_reg_df, on=['image_id', 'idx'], how='inner')


def do_tree_seg(box_df, max_seg, reg_deg, min_len=10, interp_type='linear'):
    x_seg_df = dp.tree_seg(box_df, '0', max_seg=max_seg, reg_deg=reg_deg,
                            min_len=min_len, interp_type=interp_type)
    y_seg_df = dp.tree_seg(box_df, '1', max_seg=max_seg, reg_deg=reg_deg,
                            min_len=min_len, interp_type=interp_type)
    return pd.merge(x_seg_df, y_seg_df, on=['image_id', 'idx'], how='inner', suffixes=['x', 'y'])


# merge regression results with orininal box data
# drop data where the length < 'arg' (either x or y) from one person
# but the len arg is defined in the regression function 
def valid_merge(xy_df, raw_df, inner=False):
    how_opt = 'inner' if inner else 'outer'
    valid_p = xy_df['idx'].unique()
    valid_df = raw_df[raw_df['idx'].isin(valid_p)]
    return pd.merge(xy_df, valid_df, on=['image_id', 'idx'], how=how_opt)


# cal the diff between reg and raw data
# only the values of points in 'high_score' is calculated 
# which means missing points is not considered
def cal_reg_diff(xy_df, raw_df, file_name):
    diff_df = valid_merge(xy_df, raw_df, inner=True)
    diff_df['x_diff'] = diff_df['0'] - diff_df['0reg']
    diff_df['y_diff'] = diff_df['1'] - diff_df['1reg']

    # mean diff
    x_diff_df = diff_df.groupby(['idx'])['x_diff'].agg([abs_mean, 'var']).reset_index()
    y_diff_df = diff_df.groupby(['idx'])['y_diff'].agg([abs_mean, 'var']).reset_index()
    xy_diff_df = pd.merge(x_diff_df, y_diff_df, on='idx', how='outer')
    xy_diff_df['file'] = file_name

    return xy_diff_df


src_dir = "test/"
# iou_type = 'giou'
score_thre = 2.6
# interp_type = 'previous'
linear = False
output_suffix = '-reg=' + str(linear) + \
    '-score-thre=' + str(score_thre)
max_segment_num = 5
segment_reg_deg = 2
valid_min_frame = 10
interp_method = 'previous'


if __name__ == '__main__':
    statis_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        # drop data with lower scores
        # using higher threshold here
        high_score = boxes[boxes['score'] > score_thre]

        # xy_reg = do_poly_reg(high_score, lin_reg=linear, min_len=valid_min_frame)
        # xy_reg = do_tree_reg(high_score, min_len=valid_min_frame)
        # reg_res = valid_merge(xy_reg, high_score)
        # draw_3d_reg(reg_res, fname)
        # draw_2d_reg(reg_res, fname)

        xy_seg = do_tree_seg(high_score, max_segment_num, segment_reg_deg, 
                             min_len=valid_min_frame, interp_type=interp_method)
        reg_res = valid_merge(xy_seg, high_score, inner=True)

        reg_res['segx'] = reg_res['segx'].astype(str)
        reg_res['segy'] = reg_res['segy'].astype(str)
        reg_res['seg_comb'] = reg_res['segx'] + '+' + reg_res['segy']

        draw_3d_reg(reg_res, fname, segmented=True)
        draw_2d_reg(reg_res, fname, segmented=True)

        xy_diff = cal_reg_diff(xy_seg, high_score, fname)
        statis_res = pd.concat([statis_res, xy_diff], axis=0)

        print()

        # polynomial regression
        # x_reg = dp.poly_regress(high_score, '0', linear=linear)
        # y_reg = dp.poly_regress(high_score, '1', linear=linear)

        # decesion tree regression
        # x_reg = dp.tree_reg(high_score, '0')
        # y_reg = dp.tree_reg(high_score, '1')

        # xy_reg = pd.merge(x_reg, y_reg, on=['image_id', 'idx'], how='inner')

        # decesion tree segmentation and polynomial regression
        # x_seg = dp.tree_seg(high_score, '0', max_seg=5, reg_deg=2)
        # y_seg = dp.tree_seg(high_score, '1', max_seg=5, reg_deg=2)
        # xy_seg = pd.merge(x_seg, y_seg, on=['image_id', 'idx', 'seg'], how='inner')

        # drop data where the length < 10 (either x or y) from one person
        # valid_p = xy_reg['idx'].unique()
        # valid_df = high_score[high_score['idx'].isin(valid_p)]
        # reg_res = pd.merge(xy_reg, valid_df, on=['image_id', 'idx'], how='outer')

        # cal the diff between reg and raw data
        # only the values of points in 'high_score' is calculated 
        # which means missing points is not considered
        # diff = pd.merge(xy_reg, valid_df, on=['image_id', 'idx'], how='inner')
        # diff['x_diff'] = diff['0'] - diff['0reg']
        # # diff['x_diff'] = np.abs(diff['0'] - diff['0reg'])
        # diff['y_diff'] = diff['1'] - diff['1reg']

        # # mean diff
        # x_diff = diff.groupby(['idx'])['x_diff'].agg([abs_mean, 'var']).reset_index()
        # y_diff = diff.groupby(['idx'])['y_diff'].agg([abs_mean, 'var']).reset_index()
        # xy_diff = pd.merge(x_diff, y_diff, on='idx', how='outer')
        # xy_diff['file'] = fname
        # statis_res = pd.concat([statis_res, xy_diff], axis=0)

        # draw fig
        # fig = px.scatter_3d(diff, x='x_diff', y='y_diff', z='image_id', symbol='idx', color='score')
        # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-xy-reg-diff.html')
        
    statis_res.to_excel('statis_res' + output_suffix + '.xlsx')
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


