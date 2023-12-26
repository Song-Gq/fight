import numpy as np
import plotly.express as px
import plotly
import pandas as pd
import data_process as dp
# import xlwt
import os
from datetime import datetime
import re


def draw_3d_reg(reg_df, json_name, segmented=False, xy_cols=None):
    # avoid using [] as the default value of xy_cols
    if xy_cols is None:
        xy_cols = ['0', '1']
    color_col = 'idx' if segmented else 'score'
    symb_col = 'seg_comb' if segmented else 'idx'
    # output_name = re.sub('[a-zA-Z_.]+', '', json_name)
    output_name = re.sub(r'^(AlphaPose_)|(\.json)$', '', json_name)
    output_name = output_name + '-' + xy_cols[0][0: xy_cols[0].rfind('x') - 1] +'-xy'
    # 3d scatter
    fig = px.scatter_3d(reg_df, x=xy_cols[0]+'reg', y=xy_cols[1]+'reg', z='image_id', symbol=symb_col, color=color_col)
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-reg.html', auto_open=False)

    fig = px.scatter_3d(reg_df, x=xy_cols[0], y=xy_cols[1], z='image_id', symbol=symb_col, color=color_col)
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '.html', auto_open=False)


def draw_2d_reg(reg_df, json_name, segmented=False, xy_cols=None):
    # avoid using [] as the default value of xy_cols
    if xy_cols is None:
        xy_cols = ['0', '1']
    subplot_row_x = 'segx' if segmented else None
    subplot_row_y = 'segy' if segmented else None
    # output_name = re.sub('[a-zA-Z_.]+', '', json_name)
    output_name = re.sub(r'^(AlphaPose_)|(\.json)$', '', json_name)
    reg_df.sort_values(by=['idx', 'image_id'], axis=0, inplace=True)

    # 2d line
    fig = px.line(reg_df, x='image_id', y=xy_cols[0]+'reg', facet_col='idx', facet_row=subplot_row_x)
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-' + xy_cols[0] + '-reg.html', auto_open=False)

    fig = px.line(reg_df, x='image_id', y=xy_cols[0], facet_col='idx', facet_row=subplot_row_x)
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-' + xy_cols[0] + '.html', auto_open=False)

    fig = px.line(reg_df, x='image_id', y=xy_cols[1]+'reg', facet_col='idx', facet_row=subplot_row_y)
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-' + xy_cols[1] + '-reg.html', auto_open=False)

    fig = px.line(reg_df, x='image_id', y=xy_cols[1], facet_col='idx', facet_row=subplot_row_y)
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-' + xy_cols[1] + '.html', auto_open=False)


def draw_statis(res_df, output_name):
    df_draw = res_df.copy(deep=True)
    # df_draw['cat'] = df_draw['file'].str.replace('[a-zA-Z0-9_.]+', '', regex=True)
    df_draw['cat'] = df_draw['file'].str.replace(r'[0-9]+', '', regex=True)

    fig = px.scatter(df_draw, x='var_x', y='var_y', color='cat', marginal_x='box', marginal_y='box', hover_name='file')
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-statis-scatter.html', auto_open=False)
    
    fig = px.histogram(df_draw, x='var_x', color='cat', marginal='rug', hover_name='file')
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-statis-x-hist.html', auto_open=False)

    fig = px.histogram(df_draw, x='var_y', color='cat', marginal='rug', hover_name='file')
    plotly.offline.plot(fig, filename=OUTPUT_DIR + output_name + '-statis-y-hist.html', auto_open=False)


def abs_mean(x):
    return abs(x).mean()


def do_poly_reg(box_df, reg_deg=2, min_len=10):
    x_reg_df = dp.poly_regress(box_df, '0', reg_deg=reg_deg, min_len=min_len)
    y_reg_df = dp.poly_regress(box_df, '1', reg_deg=reg_deg, min_len=min_len)
    return pd.merge(x_reg_df, y_reg_df, on=['image_id', 'idx'], how='inner')


def do_tree_reg(box_df, min_len=10):
    x_reg_df = dp.tree_reg(box_df, '0', min_len=min_len)
    y_reg_df = dp.tree_reg(box_df, '1', min_len=min_len)
    return pd.merge(x_reg_df, y_reg_df, on=['image_id', 'idx'], how='inner')


def do_tree_seg(box_df, max_seg, reg_deg, min_len=10, interp_type='linear', xy_cols=None):
    # avoid using [] as the default value of xy_cols
    if xy_cols is None:
        xy_cols = ['0', '1']
    x_seg_df = dp.tree_seg(box_df, xy_cols[0], max_seg=max_seg, reg_deg=reg_deg,
                            min_len=min_len, interp_type=interp_type, xy_cols=xy_cols)
    y_seg_df = dp.tree_seg(box_df, xy_cols[1], max_seg=max_seg, reg_deg=reg_deg,
                            min_len=min_len, interp_type=interp_type, xy_cols=xy_cols)
    if x_seg_df.shape[0] > min_len and y_seg_df.shape[0] > min_len:
        return pd.merge(x_seg_df, y_seg_df, on=['image_id', 'idx'], how='inner', suffixes=['x', 'y'])
    else:
        return None


# merge regression results with orininal box data
# drop data where the length < 'arg' (either x or y) from one person
# but the len arg is defined in the regression function 
def valid_merge(xy_df, raw_df, inner=False, id_col='idx'):
    how_opt = 'inner' if inner else 'outer'
    valid_p = xy_df[id_col].unique()
    # # split the comb data into ids
    # if id_col == 'comb':
    #     new_valid_p = []
    #     for p in valid_p:
    #         ps = p.split('+')
    #         new_valid_p = new_valid_p + ps
    #     valid_p = new_valid_p
    valid_df = raw_df[raw_df[id_col].isin(valid_p)]
    return pd.merge(xy_df, valid_df, on=['image_id', id_col], how=how_opt)


# cal the diff between reg and raw data
# only the values of points in 'high_score' is calculated 
# which means missing points is not considered
def cal_reg_diff(xy_df, raw_df, file_name, data_type='xy', xy_cols=None):
    # avoid using [] as the default value of xy_cols
    if xy_cols is None:
        xy_cols = ['0', '1']
    if data_type == 'xy':
        diff_df = valid_merge(xy_df, raw_df, inner=True)
        diff_df['x_diff'] = diff_df[xy_cols[0]] - diff_df[xy_cols[0]+'reg']
        diff_df['y_diff'] = diff_df[xy_cols[1]] - diff_df[xy_cols[1]+'reg']

        # mean diff
        x_diff_df = diff_df.groupby(['idx'])['x_diff'].agg([abs_mean, 'var']).reset_index()
        y_diff_df = diff_df.groupby(['idx'])['y_diff'].agg([abs_mean, 'var']).reset_index()
        xy_diff_df = pd.merge(x_diff_df, y_diff_df, on='idx', how='outer')
        xy_diff_df['file'] = re.sub(r'^(AlphaPose_)|(\.json)$', '', file_name)
        return xy_diff_df

    # data_type == 'iou' or 'scalar'
    id_col = 'comb' if data_type == 'iou' else 'idx'

    diff_df = valid_merge(xy_df, raw_df, inner=True, id_col=id_col)
    diff_df[data_type + '_diff'] = diff_df[data_type] - diff_df[data_type + 'reg']

    # mean diff
    res_diff_df = diff_df.groupby([id_col])[data_type + '_diff'].agg([abs_mean, 'var']).reset_index()
    res_diff_df['file'] = re.sub(r'^(AlphaPose_)|(\.json)$', '', file_name)
    return res_diff_df


# turn numbers in 'segx' and 'segy' into string
# and create a new column to store the combination of 'segx' and 'segy'
def rename_seg(seg_df):
    seg_df['segx'] = seg_df['segx'].astype(str)
    seg_df['segy'] = seg_df['segy'].astype(str)
    seg_df['seg_comb'] = seg_df['segx'] + '+' + seg_df['segy']
    return seg_df


# do segmentation and regression based on x, y location data of a person
def start_location_reg(src_dir, norm=False):
    print('starting regression on location data')
    statis_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        print('processing file ' + fname)
        if keys is None or boxes is None:
            continue
        # drop data with lower scores
        # using higher threshold here
        # high_score = boxes[boxes['score'] > score_thre]
        high_score = dp.get_high_score(boxes, upper_limit=340, 
                                       min_p_len=VALID_MIN_FRAME, 
                                       lower_confi=LOWER_CONFIDENCE)
        if high_score.shape[0] > 0:
            # normalize the x, y location data
            if norm:
                high_score = dp.xy_normalize(high_score, keys)
                for box_col in range(0, 4):
                    high_score[str(box_col)] = high_score[str(box_col) + 'norm']

            # fig = px.line(high_score, x='image_id', y='body_metric_roll', facet_col='idx')
            # plotly.offline.plot(fig, filename=output_dir + re.sub('[a-zA-Z_.]+', '', fname) + '-metric_roll.html')

            xy_seg = do_tree_seg(high_score, MAX_SEGMENT_NUM, SEGMENT_REG_DEG, 
                                min_len=VALID_MIN_FRAME, interp_type=INTERP_METHOD)
            if xy_seg is not None:
                reg_res = valid_merge(xy_seg, high_score, inner=True)
                reg_res = rename_seg(reg_res)

                draw_3d_reg(reg_res, fname, segmented=True)
                draw_2d_reg(reg_res, fname, segmented=True)

                xy_diff = cal_reg_diff(xy_seg, high_score, fname)
                statis_res = pd.concat([statis_res, xy_diff], axis=0)
    # statis_res.to_excel(output_dir + 'statis_res.xlsx')
    statis_res.to_csv(OUTPUT_DIR + 'statis_res.csv')
    draw_statis(statis_res, 'location')


# do segmentation and regression based on iou data of a combination of persons
def start_iou_reg(src_dir):
    print('starting regression on iou data')
    iou_statis_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        print('processing file ' + fname)
        if keys is None or boxes is None:
            continue
        # drop data with lower scores
        # using higher threshold here
        # high_score = boxes[boxes['score'] > score_thre]
        high_score = dp.get_high_score(boxes, upper_limit=340, 
                                       min_p_len=VALID_MIN_FRAME, 
                                       lower_confi=LOWER_CONFIDENCE)
        if high_score.shape[0] > 0:
            # do segmentation and regression for iou data
            # fft_df is useless here
            iou_df, _ = dp.comb_iou_fft(
                high_score, iou_type=IOU_TYPE, interp_type=INTERP_METHOD, fill0=False)
            if iou_df.shape[0] > VALID_MIN_FRAME:
                iou_seg = dp.tree_seg(iou_df, 'iou', max_seg=MAX_SEGMENT_NUM, reg_deg=SEGMENT_REG_DEG,
                                    min_len=VALID_MIN_FRAME, interp_type=INTERP_METHOD)
                if iou_seg.shape[0] > VALID_MIN_FRAME:
                    # iou_reg_res = valid_merge(iou_seg, iou_df, inner=True, id_col='comb')
                    
                    # fig = px.line(iou_reg_res, x='image_id', y='ioureg', facet_col='comb', facet_row='seg')
                    # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-iou-reg.html')

                    # fig = px.line(iou_reg_res, x='image_id', y='iou', facet_col='comb', facet_row='seg')
                    # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-iou.html')

                    iou_diff = cal_reg_diff(iou_seg, iou_df, fname, data_type='iou')
                    iou_statis_res = pd.concat([iou_statis_res, iou_diff], axis=0)
    # iou_statis_res.to_excel(output_dir + 'iou_statis_res.xlsx')
    iou_statis_res.to_csv(OUTPUT_DIR + 'iou_statis_res.csv')
    # iou_statis_res['cat'] = iou_statis_res['file'].str.replace('[a-zA-Z0-9_.]+', '', regex=True)
    iou_statis_res['cat'] = iou_statis_res['file'].str.replace(r'[0-9]+', '', regex=True)
    fig = px.histogram(iou_statis_res, x='var', color='cat', marginal='rug', hover_name='file')
    plotly.offline.plot(fig, filename=OUTPUT_DIR + 'statis-iou-hist.html', auto_open=False)


# for keypoint features where the data are like x, y location vectors
# for example, key_nums = [5, 9] means to calculate the relative position of 
# left wrist(9) to left shoulder(5) (x9-x5, y9-y5)
def xy_feature_reg(raw_df, key_nums, feature_name, res_df, json_name):
    temp_df = raw_df.copy()
    temp_df[feature_name + '_x'] = temp_df[str(key_nums[1]*3)] - temp_df[str(key_nums[0]*3)]
    temp_df[feature_name + '_y'] = temp_df[str(key_nums[1]*3+1)] - temp_df[str(key_nums[0]*3+1)]

    feature_xy = do_tree_seg(temp_df, MAX_SEGMENT_NUM, SEGMENT_REG_DEG, 
                        min_len=VALID_MIN_FRAME, interp_type=INTERP_METHOD, 
                        xy_cols=[feature_name + '_x', feature_name + '_y'])
    if feature_xy is not None:
        feature_xy = valid_merge(feature_xy, temp_df, inner=True, id_col='idx')
        feature_xy = rename_seg(feature_xy)
        
        draw_3d_reg(feature_xy, json_name, segmented=True, 
                    xy_cols=[feature_name + '_x', feature_name + '_y'])
        draw_2d_reg(feature_xy, json_name, segmented=True, 
                    xy_cols=[feature_name + '_x', feature_name + '_y'])
        
        feature_diff = cal_reg_diff(feature_xy, temp_df[['image_id', 'idx']], json_name, 
                    xy_cols=[feature_name + '_x', feature_name + '_y'])
        
        # calculate the speed of vector changes
        raw_sorted = temp_df.sort_values(by=['idx', 'image_id'])
        vector_speed = raw_sorted.groupby('idx').diff().fillna(0.)
        vector_speed[feature_name + '_speed'] = np.sqrt(vector_speed[feature_name + '_x']**2 + \
            vector_speed[feature_name + '_y']**2) / vector_speed['image_id']
        vector_speed = pd.concat([vector_speed[[feature_name + '_speed']], raw_sorted[['image_id', 'idx']]], axis=1)
        
        # to keep the 'tree_seg' function compatible, copy a column called 'scalar'
        vector_speed['scalar'] = vector_speed[feature_name + '_speed']
        vector_speed = vector_speed.dropna(subset=['scalar'], axis=0)
        speed_seg = dp.tree_seg(vector_speed, 'scalar', max_seg=MAX_SEGMENT_NUM, reg_deg=SEGMENT_REG_DEG,
                                    min_len=VALID_MIN_FRAME, interp_type=INTERP_METHOD)
        # empty dataframe
        if speed_seg.shape[0] == 0:
            return res_df
        speed_diff = cal_reg_diff(speed_seg, vector_speed[['image_id', 'idx', 'scalar']], json_name, data_type='scalar')

        return [pd.concat([res_df[0], feature_diff], axis=0),
                pd.concat([res_df[1], speed_diff], axis=0)]
    return res_df


# do segmentation and regression based on keypoints data of a person
def start_key_reg(src_dir, norm=False):
    print('starting regression on keypoints data')
    # left/right arm: x, y location vectors relative to shoulders
    # key 5 left shoulder 9 left wrist
    # key 6 left shoulder 10 left wrist
    # key 11 left hip, 15 left ankle
    # key 12 right hip, 16 right ankle
    vector_features = {'left_arm': [5, 9], 'right_arm': [6, 10], 
                       'left_leg': [11, 15], 'right_leg': [12, 16]}
    # each element contains two dataframes
    # which stores the statistical results of location vectors and speed sperately
    vector_res = {}
    for k in vector_features:
        vector_res[k] = [pd.DataFrame(), pd.DataFrame()]

    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        print('processing file ' + fname)
        if keys is None or boxes is None:
            continue
        # high_score = boxes[boxes['score'] > score_thre]
        high_score = dp.get_high_score(keys, upper_limit=340, 
                                       min_p_len=VALID_MIN_FRAME, 
                                       lower_confi=LOWER_CONFIDENCE)
        if high_score.shape[0] > 0:
            # normalize the x, y location data
            if norm:
                high_score = dp.key_normalize(high_score)
                key_cols = list(range(0, 77, 3)) + list(range(1, 77, 3))
                for key_col in key_cols:
                    high_score[str(key_col)] = high_score[str(key_col) + 'norm']

            for k, v in vector_features.items():
                vector_res[k] = xy_feature_reg(
                    high_score, v, k, vector_res[k], fname)
    
    for k, v in vector_res.items():
        # v[0].to_excel(output_dir + k + '_statis_res.xlsx')
        v[0].to_csv(OUTPUT_DIR + k + '_statis_res.csv')
        draw_statis(v[0], k)

        # v[1].to_excel(output_dir + k + '_statis_speed_res.xlsx')
        v[1].to_csv(OUTPUT_DIR + k + '_statis_speed_res.csv')
        # v[1]['cat'] = v[1]['file'].str.replace('[a-zA-Z0-9_.]+', '', regex=True)
        v[1]['cat'] = v[1]['file'].str.replace(r'[0-9]+', '', regex=True)
        fig = px.histogram(v[1], x='var', color='cat', marginal='rug', hover_name='file')
        plotly.offline.plot(fig, filename=OUTPUT_DIR + k + '_statis_speed_hist.html', auto_open=False)

        # # key 6 left shoulder 10 left wrist

        # # key 5 left shoulder 9 left wrist
        # speed['left_arm_x'] = speed['15'] - speed['27']
        # speed['left_arm_y'] = speed['16'] - speed['28']
        # # key 6 left shoulder 10 left wrist
        # speed['right_arm_x'] = speed['18'] - speed['19']
        # speed['right_arm_y'] = speed['30'] - speed['31']

        # # left wrist relative position
        # left_arm_xy = do_tree_seg(speed, max_segment_num, segment_reg_deg, 
        #                 min_len=valid_min_frame, interp_type=interp_method, 
        #                 xy_col=['left_arm_x', 'left_arm_y'])
        # left_arm_xy = valid_merge(left_arm_xy, speed, inner=True, id_col='idx')
        # left_arm_xy = rename_seg(left_arm_xy)

        # draw_3d_reg(left_arm_xy, fname, segmented=True)
        # draw_2d_reg(left_arm_xy, fname, segmented=True)
        # left_arm_diff = cal_reg_diff(left_arm_xy, speed, fname)

        # # right wrist relative position
        # right_arm_xy = do_tree_seg(speed, max_segment_num, segment_reg_deg, 
        #                 min_len=valid_min_frame, interp_type=interp_method, 
        #                 xy_col=['right_arm_x', 'right_arm_y'])
        # right_arm_xy = valid_merge(right_arm_xy, speed, inner=True, id_col='idx')
        # right_arm_xy = rename_seg(right_arm_xy)

        # draw_3d_reg(right_arm_xy, fname, segmented=True)
        # draw_2d_reg(right_arm_xy, fname, segmented=True)
        # right_arm_diff = cal_reg_diff(right_arm_xy, speed, fname)

        # statis_res = pd.concat([statis_res, left_arm_diff], axis=0)


# args
SRC_DIR = "fight-sur/"
# score_thre = 2.6
# linear = False
MAX_SEGMENT_NUM = 1
SEGMENT_REG_DEG = 2
VALID_MIN_FRAME = 5
# for x, y location segmentation and regression
# and for calculating iou
INTERP_METHOD = 'previous'
IOU_TYPE = 'giou'
NORMALIZATION = True
# rolling_window_frame = 100
LOWER_CONFIDENCE = True


OUTPUT_SUFFIX = '-seg_num=' + str(MAX_SEGMENT_NUM) + \
    '-reg_deg=' + str(SEGMENT_REG_DEG) + \
    '-normalization=' + str(NORMALIZATION) + \
    '-valid_minf=' + str(VALID_MIN_FRAME)
OUTPUT_DIR = 'fightDetect/fig/' + SRC_DIR +  \
    datetime.now().strftime(r"%Y-%m-%d.%H-%M-%S") + \
    OUTPUT_SUFFIX + '/'


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_location_reg(SRC_DIR, norm=NORMALIZATION)
    start_iou_reg(SRC_DIR)
    start_key_reg(SRC_DIR, norm=NORMALIZATION)
