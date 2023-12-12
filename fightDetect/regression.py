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


def draw_statis(res_df):
    df_draw = res_df.copy(deep=True)
    df_draw['cat'] = df_draw['file'].str.replace('[a-zA-Z0-9_.]+', '', regex=True)

    fig = px.scatter(df_draw, x='var_x', y='var_y', color='cat', marginal_x='box', marginal_y='box', hover_name='file')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + 'statis-scatter.html')
    
    fig = px.histogram(df_draw, x='var_x', color='cat', marginal='rug', hover_name='file')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + 'statis-x-hist.html')

    fig = px.histogram(df_draw, x='var_y', color='cat', marginal='rug', hover_name='file')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + 'statis-y-hist.html')


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
def cal_reg_diff(xy_df, raw_df, file_name, data_type='xy'):
    if data_type == 'xy':
        diff_df = valid_merge(xy_df, raw_df, inner=True)
        diff_df['x_diff'] = diff_df['0'] - diff_df['0reg']
        diff_df['y_diff'] = diff_df['1'] - diff_df['1reg']

        # mean diff
        x_diff_df = diff_df.groupby(['idx'])['x_diff'].agg([abs_mean, 'var']).reset_index()
        y_diff_df = diff_df.groupby(['idx'])['y_diff'].agg([abs_mean, 'var']).reset_index()
        xy_diff_df = pd.merge(x_diff_df, y_diff_df, on='idx', how='outer')
        xy_diff_df['file'] = file_name
        return xy_diff_df
    # data_type == 'iou'
    else:
        diff_df = valid_merge(xy_df, raw_df, inner=True, id_col='comb')
        diff_df['iou_diff'] = diff_df['iou'] - diff_df['ioureg']

        # mean diff
        iou_diff_df = diff_df.groupby(['comb'])['iou_diff'].agg([abs_mean, 'var']).reset_index()
        iou_diff_df['file'] = file_name
        return iou_diff_df


# do segmentation and regression based on x, y location data of a person
def start_location_reg(norm=False):
    statis_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        # drop data with lower scores
        # using higher threshold here
        high_score = boxes[boxes['score'] > score_thre]

        # normalize the x, y location data
        if norm:
            high_score = dp.xy_normalize(high_score, keys)
            for box_col in range(0, 4):
                high_score[str(box_col)] = high_score[str(box_col) + 'norm']

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
    statis_res.to_excel('statis_res' + output_suffix + '.xlsx')
    draw_statis(statis_res)


# do segmentation and regression based on iou data of a combination of persons
def start_iou_reg():
    iou_statis_res = pd.DataFrame()
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        # drop data with lower scores
        # using higher threshold here
        high_score = boxes[boxes['score'] > score_thre]
        # do segmentation and regression for iou data
        # fft_df is useless here
        iou_df, fft_df = dp.comb_iou_fft(
            high_score, iou_type=iou_type, interp_type=interp_method, fill0=False)
        if iou_df.shape[0] > valid_min_frame:
            iou_seg = dp.tree_seg(iou_df, 'iou', max_seg=max_segment_num, reg_deg=segment_reg_deg,
                                min_len=valid_min_frame, interp_type=interp_method)
            iou_reg_res = valid_merge(iou_seg, iou_df, inner=True, id_col='comb')
            
            # fig = px.line(iou_reg_res, x='image_id', y='ioureg', facet_col='comb', facet_row='seg')
            # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-iou-reg.html')

            # fig = px.line(iou_reg_res, x='image_id', y='iou', facet_col='comb', facet_row='seg')
            # plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-iou.html')

            iou_diff = cal_reg_diff(iou_seg, iou_df, fname, data_type='iou')
            iou_statis_res = pd.concat([iou_statis_res, iou_diff], axis=0)
    iou_statis_res.to_excel('iou_statis_res' + output_suffix + '.xlsx')
    iou_statis_res['cat'] = iou_statis_res['file'].str.replace('[a-zA-Z0-9_.]+', '', regex=True)
    fig = px.histogram(iou_statis_res, x='var', color='cat', marginal='rug', hover_name='file')
    plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + 'statis-iou-hist.html')


src_dir = "test-single/"
# iou_type = 'giou'
score_thre = 2.6
# interp_type = 'previous'
linear = False
output_suffix = '-reg=' + str(linear) + \
    '-score-thre=' + str(score_thre)
max_segment_num = 5
segment_reg_deg = 2
valid_min_frame = 10
# for x, y location segmentation and regression
# and for calculating iou
interp_method = 'previous'
iou_type = 'giou'
normalization = True


if __name__ == '__main__':
    start_location_reg(norm=normalization)
    # start_iou_reg()

    # statis_res = pd.DataFrame()
    # iou_statis_res = pd.DataFrame()
    # for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
    #     # drop data with lower scores
    #     # using higher threshold here
    #     high_score = boxes[boxes['score'] > score_thre]

    #     xy_seg = do_tree_seg(high_score, max_segment_num, segment_reg_deg, 
    #                          min_len=valid_min_frame, interp_type=interp_method)
    #     reg_res = valid_merge(xy_seg, high_score, inner=True)

    #     reg_res['segx'] = reg_res['segx'].astype(str)
    #     reg_res['segy'] = reg_res['segy'].astype(str)
    #     reg_res['seg_comb'] = reg_res['segx'] + '+' + reg_res['segy']

    #     # draw_3d_reg(reg_res, fname, segmented=True)
    #     # draw_2d_reg(reg_res, fname, segmented=True)

    #     xy_diff = cal_reg_diff(xy_seg, high_score, fname)
    #     statis_res = pd.concat([statis_res, xy_diff], axis=0)

    #     # do segmentation and regression for iou data
    #     # fft_df is useless here
    #     iou_df, fft_df = dp.comb_iou_fft(
    #         high_score, iou_type=iou_type, interp_type=interp_method, fill0=False)
    #     iou_seg = dp.tree_seg(iou_df, 'iou', max_seg=max_segment_num, reg_deg=segment_reg_deg,
    #                         min_len=valid_min_frame, interp_type=interp_method)
    #     iou_reg_res = valid_merge(iou_seg, high_score, inner=True, id_col='comb')
        
    #     fig = px.line(iou_reg_res, x='image_id', y='ioureg', facet_col='comb', facet_row='seg')
    #     plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-iou-reg.html')

    #     fig = px.line(iou_reg_res, x='image_id', y='iou', facet_col='comb', facet_row='seg')
    #     plotly.offline.plot(fig, filename='fightDetect/fig/' + src_dir + fname + '-iou.html')

    #     iou_diff = cal_reg_diff(iou_seg, high_score, fname, data_type='iou')
    #     iou_statis_res = pd.concat([iou_statis_res, iou_diff], axis=0)
    #     print()
        
    # statis_res.to_excel('statis_res' + output_suffix + '.xlsx')
    # iou_statis_res.to_excel('iou_statis_res' + output_suffix + '.xlsx')
    # draw_statis(statis_res)
    # draw_statis(iou_statis_res)
    # print()
