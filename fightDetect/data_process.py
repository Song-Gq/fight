import numpy as np
import pandas as pd
from scipy.fftpack import fft, fft2
# from scipy.signal import stft
from scipy.interpolate import interp1d
import os
import itertools
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from huang_thresholding import HuangThresholding


def gen_key_df(json_dir):
    df = pd.read_json(json_dir)
    # split the column of <list> 'keypoints' into multiple cols
    # empty dataframe
    if df.shape[0] == 0:
        return None
    keys = pd.DataFrame(df.keypoints.tolist(), index=df.index)
    keys.columns = keys.columns.map(str)
    keys = pd.concat([df, keys], axis=1)
    keys.drop(columns=keys.columns[[2, 4]], axis=1, inplace=True)
    return keys

    
def gen_box_df(json_dir):
    df = pd.read_json(json_dir)
    # split the column of <list> 'box'
        # empty dataframe
    if df.shape[0] == 0:
        return None
    boxes = pd.DataFrame(df.box.tolist(), index=df.index)
    boxes.columns = boxes.columns.map(str)
    boxes = pd.concat([df, boxes], axis=1)
    boxes.drop(columns=boxes.columns[[2, 4]], axis=1, inplace=True)

    # convert the original x, y of the top-left to of the center of the box
    boxes['centerx'] = boxes['0'] + boxes['2']/2
    boxes['centery'] = boxes['1'] + boxes['3']/2
    boxes.drop(columns=['0', '1'], axis=1, inplace=True)
    boxes.rename(columns={'centerx': '0', 'centery': '1'}, inplace=True)
    return boxes


# here the x, y is the center of the box
def box_area_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2]//2.0, box1[1] - box1[3]//2.0
    x1max, y1max = box1[0] + box1[2]//2.0, box1[1] + box1[3]//2.0
    s1 = box1[2] * box1[3]
 
    x2min, y2min = box2[0] - box2[2]//2.0, box2[1] - box2[3]//2.0
    x2max, y2max = box2[0] + box2[2]//2.0, box2[1] + box2[3]//2.0
    s2 = box2[2] * box2[3]
 
    inter_xmin = np.maximum(x1min, x2min)
    inter_ymin = np.maximum(y1min, y2min)
    inter_xmax = np.minimum(x1max, x2max)
    inter_ymax = np.minimum(y1max, y2max)

    inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
    inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # C: the smallest enclosing convex object
    c_xmin = np.minimum(x1min, x2min)
    c_ymin = np.minimum(y1min, y2min)
    c_xmax = np.maximum(x1max, x2max)
    c_ymax = np.maximum(y1max, y2max)
    c_area = (c_xmax - c_xmin) * (c_ymax - c_ymin)
    return intersection, union, c_area


def box_iou(box1, box2):
    inter, uni, _ = box_area_xywh(box1, box2)
    return inter / uni


def box_giou(box1, box2):
    inter, uni, c = box_area_xywh(box1, box2)
    return inter / uni - uni / c


def cal_iou(box_df, id1, id2, kind='giou'):
    # box ouput format: x, y, w, h
    p1 = box_df[box_df['idx'] == id1]
    p2 = box_df[box_df['idx'] == id2]
    p1xp2 = pd.merge(p1, p2, on='image_id', how='inner')
    if not p1xp2.empty:
        # using giou
        if kind == 'giou':
            p1xp2['iou'] = p1xp2.apply(lambda x: box_giou(
                [x['0_x'], x['1_x'], x['2_x'], x['3_x']], 
                [x['0_y'], x['1_y'], x['2_y'], x['3_y']]), 
                axis=1)
        # using iou
        else:
            p1xp2['iou'] = p1xp2.apply(lambda x: box_iou(
                [x['0_x'], x['1_x'], x['2_x'], x['3_x']], 
                [x['0_y'], x['1_y'], x['2_y'], x['3_y']]), 
                axis=1)
    return p1xp2


# calculate the speed of keypoints
def key_speed(key_df):
    speed = key_df.sort_values(by=['idx', 'image_id'])
    diff = speed.groupby('idx').diff().fillna(0.)
    confi_cols = list(range(2, 78, 3))
    for c in confi_cols:
        diff[str(c)] = np.sqrt(diff[str(c-1)]**2 + diff[str(c-2)]**2) / diff['image_id']
    diff = diff[[str(c) for c in range(2, 78, 3)]]
    diff = diff.add_suffix('_speed')
    return pd.concat([speed, diff], axis=1)


def do_fft(df, col_name, total_len, fft_col_name='y'):
    # do fft
    fft_y = fft(list(df[col_name]))
    abs_y = np.abs(fft_y)
    norm_y = abs_y / (total_len + 1)

    df[fft_col_name] = norm_y

    # # stft
    # fs = 25
    # nperseg = 100
    # f2, t2, Zxx = stft(list(df['iou']), fs, nperseg=nperseg, window='hann')
    # plt.pcolormesh(t2, f2, np.abs(Zxx), vmin=0, vmax=0.05, shading='gouraud')
    # plt.show()
    # print()

    return df


# discarded
def do_fft2(df, col1, col2, total_len):
    arr_2d = df[[col1, col2]].values
    arr_2d.reshape(-1, 2)
    # do fft in 2 dimensions
    fft_y = fft2(arr_2d)
    abs_y = np.abs(fft_y)
    norm_y = abs_y / (total_len + 1)

    norm_y = pd.DataFrame(norm_y)
    df['x_fft'] = norm_y[0]
    df['y_fft'] = norm_y[1]
    return df


def do_interp(raw_df, frame_col, val_col, total_len, interp_kind='linear', fill0=True, min_p_len=10):
    frame = raw_df[frame_col]
    value = raw_df[val_col]
    # linear interpolation
    frame_min = frame.min()
    frame_max = frame.max()
    # valid frame > 10
    if frame.shape[0] > min_p_len:
        frame_len = frame_max - frame_min + 1
        x = np.linspace(frame_min, frame_max, num=frame_len)
        # fill with Nan
        if interp_kind == 'na':
            interp_res = pd.DataFrame()
            interp_res['image_id'] = frame
            interp_res[val_col] = value
            df_fill = pd.DataFrame()
            df_fill['image_id'] = np.linspace(0, total_len, total_len + 1)
            df_fill = pd.merge(df_fill, interp_res, on='image_id', how='outer')
            return df_fill
            # exist_ids = frame.to_list()
            # all_ids = list(x)
            # miss_ids = [i for i in all_ids if i not in exist_ids]
            # interp_res = pd.DataFrame()
            # interp_res['image_id'] = miss_ids
            # interp_res = pd.concat([raw_df[[frame_col, val_col]], interp_res])
            # interp_res.sort_values(by='image_id', inplace=True)

        else:
            # interpolate values that are missing
            f1 = interp1d(frame, value, kind=interp_kind)
            y = f1(x)
            interp_res = pd.DataFrame()
            interp_res['image_id'] = x
            interp_res[val_col] = y

        # fill zeroes in the front and end
        if fill0:
            df_fill = pd.DataFrame()
            df_fill['image_id'] = np.linspace(0, total_len, total_len + 1)
            df_fill = pd.merge(df_fill, interp_res, on='image_id', how='outer')
            df_fill.fillna(0, inplace=True)
            return df_fill
        return interp_res
    return None


def iter_files(files_dir):
    json_dir = os.fsencode(files_dir)
    for f in os.listdir(json_dir):
        fname = os.fsdecode(f)
        if fname.endswith(".json"): 
            keys = gen_key_df(files_dir + fname)
            boxes = gen_box_df(files_dir + fname)
            yield keys, boxes, fname


def get_comb(ser):
    p_ids = ser.unique()
    return list(itertools.combinations(p_ids, 2))


def comb_iou_fft(box_df, iou_type, interp_type, fill0=True, min_p_len=10):
    # to store iou results
    res = pd.DataFrame()
    # to store fft results
    fft_df = pd.DataFrame()
    video_len = box_df['image_id'].max()

    combs2 = get_comb(box_df['idx'])
    for comb in combs2:
        iou_df = cal_iou(box_df, comb[0], comb[1], kind=iou_type)
        # the two persons have common frames
        if not iou_df.empty:
            col_name = str(comb[0]) + '+' + str(comb[1])
            # # append iou data
            # iou_df['comb'] = col_name
            # res = pd.concat([res, iou_df])

            # do interpolation
            res_interp = do_interp(iou_df, 'image_id', 'iou', 
                                   video_len, interp_type, 
                                   fill0=fill0, min_p_len=min_p_len)
            if res_interp is not None:
                # append iou data
                res_interp['comb'] = col_name
                res = pd.concat([res, res_interp])

                # do fft and append
                fft_iou = do_fft(res_interp, 'iou', video_len)
                fft_df = pd.concat([fft_df, fft_iou])

            # do fft and append
            # fft_iou = do_fft(iou_df['image_id'], iou_df['iou'], video_len, interp_type)
            # if fft_iou is not None:
            #     fft_iou['comb'] = col_name
            #     fft_df = pd.concat([fft_df, fft_iou])

            # # do fft and append
            # fft_iou = do_fft(iou_df['image_id'], iou_df['iou'], video_len, interp_type)
            # if fft_iou is not None:
            #     fft_iou['comb'] = col_name
            #     fft_df = pd.concat([fft_df, fft_iou])

    # actually, these 2 dfs are the same
    # to keep old uses of the function no need to change
    return res, fft_df


def do_xy_fft(box_df, interp_type, min_p_len=10):
    # box ouput format: x, y, w, h
    # to store fft results
    fft_df = pd.DataFrame()
    video_len = box_df['image_id'].max()

    p_ids = box_df['idx'].unique()
    for p in p_ids:
        p_df = box_df[box_df['idx'] == p]
        # valid frame > 10
        if p_df.shape[0] > min_p_len:
            # do interpolation
            x_interp = do_interp(p_df, 'image_id', '0', video_len, interp_type)
            y_interp = do_interp(p_df, 'image_id', '1', video_len, interp_type)
            # using outer merge might be wrong!!!
            # generating duplicate data??? (multiple lines for one frame)
            xy_interp = pd.merge(x_interp, y_interp, on='image_id', how='outer')
            xy_interp.fillna(0, inplace=True)

            # do fft and append
            fft_x = do_fft(xy_interp, '0', video_len, fft_col_name='x_fft')
            fft_xy = do_fft(fft_x, '1', video_len, fft_col_name='y_fft')

            # fft_xy = do_fft2(xy_interp, '0', '1', video_len)
            fft_xy['idx'] = p
            fft_df = pd.concat([fft_df, fft_xy])

    return fft_df


def poly_regress(box_df, val_col, reg_deg=2, min_len=10):
    res_df = pd.DataFrame()
    video_len = box_df['image_id'].max()
    p_ids = box_df['idx'].unique()
    for p in p_ids:
        p_df = box_df[box_df['idx'] == p]
        # valid frame > min_len
        if p_df.shape[0] > min_len:
            # avoid exponential explosion
            # 2-degree does noe match the real scene that a man's speed cannot grow at a squared way
            pf = PolynomialFeatures(degree=reg_deg, interaction_only=False)
            x_poly = pf.fit_transform(p_df['image_id'].values.reshape(-1, 1))
            
            lr = LinearRegression()
            lr.fit(x_poly, p_df[val_col])

            xx = np.linspace(0, video_len + 1, video_len + 2)
            xx_poly = pf.transform(xx.reshape(-1, 1))
            yy_poly = lr.predict(xx_poly)

            p_res = pd.DataFrame()
            p_res['image_id'] = xx
            p_res[val_col + 'reg'] = yy_poly
            p_res['idx'] = p

            res_df = pd.concat([res_df, p_res])

    return res_df


def tree_reg(box_df, val_col, min_len=10):
    res_df = pd.DataFrame()
    video_len = box_df['image_id'].max()
    p_ids = box_df['idx'].unique()
    for p in p_ids:
        p_df = box_df[box_df['idx'] == p]
        # valid frame > min_len
        if p_df.shape[0] > min_len:
            x = p_df['image_id'].values.reshape(-1, 1)
            y = p_df[val_col].values

            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(x, y)

            xx = np.linspace(0, video_len + 1, video_len + 2)
            y_pred = tree.predict(xx.reshape(-1, 1))
            
            p_res = pd.DataFrame()
            p_res['image_id'] = xx
            p_res[val_col + 'reg'] = y_pred
            p_res['idx'] = p

            res_df = pd.concat([res_df, p_res])

    return res_df


# calculate the gradient and segment data using decision trees
# and then using regression to fit every segment
def tree_seg(box_df, val_col, max_seg=3, reg_deg=2, min_len=10, interp_type='linear', xy_cols=None):
    # avoid using [] as the default value of xy_cols
    if xy_cols is None:
        xy_cols = ['0', '1']
    # the value column is iou data
    if val_col == 'iou':
        # then the identity column is 'comb', combination of pedestrians
        id_col = 'comb'
    # data like speed
    elif val_col == 'scalar':
        id_col = 'idx'
    # the value column is x, y location data
    else:
        # then the identity column is 'idx', represents the pedestrians
        id_col = 'idx'
            
    res_df = pd.DataFrame()
    video_len = int(box_df['image_id'].max())
    p_ids = box_df[id_col].unique()

    for p in p_ids:
        p_df = box_df[box_df[id_col] == p]
        # valid frame > min_len
        if p_df.shape[0] > min_len:
            # iou data have already been interpolated
            if val_col == 'iou':
                xy_interp = p_df
            # scalar data like speed. needs interpolation
            elif val_col == 'scalar':
                xy_interp = do_interp(p_df, 'image_id', val_col, video_len, interp_type, fill0=False, min_p_len=min_len)
            else:
                # do interpolation
                # the x, y here represent the axes in the video frame
                # 0 is not filled in the front and end of the time series
                x_interp = do_interp(p_df, 'image_id', xy_cols[0], video_len, interp_type, fill0=False, min_p_len=min_len)
                y_interp = do_interp(p_df, 'image_id', xy_cols[1], video_len, interp_type, fill0=False, min_p_len=min_len)
                xy_interp = pd.merge(x_interp, y_interp, on='image_id', how='inner')
                xy_interp.fillna(0, inplace=True)
                
            # the x, y here represent frame number and value seperately
            if xy_interp is not None:
                # disable segmenting by decesion tree
                if max_seg == 1:
                    x = xy_interp['image_id'].values.reshape(-1, 1)
                    y = xy_interp[val_col]

                    # using polynomial regression to fit
                    pf = PolynomialFeatures(degree=reg_deg)
                    x_poly = pf.fit_transform(x)
                    
                    lr = LinearRegression()
                    lr.fit(x_poly, y)

                    xx = np.linspace(0, video_len + 1, video_len + 2)
                    xx_poly = pf.transform(xx.reshape(-1, 1))
                    yy_poly = lr.predict(xx_poly)

                    p_res = pd.DataFrame()
                    p_res['image_id'] = xx
                    p_res[val_col + 'reg'] = yy_poly
                    p_res[id_col] = p
                    p_res['seg'] = 0
                    res_df = pd.concat([res_df, p_res])

                # max segment >= 2
                else:
                    x = xy_interp['image_id'].values
                    y = xy_interp[val_col].values
                    # calculate the gradient of y
                    dy = np.gradient(y, x)

                    tree = DecisionTreeRegressor(max_leaf_nodes=max_seg)
                    tree.fit(x.reshape(-1, 1), dy.reshape(-1, 1))
                    dy_pred = tree.predict(x.reshape(-1, 1))

                    # iter the segments by decision trees
                    seg_idx = 0
                    for seg_val in np.unique(dy_pred):
                        msk = dy_pred == seg_val
                        x_seg = x[msk].reshape(-1, 1)
                        y_seg = y[msk].reshape(-1, 1)

                        # drop segments that is too short
                        if x_seg.shape[0] > min_len:
                            seg_start = int(x_seg.min())
                            seg_end = int(x_seg.max())
                            seg_len = seg_end - seg_start + 1

                            # using polynomial regression to fit every segment
                            pf = PolynomialFeatures(degree=reg_deg)
                            x_poly = pf.fit_transform(x_seg)
                            
                            lr = LinearRegression()
                            lr.fit(x_poly, y_seg)

                            xx_seg = np.linspace(seg_start, seg_end, seg_len)
                            xx_poly = pf.transform(xx_seg.reshape(-1, 1))
                            yy_poly = lr.predict(xx_poly)

                            seg_res = pd.DataFrame()
                            seg_res['image_id'] = xx_seg
                            seg_res[val_col + 'reg'] = yy_poly
                            seg_res[id_col] = p
                            seg_res['seg'] = seg_idx
                            # seg_res['seg'] = val_col + '+' + str(seg_idx)
                            # if val_col == '0':
                            #     seg_res['x_seg'] = seg_idx
                            # else:
                            #     seg_res['y_seg'] = seg_idx

                            res_df = pd.concat([res_df, seg_res])
                        seg_idx = seg_idx + 1
    return res_df


def cal_metric(box_key_df):
    box_key = box_key_df.copy()
    # key 18 Neck 19 Hip
    box_key['Neck2Hip_x'] = np.abs(box_key['54'] - box_key['57'])
    box_key['Neck2Hip_y'] = np.abs(box_key['55'] - box_key['58'])
    box_key['Neck2Hip'] = np.sqrt(box_key['Neck2Hip_x']**2 + box_key['Neck2Hip_y']**2)
    # key 11 Left Hip 15 Left Ankle
    box_key['LHip2Ankle_x'] = np.abs(box_key['33'] - box_key['45'])
    box_key['LHip2Ankle_y'] = np.abs(box_key['34'] - box_key['46'])
    box_key['LHip2Ankle'] = np.sqrt(box_key['LHip2Ankle_x']**2 + box_key['LHip2Ankle_y']**2)
    # key 12 Right Hip 16 Left Ankle
    box_key['RHip2Ankle_x'] = np.abs(box_key['36'] - box_key['48'])
    box_key['RHip2Ankle_y'] = np.abs(box_key['37'] - box_key['49'])
    box_key['RHip2Ankle'] = np.sqrt(box_key['RHip2Ankle_x']**2 + box_key['RHip2Ankle_y']**2)
    # key 5 Left Shoulder 9 Left Wrist
    box_key['LShoulder2Wrist_x'] = np.abs(box_key['15'] - box_key['27'])
    box_key['LShoulder2Wrist_y'] = np.abs(box_key['16'] - box_key['28'])
    box_key['LShoulder2Wrist'] = np.sqrt(box_key['LShoulder2Wrist_x']**2 + box_key['LShoulder2Wrist_y']**2)
    # key 6 Right Shoulder 10 Right Wrist
    box_key['RShoulder2Wrist_x'] = np.abs(box_key['18'] - box_key['30'])
    box_key['RShoulder2Wrist_y'] = np.abs(box_key['19'] - box_key['31'])
    box_key['RShoulder2Wrist'] = np.sqrt(box_key['RShoulder2Wrist_x']**2 + box_key['RShoulder2Wrist_y']**2)

    candidate_cols = ['Neck2Hip', 'LHip2Ankle', 'RHip2Ankle', 'LShoulder2Wrist', 'RShoulder2Wrist']
    # candidate_cols = ['Neck2Hip']
    for candi in candidate_cols:
        box_key[candi].replace(0, np.nan, inplace=True)
    box_key['body_metric'] = box_key[candidate_cols].mean(axis=1)

    # use mean metric of a detected person
    p_ids = box_key['idx'].unique()
    for p in p_ids:
        p_df = box_key[box_key['idx'] == p]
        p_metric = p_df['body_metric'].mean()
        box_key.loc[box_key['idx'] == p, 'p_metric'] = p_metric

    return box_key


def xy_normalize(box_df, key_df):
    # box points: 0-3
    # keypoints: 0k-3k, 5-77
    box_key = pd.merge(
        box_df, key_df, on=['image_id', 'idx'], 
        how='inner', suffixes=['', 'k'])
    box_key = cal_metric(box_key)
    # # key 18 Neck 19 Hip
    # box_key['Neck2Hip_x'] = np.abs(box_key['54'] - box_key['57'])
    # box_key['Neck2Hip_y'] = np.abs(box_key['55'] - box_key['58'])
    # box_key['Neck2Hip'] = np.sqrt(box_key['Neck2Hip_x']**2 + box_key['Neck2Hip_y']**2)
    # # key 11 Left Hip 15 Left Ankle
    # box_key['LHip2Ankle_x'] = np.abs(box_key['33'] - box_key['45'])
    # box_key['LHip2Ankle_y'] = np.abs(box_key['34'] - box_key['46'])
    # box_key['LHip2Ankle'] = np.sqrt(box_key['LHip2Ankle_x']**2 + box_key['LHip2Ankle_y']**2)
    # # key 12 Right Hip 16 Left Ankle
    # box_key['RHip2Ankle_x'] = np.abs(box_key['36'] - box_key['48'])
    # box_key['RHip2Ankle_y'] = np.abs(box_key['37'] - box_key['49'])
    # box_key['RHip2Ankle'] = np.sqrt(box_key['RHip2Ankle_x']**2 + box_key['RHip2Ankle_y']**2)
    # # key 5 Left Shoulder 9 Left Wrist
    # box_key['LShoulder2Wrist_x'] = np.abs(box_key['15'] - box_key['27'])
    # box_key['LShoulder2Wrist_y'] = np.abs(box_key['16'] - box_key['28'])
    # box_key['LShoulder2Wrist'] = np.sqrt(box_key['LShoulder2Wrist_x']**2 + box_key['LShoulder2Wrist_y']**2)
    # # key 6 Right Shoulder 10 Right Wrist
    # box_key['RShoulder2Wrist_x'] = np.abs(box_key['18'] - box_key['30'])
    # box_key['RShoulder2Wrist_y'] = np.abs(box_key['19'] - box_key['31'])
    # box_key['RShoulder2Wrist'] = np.sqrt(box_key['RShoulder2Wrist_x']**2 + box_key['RShoulder2Wrist_y']**2)

    # candidate_cols = ['Neck2Hip', 'LHip2Ankle', 'RHip2Ankle', 'LShoulder2Wrist', 'RShoulder2Wrist']
    # # candidate_cols = ['Neck2Hip']
    # for candi in candidate_cols:
    #     box_key[candi].replace(0, np.nan, inplace=True)
    # box_key['body_metric'] = box_key[candidate_cols].mean(axis=1)
    
    # # calculate mean with rolling window
    # # interpolate first
    # body_interp = pd.DataFrame()
    # video_len = box_key['image_id'].max()
    # p_ids = box_key['idx'].unique()
    # for p in p_ids:
    #     p_df = box_key[box_key['idx'] == p]
    #     if p_df.shape[0] > min_p_len:
    #         p_interp = do_interp(p_df, 'image_id', 'body_metric', video_len, interp_kind=interp_type, fill0=False)
    #         p_interp['idx'] = p
    #         body_interp = pd.concat([body_interp, p_interp])

    # body_interp['body_metric_roll'] = body_interp['body_metric'].rolling(window, center=True, min_periods=2).mean()
    # box_key = pd.merge(box_key, body_interp[['image_id', 'idx', 'body_metric_roll']], on=['image_id', 'idx'], how='inner')

    # for box_col in range(0, 4):
    #     box_key[str(box_col) + 'norm'] = box_key[str(box_col)] / box_key['body_metric_roll']

    # use mean metric of a whole video
    # vid_metric = box_key['body_metric'].mean()
    # for box_col in range(0, 4):
    #     box_key[str(box_col) + 'norm'] = box_key[str(box_col)] / vid_metric

    # # use mean metric of a detected person
    # p_ids = box_key['idx'].unique()
    # for p in p_ids:
    #     p_df = box_key[box_key['idx'] == p]
    #     p_metric = p_df['body_metric'].mean()
    #     box_key.loc[box_key['idx'] == p, 'p_metric'] = p_metric

    # just to test and compare
    b_TEST = False
    EACH_FRAME = True
    MIN_P_LEN = 30
    WINDOW_SIZE = 1
    INTERP_TYPE = 'previous'

    if b_TEST:
        if EACH_FRAME:
            # just to test and compare
            # calculate mean with rolling window
            # no winodw
            if WINDOW_SIZE == 1:
                for box_col in range(0, 4):
                    box_key[str(box_col) + 'norm'] = box_key[str(box_col)] / box_key['body_metric']

            else:
                # interpolate first
                body_interp = pd.DataFrame()
                video_len = box_key['image_id'].max()
                p_ids = box_key['idx'].unique()
                for p in p_ids:
                    p_df = box_key[box_key['idx'] == p]
                    if p_df.shape[0] > MIN_P_LEN:
                        p_interp = do_interp(p_df, 'image_id', 'body_metric', video_len, interp_kind=INTERP_TYPE, fill0=False)
                        p_interp['idx'] = p
                        body_interp = pd.concat([body_interp, p_interp])

                body_interp['body_metric_roll'] = body_interp['body_metric'].rolling(WINDOW_SIZE, center=True, min_periods=2).mean()
                box_key = pd.merge(box_key, body_interp[['image_id', 'idx', 'body_metric_roll']], on=['image_id', 'idx'], how='inner')

                for box_col in range(0, 4):
                    box_key[str(box_col) + 'norm'] = box_key[str(box_col)] / box_key['body_metric_roll']

        else:
            # just to test and compare
            # use mean metric of a whole video
            vid_metric = box_key['body_metric'].mean()
            for box_col in range(0, 4):
                box_key[str(box_col) + 'norm'] = box_key[str(box_col)] / vid_metric

    else:
        # use mean metric of a detected person
        for box_col in range(0, 4):
            box_key[str(box_col) + 'norm'] = box_key[str(box_col)] / box_key['p_metric']
    
    return box_key


def key_normalize(key_df):
    box_key = cal_metric(key_df)
    key_cols = list(range(0, 77, 3)) + list(range(1, 77, 3))
    for key_col in key_cols:
        box_key[str(key_col) + 'norm'] = box_key[str(key_col)] / box_key['p_metric']

    return box_key


def get_high_score(vid_df, upper_limit=340, min_p_len=10, lower_confi=False):
    # data range from [0 to upper_limit/100.0]
    # histogram bins from [0.00, 0.01) to [3.39, 3.40]
    histogram_data, _ = np.histogram(vid_df['score'], bins=[b/100.0 for b in range(0, upper_limit+1)])
    huang_thresholding = HuangThresholding(histogram_data, upper_limit)
    vid_score_thre = huang_thresholding.find_threshold() / 100.0

    # use lower confidence threshold for low quality videos
    if lower_confi:
        vid_q1 = vid_df['score'].quantile(0.25)
        vid_q3 = vid_df['score'].quantile(0.75)
        vid_score_thre = vid_score_thre - vid_q3 + vid_q1

    # vid_df['score_thre'] = vid_score_thre
    # delete detected persons that has a lower confidence
    p_ids = vid_df['idx'].unique()
    high_score_p = []
    for p in p_ids:
        p_df = vid_df[vid_df['idx'] == p]
        # data length too short
        if p_df.shape[0] > min_p_len:
            # compare the 1/4 quantile of a person's score and the huang_thre
            p_q1 = p_df['score'].quantile(0.25)
            if p_q1 > vid_score_thre:
                high_score_p.append(p)
    vid_df['score_thre'] = vid_score_thre
    return vid_df[vid_df['idx'].isin(high_score_p)]
