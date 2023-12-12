import numpy as np
import pandas as pd
from scipy.fftpack import fft, fft2
from scipy.signal import stft
from scipy.interpolate import interp1d
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


def gen_key_df(json_dir):
    df = pd.read_json(json_dir)
    # split the column of <list> 'keypoints' into multiple cols
    keys = pd.DataFrame(df.keypoints.tolist(), index=df.index)
    keys.columns = keys.columns.map(str)
    keys = pd.concat([df, keys], axis=1)
    keys.drop(columns=keys.columns[[2, 4]], axis=1, inplace=True)
    return keys

    
def gen_box_df(json_dir):
    df = pd.read_json(json_dir)
    # split the column of <list> 'box'
    boxes = pd.DataFrame(df.box.tolist(), index=df.index)
    boxes.columns = boxes.columns.map(str)
    boxes = pd.concat([df, boxes], axis=1)
    boxes.drop(columns=boxes.columns[[2, 4]], axis=1, inplace=True)
    return boxes


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
    inter, uni, c = box_area_xywh(box1, box2)
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
            p1xp2['iou'] = p1xp2.apply(lambda x: box_giou(
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


def comb_iou_fft(box_df, iou_type, interp_type, fill0=True):
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
            res_interp = do_interp(iou_df, 'image_id', 'iou', video_len, interp_type, fill0=fill0)
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


def poly_regress(box_df, val_col, linear=False, min_len=10):
    res_df = pd.DataFrame()
    video_len = box_df['image_id'].max()
    p_ids = box_df['idx'].unique()
    for p in p_ids:
        p_df = box_df[box_df['idx'] == p]
        # valid frame > min_len
        if p_df.shape[0] > min_len:
            # avoid exponential explosion
            # 2-degree does noe match the real scene that a man's speed cannot grow at a squared way
            pf = PolynomialFeatures(degree=3, interaction_only=linear)
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
def tree_seg(box_df, val_col, max_seg=3, reg_deg=2, min_len=10, interp_type='linear', xy_col=['0', '1']):
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
    video_len = box_df['image_id'].max()
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
                xy_interp = do_interp(p_df, 'image_id', val_col, video_len, interp_type, fill0=False)
            else:
                # do interpolation
                # the x, y here represent the axes in the video frame
                # 0 is not filled in the front and end of the time series
                x_interp = do_interp(p_df, 'image_id', xy_col[0], video_len, interp_type, fill0=False)
                y_interp = do_interp(p_df, 'image_id', xy_col[1], video_len, interp_type, fill0=False)
                xy_interp = pd.merge(x_interp, y_interp, on='image_id', how='inner')
                xy_interp.fillna(0, inplace=True)
                
            # the x, y here represent frame number and value seperately
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


def xy_normalize(box_df, key_df, window=10, interp_type='linear', min_p_len=10):
    # box points: 0-3
    # keypoints: 0k-3k, 5-77
    box_key = pd.merge(
        box_df, key_df, on=['image_id', 'idx'], 
        how='inner', suffixes=['', 'k'])
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

    # check if vid_metric is a scalar!!!
    vid_metric = box_key['body_metric'].mean()
    for box_col in range(0, 4):
        box_key[str(box_col) + 'norm'] = box_key[str(box_col)] / vid_metric
    
    return box_key
