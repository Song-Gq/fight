import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import stft
from scipy.interpolate import interp1d
import os


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


def do_fft(frame, value, total_len, iterp_kind='linear'):
    # linear interpolation
    frame_min = frame.min()
    frame_max = frame.max()
    # valid frame > 10
    if frame.shape[0] > 10:
        frame_len = frame_max - frame_min + 1
        x = np.linspace(frame_min, frame_max, num=frame_len)
        # interpolate values that are missing
        f1 = interp1d(frame, value, kind=iterp_kind)
        y = f1(x)

        fft_res = pd.DataFrame()
        fft_res['image_id'] = x
        fft_res['y'] = y

        # fill zeroes in the front and end
        df_fill = pd.DataFrame()
        df_fill['image_id'] = np.linspace(0, total_len, total_len + 1)
        df_fill = pd.merge(df_fill, fft_res, on='image_id', how='outer')
        df_fill.fillna(0, inplace=True)

        # do fft
        fft_y = fft(list(df_fill['y']))
        abs_y = np.abs(fft_y)
        norm_y = abs_y / (total_len + 1)

        # # stft
        # fs = 25
        # nperseg = 100
        # f2, t2, Zxx = stft(list(df_fill['y']), fs, nperseg=nperseg, window='hann')
        # plt.pcolormesh(t2, f2, np.abs(Zxx), vmin=0, vmax=0.05, shading='gouraud')
        # plt.show()
        # print()

        df_fill['y'] = norm_y
        return df_fill
    return None


def iter_files(files_dir):
    json_dir = os.fsencode(files_dir)
    for f in os.listdir(json_dir):
        fname = os.fsdecode(f)
        if fname.endswith(".json"): 
            keys = gen_key_df(files_dir + fname)
            boxes = gen_box_df(files_dir + fname)
            yield keys, boxes, fname

