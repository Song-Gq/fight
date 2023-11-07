from math import sqrt
import os
import itertools
from data_process import *
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV,train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score


def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])


# def iter_comb(box_df, json_name):
#     # to store iou results
#     res = pd.DataFrame()
#     # to store fft results
#     fft_df = pd.DataFrame()
#     video_len = box_df['image_id'].max()
#     # fft_df['image_id'] = np.linspace(0, video_len, video_len + 1)

#     p_ids = box_df['idx'].unique()
#     combs2 = list(itertools.combinations(p_ids, 2))
#     for comb in combs2:
#         iou_df = cal_iou(box_df, comb[0], comb[1])
#         # the two persons have common frames
#         if not iou_df.empty:
#             col_name = str(comb[0]) + '+' + str(comb[1])
#             # append iou data
#             iou_df['comb'] = col_name
#             res = pd.concat([res, iou_df])

#             # do fft and append
#             fft_iou = do_fft(iou_df['image_id'], iou_df['iou'], video_len)
#             if fft_iou is not None:
#                 fft_iou['comb'] = col_name
#                 fft_df = pd.concat([fft_df, fft_iou])
#                 # fft_df = pd.merge(fft_df, fft_iou, on='image_id', how='outer')
#                 # fft_df = pd.concat([fft_df, fft_iou[col_name]], axis=1)
#     draw_iou(res, json_name)
#     # fft_df.fillna(0, inplace=True)
#     draw_fft(fft_df, json_name)


# def start_loop(dir):
#     json_dir = os.fsencode(dir)
#     for f in os.listdir(json_dir):
#         fname = os.fsdecode(f)
#         if fname.endswith(".json"): 
#             boxes = gen_box_df('fightDetect/data/' + src_dir + fname)
#             # drop boxes with smaller scores
#             high_score = boxes[boxes['score'] > 2]
#             # merged = cal_iou(boxes, 2, 3)
#             # draw_iou(merged, fname)
#             iter_comb(high_score, fname)
#             print()
#             # draw_fig(keys, fname)
#     print()


src_dir = "test/"
iou_type = 'giou'
score_thre = 2
interp_type = 'previous'
output_suffix = '-' + iou_type + \
    '-interp=' + interp_type + \
    '-score-thre=' + str(score_thre)


if __name__ == '__main__':
    for keys, boxes, fname in iter_files('fightDetect/data/' + src_dir):
        high_score = boxes[boxes['score'] > 2]
        # iter_comb(high_score, fname)


        print()


# if __name__ == '__main__':
    # src_name = 'fight-sur/fi001.json'
    # df = pd.read_json('fightDetect/data/' + src_name)

    # # split the column of <list> 'keypoints' into multiple cols
    # keys = pd.DataFrame(df.keypoints.tolist(), index=df.index)
    # keys.columns = keys.columns.map(str)
    # keys = pd.concat([df, keys], axis=1)
    # keys.drop(columns=keys.columns[[2, 4]], axis=1, inplace=True)

    # # calculate the speed
    # speed = keys.sort_values(by=['idx', 'image_id'])
    # diff = speed.groupby('idx').diff().fillna(0.)
    # confi_cols = list(range(2, 78, 3))
    # for c in confi_cols:
    #     diff[str(c)] = np.sqrt(diff[str(c-1)]**2 + diff[str(c-2)]**2) / diff['image_id']
    # diff = diff[[str(c) for c in range(2, 78, 3)]]
    # diff = diff.add_suffix('_speed')
    # speed = pd.concat([speed, diff], axis=1)
