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


# def iter_comb(box_df, json_name):
#     # to store iou results
#     res = pd.DataFrame()
#     # to store fft results
#     fft_df = pd.DataFrame()
#     video_len = box_df['image_id'].max()
#     # fft_df['image_id'] = np.linspace(0, video_len, video_len + 1)

#     combs2 = dp.get_comb(box_df['idx'])
#     for comb in combs2:
#         iou_df = dp.cal_iou(box_df, comb[0], comb[1], kind=iou_type)
#         # the two persons have common frames
#         if not iou_df.empty:
#             col_name = str(comb[0]) + '+' + str(comb[1])
#             # append iou data
#             iou_df['comb'] = col_name
#             res = pd.concat([res, iou_df])

#             # do fft and append
#             fft_iou = dp.do_fft(iou_df['image_id'], iou_df['iou'], video_len, interp_type)
#             if fft_iou is not None:
#                 fft_iou['comb'] = col_name
#                 fft_df = pd.concat([fft_df, fft_iou])
#                 # fft_df = pd.merge(fft_df, fft_iou, on='image_id', how='outer')
#                 # fft_df = pd.concat([fft_df, fft_iou[col_name]], axis=1)
#     draw_iou(res, json_name)
#     # fft_df.fillna(0, inplace=True)
#     draw_fft(fft_df, json_name)


src_dir = "test/"
iou_type = 'giou'
score_thre = 2
interp_type = 'previous'
output_suffix = '-' + iou_type + \
    '-interp=' + interp_type + \
    '-score-thre=' + str(score_thre)


if __name__ == '__main__':
    for keys, boxes, fname in dp.iter_files('fightDetect/data/' + src_dir):
        high_score = boxes[boxes['score'] > score_thre]
        iou_df, fft_df = dp.comb_iou_fft(high_score, iou_type=iou_type, 
                                         interp_type=interp_type)
        draw_iou(iou_df, fname)
        draw_fft(fft_df, fname)
        
        print()

