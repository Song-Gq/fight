from math import sqrt
import pandas as pd
import data_process as dp


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
        
        # to store iou results
        res = pd.DataFrame()
        combs2 = dp.get_comb(high_score['idx'])
        for comb in combs2:
            iou_df = dp.cal_iou(high_score, comb[0], comb[1], kind=iou_type)
            # the two persons have common frames
            if not iou_df.empty:
                col_name = str(comb[0]) + '+' + str(comb[1])

        print()

