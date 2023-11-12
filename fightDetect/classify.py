from math import sqrt
import pandas as pd
import data_process as dp
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
import numpy as np
import matplotlib.pyplot as plt


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
            DTW[(i, j)] = dist + min(
                DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])


src_dir = "train/"
iou_type = 'giou'
score_thre = 2
interp_type = 'previous'
output_suffix = '-' + iou_type + \
    '-interp=' + interp_type + \
    '-score-thre=' + str(score_thre)


if __name__ == '__main__':
    datasets = [dict(), dict()]
    subsets = ['norm/', 'abnorm/']
    # iterate files
    for i in range(0, 2):
        for keys, boxes, fname in dp.iter_files(
            'fightDetect/data/' + src_dir + subsets[i]):
            high_score = boxes[boxes['score'] > score_thre]
            iou_df, fft_df = dp.comb_iou_fft(
                high_score, iou_type=iou_type, interp_type=interp_type)
            
            # split iou_df into multiple dfs by 'comb'
            ts_df = iou_df[['iou', 'comb']]
            ts_dict = dict()
            for x, y in ts_df.groupby('comb'):
                ts_dict[x] = y
            datasets[i][fname] = ts_dict
    
    # cal the max len of the dfs
    max_vid_len = 0
    for sub in datasets:
        for k, v in sub.items():
            # each v is a dict of dfs
            for k2, v2 in v.items():
                # each v2 is a df of one comb
                # len starts from 1
                max_vid_len = np.maximum(max_vid_len, v2.shape[0])

    # make all time series the same length by filling zeroes
    for i in [0, 1]:
        for k, v in datasets[i].items():
            # each v is a dict of dfs
            for k2, v2 in v.items():
                # each v2 is a df of one comb
                to_fill = pd.DataFrame(
                    columns=['iou', 'comb'], index=list(
                        range(v2.shape[0], max_vid_len)))
                to_fill['comb'] = v2.loc[0, 'comb']
                to_fill.fillna(0, inplace=True)
                datasets[i][k][k2] = pd.concat([v2, to_fill])

                # comb_fill['image_id'] = np.linspace(0, max_vid_len - 1, max_vid_len)
                # comb_fill = pd.merge(comb_fill, comb_df, on='count', how='outer')
                # comb_fill.fillna(0, inplace=True)
                # sub[k][k2] = comb_fill.drop(columns=['count'], axis=1, inplace=True)
    
    # fit the dfs into array
    # ts_arr is the time series of X(iou value),
    # and class_gt is the ground truth of classes
    ts_arr = np.zeros((0))
    class_gt = list()
    # video file name
    vid_label = list()
    # combination labels of detected persons
    comb_label = list()
    for i in [0, 1]:
        for k, v in datasets[i].items():
            # each v is a dict of dfs (from one video)
            for k2, v2 in v.items():
                # each v2 is a df of one comb
                # check if the length of v2 equals max_vid_len
                if v2.shape[0] != max_vid_len:
                    raise ValueError('something wrong here!!!')
                ts_arr = np.append(ts_arr, np.array(v2['iou']), axis=0)
                class_gt.append(i)
                vid_label.append(k)
                comb_label.append(k2)
    ts_arr = ts_arr.reshape(-1, max_vid_len)

    seed = 0
    cluster_num = 4
    np.random.seed(seed)
    np.random.shuffle(ts_arr)
    np.random.seed(seed)
    np.random.shuffle(class_gt)
    np.random.seed(seed)
    np.random.shuffle(vid_label)
    np.random.seed(seed)
    np.random.shuffle(comb_label)

    # euclidean k-means 
    km = TimeSeriesKMeans(n_clusters=cluster_num, verbose=True, random_state=seed)
    y_pred = km.fit_predict(ts_arr)

    plt.figure()
    for yi in range(cluster_num):
        plt.subplot(3, cluster_num, yi + 1)
        for xx in ts_arr[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, max_vid_len)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 0:
            plt.title("Euclidean $k$-means")

    # check the prediction results
    pred_res = pd.DataFrame(data={
        'y_pred': y_pred, 
        'class_gt': class_gt, 
        'vid_label': vid_label, 
        'comb_label': comb_label
    })
    res_sorted = pred_res.sort_values(by=['vid_label', 'comb_label'])
    res_by_vid = pred_res.groupby('vid_label')
    sample_num = res_by_vid.count()
    positive_num = res_by_vid['y_pred'].sum()

    # DBA-k-means
    dba_km = TimeSeriesKMeans(n_clusters=cluster_num,
                            n_init=2,
                            metric="dtw",
                            verbose=True,
                            max_iter_barycenter=10,
                            random_state=seed, 
                            n_jobs=-1)
    y_pred = dba_km.fit_predict(ts_arr)

    for yi in range(cluster_num):
        plt.subplot(3, cluster_num, cluster_num + 1 + yi)
        for xx in ts_arr[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, max_vid_len)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 0:
            plt.title("DBA $k$-means")
    
    # check the prediction results
    pred_res = pd.DataFrame(data={
        'y_pred': y_pred, 
        'class_gt': class_gt, 
        'vid_label': vid_label, 
        'comb_label': comb_label
    })
    res_sorted = pred_res.sort_values(by=['vid_label', 'comb_label'])
    res_by_vid = pred_res.groupby('vid_label')
    sample_num = res_by_vid.count()
    positive_num = res_by_vid['y_pred'].sum()

    # Soft-DTW-k-means
    print("Soft-DTW k-means")
    sdtw_km = TimeSeriesKMeans(n_clusters=cluster_num,
                            metric="softdtw",
                            metric_params={"gamma": .01},
                            verbose=True,
                            random_state=seed, 
                            n_jobs=-1)
    y_pred = sdtw_km.fit_predict(ts_arr)

    for yi in range(2):
        plt.subplot(3, cluster_num, cluster_num * 2 + 1 + yi)
        for xx in ts_arr[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, max_vid_len)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 0:
            plt.title("Soft-DTW $k$-means")

    # check the prediction results
    pred_res = pd.DataFrame(data={
        'y_pred': y_pred, 
        'class_gt': class_gt, 
        'vid_label': vid_label, 
        'comb_label': comb_label
    })
    res_sorted = pred_res.sort_values(by=['vid_label', 'comb_label'])
    res_by_vid = pred_res.groupby('vid_label')
    sample_num = res_by_vid.count()
    positive_num = res_by_vid['y_pred'].sum()

    plt.show()
    print()

    # # to store iou results
    # res = pd.DataFrame()
    # combs2 = dp.get_comb(high_score['idx'])
    # for comb in combs2:
    #     iou_df = dp.cal_iou(high_score, comb[0], comb[1], kind=iou_type)
    #     # the two persons have common frames
    #     if not iou_df.empty:
    #         col_name = str(comb[0]) + '+' + str(comb[1])
