import pandas as pd
import re
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import numpy as np


def read_from_csv(csv_dir):
    # read csvs
    key_dfs = {}
    speed_dfs = {}
    box_dfs = {}
    for kf in key_files:
        key_dfs[re.sub('(_statis_res.csv)$', '', kf)] = pd.read_csv(csv_dir + kf, index_col=0)
    for sf in speed_files:
        speed_dfs[re.sub('(_statis_speed_res.csv)$', '', sf)] = pd.read_csv(csv_dir + sf, index_col=0)
    for bf in box_files:
        box_dfs[re.sub('(statis_res.csv)$', 'box', bf)] = pd.read_csv(csv_dir + bf, index_col=0)

    # split column 'comb' into 'idx1' and 'idx2'
    iou_p_df = []
    iou_df = box_dfs['iou_box']
    iou_df['idx1'] = iou_df['comb'].map(lambda x: x.split('+')[0])
    iou_df['idx2'] = iou_df['comb'].map(lambda x: x.split('+')[1])
    flist = iou_df['file'].unique()
    for f in flist:
        f_df = iou_df[iou_df['file'] == f]
        plist = list(f_df['idx1'].unique()) + list(f_df['idx2'].unique())
        for p in plist:
            p_var_max = f_df[(f_df['idx1'] == p) | (f_df['idx2'] == p)]['var'].max()
            # idx, iou_var_max, file
            iou_p_df.append([int(p), p_var_max, f])
    iou_p_df = pd.DataFrame(iou_p_df, columns=['idx', 'var', 'file'])

    # merge all dfs
    common_cols = ['idx', 'file']
    merged = pd.merge(box_dfs['box'][common_cols + ['var_x', 'var_y']],
                      iou_p_df,
                      on=common_cols)
    for k, v in key_dfs.items():
        merged = pd.merge(merged, v[common_cols + ['var_x', 'var_y']],
                          on=common_cols, suffixes=['', '_' + k])
    for k, v in speed_dfs.items():
        merged = pd.merge(merged, v[common_cols + ['var']],
                    on=common_cols, suffixes=['', '_speed_' + k])
    merged.rename(columns={'var': 'var_iou'}, inplace=True)

    # read label csv
    label_df = pd.read_csv(csv_dir + 'labels.csv')
    label_df = label_df.loc[:, ~label_df.columns.str.contains("^Unnamed")]

    # label the data with 2 classes
    label_df['label'] = label_df['class']
    label_df.loc[label_df['pclass'] == 0, 'label'] = 'normal'
    label_df['label'] = label_df['label'].replace(['sleep', 'fall', 'wander', 'gather'], 'normal')
    label_df['label'] = label_df['label'].replace(['weapon', 'fight', 'chase'], 'abnormal')

    # merge columns 'idx' and 'file' into one
    label_df['person'] = label_df['file'] + '+' + label_df['idx'].map(str)
    merged['person'] = merged['file'] + '+' + merged['idx'].map(str)

    # delete data of quality 3
    low_quality = label_df[label_df['quality'] == 3]
    low_files = low_quality['file'].unique()

    filtered = merged[~merged['file'].isin(low_files)]
    label_filtered = label_df[~label_df['file'].isin(low_files)]
    return filtered, label_filtered


def prepare_data(feature_df, ground_truth):
    # this will excludes data whose 'idx' is not in the label dataframe
    merged_data = pd.merge(feature_df, ground_truth[['idx', 'file', 'label']],
                           on=['idx', 'file'])
    merged_data.drop(['person', 'idx', 'file'], axis=1, inplace=True)

    lbc = LabelEncoder()
    merged_data['label_encoded'] = lbc.fit_transform(merged_data['label'])

    ohe = OneHotEncoder()
    x_ohe = ohe.fit_transform(merged_data['label_encoded'].values.reshape(-1, 1)).toarray()
    df_ohe = pd.DataFrame(x_ohe, columns=['label' + str(i) for i in range(x_ohe.shape[1])])
    data = pd.concat([merged_data, df_ohe], axis=1)
    
    data_x = data.iloc[:, 0:15]
    data_y = data.iloc[:, 16]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

    xgb_model_0 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,
        seed=0,
        nthread=-1
    )

    xgb_model_0.fit(x_train, y_train)
    acc = accuracy_score(y_test, xgb_model_0.predict(x_test))
    print("accuracy_score:" + str(acc))


key_names = ['left_leg', 'left_arm', 'right_leg', 'right_arm']
key_files =  [fname + '_statis_res.csv' for fname in key_names]
speed_files = [fname + '_statis_speed_res.csv' for fname in key_names]
box_files = ['statis_res.csv', 'iou_statis_res.csv']

# args
src_dir = "test-plus/"


if __name__ == '__main__':
    features, gt = read_from_csv('fightDetect/csv/' + src_dir)
    prepare_data(features, gt)

