import pandas as pd
import re
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly
# import numpy as np


def read_from_csv(csv_dir, label_file=True):
    # read csvs
    key_dfs = {}
    speed_dfs = {}
    box_dfs = {}
    for kf in KEY_FILES:
        kf_df = pd.read_csv(csv_dir + kf, index_col=0)
        key_dfs[re.sub('(_statis_res.csv)$', '', kf)] = kf_df
        print(kf, ' file num: ', len(kf_df['file'].unique()))
    for sf in SPEED_FILES:
        sf_df = pd.read_csv(csv_dir + sf, index_col=0)
        speed_dfs[re.sub('(_statis_speed_res.csv)$', '', sf)] = sf_df
        print(sf, ' file num: ', len(sf_df['file'].unique()))
    for bf in BOX_FILES:
        bf_df = pd.read_csv(csv_dir + bf, index_col=0)
        box_dfs[re.sub('(statis_res.csv)$', 'box', bf)] = bf_df
        print(bf, ' file num: ', len(bf_df['file'].unique()))

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
            p_var_min = f_df[(f_df['idx1'] == p) | (f_df['idx2'] == p)]['var'].min()
            p_var_mid = f_df[(f_df['idx1'] == p) | (f_df['idx2'] == p)]['var'].median()
            # idx, iou_var_max, iou_var_min, iou_var_mid, file
            iou_p_df.append([int(p), p_var_max, p_var_min, p_var_mid, f])
    iou_p_df = pd.DataFrame(iou_p_df, columns=['idx', 'iou_var_max', 'iou_var_min', 'iou_var_mid', 'file'])

    # merge all dfs
    common_cols = ['idx', 'file']
    # the file numbers in iou data can be dramatically less than others
    # thus using the 'outer' merge is important
    # leaving some NaN values though
    merged = pd.merge(box_dfs['box'][common_cols + ['var_x', 'var_y']],
                      iou_p_df,
                      on=common_cols, 
                      how='outer')
    for k, v in key_dfs.items():
        merged = pd.merge(merged, v[common_cols + ['var_x', 'var_y']],
                          on=common_cols, suffixes=['', '_' + k])
    for k, v in speed_dfs.items():
        merged = pd.merge(merged, v[common_cols + ['var']],
                    on=common_cols, suffixes=['', '_speed_' + k])
    merged.rename(columns={'var': 'var_speed_left_leg'}, inplace=True)

    if not label_file:
        return merged, None
    
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


def fit_xgb(feature_df, ground_truth, subclass=False):
    print('person')
    # this will excludes data whose 'idx' is not in the label dataframe
    merged_data = pd.merge(feature_df, ground_truth[['idx', 'file', 'label']],
                           on=['idx', 'file'])
    random_state = 7
    print('random_state: ', random_state)
    im_type_list = ['weight', 'gain', 'cover']

    if subclass:
        merged_data['class'] = merged_data['file'].str.replace(r'[0-9]+', '', regex=True)

    if K_FOLDS > 1:
        acc_list = []
        file_list = merged_data['file'].unique()
    
        lbc = LabelEncoder()
        if subclass:
            merged_data['label_encoded'] = lbc.fit_transform(merged_data['class'])
            merged_data.drop(['person', 'idx', 'label'], axis=1, inplace=True)
        else:
            merged_data['label_encoded'] = lbc.fit_transform(merged_data['label'])
            merged_data.drop(['person', 'idx'], axis=1, inplace=True)

        ohe = OneHotEncoder()
        x_ohe = ohe.fit_transform(merged_data['label_encoded'].values.reshape(-1, 1)).toarray()
        df_ohe = pd.DataFrame(x_ohe, columns=['label' + str(i) for i in range(x_ohe.shape[1])])
        data = pd.concat([merged_data, df_ohe], axis=1)

        im_df = pd.DataFrame()
        for im_type in im_type_list:
            count = 1
            kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(file_list):
                train_files = file_list[train_index]
                test_files = file_list[test_index]
                train_data = data.loc[data['file'].isin(train_files)]
                test_data = data.loc[data['file'].isin(test_files)]
                x_train = train_data.iloc[:, 1:18]
                x_test = test_data.iloc[:, 1:18]
                y_train = train_data.iloc[:, 19]
                y_test = test_data.iloc[:, 19]
                
                # train_samples = y_train.value_counts()
                # test_samples = y_test.value_counts()
                # weight = train_samples[1]/train_samples[0]
                weight = 1

                xgb_model_0 = xgb.XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=100,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    scale_pos_weight=weight,
                    seed=0,
                    nthread=-1, 
                    importance_type=im_type
                )

                xgb_model_0.fit(x_train, y_train)
                acc = accuracy_score(y_test, xgb_model_0.predict(x_test))
                # plot_importance(xgb_model_0)
                # plt.show()
                acc_list.append(acc)
                print("accuracy_score:" + str(acc))

                importances = xgb_model_0.feature_importances_
                importances = pd.DataFrame({'value': importances, 'feature': data.columns[1:18]})
                importances['count'] = count
                importances['type'] = im_type
                im_df = pd.concat([im_df, importances], axis=0)
                count = count + 1
            print('avg :', sum(acc_list)/len(acc_list))
        fig = px.bar(im_df, x='feature', y='value', facet_row='count', color='type', barmode='group', 
                     labels={'feature': '特征', 'value': '重要度', 'count': '折数', 'type': '重要度类型'})
        fig.update_layout(font=dict(size=18))
        plotly.offline.plot(fig, filename='fightDetect/val/importance.html', auto_open=False)
        # im_df.columns = data.columns.tolist()[1:18]


    else:
        merged_data.drop(['person', 'idx', 'file'], axis=1, inplace=True)
        lbc = LabelEncoder()
        merged_data['label_encoded'] = lbc.fit_transform(merged_data['label'])

        ohe = OneHotEncoder()
        x_ohe = ohe.fit_transform(merged_data['label_encoded'].values.reshape(-1, 1)).toarray()
        df_ohe = pd.DataFrame(x_ohe, columns=['label' + str(i) for i in range(x_ohe.shape[1])])
        data = pd.concat([merged_data, df_ohe], axis=1)
        
        data_x = data.iloc[:, 0:17]
        data_y = data.iloc[:, 18]

        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y, random_state=random_state, test_size=0.25)

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


def fit_xgb_vid(feature_df):
    print('vid')
    data = feature_df.copy()
    data.drop(['idx'], axis=1, inplace=True)

    var_max = data.groupby('file').max()
    var_mid = data.groupby('file').median()
    var_avg = data.groupby('file').mean()

    var_agg = pd.merge(var_max, var_mid, on='file')
    var_agg = pd.merge(var_avg, var_agg, on='file')
    var_agg.reset_index(inplace=True)

    # this will excludes data whose 'idx' is not in the label dataframe
    var_agg['label'] = var_agg['file'].str.replace(r'[0-9]+', '', regex=True)
    lbc = LabelEncoder()
    var_agg['label_encoded'] = lbc.fit_transform(var_agg['label'])
    var_agg.drop(['label', 'file'], axis=1, inplace=True)

    # ohe = OneHotEncoder()
    # x_ohe = ohe.fit_transform(data['label_encoded'].values.reshape(-1, 1)).toarray()
    # df_ohe = pd.DataFrame(x_ohe, columns=['label' + str(i) for i in range(x_ohe.shape[1])])
    # data = pd.concat([data, df_ohe], axis=1)
    
    data_x = var_agg.iloc[:, 0:51]
    data_y = var_agg.iloc[:, 51]
    random_state = 0
    print('random_state: ', random_state)
    if K_FOLDS == 1:
        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y, random_state=random_state, test_size=0.2)
        
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
            seed=random_state,
            nthread=-1
        )

        xgb_model_0.fit(x_train, y_train)
        acc = accuracy_score(y_test, xgb_model_0.predict(x_test))
        print("accuracy_score:" + str(acc))

    else:
        acc_list = []
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)
        for train_index, test_index in kf.split(data_x, data_y):
            x_train = data_x.loc[train_index]
            x_test = data_x.loc[test_index]
            y_train = data_y.loc[train_index]
            y_test = data_y.loc[test_index]

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
                seed=random_state,
                nthread=-1
            )

            xgb_model_0.fit(x_train, y_train)
            acc = accuracy_score(y_test, xgb_model_0.predict(x_test))
            acc_list.append(acc)
            print("accuracy_score:" + str(acc))
        print('avg :', sum(acc_list)/len(acc_list))


KEY_NAMES = ['left_leg', 'left_arm', 'right_leg', 'right_arm']
KEY_FILES =  [fname + '_statis_res.csv' for fname in KEY_NAMES]
SPEED_FILES = [fname + '_statis_speed_res.csv' for fname in KEY_NAMES]
BOX_FILES = ['statis_res.csv', 'iou_statis_res.csv']

# args
SRC_DIR = "private-correct/"
K_FOLDS = 5


if __name__ == '__main__':
    # classify individuals 
    features, gt = read_from_csv('fightDetect/csv/' + SRC_DIR, label_file=True)
    fit_xgb(features, gt, subclass=False)

    # classify videos
    # features, _ = read_from_csv('fightDetect/csv/' + SRC_DIR, label_file=False)
    # fit_xgb_vid(features)


