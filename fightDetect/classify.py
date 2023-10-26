import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    src_name = 'fight-sur/fi001.json'
    df = pd.read_json('fightDetect/data/' + src_name)

    # split the column of <list> 'keypoints' into multiple cols
    keys = pd.DataFrame(df.keypoints.tolist(), index=df.index)
    keys.columns = keys.columns.map(str)
    keys = pd.concat([df, keys], axis=1)
    keys.drop(columns=keys.columns[[2, 4]], axis=1, inplace=True)

    # calculate the speed
    speed = keys.sort_values(by=['idx', 'image_id'])
    diff = speed.groupby('idx').diff().fillna(0.)
    confi_cols = list(range(2, 78, 3))
    for c in confi_cols:
        diff[str(c)] = np.sqrt(diff[str(c-1)]**2 + diff[str(c-2)]**2) / diff['image_id']
    diff = diff[[str(c) for c in range(2, 78, 3)]]
    diff = diff.add_suffix('_speed')
    speed = pd.concat([speed, diff], axis=1)

    print()
