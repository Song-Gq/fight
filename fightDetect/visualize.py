import numpy as np
import plotly.express as px
import plotly
import pandas as pd

src_name = "AlphaPose_空手扭打11.json"

if __name__ == '__main__':
    df = pd.read_json('data/' + src_name)
    df['image_id'] = df['image_id'].str.removesuffix('.jpg').astype(int)

    # split the column of <list> 'keypoints' into multiple cols
    keys = pd.DataFrame(df.keypoints.tolist(), index=df.index)
    keys.columns = keys.columns.map(str)
    keys = pd.concat([df, keys], axis=1)
    # delete keypoints 26-68
    keys.drop(columns=keys.columns[[2, 4] + list(range(81, 210))], axis=1, inplace=True)

    # split the column of <list> 'box'
    boxes = pd.DataFrame(df.box.tolist(), index=df.index)
    boxes.columns = boxes.columns.map(str)
    boxes = pd.concat([df, boxes], axis=1)
    boxes.drop(columns=boxes.columns[[2, 4]], axis=1, inplace=True)

    # 9. left hand: 27-x, 28-y, 29-confidence
    fig = px.scatter_3d(keys, x='27', y='28', z='image_id', color='29', symbol='idx')
    plotly.offline.plot(fig, filename='fig/' + src_name + '-left-hand.html')

    # 15. left ankle: 45-x, 46-y, 47-confidence
    fig = px.scatter_3d(keys, x='45', y='46', z='image_id', color='47', symbol='idx')
    plotly.offline.plot(fig, filename='fig/' + src_name + '-left-ankle.html')

    # calculate the speed
    speed = keys.sort_values(by=['idx', 'image_id'])
    diff = speed.groupby('idx').diff().fillna(0.)
    confi_cols = list(range(2, 75, 3))
    for c in confi_cols:
        diff[str(c)] = np.sqrt(diff[str(c-1)]**2 + diff[str(c-2)]**2) / diff['image_id']
    diff = diff[[str(c) for c in range(2, 75, 3)]]
    diff = diff.add_suffix('_speed')
    speed = pd.concat([speed, diff], axis=1)

    # speed.fillna(0.)
    fig = px.scatter_3d(speed[speed['29'] > 0.5], x='27', y='28', z='image_id', color='29_speed', symbol='idx')
    plotly.offline.plot(fig, filename='fig/' + src_name + '-left-hand-speed.html')

    fig = px.scatter_3d(speed[speed['47'] > 0.5], x='45', y='46', z='image_id', color='47_speed', symbol='idx')
    plotly.offline.plot(fig, filename='fig/' + src_name + '-left-ankle-speed.html')

    # fig.show()
    print()


