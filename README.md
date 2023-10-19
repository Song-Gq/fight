## AlphaPose安装

按照AlphaPose/docs/INSTALL.md步骤即可（使用Python3.7）

使用Python3.9时，在`python setup.py build develop`过程中，可能会出现以下错误。

- 会报setuptools的deprecated方法错误，执行以下操作：

  `pip install setuptools==65.6.3`

- 在安装halpecocotools时，报NoneType错误。

  手动`pip install halpecocotools`仍失败，报GCC文件不存在错误。

  可参照此处解决方案：https://github.com/HaoyiZhu/HalpeCOCOAPI/issues/1

  > 执行下面手动安装命令：
  >
  > `pip3 install git+https://github.com/Ambrosiussen/HalpeCOCOAPI.git#subdirectory=PythonAPI`
  >
  > 然后清除build文件：
  >
  > `python setup.py clean --all`
  >
  > 重新执行build develop

- 运行时报numpy的deprecated错误，报opencv的numpy错误，安装旧版本：

  `pip install numpy==1.21.5 scipy==1.7.3 opencv-python==4.7.0.68`

## 数据可视化

`pip install plotly pandas==1.3.5`

## 阈值调整

针对fight-sur数据集较弱的视频质量，对AlphaPose中YOLO和跟踪的阈值进行调整：

`detector/yolox_cfg.py -> cfg.CONF_THRES = 0.05 (default = 0.1)`
`detector/yolox_cfg.py -> cfg.NMS_THRES = 0.1 (default = 0.6)`
`trackers/tracker_cfg.py -> cfg.conf_thres = 0.05 (default = 0.5)`