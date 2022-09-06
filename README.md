## 程序启动步骤

1. 水果识别：  (yolov5_ros/scripts/detect_fruit.py)      
```bash
roslaunch yolov5_ros yolov5_fruit.launch    # 启动摄像头
rostopic list    #查找当前话题
# 输入指令，1：识别Red_apples  ; 2：识别Green_apple ; 3：识别Orange  ; 4 ：识别mango
rostopic pub -1 /yolov5_test std_msgs/Int64 "data: 3"   
```
2. 医疗废弃物识别：(yolov5_ros/scripts/detect_medical.py)    
```bash
roslaunch yolov5_ros yolov5_medical.launch
```
3. 电器柜识别：(yolov5_ros/scripts/detect_dqg.py)  
```bash
roslaunch usb_cam usb_cam-test.launch
rostopic list      #修改yolov5_dqg.launch文件的 'image_topic'话题名
roslaunch yolov5_ros yolov5_dqg.launch 
```
***
## 环境配置

1. Install  (install requirements.txt in a _Python>=3.7.0_ environment, including _PyTorch>=1.7_.)
```bash
cd yolov5
pip install -r requirements.txt  # install
```
2. 国内常用下载源  
```bash

https://pypi.tuna.tsinghua.edu.cn/simple  # 清华

http://mirrors.aliyun.com/pypi/simple/  #阿里云

https://pypi.mirrors.ustc.edu.cn/simple/ #中国科技大学

http://pypi.hustunique.com/  # 华中理工大学

http://pypi.sdutlinux.org/  # 山东理工大学

http://pypi.douban.com/simple/   #豆瓣
```