#!/usr/bin/env python2
#!coding=utf-8
 
#right code !
#write by leo at 2018.04.26
#function: 
#display the frame from another node.
 
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time

cam0_path  = '/home/grantli/test_ws/src/yolov5_ros/img/cam0/'    
cam1_path  = '/home/grantli/test_ws/src/yolov5_ros/img/cam1/'
fourcc = cv2.VideoWriter_fourcc(*'XVID')


def callback(data):
    starttime = time.time()
    # define picture to_down' coefficient of ratio
    scaling_factor = 0.5
    global count,bridge
    count = count + 1
    # if count == 1:
    #     count = 0
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    # print(type(cv_img))
    out = cv2.VideoWriter(cam1_path +  str(count + 1) +  'output.mp4', fourcc, 20.0, (640,  480))
    out.write(cv_img)
        # time.sleep(5)
        # cv2.waitKey(3000)
    cap = cv2.VideoCapture(cam1_path +  str(count) +  'output.mp4')
    if count == 2:
        count = 1
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     cv2.imshow('frame', frame)
        #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        #     print(w, h, fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(w, h, fps)
        # timestr = "%.6f" %  data.header.stamp.to_sec()
        # image_name = "test" + ".jpg"
        # cv2.imwrite(cam0_path + image_name, cv_img)
        # cv_img = cv2.imread(cam0_path + image_name)
        # print(type(cv_img))
        # ret, frame = cv_img.read()
        # print(type(ret))
    cv2.imshow("frame" , cv_img)
    cv2.waitKey(1)
    endtime = time.time()
    print(endtime - starttime)
    # else:
    #     pass
 
def displayWebcam():
    rospy.init_node('webcam_display', anonymous=True)
 
    # make a video_object and init the video object
    global count,bridge
    count = 0
    bridge = CvBridge()
    rospy.Subscriber('/usb_cam/image_raw', Image, callback)
    rospy.spin()
 
if __name__ == '__main__':
    displayWebcam()

    # cap = cv2.VideoCapture(0)
    # print(type(cap))
    # while(True):
    #     ret, frame = cap.read()    
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()

 