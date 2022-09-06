'''
获取数据集，通过摄像头拍照获取，然后保存到Images目录中
'''
# -*- coding: utf-8 -*-
import cv2

cap = cv2.VideoCapture(0)

# 图像计数 从1开始
img_count = 1

while (1):
    # get a frame
    ret, frame = cap.read()
    if ret:
        # show a frame
        cv2.imshow("capture", frame)
        # 等待按键事件发生 等待1ms
        key = cv2.waitKey(1)
        if key == ord('q'):
            # 如果按键为q 代表quit 退出程序
            print("程序正常退出..")
            break
        elif key == ord('f'):
            # 如果f键按下，则进行图片保存;并命名图片为图片序号.jpg
            cv2.imwrite("Images/{}.jpg".format(img_count), frame)
            print("保存图片{}.jpg".format(img_count))
            # 图片编号计数自增1
            img_count += 1
    else:
        print("图像数据获取失败！！")
        break
cap.release()
cv2.destroyAllWindows()
