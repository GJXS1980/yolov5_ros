# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import  rospy
from std_msgs.msg import Int64, Float32MultiArray


class detect:
    def __init__(self):
        self.com = 0
        self.per_label = None
    @torch.no_grad()
    def run(
            self,
            weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.70,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        source = str(source)  # 输入的路径变为字符串
        save_img = not nosave and not source.endswith('.txt')  # 是否保存图片和txt文件
        # 提取文件后缀名是否符合要求的文件，例如：是否格式是jpg, png, asf, avi等
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # 判断文件是否是视频流 # Path()提取文件名 例如：Path("./data/test_images/bus.jpg")
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # .lower()转化成小写
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories # 预测路径是否存在，不存在新建，按照实验文件以此递增新建
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader # 使用视频流或者页面
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt) # 直接从source文件下读取图片
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs    # 保存的路径

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False # 可视化文件路径
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2 # 预测的时间
            """
            pred.shape=(1, num_boxes, 5+num_class)
            h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
            num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
            pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
            pred[..., 4]为objectness置信度
            pred[..., 5:-1]为分类结果
            """

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3# 预测+NMS的时间
            """
            pred: 网络的输出结果
            conf_thres:置信度阈值
            ou_thres:iou阈值
            classes: 是否只保留特定的类别
            agnostic_nms: 进行nms是否也去除不同类别之间的框
            max-det: 保留的最大检测框数量
            ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
            pred是一个列表list[torch.tensor], 长度为batch_size
            每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
            """

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            rospy.init_node("yolov5_cam", anonymous=True)
            rospy.Subscriber("yolov5_test", Int64,  self.Command_frame)
            data_pub = rospy.Publisher("yolov5_data", Float32MultiArray, queue_size=1)
            
            for i, det in enumerate(pred):  # per image # 对每张图片做处理
                seen += 1
                
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count  # 如果输入源是webcam则batch_size>=1 取出dataset中的一张图片
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                    # s: 输出信息 初始为 ''
                    # im0: 原始图片 letterbox + pad 之前的图片
                    # frame: 视频流

                p = Path(p)  # to Path  # 当前路径yolov5/data/images/
                save_path = str(save_dir / p.name)  # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\bus.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  #设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
                s += '%gx%g ' % im.shape[2:]  # print string 像素640*480
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop # 保存截图
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()# 将预测信息映射到原图

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string 输出标签和个数：4 persons, 1 bus, 1 skateboard

                    # Write results 保存结果： txt/图片画框/crop-image
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            
                            #获取中心坐标
                            real_label = f'{names[c]}'
                            pub_xy = Float32MultiArray()
                            if real_label == self.per_label:
                                x1 = (int(xyxy[0]), int(xyxy[1]))
                                x2 = (int(xyxy[2]), int(xyxy[3]))
                                a = 80
                                b = 80
                                a_test = x2[0]-x1[0]
                                b_test = x2[1]-x1[1]
                                i_x = a/a_test
                                i_y = b/b_test
                                x = a_test/2 + x1[0]
                                y = b_test/2 + x1[1]
                                print("label：" + f' {names[c]}')
                                print("conf：",f' {conf:.2f}')
                                # print("label and conf:" + label)
                                print("x：",x)
                                print("y：",y)
                                print("x系数：",i_x)
                                print("y系数：",i_y)
                                pub_xy.data = [x,y]
                                data_pub.publish(pub_xy)
                            
                        

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                cv2.putText(im0, ("fps:" + str(1/(t3 - t2))), (10, 20), 0, 0.7, (0, 255, 0), 2)
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
        
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            print("------------")
            

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    
    def parse_opt(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
        self.parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
        self.parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        self.parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        self.parser.add_argument('--conf-thres', type=float, default=0.70, help='confidence threshold')
        self.parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        self.parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        self.parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--view-img', action='store_true', help='show results')
        self.parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        self.parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        self.parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        self.parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        self.parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        self.parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', action='store_true', help='augmented inference')
        self.parser.add_argument('--visualize', action='store_true', help='visualize features')
        self.parser.add_argument('--update', action='store_true', help='update all models')
        self.parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        self.parser.add_argument('--name', default='exp', help='save results to project/name')
        self.parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        self.parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        self.parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        self.parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        self.parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = self.parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt

    def Command_frame(self,data):
            self.com = data.data
            if self.com ==1:
                self.per_label = 'Red_apples'
            elif self.com == 2:
                self.per_label ='Green_apples'
            elif self.com == 3:
                self.per_label ='Orange'
            elif self.com == 4:
                self.per_label ='mango'

    def main(self,opt):
        check_requirements(exclude=('tensorboard', 'thop'))
        self.run(**vars(opt))


if __name__ == "__main__":
    d = detect()
    opt = d.parse_opt()
    d.main(opt)
