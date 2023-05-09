import json
import os
import pathlib

import numpy
from PyQt5 import uic
from PyQt5.Qt import QThread
from PyQt5.QtCore import pyqtSignal, QRectF, QPoint, Qt, QEvent, QRegExp
from PyQt5.QtGui import QPixmap, QImage, QIcon, QStandardItemModel, QStandardItem, QCloseEvent, QRegExpValidator
from PyQt5.QtWidgets import QApplication, QLabel, QFileDialog, QWidget, \
    QGraphicsPixmapItem, QGraphicsScene, QMessageBox, QHeaderView, QAbstractItemView, QTableView, QMainWindow, QDialog, \
    QGridLayout

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch

# ----------------------------------------------------------------------------------
# 添加时间计算相关的包
from time import strftime, gmtime

# ----------------------------------------------------------------------------------


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, cv2,
                                  check_imshow, increment_path, strip_optimizer, colorstr, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker


class trackThread(QThread):
    opt = None
    sta = None
    preImg = pyqtSignal(numpy.ndarray)
    counts = pyqtSignal(list)
    isOver = pyqtSignal(bool)
    msgs = pyqtSignal(str)
    logs = pyqtSignal(str)
    timing = pyqtSignal(dict)
    zoneInfo = None

    # zoneInfo = None

    def __init__(self):
        super().__init__()

    def setSta(self, sta):
        self.sta = sta

    def setOpt(self, opt):
        self.opt = json.loads(opt)

    def run(self) -> None:
        # while True:
        #     if self.sta:
        #         print("收到")
        #         print(self.opt)
        #     else:
        #         print("未收到")
        #         break
        #     time.sleep(1)
        print(self.opt['weights'])
        print(self.opt['source'])
        print(self.opt['save-vid'])
        print(self.opt['zone'])
        print(self.opt['vid_stride'])
        if self.opt['zone']:
            self.zoneInfo = self.opt['zoneInfo']
        self.tracking(yolo_weights=pathlib.Path(self.opt['weights']), source=self.opt['source'],
                      save_vid=self.opt['save-vid'], zone=self.opt['zone'], zonInfo=self.zoneInfo,
                      vid_stride=self.opt['vid_stride'])

    @torch.no_grad()
    def tracking(
            self,
            source='0',
            yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
            reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
            tracking_method='strongsort',
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.45,  # confidence threshold
            iou_thres=0.25,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            show_vid=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            save_vid=False,  # save confidences in --save-txt labels
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/track',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=1,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
            zone=False,
            zonInfo=None,
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        if not isinstance(yolo_weights, list):  # single yolo model
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(yolo_weights[0]).stem
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + reid_weights.stem
        save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 模型载入，yolo的检测模型，sort的专用追迹模型
        device = select_device(device)
        model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        # 载入数据：网页捕获和常规文件
        # ----------------------------------------------------------------------------------
        # temp_cap:暂时读取视频，获取信息，读取后需要释放掉
        if webcam:
            temp_cap = cv2.VideoCapture(0)
        else:
            temp_cap = cv2.VideoCapture(source)
        v_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频的帧率
        # 尝试将帧率与vid_stride关联起来
        v_fps = int(temp_cap.get(cv2.CAP_PROP_FPS) / vid_stride)
        if v_fps == 0:
            v_fps = 30 / vid_stride
        # 获取视频的总帧数(已弃用)
        # v_frames = temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        v_w = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        v_h = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # 释放视频
        temp_cap.release()
        # ----------------------------------------------------------------------------------
        if webcam:  # 如果设备是网络摄像机
            show_vid = check_imshow()
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            nr_sources = len(dataset)
        else:
            # 测试，增加vid_stride
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # Create as many strong sort instances as there are video sources
        # 创建和视频源数量一致的追踪器实例
        tracker_list = []
        for i in range(nr_sources):
            # 根据模式，创建相应的追踪器的实例
            # 详见trackers/multi_tracker_zoo.py
            tracker = create_tracker(tracking_method, reid_weights, device, half)
            # 追踪器加入表中
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                # 对未执行预热的模型的模型进行预热
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        outputs = [None] * nr_sources

        # Run tracking
        # 运行追踪，开始进行追踪
        # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        # dt：存储信息用的[准备时间，识别时间，净化时间，追迹时间],seen算是一种索引？计数？
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

        # ----------------------------------------------------------------------------------
        # 字典，通过id作为索引，记录下每一个id的出现帧数
        FaceingTimeCount = {}
        FaceingTiming = {}
        # ----------------------------------------------------------------------------------

        # 对数据集进行枚举操作，检测每一帧/图片
        # enumerate()函数返回 索引 与 对应位数据
        vc_temp = None
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            if not self.sta and frame_idx != 0:
                break
            vc_temp = vid_cap
            # ----------------------------------------------------------------------------------------------------------
            if zone:
                h1 = zonInfo[0][1] / v_h
                w1 = zonInfo[0][0] / v_w

                h2 = zonInfo[1][1] / v_h
                w2 = zonInfo[1][0] / v_w

                h3 = zonInfo[2][1] / v_h
                w3 = zonInfo[2][0] / v_w

                h4 = zonInfo[3][1] / v_h
                w4 = zonInfo[3][0] / v_w

                if webcam:
                    for b in range(0, im.shape[0]):
                        mask = numpy.zeros([im[b].shape[1], im[b].shape[2]], dtype=numpy.uint8)
                        # mask[round(img[b].shape[1] * hl1):img[b].shape[1], round(img[b].shape[2] * wl1):img[b].shape[2]] = 255
                        pts = numpy.array([[int(im[b].shape[2] * w1), int(im[b].shape[1] * w1)],  # pts1
                                           [int(im[b].shape[2] * w2), int(im[b].shape[1] * h2)],  # pts2
                                           [int(im[b].shape[2] * w3), int(im[b].shape[1] * h3)],  # pts3
                                           [int(im[b].shape[2] * w4), int(im[b].shape[1] * h4)]], numpy.int32)
                        mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
                        imgc = im[b].transpose((1, 2, 0))
                        imgc = cv2.add(imgc, numpy.zeros(numpy.shape(imgc), dtype=numpy.uint8), mask=mask)
                        # cv2.imshow('1',imgc)
                        im[b] = imgc.transpose((2, 0, 1))

                else:
                    mask = numpy.zeros([im.shape[1], im.shape[2]], dtype=numpy.uint8)
                    # mask[round(img.shape[1] * hl1):img.shape[1], round(img.shape[2] * wl1):img.shape[2]] = 255
                    pts = numpy.array([[int(im.shape[2] * w1), int(im.shape[1] * h1)],  # pts1
                                       [int(im.shape[2] * w2), int(im.shape[1] * h2)],  # pts2
                                       [int(im.shape[2] * w3), int(im.shape[1] * h3)],  # pts3
                                       [int(im.shape[2] * w4), int(im.shape[1] * h4)]], numpy.int32)
                    mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
                    im = im.transpose((1, 2, 0))
                    im = cv2.add(im, numpy.zeros(numpy.shape(im), dtype=numpy.uint8), mask=mask)
                    im = im.transpose((2, 0, 1))
            # ----------------------------------------------------------------------------------------------------------
            # GPU流转等待？
            t1 = time_sync()
            # 数据转入CDUA设备
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            # 记录准备时间
            dt[0] += t2 - t1

            # Inference
            # 推理阶段

            # 递增文件目录
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            # 交由YOLO进行识别
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            # 记录识别时间
            dt[1] += t3 - t2

            # Apply NMS
            # 应用非极大值抑制
            # 默认置信度为.25，IoU阈值为.45
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # 记录清理候选框的时间
            dt[2] += time_sync() - t3

            # Process detections
            # 处理检测到的目标
            # 枚举并检测目标
            for i, det in enumerate(pred):  # detections per image
                # 初始的，用于在帧上显示的“Count：”和后期附加检测总数的变量Count
                Count = "Counts: "
                counttuple = [torch.tensor(0), torch.tensor(0)]
                # 展示在Label上的时间可能不便于观察，故在视频右上角展示计时
                if vid_stride != 1:
                    Timing = f"Timing({vid_stride}x): "
                else:
                    Timing = f"Timing({vid_stride}x): "
                seen += 1
                # 如果输入源为网络摄像头
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    # ----------------------------------------------------------------------------------------------
                    if zone:
                        cv2.putText(im0, "Detection_Region", (int(im0.shape[1] * w1 - 5), int(im0.shape[0] * h1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (255, 255, 0), 2, cv2.LINE_AA)

                        pts = np.array([[int(im0.shape[1] * w1), int(im0.shape[0] * h1)],  # pts1
                                        [int(im0.shape[1] * w2), int(im0.shape[0] * h2)],  # pts2
                                        [int(im0.shape[1] * w3), int(im0.shape[0] * h3)],  # pts3
                                        [int(im0.shape[1] * w4), int(im0.shape[0] * h4)]], np.int32)  # pts4
                        # pts = pts.reshape((-1, 1, 2))
                        zeros = np.zeros((im0.shape), dtype=np.uint8)
                        mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                        im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                        cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
                    # ----------------------------------------------------------------------------------------------
                    p = Path(p)  # to Path
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    # getattr函数：
                    #     从对象中获取命名属性;getattr（x， 'y'） 等价于 x.y。
                    #     当给定默认参数时，当属性未给出时返回
                    #     存在;如果没有它，在这种情况下会引发异常。
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    if zone:
                        cv2.putText(im0, "Detection_Region", (int(im0.shape[1] * w1 - 5), int(im0.shape[0] * h1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (255, 255, 0), 2, cv2.LINE_AA)
                        pts = numpy.array([[int(im0.shape[1] * w1), int(im0.shape[0] * h1)],  # pts1
                                           [int(im0.shape[1] * w2), int(im0.shape[0] * h2)],  # pts2
                                           [int(im0.shape[1] * w3), int(im0.shape[0] * h3)],  # pts3
                                           [int(im0.shape[1] * w4), int(im0.shape[0] * h4)]], numpy.int32)  # pts4
                        # pts = pts.reshape((-1, 1, 2))
                        zeros = numpy.zeros((im0.shape), dtype=numpy.uint8)
                        mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                        im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)

                        cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
                    p = Path(p)  # to Path
                    # video file
                    # 识别是否为视频文件  （以视频格式后缀名）
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    # 不是视频的话，就是一文件夹的图片
                    else:
                        txt_file_name = p.parent.name  # get folder name containing current img
                        save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                curr_frames[i] = im0

                txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop

                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                # hasattr()函数用于判断对象是否包含对应的属性
                # 用于检测完成后的追迹
                if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                    if (prev_frames[i] is not None) and (curr_frames[i] is not None):  # camera motion compensation
                        tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if (det is not None) and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                    # Print results
                    # 在命令行中打印结果
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        # 将识别到的结果加入到Count变量，后期以实现展示
                        # 以特殊字符’\n‘进行分割
                        Count += '\n' + f"{names[int(c)]}: {n}"
                        if names[int(c)] == 'person':
                            counttuple[0] = n
                        elif names[int(c)] == 'dir_face':
                            counttuple[1] = n
                    self.counts.emit(counttuple)

                    # pass detections to strongsort
                    # 将检测的目标传送给StrongSort，进行追迹
                    # t4存储GPU启动时序
                    t4 = time_sync()
                    # strongsort的更新函数，见trackers/strong_sort/strong_sort.py
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                    # t5存储GPU结束时序
                    t5 = time_sync()
                    # dt的最后参数为GPU时间（识别时间）
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    # 检测框绘制和可视化
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                            # 0，1，2，3位为检测框的中心位置与长宽（xywh）
                            bboxes = output[0:4]
                            # 4位为目标的ID
                            id = output[4]
                            # 5位为类别
                            cls = output[5]

                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                            # 保存视频/保存截图/展示视频
                            if save_vid or save_crop or show_vid:  # Add bbox to image 在图像上附加检测框
                                c = int(cls)  # integer class   类别
                                id = int(id)  # integer id  ID
                                # -------------------------------------------------------------------------
                                # 大概可以在此处插入->计时代码，在将计时结果输出到标签上
                                # 计时方式：记录目标ID出现的帧的数量，用获得的帧的数量比上视频帧的数量，
                                #         用这个比值乘上视频时间长度，可以大体估算出时间
                                # 检测类别是否为我们需要的
                                if names[c] == 'dir_face':
                                    # 将新id写入字典
                                    if str(id) not in FaceingTimeCount.keys():
                                        FaceingTimeCount[str(id)] = 0
                                        FaceingTiming[str(id)] = ''
                                    FaceingTimeCount[str(id)] += 1

                                    # time_id = (FaceingTimeCount[str(id)] / v_frames) * (v_frames / v_fps)
                                    time_id = (FaceingTimeCount[str(id)] / v_fps) * vid_stride

                                    time_id = strftime("%H:%M:%S", gmtime(time_id))

                                    label = None if hide_labels else (f'{id} {names[c]} {time_id}' if hide_conf else \
                                                                          (
                                                                              f'{id} {conf:.2f} {time_id}' if hide_class else
                                                                              f'{id} {names[c]} {conf:.2f} {time_id}'))
                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                    Timing += '\n' + f"{id} {time_id}"
                                    FaceingTiming[str(id)] = time_id
                                # -------------------------------------------------------------------------
                                # 不必要的就不输出时间
                                else:
                                    label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                          (
                                                                              f'{id} {conf:.2f} ' if hide_class else
                                                                              f'{id} {names[c]} {conf:.2f}'))
                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                # cv2.putText(im0, f"{n} {names[int(c)]}{'s' * (n > 1)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                #             (0, 0, 255), 2)
                                # #
                                # y0, dy = 30, 40
                                # for dus, txt in enumerate(Count.split('\n')):
                                #     y = y0 + dus * dy
                                #     cv2.putText(im0, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, 2)
                                # #
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                        c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                    sstemp = f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)'
                    LOGGER.info(sstemp)
                    self.logs.emit(sstemp)

                else:
                    # strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')
                    self.logs.emit('No detections')
                self.timing.emit(FaceingTiming)
                # Stream results
                # 流式媒体的结果
                im0 = annotator.result()
                # 播放识别过程
                if show_vid:
                    # cv2.putText(im0, f"{n} {names[int(c)]}{'s' * (n > 1)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    #             (0, 0, 255), 2)
                    # 生成计数，并且该计数将会伴随该图片保存（设置了--save-vid参数）
                    # y0, dy = 30, 40
                    # # 用’\n‘来分离字符串，换行展示
                    # for dus, txt in enumerate(Count.split('\n')):
                    #     y = y0 + dus * dy
                    #     cv2.putText(im0, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, 2)
                    # # -------------------------------------------------------------------------------------------
                    # for dus, txt in enumerate(Timing.split('\n')):
                    #     y = y0 + dus * dy
                    #     cv2.putText(im0, txt, (v_width - 300, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, 2)
                    # # -------------------------------------------------------------------------------------------
                    # cv2.imshow(str(p), im0)
                    self.preImg.emit(im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # 保存视频，用来存储图像和识别的结果
                if save_vid:
                    # 之前未展示计数，则生成视频时附加上
                    # cv2.putText(im0, f"{n} {names[int(c)]}{'s' * (n > 1)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    #             (0, 0, 255), 2)
                    y0, dy = 30, 40
                    for dus, txt in enumerate(Count.split('\n')):
                        y = y0 + dus * dy
                        cv2.putText(im0, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, 2)
                        # --------------------------------------------------------------------------------------
                    for dus, txt in enumerate(Timing.split('\n')):
                        y = y0 + dus * dy
                        cv2.putText(im0, txt, (v_width - 300, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, 2)
                    # --------------------------------------------------------------------------------------
                    # 创建新的视频文件
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

                prev_frames[i] = curr_frames[i]
        vc_temp.release()
        # Print results
        # 在命令行中，打印识别结果
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_vid:
            s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
        # vid_cap.release()
        # 进程结束
        self.isOver.emit(True)


class trackUi(QMainWindow):
    ui = None
    fileChos, filePathShow = None, None
    filepath = '0'
    modelPath = None
    trackingThread = None
    stopBtn = None
    startBtn = None
    modelSelect = None
    isSave = False

    def __init__(self):
        super().__init__()
        self.vidStrider = None
        # 区域选择
        self.we = None
        self.zone = None
        self.zoneInfo = None
        # 区域选择复选框
        self.zoneChos = None
        self.tableModel = None
        self.tableShow = None
        self.saveVid = None
        self.textBox = None
        self.dir_faceCountLabel = None
        self.personCountLabel = None
        self.vidShow = None
        self.modelPathShow = None
        self.vidStride = 1
        self.ui_init()

    def ui_init(self):
        self.ui = uic.loadUi('track_ui.ui')
        self.ui.setWindowIcon(QIcon('PYApplication_16x.ico'))

        self.fileChos = self.ui.file
        self.fileChos.clicked.connect(self.getVidPath)

        self.filePathShow = self.ui.filePath

        self.modelSelect = self.ui.modelSelect
        self.modelSelect.clicked.connect(self.getModPath)

        self.modelPathShow = self.ui.modelPath

        self.startBtn = self.ui.startBtn
        self.startBtn.clicked.connect(self.startYoloThread)

        self.stopBtn = self.ui.stopBtn
        self.stopBtn.clicked.connect(self.stopYoloThread)

        self.vidShow = self.ui.graphicsView

        self.personCountLabel = self.ui.personCount
        self.dir_faceCountLabel = self.ui.dir_faceCount

        self.textBox = self.ui.textBrowser

        self.saveVid = self.ui.saveVid
        self.saveVid.stateChanged.connect(self.saveVidCheckBox)

        self.zoneChos = self.ui.zoneCheckBox
        self.zoneChos.stateChanged.connect(self.zoneCheckBox)

        self.tableShow = self.ui.tableView
        self.initTableShow()

        self.vidStrider = self.ui.vidStrider
        reg = QRegExp('[0-9]+$')
        validator = QRegExpValidator()
        validator.setRegExp(reg)
        self.vidStrider.setValidator(validator)

    def getVidPath(self):
        filename = QFileDialog.getOpenFileName(self.ui,
                                               '获取视频源',
                                               './',
                                               "Video \
                                               (*.asf *.avi *.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv')")
        self.filePathShow.setText(filename[0])
        if filename[0] == '' or len(filename[0]) == 0:
            self.filepath = 0
        else:
            self.filepath = filename[0]

    def getModPath(self):
        modelPath = QFileDialog.getOpenFileName(self.ui,
                                                '选择识别模型',
                                                './',
                                                "模型 (*.pt)")
        self.modelPathShow.setText(modelPath[0])
        self.modelPath = modelPath[0]

    def startYoloThread(self):
        if self.trackingThread is None:
            if self.modelPath != '' and self.modelPath is not None:
                if self.zone:
                    self.we = zoneChosShowDialog(ffp=self.filepath)
                    # self.we.getFilePath(self.filePath)
                    self.we.posC.connect(self.getPos)
                    self.we.exec()
                self.tableClearn()
                self.trackingThread = trackThread()  # 创建线程
                self.trackingThread.preImg.connect(self.showVid)
                self.trackingThread.counts.connect(self.showCount)
                self.trackingThread.isOver.connect(self.resetThreadSta)
                self.trackingThread.logs.connect(self.showLog)
                self.trackingThread.timing.connect(self.tableRe)
                self.vidStride = self.getVidStride()
                opts = json.dumps({
                    "weights": self.modelPath,
                    "source": self.filepath,
                    "save-vid": self.isSave,
                    "zone": self.zone,
                    "zoneInfo": self.zoneInfo,
                    "vid_stride": self.vidStride
                })
                self.trackingThread.setOpt(opts)
                self.trackingThread.daemon = True
                self.trackingThread.start()  # 开始线程
                self.trackingThread.setSta(True)
            else:
                QMessageBox.warning(self, "警告", "请选择一个可用的模型", QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.Yes)
        else:
            print("线程已存在，请等待执行完成")

    def stopYoloThread(self):
        if self.trackingThread is None:
            print('进程不存在，任务未开始')
        else:
            if self.trackingThread.isRunning():
                print("进程存在，正在结束")
                self.trackingThread.setSta(False)
                self.trackingThread = None
            else:
                print("进程已自然结束")
                self.trackingThread = None

    def showVid(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        zoomscale = min(380 / y, 720 / x)  # 图片放缩尺度
        frame = QImage(img, x, y, x * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)  # 创建像素图元
        item.setScale(zoomscale)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.vidShow.setScene(scene)  # 将场景添加至视图

    def showCount(self, cps):
        self.personCountLabel.setText(str(cps[0].item()))
        self.dir_faceCountLabel.setText(str(cps[1].item()))

    def resetThreadSta(self, ts):
        if ts:
            self.trackingThread = None

    def showLog(self, msg):
        self.textBox.append(msg + "<br>")
        self.textBox.repaint()

    def saveVidCheckBox(self):
        if self.saveVid.isChecked():
            self.isSave = True
        else:
            self.isSave = False

    def zoneCheckBox(self):
        if self.zoneChos.isChecked():
            self.zone = True
        else:
            self.zone = False

    def initTableShow(self):
        # 创建一个 0行3列 的标准模型
        self.tableModel = QStandardItemModel(0, 2)
        # 设置表头标签
        self.tableModel.setHorizontalHeaderLabels(['正脸ID', '时间'])
        self.tableShow.setModel(self.tableModel)
        self.tableShow.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 所有列自动拉伸，充满界面

        self.tableShow.setSelectionMode(QAbstractItemView.SingleSelection)  # 设置只能选中整行
        self.tableShow.setEditTriggers(QTableView.NoEditTriggers)  # 不可编辑
        self.tableShow.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置只能选中一行

        # self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置只能选中整行
        # self.tableView.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 设置只能选中多行

    def tableClearn(self):
        # 会全部清空，包括那个标准表头
        self.tableModel.clear()
        # 所以重新设置标准表头 自己将一下代码注释 尝试
        self.tableModel.setHorizontalHeaderLabels(['正脸ID', '时间'])

    def tableRe(self, dic):
        self.tableClearn()
        for item in dic.items():
            C1 = QStandardItem('%s' % str(item[0]))
            C2 = QStandardItem('%s' % str(item[1]))
            self.tableModel.appendRow([C1, C2])

    def getVidStride(self):
        vs = self.vidStrider.text()
        if vs == '' or len(vs) == 0:
            vs = 1
        elif int(vs) > 10:
            vs = 10
        elif int(vs) < 1:
            vs = 1
        return int(vs)

    def getPos(self, ls):
        self.zoneInfo = ls


class zoneChosShowDialog(QDialog):
    posC = pyqtSignal(list)
    filePath = None
    posList = [(0, 0), (0, 0), (0, 0), (0, 0)]
    posCount = 0

    def __init__(self, ffp):
        QWidget.__init__(self)
        self.filePath = ffp
        self.resize(1280, 720)
        self.setWindowTitle('区域选的')
        layout = QGridLayout(self)

        # self.getImageButton = QPushButton('OK')
        # layout.addWidget(self.getImageButton)
        # self.getImageButton.clicked.connect(self.resimac)

        self.resim1 = QLabel()
        layout.addWidget(self.resim1)
        self.resim1.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # I'm assuming the following...
        self.resim1.setScaledContents(True)
        self.resim1.setFixedSize(1024, 768)

        # install an event filter to "capture" mouse events (amongst others)
        self.resim1.installEventFilter(self)
        print(self.filePath)
        self.resimac()

    def closeEvent(self, a0: QCloseEvent) -> None:
        """
        重写QWidget类的closrEvent方法，在窗口被关闭的时候自动触发
        """
        super().closeEvent(a0)  # 先添加父类的方法，以免导致覆盖父类方法（这是重点！！！）
        self.posC.emit(self.posList)

    # def close(self):
    #     self.posC.emit(self.posList)
    #     super().close()

    def resimac(self):
        if self.filePath is None:
            return
        if self.filePath.isdigit():
            self.filePath = int(self.filePath)
        cap = cv2.VideoCapture(self.filePath)
        if cap.isOpened():
            print('vid get' + str(self.filePath))
        ret, frame = cap.read()
        height, width, depth = frame.shape
        cvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
        self.resim1.setPixmap(QPixmap(cvimg))
        cap.release()

    def eventFilter(self, source, event):
        # if the source is our QLabel, it has a valid pixmap, and the event is
        # a left click, proceed in trying to get the event position
        if (source == self.resim1 and source.pixmap() and not source.pixmap().isNull() and
                event.type() == QEvent.MouseButtonPress and
                event.button() == Qt.LeftButton):
            self.getClickedPosition(event.pos())
        return super().eventFilter(source, event)

    def getClickedPosition(self, pos):
        # consider the widget contents margins
        contentsRect = QRectF(self.resim1.contentsRect())
        if pos not in contentsRect:
            # outside widget margins, ignore!
            return

        # adjust the position to the contents margins
        pos -= contentsRect.topLeft()

        pixmapRect = self.resim1.pixmap().rect()
        if self.resim1.hasScaledContents():
            x = pos.x() * pixmapRect.width() / contentsRect.width()
            y = pos.y() * pixmapRect.height() / contentsRect.height()
            pos = QPoint(x, y)
        else:
            align = self.resim1.alignment()
            # for historical reasons, QRect (which is based on integer values),
            # returns right() as (left+width-1) and bottom as (top+height-1),
            # and so their opposite functions set/moveRight and set/moveBottom
            # take that into consideration; using a QRectF can prevent that; see:
            # https://doc.qt.io/qt-5/qrect.html#right
            # https://doc.qt.io/qt-5/qrect.html#bottom
            pixmapRect = QRectF(pixmapRect)

            # the pixmap is not left aligned, align it correctly
            if align & Qt.AlignRight:
                pixmapRect.moveRight(contentsRect.x() + contentsRect.width())
            elif align & Qt.AlignHCenter:
                pixmapRect.moveLeft(contentsRect.center().x() - pixmapRect.width() / 2)
            # the pixmap is not top aligned (note that the default for QLabel is
            # Qt.AlignVCenter, the vertical center)
            if align & Qt.AlignBottom:
                pixmapRect.moveBottom(contentsRect.y() + contentsRect.height())
            elif align & Qt.AlignVCenter:
                pixmapRect.moveTop(contentsRect.center().y() - pixmapRect.height() / 2)

            if not pos in pixmapRect:
                # outside image margins, ignore!
                return
            # translate coordinates to the image position and convert it back to
            # a QPoint, which is integer based
            pos = (pos - pixmapRect.topLeft()).toPoint()

        # print('X={}, Y={}'.format(pos.x(), pos.y()))
        if self.posCount < 3:
            print('X={}, Y={}'.format(pos.x(), pos.y()))
            self.posList[self.posCount] = (pos.x(), pos.y())
            self.posCount += 1
        else:
            print('X={}, Y={}'.format(pos.x(), pos.y()))
            self.posList[self.posCount] = (pos.x(), pos.y())
            print(self.posList)
            self.close()

    def getFilePath(self, str):
        self.filePath = str


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = trackUi()
    w.ui.show()

    sys.exit(app.exec_())
