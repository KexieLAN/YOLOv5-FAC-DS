import argparse

import json
import os
import pathlib
import sys, time

import numpy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QCheckBox, QTextBrowser, QTextEdit, QFileDialog, QWidget, \
    QGraphicsPixmapItem, QGraphicsScene, QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.Qt import QThread

import platform
import torch

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
import torch.backends.cudnn as cudnn

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

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker


class trackThread(QThread):
    opt = None
    sta = None
    preImg = pyqtSignal(numpy.ndarray)
    counts = pyqtSignal(list)
    isOver = pyqtSignal(str)
    msgs = pyqtSignal(str)
    logs = pyqtSignal(str)

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
        self.tracking(yolo_weights=pathlib.Path(self.opt['weights']), source=self.opt['source'],
                      save_vid=self.opt['save-vid'])

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
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            show_vid=False,  # show results
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
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
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
        # ----------------------------------------------------------------------------------

        # 对数据集进行枚举操作，检测每一帧/图片
        # enumerate()函数返回 索引 与 对应位数据
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            if not self.sta:
                break
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
                    Timing = "Timing: "
                seen += 1
                # 如果输入源为网络摄像头
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
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
                                    FaceingTimeCount[str(id)] += 1

                                    # time_id = (FaceingTimeCount[str(id)] / v_frames) * (v_frames / v_fps)
                                    time_id = (FaceingTimeCount[str(id)] / v_fps)

                                    time_id = strftime("%H:%M:%S", gmtime(time_id))

                                    label = None if hide_labels else (f'{id} {names[c]} {time_id}' if hide_conf else \
                                                                          (
                                                                              f'{id} {conf:.2f} {time_id}' if hide_class else
                                                                              f'{id} {names[c]} {conf:.2f} {time_id}'))
                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                    Timing += '\n' + f"{id} {time_id}"
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


class trackUi(QWidget):
    ui = None
    fileChos, filePathShow = None, None
    filepath = 0
    modelPath = None
    trackingThread = None
    stopBtn = None
    startBtn = None
    modelSelect = None
    isSave = False

    def __init__(self):
        super().__init__()
        self.saveVid = None
        self.textBox = None
        self.dir_faceCountLabel = None
        self.personCountLabel = None
        self.vidShow = None
        self.modelPathShow = None
        self.ui_init()

    def ui_init(self):
        self.ui = uic.loadUi('track_ui.ui')

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
        self.saveVid.stateChanged.connect(self.checkBox)

    def getVidPath(self):
        filename = QFileDialog.getOpenFileName(self.ui,
                                               '获取视频源',
                                               './',
                                               "Video \
                                               (*.asf *.avi *.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv')")
        self.filePathShow.setText(filename[0])
        if filename[0] == '':
            self.filepath = 0
            print('123')
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
                self.trackingThread = trackThread()  # 创建线程
                self.trackingThread.preImg.connect(self.showVid)
                self.trackingThread.counts.connect(self.showCount)
                self.trackingThread.isOver.connect(self.resetThreadSta)
                self.trackingThread.logs.connect(self.showLog)
                opts = json.dumps({
                    "weights": self.modelPath,
                    "source": self.filepath,
                    "save-vid": self.isSave
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

    def resetThreadSta(self, ms):
        self.trackingThread = None

    def showLog(self, msg):
        self.textBox.append(msg + "<br>")
        self.textBox.repaint()

    def checkBox(self):
        if self.saveVid.isChecked():
            self.isSave = True
        else:
            self.isSave = False


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = trackUi()
    w.ui.show()

    app.exec()
