# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
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
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 显示计数
            Count = "Counts: "
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # 扩充变量Count
                    Count += '\n' + f"{names[int(c)]}: {n}"

                # Write results
                # 仅在设置了保存文本(save_txt)，保存图像(save_img)和保存目标(save_corp)时生效
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
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # 流媒体格式的文件，如视频流，网络视频流
            im0 = annotator.result()
            # view_img参数，展示当前的识别进度和识别结果
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 用于展示特定结果与计数的代码
                # 图片，文字，位置，文字类型，字体大小，颜色，粗细
                # cv2.putText(im0, f"{n} {names[int(c)]}{'s' * (n > 1)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            # (0, 0, 255), 2)
                y0, dy = 30, 40
                for dus, txt in enumerate(Count.split('\n')):
                    y = y0 + dus * dy
                    cv2.putText(im0, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, 2)
                #     cv2.putText(im0, Count, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                # 左上角显示检测标签和数量
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 保存检测结果
            if save_img:
                if not view_img:
                    # # 需用循环的方式显示多行,因为cv2.putText对换行转义符'\n'显示为'?'
                    y0, dy = 30, 40
                    for dus, txt in enumerate(Count.split('\n')):
                        y = y0 + dus * dy
                        cv2.putText(im0, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, 2)
                    #     cv2.putText(im0, Count, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    # 左上角显示检测标签和数量
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
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # 权重文件
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    # 文件源，通常为文件夹，也可单独指定文件
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # 配置文件，为yaml格式
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # 执行检测前，会将图片resize成*640*640大小，可以修改
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 置信度阈值，用于限制或提高网络的敏感度（或者说将更多或更少可疑目标端上来）
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    # 调节IoU阈值
    # 通过NMS（非极大值抑制）来清理boundingbox   https://blog.csdn.net/mechleechan/article/details/88365039
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    # 最大目标检测数(置信度排序)，额，随意
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 设备选择，默认采用CUDA设备，可以手动指定多GPU
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 一张一张展示图片，这功能不好使                       启动该功能的开关，命令行中为“ --view_img ”
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 检测结果存储为.txt文件    “ --save_txt ”   文件会存储标签与box坐标
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 与save-txt配合使用，保存置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 裁切目标
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 别存了，生成个文件夹意思一下
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 检测的分类类别，可以指定检测特定类别
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 跨类别检测
    # agnostic-nms是跨类别nms，比如待检测图像中有一个长得很像排球的足球，pt文件的分类中有足球和排球两种，
    # 那在识别时这个足球可能会被同时框上2个框：一个是足球，一个是排球。
    # 开启agnostic-nms后，那只会框出一个框
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 增强方式,可能为马赛克数据增强
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 特征图可视化，生成两个文件，图片与numpy
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 去除优化器
    parser.add_argument('--update', action='store_true', help='update all models')
    # 检测结果保存路径
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 检测结果保存名称
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 这个参数的意思就是每次预测模型的结果是否保存在原来的文件夹，
    # 如果指定了这个参数的话，那么本次预测的结果还是保存在上一次保存的文件夹里；如果不指定就是每次预测结果保存一个新的文件夹下。
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 调节box的粗细，可用来防遮挡
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    # 隐藏标签
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 隐藏置信度
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 使用16位精度(半精度推理)。一般在training中关闭half
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 是否使用OpenCV DNN进行ONNX推理
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 设置视频文件的步进？
    # 以一定的帧间隔读取视频？
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


# 可能有用的计时代码
# import cv2
# import torch
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import non_max_suppression, scale_coords
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device
#
# # 设置参数
# weights = 'yolov5s.pt'  # 模型权重路径
# img_size = 640  # 输入图像尺寸
# conf_thres = 0.4  # 置信度阈值
# iou_thres = 0.5  # NMS阈值
# device = select_device('')  # 使用CPU或GPU
# model = attempt_load(weights, map_location=device).autoshape()  # 加载模型并自适应输入形状
# names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称
#
# # 加载视频
# cap = cv2.VideoCapture('test.mp4')
# fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
#
# # 初始化变量
# prev_boxes = None  # 上一帧的检测结果
# prev_time = None  # 上一帧的时间
# total_time = 0  # 检测目标总时间
#
# while True:
#     ret, img0 = cap.read()
#     if not ret:
#         break
#
#     # 图像预处理
#     img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (img_size, img_size))
#     img = torch.from_numpy(img).to(device).float() / 255.0
#     img = img.permute(2, 0, 1).unsqueeze(0)
#
#     # 目标检测
#     pred = model(img)[0]
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
#
#     # 处理检测结果
#     boxes = []
#     for i, det in enumerate(pred):
#         if len(det):
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
#             for *xyxy, conf, cls in reversed(det):
#                 label = f'{names[int(cls)]} {conf:.2f}'
#                 plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=3)
#                 boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
#
#     # 计算检测目标时间
#     if prev_boxes is not None and len(boxes) > 0:
#         overlap = 0
#         for box in boxes:
#             for prev_box in prev_boxes:
#                 x1 = max(box[0], prev_box[0])
#                 y1 = max(box[1], prev_box[1])
#                 x2 = min(box[2], prev_box[2])
#                 y2 = min(box[3], prev_box[3])
#                 if x2 > x1 and y2 > y1:
#                     overlap += (x2 - x1) * (y2 - y1)
#         if overlap > 0:
#             curr_time = 1 / fps * (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
#             if prev_time is not None:
#                 total_time += curr_time - prev_time
#             prev_time = curr_time
#
#     # 更新变量
#     prev_boxes = boxes
#
#     # 显示结果
#     cv2.imshow('result', img0)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# # 输出结果
# print(f'Total time: {total_time:.2f}s')
# print(f'Time per object: {total_time / len(prev_boxes):.2f}s')
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()