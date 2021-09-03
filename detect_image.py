import argparse
import time
from pathlib import Path
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device


def detect(opt):
    source, weights, view_img, imgsz = opt.source, opt.weights, opt.view_img, opt.img_size
    save_dir = 'output'
    file_name = Path(source).name

    # Load model
    device = select_device(opt.device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
  
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    img = cv2.imread(source)
    original_image = img.copy()

    t0 = time.time()
    
    # img = cv2.resize(img, (416, 416))
    img = letterbox(img)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    save_path = os.path.join(save_dir, file_name)   
    # Process detections
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img size to original_image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                plot_one_box(xyxy, original_image, label=label, color=colors(c, True), line_thickness=2)
                    
    if view_img:
        cv2.imshow("result", original_image)
        cv2.waitKey(0)  # 1 millisecond

    # Save results (image with detections)
    cv2.imwrite(save_path, original_image)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='input/fruits.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    args = parser.parse_args()
    
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        detect(opt=args)
