#!/usr/bin/env python3
"""
runs NanoDet on npu core 0 and YOLOv8n on npu core 1 simultaneously.

Usage:
    # Image inference
    python3 dual_infer.py \
        --input test.jpg \
        --nanodet_model nanodet-plus-m-1.5x_416.rknn \
        --yolo_model yolov8n.rknn

    # Video inference
    python3 dual_infer.py \
        --input video.mp4 \
        --nanodet_model nanodet.rknn \
        --yolo_model yolov8n.rknn \
        --save

    # Benchmark (500 iterations on one image)
    python3 dual_infer.py \
        --input test.jpg \
        --nanodet_model nanodet.rknn \
        --yolo_model yolov8n.rknn \
        --benchmark
"""

import argparse
import math
import os
import time
import threading

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("ERROR: rknnlite not found. Install rknn-toolkit2-lite for RK35xx.")
    exit(1)


#coco classes

COCO_CLASSES_NANODET = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

COCO_CLASSES_YOLO = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

class NanoDetRKNN:
    """nanodet inference on a specific NPU core."""

    def __init__(
        self,
        model_path,
        input_shape=(416, 416),
        num_classes=80,
        reg_max=7,
        strides=[8, 16, 32, 64],
        score_threshold=0.45,
        nms_threshold=0.6,
        core_mask=None,
    ):
        self.model_path = model_path
        self.input_shape = input_shape  # (width, height)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.rknn = RKNNLite()

        print(f"[NanoDet] Loading model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"[NanoDet] Load RKNN model failed! ret={ret}")

        try:
            if core_mask is not None:
                ret = self.rknn.init_runtime(core_mask=core_mask)
            else:
                ret = self.rknn.init_runtime()
        except TypeError:
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"[NanoDet] Init runtime failed! ret={ret}")

        print(f"[NanoDet] Model loaded successfully")
        self.generate_anchors()

    def generate_anchors(self):
        anchors_per_level = []
        strides_per_level = []
        for stride in self.strides:
            h = math.ceil(self.input_shape[1] / stride)
            w = math.ceil(self.input_shape[0] / stride)
            shift_x = np.arange(0, w, dtype=np.float32)
            shift_y = np.arange(0, h, dtype=np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            anchor = np.stack([shift_x, shift_y], axis=-1).reshape(-1, 2)
            anchors_per_level.append(anchor)
            strides_per_level.append(np.full(anchor.shape[0], stride, dtype=np.float32))
        self.all_anchors = np.concatenate(anchors_per_level, axis=0)
        self.all_strides = np.concatenate(strides_per_level, axis=0)

    def preprocess(self, image):
        input_h, input_w = self.input_shape[1], self.input_shape[0]
        img = cv2.resize(image, (input_w, input_h))
        img = np.expand_dims(img, axis=0)  # HWC -> NHWC (BGR kept)
        return img

    def postprocess(self, outputs, image_shape):
        output = outputs[0]
        if output.ndim == 3:
            output = output[0]

        cls_scores = output[:, :self.num_classes]
        reg_preds = output[:, self.num_classes:]

        scores = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)

        valid_mask = scores > self.score_threshold
        if not valid_mask.any():
            return []

        valid_scores = scores[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        valid_anchors = self.all_anchors[valid_mask]
        valid_strides = self.all_strides[valid_mask]
        valid_reg = reg_preds[valid_mask]

        dis_pred = valid_reg.reshape(-1, 4, self.reg_max + 1)
        dis_pred = self._softmax(dis_pred, axis=-1)
        project = np.arange(0, self.reg_max + 1, dtype=np.float32).reshape(1, 1, -1)
        dis_pred = (dis_pred * project).sum(axis=-1)

        x1 = (valid_anchors[:, 0] - dis_pred[:, 0]) * valid_strides
        y1 = (valid_anchors[:, 1] - dis_pred[:, 1]) * valid_strides
        x2 = (valid_anchors[:, 0] + dis_pred[:, 2]) * valid_strides
        y2 = (valid_anchors[:, 1] + dis_pred[:, 3]) * valid_strides

        scale_x = image_shape[1] / self.input_shape[0]
        scale_y = image_shape[0] / self.input_shape[1]
        x1 *= scale_x; x2 *= scale_x
        y1 *= scale_y; y2 *= scale_y

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._multiclass_nms(boxes, valid_scores, valid_class_ids)

        detections = []
        for idx in keep:
            detections.append([
                boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3],
                valid_scores[idx], valid_class_ids[idx],
            ])
        return detections

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _multiclass_nms(self, boxes, scores, class_ids):
        keep = []
        for c in np.unique(class_ids):
            mask = class_ids == c
            indices = np.where(mask)[0]
            nms_keep = self._nms(boxes[mask], scores[mask])
            keep.extend(indices[nms_keep])
        return keep

    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= self.nms_threshold)[0] + 1]
        return keep

    def inference(self, image):
        img = self.preprocess(image)
        t0 = time.time()
        outputs = self.rknn.inference(inputs=[img], data_format="nhwc")
        inference_time = time.time() - t0
        if outputs is None:
            raise RuntimeError("[NanoDet] RKNN inference returned None")
        detections = self.postprocess(outputs, image.shape)
        return detections, inference_time

    def release(self):
        self.rknn.release()

class YOLOv8RKNN:
    """YOLOv8 inference on a specific NPU core."""

    def __init__(
        self,
        model_path,
        input_size=(640, 640),
        num_classes=80,
        score_threshold=0.25,
        nms_threshold=0.45,
        quantized=False,
        core_mask=None,
    ):
        self.model_path = model_path
        self.input_size = input_size  # (height, width)
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.quantized = quantized

        self.rknn = RKNNLite()

        print(f"[YOLOv8] Loading model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"[YOLOv8] Load RKNN model failed! ret={ret}")

        try:
            if core_mask is not None:
                ret = self.rknn.init_runtime(core_mask=core_mask)
            else:
                ret = self.rknn.init_runtime()
        except TypeError:
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"[YOLOv8] Init runtime failed! ret={ret}")

        print(f"[YOLOv8] Model loaded successfully ({'quantized' if quantized else 'float'})")

    def letterbox(self, im):
        target_h, target_w = self.input_size
        h, w = im.shape[:2]
        r = min(target_h / h, target_w / w)
        new_h, new_w = int(h * r), int(w * r)
        im = cv2.resize(im, (new_w, new_h))
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top = pad_h // 2
        left = pad_w // 2
        out = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        out[top:top + new_h, left:left + new_w] = im
        return out, r, (left, top)

    def _dfl(self, position):
        n, c, h, w = position.shape
        mc = c // 4
        y = position.reshape(n, 4, mc, h, w)
        y = np.exp(y - np.max(y, axis=2, keepdims=True))
        y /= np.sum(y, axis=2, keepdims=True)
        acc = np.arange(mc).reshape(1, 1, mc, 1, 1)
        return (y * acc).sum(2)

    def _box_process(self, position):
        input_h, input_w = self.input_size
        feat_h, feat_w = position.shape[2:]
        grid_x, grid_y = np.meshgrid(np.arange(feat_w), np.arange(feat_h))
        grid = np.stack((grid_x, grid_y), axis=0).reshape(1, 2, feat_h, feat_w)
        stride_h = input_h / feat_h
        stride_w = input_w / feat_w
        stride = np.array([stride_w, stride_h]).reshape(1, 2, 1, 1)
        position = self._dfl(position)
        box1 = grid + 0.5 - position[:, 0:2]
        box2 = grid + 0.5 + position[:, 2:4]
        return np.concatenate((box1 * stride, box2 * stride), axis=1)

    def _organize_outputs(self, outputs):
        boxes = {}
        classes = {}
        for out in outputs:
            _, c, h, w = out.shape
            key = (h, w)
            if c == 64 or c == 4:
                boxes[key] = out
            elif c == self.num_classes:
                classes[key] = out
        ordered = []
        for k in sorted(boxes.keys(), key=lambda x: x[0]):
            ordered.append(boxes[k])
            ordered.append(classes[k])
        return ordered

    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.nms_threshold)[0] + 1]
        return keep

    def postprocess(self, outputs, ratio, pad):
        outputs = self._organize_outputs(outputs)
        boxes_all, cls_all, scores_all = [], [], []

        for i in range(3):
            box_out = outputs[i * 2]
            cls_out = outputs[i * 2 + 1]
            boxes = self._box_process(box_out)
            boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
            cls_out = cls_out.transpose(0, 2, 3, 1).reshape(-1, self.num_classes)
            cls_score = np.max(cls_out, axis=1)
            cls_id = np.argmax(cls_out, axis=1)
            mask = cls_score >= self.score_threshold
            boxes_all.append(boxes[mask])
            cls_all.append(cls_id[mask])
            scores_all.append(cls_score[mask])

        boxes = np.concatenate(boxes_all)
        classes = np.concatenate(cls_all)
        scores = np.concatenate(scores_all)

        if len(boxes) == 0:
            return []

        final_boxes, final_cls, final_scores = [], [], []
        for c in np.unique(classes):
            idx = np.where(classes == c)
            keep = self._nms(boxes[idx], scores[idx])
            final_boxes.append(boxes[idx][keep])
            final_cls.append(classes[idx][keep])
            final_scores.append(scores[idx][keep])

        if not final_boxes:
            return []

        boxes = np.concatenate(final_boxes)
        classes = np.concatenate(final_cls)
        scores = np.concatenate(final_scores)

        # Rescale to original image coordinates
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio

        detections = []
        for b, c, s in zip(boxes, classes, scores):
            detections.append([b[0], b[1], b[2], b[3], s, c])
        return detections

    def inference(self, image):
        img_lb, ratio, pad = self.letterbox(image)
        inp = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

        if self.quantized:
            inp = np.expand_dims(inp, axis=0)
        else:
            # Feed float models in NHWC to match RKNN runtime input layout and
            # avoid per-frame NCHW->NHWC conversion warnings.
            inp = np.expand_dims(inp.astype(np.float32) / 255.0, axis=0)

        t0 = time.time()
        outputs = self.rknn.inference(inputs=[inp], data_format="nhwc")
        inference_time = time.time() - t0

        if outputs is None:
            raise RuntimeError("[YOLOv8] RKNN inference returned None")

        detections = self.postprocess(outputs, ratio, pad)
        return detections, inference_time

    def release(self):
        self.rknn.release()


#viz
def draw_detections(image, detections, class_names, color, label_prefix="",
                    score_threshold=0.0, occupied_labels=None, prefer_top=True):
    """Draw detections and place labels so they do not cover each other."""
    if occupied_labels is None:
        occupied_labels = []

    img_h, img_w = image.shape[:2]

    def overlaps(r1, r2):
        return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])

    def find_label_rect(x1, y1, x2, y2, label_w, label_h):
        step = label_h + 4
        max_steps = 20

        top_y = y1 - label_h
        bottom_y = y2
        if not prefer_top:
            top_y, bottom_y = bottom_y, top_y

        candidates = []
        for i in range(max_steps):
            candidates.append((x1, top_y - i * step))
            candidates.append((x1, bottom_y + i * step))

        for rx, ry in candidates:
            rx = max(0, min(rx, img_w - label_w))
            ry = max(0, min(ry, img_h - label_h))
            rect = (rx, ry, rx + label_w, ry + label_h)
            if all(not overlaps(rect, occ) for occ in occupied_labels):
                return rect

        # Fallback: clamp to image and place at preferred side.
        rx = max(0, min(x1, img_w - label_w))
        ry = max(0, min(top_y, img_h - label_h))
        return (rx, ry, rx + label_w, ry + label_h)

    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        label = f"{label_prefix}{class_names[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_w = lsz[0] + 6
        label_h = lsz[1] + 10

        lx1, ly1, lx2, ly2 = find_label_rect(x1, y1, x2, y2, label_w, label_h)
        occupied_labels.append((lx1, ly1, lx2, ly2))

        cv2.rectangle(image, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(image, label, (lx1 + 3, ly2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image, occupied_labels


#threaded inference
def _infer_worker(model, image, result_dict, key):
    """Worker function for threaded inference."""
    try:
        detections, inf_time = model.inference(image)
        result_dict[key] = (detections, inf_time)
    except Exception as e:
        result_dict[key] = ([], 0.0)
        print(f"[{key}] Inference error: {e}")


def dual_inference(nanodet, yolov8, image):
    """Run both models in parallel on separate NPU cores via threads."""
    results = {}
    t_nano = threading.Thread(target=_infer_worker,
                              args=(nanodet, image, results, "nanodet"))
    t_yolo = threading.Thread(target=_infer_worker,
                              args=(yolov8, image, results, "yolov8"))
    t_nano.start()
    t_yolo.start()
    t_nano.join()
    t_yolo.join()
    return results["nanodet"], results["yolov8"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual RKNN Inference: NanoDet (Core 0) + YOLOv8 (Core 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image
  python3 dual_infer.py --input test.jpg \\
      --nanodet_model nanodet.rknn --yolo_model yolov8n.rknn

  # Video with save
  python3 dual_infer.py --input video.mp4 --save \\
      --nanodet_model nanodet.rknn --yolo_model yolov8n.rknn

  # Benchmark
  python3 dual_infer.py --input test.jpg --benchmark \\
      --nanodet_model nanodet.rknn --yolo_model yolov8n.rknn
        """,
    )

    # ---- Common ----
    common = parser.add_argument_group("Common")
    common.add_argument("--input", type=str, required=True,
                        help="Path to input image or video")
    common.add_argument("--benchmark", action="store_true",
                        help="Run 500-iteration benchmark on a single image")
    common.add_argument("--save", action="store_true",
                        help="Save output image/video instead of displaying")

    # ---- NanoDet ----
    nano = parser.add_argument_group("NanoDet")
    nano.add_argument("--nanodet_model", type=str, required=True,
                      help="Path to NanoDet RKNN model")
    nano.add_argument("--nanodet_input_size", type=str, default="416,416",
                      help="NanoDet input size as W,H (default: 416,416)")
    nano.add_argument("--nanodet_num_classes", type=int, default=80,
                      help="NanoDet number of classes (default: 80)")
    nano.add_argument("--nanodet_reg_max", type=int, default=7,
                      help="NanoDet reg_max (default: 7)")
    nano.add_argument("--nanodet_strides", type=str, default="8,16,32,64",
                      help="NanoDet strides (default: 8,16,32,64)")
    nano.add_argument("--nanodet_score_threshold", type=float, default=0.45,
                      help="NanoDet score threshold (default: 0.45)")
    nano.add_argument("--nanodet_nms_threshold", type=float, default=0.6,
                      help="NanoDet NMS threshold (default: 0.6)")

    # ---- YOLOv8 ----
    yolo = parser.add_argument_group("YOLOv8")
    yolo.add_argument("--yolo_model", type=str, required=True,
                      help="Path to YOLOv8 RKNN model")
    yolo.add_argument("--yolo_size", type=str, default="640",
                      help="YOLOv8 input size: int or H,W (default: 640)")
    yolo.add_argument("--yolo_quantized", action="store_true",
                      help="YOLOv8 model is quantized (uint8 input)")
    yolo.add_argument("--yolo_score_threshold", type=float, default=0.45,
                      help="YOLOv8 score threshold (default: 0.45)")
    yolo.add_argument("--yolo_nms_threshold", type=float, default=0.45,
                      help="YOLOv8 NMS threshold (default: 0.45)")

    return parser.parse_args()


def parse_size(size_str):
    """Parse 'H,W' or single int to (H, W) tuple."""
    if "," in size_str:
        h, w = map(int, size_str.split(","))
        return (h, w)
    s = int(size_str)
    return (s, s)

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        return

    # Resolve NPU core masks
    core0 = getattr(RKNNLite, "NPU_CORE_0", None)
    core1 = getattr(RKNNLite, "NPU_CORE_1", None)
    print(f"[Cores] NanoDet -> NPU_CORE_0, YOLOv8 -> NPU_CORE_1")

    # ---- Init NanoDet ----
    nanodet_input = tuple(map(int, args.nanodet_input_size.split(",")))
    nanodet_strides = list(map(int, args.nanodet_strides.split(",")))
    nanodet = NanoDetRKNN(
        model_path=args.nanodet_model,
        input_shape=nanodet_input,
        num_classes=args.nanodet_num_classes,
        reg_max=args.nanodet_reg_max,
        strides=nanodet_strides,
        score_threshold=args.nanodet_score_threshold,
        nms_threshold=args.nanodet_nms_threshold,
        core_mask=core0,
    )

    # ---- Init YOLOv8 ----
    yolo_size = parse_size(args.yolo_size)
    yolov8 = YOLOv8RKNN(
        model_path=args.yolo_model,
        input_size=yolo_size,
        score_threshold=args.yolo_score_threshold,
        nms_threshold=args.yolo_nms_threshold,
        quantized=args.yolo_quantized,
        core_mask=core1,
    )

    # ---- Determine input type ----
    ext = os.path.splitext(args.input)[1].lower()
    is_video = ext in (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm")

    if is_video:
        _run_video(args, nanodet, yolov8)
    else:
        _run_image(args, nanodet, yolov8)

    nanodet.release()
    yolov8.release()
    print("\nDone!")


def _run_image(args, nanodet, yolov8):
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Failed to load image: {args.input}")
        return

    if args.benchmark:
        _run_benchmark(nanodet, yolov8, image)
        return

    # Single-frame dual inference
    (nano_dets, nano_t), (yolo_dets, yolo_t) = dual_inference(nanodet, yolov8, image)

    nano_ms = nano_t * 1000
    yolo_ms = yolo_t * 1000
    print(f"\n{'Model':<12} {'Objects':<10} {'Time (ms)':<10}")
    print(f"{'─'*32}")
    print(f"{'NanoDet':<12} {len(nano_dets):<10} {nano_ms:<10.2f}")
    print(f"{'YOLOv8':<12} {len(yolo_dets):<10} {yolo_ms:<10.2f}")

    # Draw both sets of detections (green=NanoDet, blue=YOLOv8)
    vis = image.copy()
    _, occupied = draw_detections(
        vis, nano_dets, COCO_CLASSES_NANODET, (0, 255, 0),
        label_prefix="[N] ", score_threshold=args.nanodet_score_threshold, prefer_top=True
    )
    draw_detections(
        vis, yolo_dets, COCO_CLASSES_YOLO, (255, 128, 0),
        label_prefix="[Y] ", score_threshold=args.yolo_score_threshold,
        occupied_labels=occupied, prefer_top=False
    )

    if args.save:
        base, ext_name = os.path.splitext(args.input)
        out_path = f"{base}_dual{ext_name}"
        cv2.imwrite(out_path, vis)
        print(f"Result saved to: {out_path}")
    else:
        cv2.imshow("Dual Inference (Green=NanoDet, Blue=YOLOv8)", vis)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_video(args, nanodet, yolov8):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save:
        base, _ = os.path.splitext(args.input)
        out_path = f"{base}_dual.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"Saving output to: {out_path}")

    frame_idx = 0
    nano_times, yolo_times = [], []

    print(f"\nProcessing video: {w}x{h} @ {fps:.1f} FPS, {total} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        (nano_dets, nano_t), (yolo_dets, yolo_t) = dual_inference(nanodet, yolov8, frame)
        nano_ms = nano_t * 1000
        yolo_ms = yolo_t * 1000
        nano_times.append(nano_ms)
        yolo_times.append(yolo_ms)

        print(f"\rFrame {frame_idx}/{total}  "
              f"NanoDet: {nano_ms:.1f}ms ({len(nano_dets)} obj)  "
              f"YOLOv8: {yolo_ms:.1f}ms ({len(yolo_dets)} obj)", end="")

        vis = frame.copy()
        _, occupied = draw_detections(
            vis, nano_dets, COCO_CLASSES_NANODET, (0, 255, 0),
            label_prefix="[N] ", score_threshold=args.nanodet_score_threshold, prefer_top=True
        )
        draw_detections(
            vis, yolo_dets, COCO_CLASSES_YOLO, (255, 128, 0),
            label_prefix="[Y] ", score_threshold=args.yolo_score_threshold,
            occupied_labels=occupied, prefer_top=False
        )

        if writer:
            writer.write(vis)
        else:
            cv2.imshow("Dual Inference (Green=NanoDet, Blue=YOLOv8)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nUser quit.")
                break

    cap.release()
    if writer:
        writer.release()
    if not args.save:
        cv2.destroyAllWindows()

    print(f"\n\n{'Model':<12} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print(f"{'─'*48}")
    if nano_times:
        print(f"{'NanoDet':<12} {np.mean(nano_times):<12.2f} "
              f"{np.min(nano_times):<12.2f} {np.max(nano_times):<12.2f}")
    if yolo_times:
        print(f"{'YOLOv8':<12} {np.mean(yolo_times):<12.2f} "
              f"{np.min(yolo_times):<12.2f} {np.max(yolo_times):<12.2f}")


def _run_benchmark(nanodet, yolov8, image):
    num_iters = 500
    print(f"\nRunning benchmark: {num_iters} iterations (parallel on separate cores)...")

    nano_total = 0.0
    yolo_total = 0.0

    for i in range(num_iters):
        (_, nano_t), (_, yolo_t) = dual_inference(nanodet, yolov8, image)

        nano_total += nano_t * 1000  # ms
        yolo_total += yolo_t * 1000  # ms

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{num_iters} ...")

    nano_avg = nano_total / num_iters
    yolo_avg = yolo_total / num_iters

    print("\nBenchmark Results (avg)")
    print(f"{'Model':<12} {'avg (ms)':<12}")
    print(f"{'─'*24}")
    print(f"{'NanoDet':<12} {nano_avg:<12.2f}")
    print(f"{'YOLOv8':<12} {yolo_avg:<12.2f}")

if __name__ == "__main__":
    main()