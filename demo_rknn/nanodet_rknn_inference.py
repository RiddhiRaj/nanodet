#!/usr/bin/env python3
"""
NanoDet RKNN Inference for RK3588

Usage:
    # On RK3588 device:
    python3 nanodet_rknn.py \
        --model nanodet-plus-m-1.5x_416.rknn \
        --config config/nanodet-plus-m-1.5x_416.yml \
        --image test.jpg \
        --score_threshold 0.35

"""

import argparse
import os
import time

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("Warning: rknnlite not found. If running on RK3588, install rknn-toolkit2-lite")
    print("If testing on PC, use RKNN toolkit2 instead")
    try:
        from rknn.api import RKNN as RKNNLite
    except ImportError:
        print("ERROR: Neither rknnlite nor rknn found!")
        exit(1)


class NanoDetRKNN:
    """NanoDet inference using RKNN"""

    def __init__(
        self,
        model_path,
        input_shape=(416, 416),
        num_classes=80,
        reg_max=7,
        strides=[8, 16, 32, 64],
        score_threshold=0.35,
        nms_threshold=0.6,
    ):
        self.model_path = model_path
        self.input_shape = input_shape  # (width, height)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Initialize RKNN
        self.rknn = RKNNLite()

        # Load RKNN model
        print(f"Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print("Load RKNN model failed!")
            exit(ret)

        # Init runtime environment
        print("Init runtime environment...")
        # For RK3588, cores can be: 0, 1, 2 (NPU cores)
        # Use core_mask to specify which NPU cores to use
        # 0: auto, 1: core0, 2: core1, 4: core2, 7: all cores
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            print("Init runtime environment failed!")
            exit(ret)

        print("RKNN model loaded successfully!")

        # Generate anchor points
        self.generate_anchors()

    def generate_anchors(self):
        """Generate anchor points for each stride level and build combined arrays."""
        anchors_per_level = []
        strides_per_level = []
        for stride in self.strides:
            h = self.input_shape[1] // stride
            w = self.input_shape[0] // stride
            shift_x = np.arange(0, w) + 0.5
            shift_y = np.arange(0, h) + 0.5
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            anchor = np.stack([shift_x, shift_y], axis=-1).reshape(-1, 2)
            anchors_per_level.append(anchor)
            strides_per_level.append(np.full(anchor.shape[0], stride, dtype=np.float32))

        # Concatenate all levels (matches the ONNX output order)
        self.all_anchors = np.concatenate(anchors_per_level, axis=0)   # (N, 2)
        self.all_strides = np.concatenate(strides_per_level, axis=0)   # (N,)

    def preprocess(self, image):
        """
        Preprocess image for inference.
        RKNN handles normalization internally based on mean/std values
        configured during conversion (nanodet2rknn.py passes BGR-order
        mean/std from the YAML config, so the input must stay BGR).
        """
        input_h, input_w = self.input_shape[1], self.input_shape[0]
        img = cv2.resize(image, (input_w, input_h))

        # Keep BGR order — nanodet2rknn.py configures RKNN with BGR mean/std
        # so no cv2.cvtColor conversion is needed.
        return img

    def postprocess(self, outputs, image_shape):
        """
        Post-process RKNN outputs to get bounding boxes.

        The ONNX model (_forward_onnx in NanoDetPlusHead) produces a **single**
        tensor of shape (1, N, num_classes + 4*(reg_max+1)) where:
          - N = total anchor points across all stride levels
          - cls scores already have sigmoid applied
          - values are [cls_score_0 .. cls_score_C-1, reg_0 .. reg_4*(R+1)-1]

        Args:
            outputs: List with one tensor from RKNN inference
            image_shape: Original image shape (h, w, c)

        Returns:
            List of detections: [[x1, y1, x2, y2, score, class_id], ...]
        """
        # Single-tensor output from the ONNX / RKNN model
        output = outputs[0]

        # Remove batch dimension if present
        if output.ndim == 3:
            output = output[0]  # (N, num_classes + 4*(reg_max+1))

        # Split classification scores (sigmoid already applied) and regression
        cls_scores = output[:, :self.num_classes]
        reg_preds = output[:, self.num_classes:]

        # Per-anchor best class
        scores = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)

        # Early score filter
        valid_mask = scores > self.score_threshold
        if not valid_mask.any():
            return []

        valid_scores = scores[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        valid_anchors = self.all_anchors[valid_mask]
        valid_strides = self.all_strides[valid_mask]
        valid_reg = reg_preds[valid_mask]

        # Decode distance predictions  ->  softmax  ->  expectation
        dis_pred = valid_reg.reshape(-1, 4, self.reg_max + 1)
        dis_pred = self._softmax(dis_pred, axis=-1)
        project = np.arange(0, self.reg_max + 1, dtype=np.float32).reshape(1, 1, -1)
        dis_pred = (dis_pred * project).sum(axis=-1)  # (M, 4)

        # Decode boxes: anchor centre ± distance, scaled by stride
        x1 = (valid_anchors[:, 0] - dis_pred[:, 0]) * valid_strides
        y1 = (valid_anchors[:, 1] - dis_pred[:, 1]) * valid_strides
        x2 = (valid_anchors[:, 0] + dis_pred[:, 2]) * valid_strides
        y2 = (valid_anchors[:, 1] + dis_pred[:, 3]) * valid_strides

        # Rescale to original image dimensions
        scale_x = image_shape[1] / self.input_shape[0]
        scale_y = image_shape[0] / self.input_shape[1]

        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = self.multiclass_nms(boxes, valid_scores, valid_class_ids)

        # Format output
        detections = []
        for idx in keep:
            detections.append(
                [
                    boxes[idx, 0],
                    boxes[idx, 1],
                    boxes[idx, 2],
                    boxes[idx, 3],
                    valid_scores[idx],
                    valid_class_ids[idx],
                ]
            )

        return detections

    def _softmax(self, x, axis=-1):
        """Compute softmax"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def multiclass_nms(self, boxes, scores, class_ids):
        """Multi-class NMS"""
        keep = []
        unique_classes = np.unique(class_ids)

        for c in unique_classes:
            class_mask = class_ids == c
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]

            # Simple NMS
            nms_keep = self.nms(class_boxes, class_scores, self.nms_threshold)
            keep.extend(class_indices[nms_keep])

        return keep

    def nms(self, boxes, scores, iou_threshold):
        """Non-maximum suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

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

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def inference(self, image):
        """
        Run inference on an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            detections: List of [x1, y1, x2, y2, score, class_id]
            inference_time: Inference time in seconds
        """
        # Preprocess
        img = self.preprocess(image)

        # Inference
        start_time = time.time()
        outputs = self.rknn.inference(inputs=[img])
        inference_time = time.time() - start_time

        # Postprocess
        detections = self.postprocess(outputs, image.shape)

        return detections, inference_time

    def release(self):
        
        self.rknn.release()


def visualize(image, detections, class_names, score_threshold=0.35):
    """Draw bounding boxes on image"""
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{class_names[class_id]}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

    return image


def parse_args():
    parser = argparse.ArgumentParser(description="NanoDet RKNN Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to RKNN model")
    parser.add_argument(
        "--image", type=str, default=None, help="Path to input image or video"
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default="416,416",
        help="Input size as WxH (default: 416,416)",
    )
    parser.add_argument(
        "--num_classes", type=int, default=80, help="Number of classes (default: 80)"
    )
    parser.add_argument(
        "--reg_max", type=int, default=7, help="Reg max value (default: 7)"
    )
    parser.add_argument(
        "--strides",
        type=str,
        default="8,16,32,64",
        help="Strides (default: 8,16,32,64)",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.35,
        help="Score threshold (default: 0.35)",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.6,
        help="NMS threshold (default: 0.6)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save result image"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark (100 iterations)"
    )

    return parser.parse_args()


# COCO class names
COCO_CLASSES = [
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


def main():
    args = parse_args()

    # Parse input size and strides
    input_size = tuple(map(int, args.input_size.split(",")))
    strides = list(map(int, args.strides.split(",")))

    # Initialize detector
    detector = NanoDetRKNN(
        model_path=args.model,
        input_shape=input_size,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        strides=strides,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )

    if args.benchmark:
        # Benchmark mode
        print("\nRunning benchmark (100 iterations)...")
        dummy_img = np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
        
        times = []
        for i in range(100):
            _, inference_time = detector.inference(dummy_img)
            times.append(inference_time)
            if i % 10 == 0:
                print(f"Iteration {i}/100")

        times = times[10:]  # Remove first 10 for warmup
        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        print(f"\nBenchmark Results:")
        print(f"  Average inference time: {avg_time*1000:.2f}ms")
        print(f"  FPS: {fps:.2f}")
        
    elif args.image:
        # Image inference
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return

        print(f"\nRunning inference on: {args.image}")
        image = cv2.imread(args.image)

        if image is None:
            print(f"Error: Failed to load image: {args.image}")
            return

        # Inference
        detections, inference_time = detector.inference(image)

        print(f"\nInference time: {inference_time*1000:.2f}ms ({1/inference_time:.2f} FPS)")
        print(f"Detected {len(detections)} objects")

        # Visualize
        result_img = visualize(image, detections, COCO_CLASSES, args.score_threshold)

        # Display or save
        if args.save:
            output_path = args.image.replace(".", "_nanodet.")
            cv2.imwrite(output_path, result_img)
            print(f"Result saved to: {output_path}")
        else:
            cv2.imshow("NanoDet RKNN", result_img)
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Error: Please specify --image or --benchmark")

    # Release
    detector.release()
    print("\nDone!")


if __name__ == "__main__":
    main()
