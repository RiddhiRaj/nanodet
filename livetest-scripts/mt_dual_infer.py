#!/usr/bin/env python3
"""
Multi-threaded dual-model RKNN inference pipeline.

pipeline:
capture -> preprocess -> dispatch -> inference workers (2xYOLO + 1xNanoDet)
-> raw priority queues -> postprocess workers -> post priority queues
-> sequencer (strict order + timeout) -> display

"""

import argparse
import itertools
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np

from dual_infer import (
    COCO_CLASSES_NANODET,
    COCO_CLASSES_YOLO,
    NanoDetRKNN,
    RKNNLite,
    YOLOv8RKNN,
    draw_detections,
    parse_size,
)


SENTINEL = object() # unique sentinel value for signaling shutdown or end-of-stream in queues


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    processed_image: np.ndarray
    transform_meta: Dict[str, Any]


@dataclass
class RawResult:
    frame_id: int
    timestamp: float
    model_type: str
    raw_output: Any
    transform_meta: Dict[str, Any]
    infer_ms: float
    model_ref: Any


@dataclass
class PostResult:
    frame_id: int
    model_type: str
    drawn_image: np.ndarray
    infer_ms: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="MT Dual Inference: 2xYOLOv8 workers + 1xNanoDet worker on 3 NPU cores"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Video path, camera index (e.g. 0), or IP webcam URL",
    )
    parser.add_argument("--nanodet_model", type=str, required=True, help="NanoDet RKNN model")
    parser.add_argument("--yolo_model", type=str, required=True, help="YOLOv8 RKNN model")

    parser.add_argument(
        "--nanodet_input_size",
        type=str,
        default="416,416",
        help="NanoDet input as W,H",
    )
    parser.add_argument(
        "--nanodet_num_classes",
        type=int,
        default=80,
        help="NanoDet classes count",
    )
    parser.add_argument(
        "--nanodet_reg_max",
        type=int,
        default=7,
        help="NanoDet reg_max",
    )
    parser.add_argument(
        "--nanodet_strides",
        type=str,
        default="8,16,32,64",
        help="NanoDet strides list",
    )
    parser.add_argument(
        "--nanodet_score_threshold",
        type=float,
        default=0.45,
        help="NanoDet score threshold",
    )
    parser.add_argument(
        "--nanodet_nms_threshold",
        type=float,
        default=0.6,
        help="NanoDet NMS threshold",
    )

    parser.add_argument(
        "--yolo_size",
        type=str,
        default="640",
        help="YOLO input size: int or H,W",
    )
    parser.add_argument(
        "--yolo_quantized",
        action="store_true",
        help="Set for quantized YOLO model",
    )
    parser.add_argument(
        "--yolo_score_threshold",
        type=float,
        default=0.45,
        help="YOLO score threshold",
    )
    parser.add_argument(
        "--yolo_nms_threshold",
        type=float,
        default=0.45,
        help="YOLO NMS threshold",
    )

    parser.add_argument(
        "--queue_size",
        type=int,
        default=8,
        help="Bounded input queue size (drop-oldest policy)",
    )
    parser.add_argument(
        "--timeout_ms",
        type=float,
        default=60.0,
        help="Per-frame fusion timeout in milliseconds",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=30,
        help="Print stats every N displayed frames",
    )
    return parser.parse_args()

def put_drop_oldest(q: queue.Queue, item: Any):
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def put_priority(q: queue.PriorityQueue, seq: itertools.count, frame_id: int, payload: Any):
    q.put((frame_id, next(seq), payload))


def safe_put_sentinel(q: queue.Queue, count: int):
    for _ in range(count):
        while True:
            try:
                q.put_nowait(SENTINEL)
                break
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass


def parse_input_source(input_arg: str):
    if input_arg.isdigit():
        return int(input_arg)
    return input_arg


def yolo_preprocess(image: np.ndarray, input_size, quantized: bool):
    target_h, target_w = input_size
    h, w = image.shape[:2]
    ratio = min(target_h / h, target_w / w)
    new_h, new_w = int(h * ratio), int(w * ratio)

    resized = cv2.resize(image, (new_w, new_h))
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    left = pad_w // 2

    letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    letterboxed[top : top + new_h, left : left + new_w] = resized

    inp = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    if quantized:
        inp = np.expand_dims(inp, axis=0)
    else:
        inp = np.expand_dims(inp.astype(np.float32) / 255.0, axis=0)

    meta = {"ratio": ratio, "pad": (left, top)}
    return inp, meta


def nanodet_preprocess(image: np.ndarray, input_shape):
    input_w, input_h = input_shape
    resized = cv2.resize(image, (input_w, input_h))
    inp = np.expand_dims(resized, axis=0)
    meta = {"image_shape": image.shape}
    return inp, meta


def compose_tiles(yolo_img: np.ndarray, nano_img: np.ndarray):
    if yolo_img.shape[:2] != nano_img.shape[:2]:
        nano_img = cv2.resize(nano_img, (yolo_img.shape[1], yolo_img.shape[0]))

    left = yolo_img.copy()
    right = nano_img.copy()

    cv2.putText(
        left,
        "YOLOv8n",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        right,
        "NanoDet",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    return np.hstack([left, right])


def fallback_tile(base_frame: np.ndarray, title: str):
    tile = base_frame.copy()
    cv2.putText(
        tile,
        f"{title} (stale/no result)",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    return tile


def capture_worker(
    args,
    q_in_yolo: queue.Queue,
    q_in_nano: queue.Queue,
    frame_store: Dict[int, Dict[str, Any]],
    frame_lock: threading.Lock,
    timeout_sec: float,
    capture_state: Dict[str, Any],
    capture_done: threading.Event,
    shutdown_event: threading.Event,
):
    source = parse_input_source(args.input)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Capture] ERROR: cannot open input: {args.input}")
        capture_state["total_frames"] = 0
        capture_done.set()
        shutdown_event.set()
        safe_put_sentinel(q_in_yolo, 2)
        safe_put_sentinel(q_in_nano, 1)
        return

    frame_id = 0
    yolo_size = parse_size(args.yolo_size)
    nanodet_input = tuple(map(int, args.nanodet_input_size.split(",")))

    while not shutdown_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break

        ts = time.time()
        with frame_lock:
            frame_store[frame_id] = {
                "frame": frame,
                "timestamp": ts,
                "deadline": ts + timeout_sec,
            }

        yolo_inp, yolo_meta = yolo_preprocess(frame, yolo_size, args.yolo_quantized)
        nano_inp, nano_meta = nanodet_preprocess(frame, nanodet_input)

        put_drop_oldest(
            q_in_yolo,
            FramePacket(
                frame_id=frame_id,
                timestamp=ts,
                processed_image=yolo_inp,
                transform_meta=yolo_meta,
            ),
        )
        put_drop_oldest(
            q_in_nano,
            FramePacket(
                frame_id=frame_id,
                timestamp=ts,
                processed_image=nano_inp,
                transform_meta=nano_meta,
            ),
        )
        frame_id += 1

    cap.release()
    capture_state["total_frames"] = frame_id
    capture_done.set()
    safe_put_sentinel(q_in_yolo, 2)
    safe_put_sentinel(q_in_nano, 1)
    print(f"[Capture] stopped. total_frames={frame_id}")


def inference_worker(
    name: str,
    model_type: str,
    model_ref: Any,
    in_q: queue.Queue,
    raw_pq: queue.PriorityQueue,
    pq_seq: itertools.count,
    shutdown_event: threading.Event,
):
    while True:
        if shutdown_event.is_set():
            break
        try:
            item = in_q.get(timeout=0.05)
        except queue.Empty:
            continue

        if item is SENTINEL:
            put_priority(raw_pq, pq_seq, 10**15, SENTINEL)
            break

        packet: FramePacket = item
        try:
            t0 = time.time()
            outputs = model_ref.rknn.inference(
                inputs=[packet.processed_image],
                data_format="nhwc",
            )
            infer_ms = (time.time() - t0) * 1000.0
            if outputs is None:
                outputs = []

            raw = RawResult(
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                model_type=model_type,
                raw_output=outputs,
                transform_meta=packet.transform_meta,
                infer_ms=infer_ms,
                model_ref=model_ref,
            )
            put_priority(raw_pq, pq_seq, packet.frame_id, raw)
        except Exception as exc:
            print(f"[{name}] inference error on frame {packet.frame_id}: {exc}")

    print(f"[{name}] stopped")


def postprocess_worker(
    name: str,
    model_type: str,
    raw_pq: queue.PriorityQueue,
    post_pq: queue.PriorityQueue,
    pq_seq: itertools.count,
    frame_store: Dict[int, Dict[str, Any]],
    frame_lock: threading.Lock,
    score_threshold: float,
    sentinel_target: int,
    shutdown_event: threading.Event,
):
    sentinel_seen = 0
    while True:
        if shutdown_event.is_set():
            break
        try:
            _, _, payload = raw_pq.get(timeout=0.05)
        except queue.Empty:
            continue

        if payload is SENTINEL:
            sentinel_seen += 1
            if sentinel_seen >= sentinel_target:
                put_priority(post_pq, pq_seq, 10**15, SENTINEL)
                break
            continue

        raw: RawResult = payload
        with frame_lock:
            frame_entry = frame_store.get(raw.frame_id)

        if frame_entry is None:
            continue

        vis = frame_entry["frame"].copy()
        try:
            if model_type == "yolo":
                detections = raw.model_ref.postprocess(
                    raw.raw_output,
                    raw.transform_meta["ratio"],
                    raw.transform_meta["pad"],
                )
                draw_detections(
                    vis,
                    detections,
                    COCO_CLASSES_YOLO,
                    (255, 128, 0),
                    label_prefix="[Y] ",
                    score_threshold=score_threshold,
                )
            else:
                detections = raw.model_ref.postprocess(
                    raw.raw_output,
                    raw.transform_meta["image_shape"],
                )
                draw_detections(
                    vis,
                    detections,
                    COCO_CLASSES_NANODET,
                    (0, 255, 0),
                    label_prefix="[N] ",
                    score_threshold=score_threshold,
                )

            post = PostResult(
                frame_id=raw.frame_id,
                model_type=model_type,
                drawn_image=vis,
                infer_ms=raw.infer_ms,
            )
            put_priority(post_pq, pq_seq, raw.frame_id, post)
        except Exception as exc:
            print(f"[{name}] postprocess error on frame {raw.frame_id}: {exc}")

    print(f"[{name}] stopped")


def fusion_worker(
    yolo_post_pq: queue.PriorityQueue,
    nano_post_pq: queue.PriorityQueue,
    display_q: queue.Queue,
    frame_store: Dict[int, Dict[str, Any]],
    frame_lock: threading.Lock,
    capture_state: Dict[str, Any],
    capture_done: threading.Event,
    shutdown_event: threading.Event,
    log_interval: int,
):
    next_expected = 0
    buffer: Dict[int, Dict[str, PostResult]] = {}
    last_good = {"yolo": None, "nano": None}
    yolo_done = False
    nano_done = False

    displayed = 0
    started = time.time()
    yolo_ms_hist = []
    nano_ms_hist = []

    while True:
        if shutdown_event.is_set():
            break

        drained = False
        while True:
            try:
                _, _, payload = yolo_post_pq.get_nowait()
            except queue.Empty:
                break
            drained = True
            if payload is SENTINEL:
                yolo_done = True
                continue
            post: PostResult = payload
            buffer.setdefault(post.frame_id, {})["yolo"] = post

        while True:
            try:
                _, _, payload = nano_post_pq.get_nowait()
            except queue.Empty:
                break
            drained = True
            if payload is SENTINEL:
                nano_done = True
                continue
            post: PostResult = payload
            buffer.setdefault(post.frame_id, {})["nano"] = post

        progressed = False
        while True:
            with frame_lock:
                frame_entry = frame_store.get(next_expected)

            if frame_entry is None:
                break

            slot = buffer.get(next_expected, {})
            yolo_res = slot.get("yolo")
            nano_res = slot.get("nano")
            
            if yolo_res is not None and nano_res is not None:
                yolo_img = yolo_res.drawn_image
                nano_img = nano_res.drawn_image
            elif time.time() > frame_entry["deadline"]:
                yolo_img = (
                    yolo_res.drawn_image
                    if yolo_res is not None
                    else (
                        last_good["yolo"].copy()
                        if last_good["yolo"] is not None
                        else fallback_tile(frame_entry["frame"], "YOLOv8n")
                    )
                )
                nano_img = (
                    nano_res.drawn_image
                    if nano_res is not None
                    else (
                        last_good["nano"].copy()
                        if last_good["nano"] is not None
                        else fallback_tile(frame_entry["frame"], "NanoDet")
                    )
                )
            else:
                break

            composed = compose_tiles(yolo_img, nano_img)
            put_drop_oldest(display_q, composed)

            if yolo_res is not None:
                last_good["yolo"] = yolo_res.drawn_image
                yolo_ms_hist.append(yolo_res.infer_ms)
            if nano_res is not None:
                last_good["nano"] = nano_res.drawn_image
                nano_ms_hist.append(nano_res.infer_ms)

            with frame_lock:
                frame_store.pop(next_expected, None)
            buffer.pop(next_expected, None)
            next_expected += 1
            displayed += 1
            progressed = True

            if displayed % max(log_interval, 1) == 0:
                elapsed = max(time.time() - started, 1e-6)
                out_fps = displayed / elapsed
                yolo_avg = float(np.mean(yolo_ms_hist[-log_interval:])) if yolo_ms_hist else 0.0
                nano_avg = float(np.mean(nano_ms_hist[-log_interval:])) if nano_ms_hist else 0.0
                print(
                    f"[Stats] displayed={displayed} out_fps={out_fps:.2f} "
                    f"yolo_inf_ms(avg)={yolo_avg:.2f} nano_inf_ms(avg)={nano_avg:.2f}"
                )

        total_frames = capture_state.get("total_frames")
        if (
            capture_done.is_set()
            and yolo_done
            and nano_done
            and total_frames is not None
            and next_expected >= total_frames
        ):
            break

        if not drained and not progressed:
            time.sleep(0.002)

    safe_put_sentinel(display_q, 1)
    print("[Fusion] stopped")


def display_worker(display_q: queue.Queue, shutdown_event: threading.Event):
    win = "Dual Stream | Left: YOLOv8n  Right: NanoDet  (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        if shutdown_event.is_set() and display_q.empty():
            break
        try:
            item = display_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if item is SENTINEL:
            break

        cv2.imshow(win, item)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            shutdown_event.set()
            break
    cv2.destroyAllWindows()
    print("[Display] stopped")


def main():
    args = parse_args()

    timeout_sec = args.timeout_ms / 1000.0
    q_in_yolo: queue.Queue = queue.Queue(maxsize=max(args.queue_size, 1))
    q_in_nano: queue.Queue = queue.Queue(maxsize=max(args.queue_size, 1))
    yolo_raw_pq: queue.PriorityQueue = queue.PriorityQueue()
    nano_raw_pq: queue.PriorityQueue = queue.PriorityQueue()
    yolo_post_pq: queue.PriorityQueue = queue.PriorityQueue()
    nano_post_pq: queue.PriorityQueue = queue.PriorityQueue()
    display_q: queue.Queue = queue.Queue(maxsize=2)

    frame_store: Dict[int, Dict[str, Any]] = {}
    frame_lock = threading.Lock()
    capture_state: Dict[str, Any] = {"total_frames": None}

    capture_done = threading.Event()
    shutdown_event = threading.Event()
    pq_seq = itertools.count()

    core0 = getattr(RKNNLite, "NPU_CORE_0", None)
    core1 = getattr(RKNNLite, "NPU_CORE_1", None)
    core2 = getattr(RKNNLite, "NPU_CORE_2", None)
    print("[Cores] YOLO worker-0 -> CORE_0, YOLO worker-1 -> CORE_1, NanoDet -> CORE_2")

    nanodet_input = tuple(map(int, args.nanodet_input_size.split(",")))
    nanodet_strides = list(map(int, args.nanodet_strides.split(",")))
    yolo_size = parse_size(args.yolo_size)

    yolo_a = YOLOv8RKNN(
        model_path=args.yolo_model,
        input_size=yolo_size,
        score_threshold=args.yolo_score_threshold,
        nms_threshold=args.yolo_nms_threshold,
        quantized=args.yolo_quantized,
        core_mask=core0,
    )
    yolo_b = YOLOv8RKNN(
        model_path=args.yolo_model,
        input_size=yolo_size,
        score_threshold=args.yolo_score_threshold,
        nms_threshold=args.yolo_nms_threshold,
        quantized=args.yolo_quantized,
        core_mask=core1,
    )
    nanodet = NanoDetRKNN(
        model_path=args.nanodet_model,
        input_shape=nanodet_input,
        num_classes=args.nanodet_num_classes,
        reg_max=args.nanodet_reg_max,
        strides=nanodet_strides,
        score_threshold=args.nanodet_score_threshold,
        nms_threshold=args.nanodet_nms_threshold,
        core_mask=core2,
    )

    threads = [
        threading.Thread(
            target=capture_worker,
            name="capture",
            args=(
                args,
                q_in_yolo,
                q_in_nano,
                frame_store,
                frame_lock,
                timeout_sec,
                capture_state,
                capture_done,
                shutdown_event,
            ),
        ),
        threading.Thread(
            target=inference_worker,
            name="yolo-inf-0",
            args=("yolo-inf-0", "yolo", yolo_a, q_in_yolo, yolo_raw_pq, pq_seq, shutdown_event),
        ),
        threading.Thread(
            target=inference_worker,
            name="yolo-inf-1",
            args=("yolo-inf-1", "yolo", yolo_b, q_in_yolo, yolo_raw_pq, pq_seq, shutdown_event),
        ),
        threading.Thread(
            target=inference_worker,
            name="nano-inf-0",
            args=("nano-inf-0", "nano", nanodet, q_in_nano, nano_raw_pq, pq_seq, shutdown_event),
        ),
        threading.Thread(
            target=postprocess_worker,
            name="yolo-post",
            args=(
                "yolo-post",
                "yolo",
                yolo_raw_pq,
                yolo_post_pq,
                pq_seq,
                frame_store,
                frame_lock,
                args.yolo_score_threshold,
                2,
                shutdown_event,
            ),
        ),
        threading.Thread(
            target=postprocess_worker,
            name="nano-post",
            args=(
                "nano-post",
                "nano",
                nano_raw_pq,
                nano_post_pq,
                pq_seq,
                frame_store,
                frame_lock,
                args.nanodet_score_threshold,
                1,
                shutdown_event,
            ),
        ),
        threading.Thread(
            target=fusion_worker,
            name="fusion",
            args=(
                yolo_post_pq,
                nano_post_pq,
                display_q,
                frame_store,
                frame_lock,
                capture_state,
                capture_done,
                shutdown_event,
                args.log_interval,
            ),
        ),
        threading.Thread(
            target=display_worker,
            name="display",
            args=(display_q, shutdown_event),
        ),
    ]

    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        shutdown_event.set()
        safe_put_sentinel(q_in_yolo, 2)
        safe_put_sentinel(q_in_nano, 1)
        safe_put_sentinel(display_q, 1)
    finally:
        yolo_a.release()
        yolo_b.release()
        nanodet.release()
        print("Done.")


if __name__ == "__main__":
    main()
