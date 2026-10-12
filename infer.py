import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==================== 配置区 ====================
MODEL_PATH = "best.pt"              # 模型权重路径
VIDEO_PATH = "test.mp4"             # 待推理的视频路径
OUTPUT_PATH = "output_infer.mp4"    # 输出视频路径

CONF_THRESHOLD = 0.25               # 置信度阈值
IOU_THRESHOLD = 0.45                # NMS IoU 阈值
IMG_SIZE = 640                      # 推理输入尺寸

# 关键点颜色（BGR），4个角点用不同颜色区分
KEYPOINT_COLORS = [
    (0, 255, 0),    # 关键点0: 绿色
    (0, 0, 255),    # 关键点1: 红色
    (255, 0, 0),    # 关键点2: 蓝色
    (0, 255, 255),  # 关键点3: 黄色
]
KEYPOINT_RADIUS = 5
KEYPOINT_THICKNESS = -1  # -1 表示实心圆
# =================================================


def draw_detections(frame, result):
    """
    在帧上绘制检测框、类别、置信度和关键点
    返回绘制后的帧
    """
    annotated = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes
    keypoints = result.keypoints

    # 获取各字段的 numpy 数组
    xyxy = boxes.xyxy.cpu().numpy()          # (N, 4) 绝对像素坐标
    confs = boxes.conf.cpu().numpy()         # (N,)
    cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
    names = result.names                     # {id: name}

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = float(confs[i])
        cls_id = int(cls_ids[i])
        cls_name = names.get(cls_id, str(cls_id))

        # ---- 1. 绘制检测框 ----
        color = (0, 255, 0)  # 绿色框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # ---- 2. 绘制类别和置信度标签 ----
        label = f"{cls_name} {conf:.2f}"
        # 计算文字尺寸
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        # 标签背景（在框上方）
        label_y = max(y1 - 10, text_h + 10)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # 黑色文字
            2,
            cv2.LINE_AA,
        )

        # ---- 3. 绘制关键点 ----
        if keypoints is not None and keypoints.xy is not None:
            kpts = keypoints.xy.cpu().numpy()  # (N, K, 2)
            if i < len(kpts):
                person_kpts = kpts[i]  # (K, 2)
                for k in range(len(person_kpts)):
                    kx, ky = person_kpts[k]
                    if kx > 0 and ky > 0:  # 过滤掉无效点
                        kp_color = KEYPOINT_COLORS[k % len(KEYPOINT_COLORS)]
                        cv2.circle(
                            annotated,
                            (int(kx), int(ky)),
                            KEYPOINT_RADIUS,
                            kp_color,
                            KEYPOINT_THICKNESS,
                        )

    return annotated


def main():
    # ---- 1. 加载模型 ----
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        return

    print(f"[信息] 加载模型: {MODEL_PATH}")
    model = YOLO(str(model_path))

    # ---- 2. 打开视频 ----
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"[错误] 视频文件不存在: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {VIDEO_PATH}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[信息] 视频属性: {width}x{height}, {fps:.2f} FPS, "
          f"共 {total_frames} 帧")

    # ---- 3. 创建 VideoWriter 保存输出视频 ----
    # 使用 mp4v 编码，确保输出为可播放的 MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )
    if not out.isOpened():
        print(f"[错误] 无法创建输出视频: {OUTPUT_PATH}")
        cap.release()
        return

    # ---- 4. 逐帧推理与绘制 ----
    frame_idx = 0
    print("[信息] 开始推理...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 执行推理
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
        )

        result = results[0]

        # 绘制检测框、类别、置信度、关键点
        annotated = draw_detections(frame, result)

        # 写入输出视频
        out.write(annotated)

        # 实时显示（可选，按 q 退出）
        cv2.imshow("YOLO26n Armor Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[信息] 用户中断")
            break

        # 进度打印
        if frame_idx % 30 == 0:
            print(f"  已处理 {frame_idx}/{total_frames} 帧")

    # ---- 5. 释放资源 ----
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[完成] 推理结果已保存到: {OUTPUT_PATH}")
    print(f"[完成] 共处理 {frame_idx} 帧")


if __name__ == "__main__":
    main()

