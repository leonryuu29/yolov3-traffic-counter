import cv2
import numpy as np
import onnxruntime
from sort import Sort

# =============================================================================
# --- 辅助函数 ---
# =============================================================================
def letterbox(im, new_shape=(608, 608), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    将图像缩放并填充到指定尺寸，同时保持长宽比。
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

# =============================================================================
# --- 全局参数 ---
# =============================================================================
VIDEO_PATH = "traffic_video.mp4"
ONNX_MODEL_PATH = "yolov3.onnx"
CLASSES_PATH = "coco.names"

# --- ONNX模型输入参数 ---
INPUT_WIDTH = 608
INPUT_HEIGHT = 608

# --- 检测和追踪阈值 ---
CONFIDENCE_THRESHOLD = 0.4  # 物体置信度阈值
SCORE_THRESHOLD = 0.5       # 类别分数阈值 (物体置信度 * 类别置信度)
NMS_THRESHOLD = 0.45        # 非极大值抑制(NMS)的IOU阈值

# =============================================================================
# --- 主函数 ---
# =============================================================================
if __name__ == "__main__":
    # 1. 加载类别名称
    with open(CLASSES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # 2. 加载ONNX模型并设置执行器
    print("正在加载ONNX模型...")
    try:
        # 优先尝试使用GPU
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"模型加载成功，使用: {session.get_providers()}")
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        # 如果GPU失败，则仅使用CPU
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"模型加载成功，使用: {session.get_providers()}")

    # 获取模型输入输出名称
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # 3. 初始化SORT追踪器
    # max_age: 目标在被删除前可以连续未被检测到的帧数
    # min_hits: 目标在被确认为“已追踪”前需要被连续检测到的帧数
    # iou_threshold: 用于关联检测框和追踪轨迹的IOU阈值
    sort_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # 4. 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {VIDEO_PATH}")
        exit()

    # 获取视频的宽度和高度
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 5. 循环处理视频帧
    print("开始进行车辆检测...")
    total_vehicles = 0
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 5.1 预处理：Letterbox缩放、转换为blob
        img, ratio, (dw, dh) = letterbox(frame, new_shape=(INPUT_WIDTH, INPUT_HEIGHT), auto=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 先做letterbox再转RGB
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img) # 将内存变为连续
        img = img.reshape(1, 3, INPUT_HEIGHT, INPUT_WIDTH) # 添加batch维度
        img = img.astype(np.float32)
        img = img / 255.0

        # 5.2 ONNX推理
        outputs = session.run(output_names, {input_name: img})

        # 5.3 解析ONNX输出 (手动后处理)
        # ONNX YOLOv3的输出是一个单一的张量，形状为 (batch_size, num_boxes, 5 + num_classes)
        detections = outputs[0][0] # 移除批处理维度

        boxes_for_nms = []
        confidences_for_nms = []
        class_ids_for_nms = []

        for row in detections:
            objectness_confidence = row[4]
            if objectness_confidence > CONFIDENCE_THRESHOLD:
                class_scores = row[5:]
                class_id = np.argmax(class_scores)

                # 只关注车辆相关的类别
                if classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                    final_confidence = objectness_confidence * class_scores[class_id]

                    if final_confidence > SCORE_THRESHOLD:
                        # 坐标转换：从模型输出(cx,cy,w,h)转换到原始视频帧尺寸(x,y,w,h)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]

                        # 1. 减去padding
                        cx -= dw
                        cy -= dh
                        # 2. 除以缩放比例
                        cx /= ratio[0]
                        cy /= ratio[1]
                        w /= ratio[0]
                        h /= ratio[1]

                        # 3. 转换成左上角坐标(x,y)
                        x = int(cx - w / 2)
                        y = int(cy - h / 2)

                        boxes_for_nms.append([x, y, int(w), int(h)])
                        confidences_for_nms.append(float(final_confidence))
                        class_ids_for_nms.append(class_id)

        # 5.4 应用非极大值抑制 (NMS)
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences_for_nms, SCORE_THRESHOLD, NMS_THRESHOLD)

        detections_for_sort = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes_for_nms[i]
                # 转换为SORT需要的(x1, y1, x2, y2)格式
                detections_for_sort.append([x, y, x + w, y + h, confidences_for_nms[i], class_ids_for_nms[i]])

        # 5.5 更新SORT追踪器
        # SORT需要 [x1, y1, x2, y2, score] 格式的numpy数组
        if len(detections_for_sort) > 0:
            tracked_objects = sort_tracker.update(np.array(detections_for_sort)[:, :5])
        else:
            tracked_objects = sort_tracker.update(np.empty((0, 5)))

        # 5.6 绘制结果和计数
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj[:5]
            obj_id = int(obj_id)

            # 计数
            if obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                total_vehicles += 1

            # 绘制
            color = (0, 255, 0) # 绿色
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"Vehicle {obj_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 显示总数
        cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示帧 (可选)
        # cv2.imshow("Vehicle Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 6. 释放资源
    cap.release()
    # cv2.destroyAllWindows()
    print(f"检测结束。总计车辆: {total_vehicles}")
