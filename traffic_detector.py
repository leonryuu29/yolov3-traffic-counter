# 导入必要的库
import cv2  # OpenCV库，用于图像和视频处理
import numpy as np  # NumPy库，用于高效的数值计算
import time  # 时间库，在此项目中暂未使用，但可用于性能分析
from collections import deque  # 从collections库导入deque，用于创建高效的队列来存储车辆轨迹
from sort import Sort  # 从sort.py文件中导入Sort类，这是实现SORT跟踪算法的核心

# =============================================================================
# --- 配置参数 ---
# =============================================================================
# 定义输入和输出文件的路径及名称
VIDEO_SOURCE = "traffic_video.mp4"  # 输入视频文件的路径
WEIGHTS_PATH = "yolov3.weights"  # YOLOv3模型权重文件的路径
CONFIG_PATH = "yolov3.cfg"  # YOLOv3模型配置文件（网络结构）的路径
NAMES_PATH = "coco.names"  # 包含COCO数据集中所有类别名称的文件的路径

# --- 检测与追踪参数 (已优化) ---
# 这些参数用于微调目标检测和跟踪算法的性能
CONFIDENCE_THRESHOLD = 0.1  # 置信度阈值：只有当模型对检测结果的置信度高于此值时，才被认为是有效检测
NMS_THRESHOLD = 0.4  # 非极大值抑制（NMS）阈值：用于合并重叠的边界框
SORT_MAX_AGE = 120  # SORT算法参数：一个跟踪目标在被连续多少帧未检测到后，才被认为是消失了
SORT_MIN_HITS = 0  # SORT算法参数：一个新目标需要被连续检测到多少次，才开始对其进行跟踪

# --- 业务逻辑参数 ---
# 这些参数与具体的应用场景（如车速估计和计数）相关
PIXELS_PER_METER = 70  # 像素-米转换比例：表示在视频中，每米对应多少像素。需要根据摄像机角度和高度手动校准
LINE_Y = 400  # 计数线Y坐标：在视频帧的Y轴上画一条虚拟线，用于车辆计数

# =============================================================================
# --- 辅助函数 ---
# =============================================================================

def get_speed(prev_center, current_center, time_interval):
    """
    根据车辆在连续两帧中的中心点位置，估算其速度。
    :param prev_center: 上一帧车辆中心点坐标 (x, y)
    :param current_center: 当前帧车辆中心点坐标 (x, y)
    :param time_interval: 两帧之间的时间间隔（秒）
    :return: 速度（公里/小时, km/h）
    """
    if time_interval == 0:
        return None  # 避免除以零错误
    # 计算两点之间的欧氏距离（像素距离）
    distance_pixels = np.linalg.norm(np.array(current_center) - np.array(prev_center))
    # 将像素距离根据比例尺转换为实际距离（米）
    distance_meters = distance_pixels / PIXELS_PER_METER
    # 计算速度（米/秒）
    speed_mps = distance_meters / time_interval
    # 将速度从米/秒转换为公里/小时
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def iou(boxA, boxB):
    """
    计算两个边界框的交并比（Intersection over Union, IoU）。
    IoU是衡量两个边界框重叠程度的指标。
    :param boxA: 第一个边界框 [x1, y1, x2, y2]
    :param boxB: 第二个边界框 [x1, y1, x2, y2]
    :return: IoU值（浮点数）
    """
    # 计算交集矩形的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # 计算交集区域的面积
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # 计算两个边界框各自的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # 计算并集面积并计算IoU
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val

# =============================================================================
# --- 主程序 ---
# =============================================================================

# --- 1. 初始化 ---

# 加载YOLOv3模型
# cv2.dnn.readNet会从磁盘加载预训练的YOLOv3模型权重和网络配置
net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

# 检查是否有可用的CUDA GPU，并相应地设置计算后端
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # 设置后端为CUDA
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # 设置目标设备为GPU
    print("CUDA可用，已设置YOLO使用GPU。")
else:
    print("未检测到CUDA设备，YOLO将使用CPU。")

# 加载COCO数据集的类别名称
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]
# 获取YOLO网络所有层的名称，并确定输出层的名称
output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
# 为不同类别的边界框生成一组随机颜色，用于可视化
colors = np.random.uniform(0, 255, size=(200, 3))

# 初始化SORT跟踪器
# max_age, min_hits是SORT算法的重要参数，iou_threshold用于关联检测和跟踪
mot_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=0.3)

# 打开视频文件
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {VIDEO_SOURCE}")
    exit()

# 获取视频的帧率（FPS），并计算每帧之间的时间间隔
fps = cap.get(cv2.CAP_PROP_FPS)
TIME_INTERVAL_FRAME = 1 / fps if fps > 0 else 1/30  # 如果无法获取FPS，则默认为30

print("YOLO模型加载成功,初始化SORT追踪器成功，开始进行车辆检测...")

# --- 2. 初始化用于存储状态的变量 ---

# 设置显示文本的字体
font = cv2.FONT_HERSHEY_PLAIN
# 初始化车辆计数器
vehicle_count = 0
# 创建一个集合，用于存储已经计数过的车辆ID，防止重复计数
counted_vehicles = set()
# 创建一个字典，用于存储每个被跟踪车辆的详细信息
# 结构: {obj_id: {'path': deque, 'label': '', 'color': (B,G,R), 'confidence': 0.0}}
tracked_vehicles = {}

# --- 3. 主循环：逐帧处理视频 ---

while True:
    # 从视频中读取一帧
    ret, frame = cap.read()
    # 如果ret为False，表示视频已结束或读取错误
    if not ret:
        print("视频读取完毕或发生错误。")
        break

    # 获取帧的尺寸
    height, width, _ = frame.shape

    # --- a. 目标检测 (YOLO) ---
    # 将图像转换为blob格式，这是YOLO模型的标准输入格式
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # 将blob输入到网络中
    net.setInput(blob)
    # 执行前向传播，获取检测结果
    outs = net.forward(output_layers)

    # 初始化列表，用于存储当前帧的检测信息
    detections_for_sort = []  # 存储格式化后用于SORT跟踪器的检测结果
    class_ids_current_frame = []  # 存储检测到的对象类别ID
    confidences_current_frame = []  # 存储每个检测的置信度
    boxes_current_frame = []  # 存储原始边界框坐标

    # 解析YOLO的输出
    for out in outs:
        for detection in out:
            scores = detection[5:]  # 获取所有类别的置信度分数
            class_id = np.argmax(scores)  # 找到分数最高的类别ID
            confidence = scores[class_id]  # 获取最高置信度
            # 过滤掉低置信度的检测，并且只保留我们感兴趣的车辆类别
            if confidence > CONFIDENCE_THRESHOLD and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                # 将检测到的边界框坐标从相对值（0-1）转换回像素值
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # 将处理后的检测信息存入相应列表
                boxes_current_frame.append([x, y, w, h])
                confidences_current_frame.append(float(confidence))
                class_ids_current_frame.append(class_id)

    # --- b. 非极大值抑制 (NMS) ---
    # 应用NMS来抑制弱的、重叠的边界框，只保留最佳的检测结果
    indexes = cv2.dnn.NMSBoxes(boxes_current_frame, confidences_current_frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(indexes) > 0:
        # 遍历通过NMS筛选后的检测结果
        for i in indexes.flatten():
            x, y, w, h = boxes_current_frame[i]
            # 将边界框格式化为 [x1, y1, x2, y2, confidence, class_id]，以备后续使用
            detections_for_sort.append([x, y, x + w, y + h, confidences_current_frame[i], class_ids_current_frame[i]])

    # --- c. 目标跟踪 (SORT) ---
    # 将经过NMS处理的检测结果传递给SORT跟踪器
    # 如果没有检测到目标，则传递一个空的numpy数组
    # SORT返回的是当前帧中所有被跟踪目标的列表，格式为 [x1, y1, x2, y2, obj_id]
    tracked_objects = mot_tracker.update(np.array([d[:5] for d in detections_for_sort]) if detections_for_sort else np.empty((0, 5)))

    # --- d. 处理和可视化跟踪结果 ---
    # 遍历SORT返回的每一个被跟踪的对象
    for track in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, track)  # 获取边界框坐标和唯一的对象ID
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # 计算边界框的中心点

        # 如果是新跟踪到的对象，则在tracked_vehicles字典中为其创建条目
        if obj_id not in tracked_vehicles:
            tracked_vehicles[obj_id] = {'path': deque(maxlen=30), 'label': '', 'color': colors[obj_id % 200], 'confidence': 0.0}

        # 将当前中心点添加到该对象的轨迹中
        tracked_vehicles[obj_id]['path'].append(center)
        path = tracked_vehicles[obj_id]['path']

        # 为新出现的目标关联类别标签和置信度
        # 这个过程只在轨迹点很少（刚开始跟踪）时执行一次
        if len(path) < 2:
            best_iou = 0
            best_det = None
            # 遍历当前帧的所有检测结果，找到与当前跟踪框IoU最大的那个
            for det in detections_for_sort:
                iou_val = iou([x1, y1, x2, y2], det[:4])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = det
            # 如果找到了匹配的检测框，就将该检测的类别和置信度赋给这个跟踪对象
            if best_det:
                confidence = best_det[4]
                class_id = int(best_det[5])
                tracked_vehicles[obj_id]['label'] = classes[class_id]
                tracked_vehicles[obj_id]['confidence'] = confidence

        # --- e. 车辆计数 ---
        # 当车辆的轨迹长度足够长，并且尚未被计数时，检查其是否穿过计数线
        if len(path) >= 2 and obj_id not in counted_vehicles:
            prev_point, curr_point = path[-2], path[-1]  # 获取轨迹的最新两个点
            # 判断车辆是否从上到下穿过计数线
            if prev_point[1] <= LINE_Y and curr_point[1] > LINE_Y:
                vehicle_count += 1  # 车辆计数加一
                counted_vehicles.add(obj_id)  # 将ID加入已计数集合
                tracked_vehicles[obj_id]['color'] = (0, 255, 0)  # 将边界框颜色变为绿色
            # 判断车辆是否从下到上穿过计数线 (可选逻辑)
            elif prev_point[1] >= LINE_Y and curr_point[1] < LINE_Y:
                counted_vehicles.add(obj_id)
                tracked_vehicles[obj_id]['color'] = (0, 255, 255) # 将边界框颜色变为黄色

        # 获取该对象的颜色、标签和置信度，用于可视化
        color = tracked_vehicles[obj_id]['color']
        label = tracked_vehicles[obj_id]['label']
        confidence = tracked_vehicles[obj_id]['confidence']

        # 在帧上绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 在边界框上方显示类别标签和置信度
        display_text = f"{label} {confidence:.2f}"
        cv2.putText(frame, display_text, (x1, y1 - 10), font, 2, color, 2)

        # --- f. 速度估计 ---
        # 当轨迹长度足够长时，计算并显示车辆速度
        if len(path) >= 2:
            speed = get_speed(path[-2], path[-1], TIME_INTERVAL_FRAME)
            if speed is not None:
                # 在边界框右下角显示速度
                cv2.putText(frame, f"{int(speed)}km/h", (x2 - 60, y2 - 10), font, 1.5, color, 2)

    # --- g. 在帧上绘制辅助信息 ---
    # 绘制车辆计数线
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (255, 0, 0), 2)
    # 在左上角显示总的车辆计数
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 50), font, 3, (255, 255, 255), 3)

    # 显示处理后的帧
    cv2.imshow("Traffic Detection", frame)
    # 等待按键，如果按下'ESC'键 (ASCII码为27)，则退出循环
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- 4. 清理和退出 ---
# 释放视频捕获对象
cap.release()
# 关闭所有OpenCV创建的窗口
cv2.destroyAllWindows()
# 打印最终的车辆总数
print(f"检测结束。总计车辆: {vehicle_count}")
