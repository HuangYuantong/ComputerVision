import cv2
import numpy

from Source import Tracking, Segment, Contour_inChainCode, _distance, Store

detectShadows = False  # 背景模型：是否检测阴影
# video_source = 0  # 调用摄像头
video_source = 'data/video.mp4'  # 视频源video.mp4：目标初始位于画面内
# video_source = 'data/video1.1.mp4'  # 视频源video1.1.mp4：目标初始不在画面内
video = cv2.VideoCapture(video_source)  # 创建视频流对象

target_color = (0, 255, 0)  # 跟踪目标的定位框颜色、轮廓颜色
other_color = (255, 0, 0)  # 其他轮廓颜色
location_color = (0, 0, 255)  # 跟踪目标中心点的运动轨迹颜色

chain_code_list = []  # 保存链码
location_list = []  # 保存运动轨迹

# 背景模型
# 前景255、背景0，detectShadows=True表示进行阴影检测，阴影区值为127
# 默认值：历史帧数 history = 500，邻域大小 dist2Threshold = 400.0，检测阴影 detectShadows = true
background_model = cv2.createBackgroundSubtractorKNN(history=200, detectShadows=detectShadows)
# 默认值：history = 500，像素模型匹配的方差阈值 varThreshold = 16，detectShadows = true
# background_model = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=detectShadows)

# 追踪器（在cv2拓展包中）
# tracker = cv2.TrackerMOSSE_create()  # 最快（最小平方误差 追踪器）
# tracker = cv2.TrackerCSRT_create()  # 最慢，最优，自动调整大小,但需要较好的ROI初始化（通道和空间可靠性判别 追踪器）
# init_location = (270, 270, 10, 10)  # video：目标初始(x,y,w,h)手工标定：(391, 377, 56, 89)。随机：(270, 270, 10, 10)


# 视频保存
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高
fps = int(video.get(cv2.CAP_PROP_FPS))  # 帧
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
video_result = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))


def Relocate():
    """跟踪失败时,使用最近的contour作为目标定位（并更新tracker）"""
    global tracker, chain_code, previous_bbox, masked_segmented, x, y, w, h, count
    contours = cv2.findContours(segmented, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    if not contours:
        chain_code = None
    else:
        _bbox = min(contours, key=_distance(previous_bbox))  # 按照与_previous_bbox间距离排序
        _bbox = cv2.boundingRect(_bbox)
        # 生成新的追踪器
        # tracker = cv2.TrackerMOSSE_create()  # MOSSE、CSRT
        # tracker.init(frame, _bbox)
        # 使用与_previous_bbox距离最近的前景作为目标
        previous_bbox = _bbox
        x, y, w, h = [int(i) for i in _bbox]
        masked_segmented = numpy.zeros_like(segmented)
        masked_segmented[y:y + h, x:x + w] = 255
        chain_code = Contour_inChainCode(frame, masked_segmented & segmented)


if __name__ == '__main__':
    # 使用第一帧初始化
    frame = video.read()[1]
    print('  ==============================================\n'
          '   如果要跟踪的目标在当前视野内：请框选出目标的最小外接矩形，\n'
          '   如果不在视野内：请按任意键跳过以继续！\n'
          '  ==============================================\n')
    init_location = cv2.selectROI('! press any key to skip if target not in sight !', frame)  # 选取跟踪区域
    # print(init_location)
    if (init_location[2] < 30) and (init_location[3] < 30):
        init_location = (270, 270, 10, 10)
        tracker = cv2.TrackerMOSSE_create()  # 最快
        # tracker.init(frame, init_location)
    else:
        tracker = cv2.TrackerCSRT_create()  # 最慢，最优，自动调整大小,但需要较好的ROI初始化
        tracker.init(frame, init_location)

    previous_bbox = init_location
    background_model.apply(cv2.GaussianBlur(frame, (3, 3), 0))

    # 开始遍历视频
    count = 0
    while True:
        retval, frame = video.read()
        if not retval: break
        timer = cv2.getTickCount()
        # 获得追踪、分割结果
        x, y, w, h = Tracking(frame, tracker, previous_bbox)
        segmented = Segment(frame, background_model)
        # 跟踪失败：重新定位（并更新tracker）
        # if (not x) or (count % 30 == 0):
        if not x:
            Relocate()
        # 跟踪成功:只分割定位框内的前景
        else:
            masked_segmented = numpy.zeros_like(segmented)
            masked_segmented[y:y + h, x:x + w] = 255
            chain_code = Contour_inChainCode(frame, masked_segmented & segmented)
        # 保存链码、运动轨迹（定位框中心点）
        chain_code_list.append(chain_code if chain_code
                               else chain_code_list[-1] if chain_code else ' ')  # 如果未找到边缘则为上一时刻的
        location = (int(x + w / 2), int(y + h / 2)) if x else None
        location_list.append(location)

        # 画定位框
        cv2.rectangle(frame, (x, y), (x + w, y + h), target_color, 3) if x else 0
        # 画运动轨迹
        for i in range(1, len(location_list)):
            # 画出运动矢量（方向：运动方向，速度：长度/2）
            if not (location_list[i] and location_list[i - 1]): continue
            cv2.line(frame, location_list[i - 1],
                     tuple(((numpy.array(location_list[i]) + numpy.array(location_list[i - 1])) / 2).astype(int)),
                     location_color, 3)

        # 输出、保存
        count += 1
        cv2.putText(frame, 'Green: outline and bbox of target', (30, 50), cv2.FONT_ITALIC, 0.7, target_color, 2)
        cv2.putText(frame, 'Red  : motion vector of target', (30, 75), cv2.FONT_ITALIC, 0.7, location_color, 2)
        cv2.putText(frame, 'Blue : outline of other foreground', (30, 100), cv2.FONT_ITALIC, 0.7, other_color, 2)
        cv2.putText(frame, f'{int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))}FPS',
                    (10, 125), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
        cv2.imshow('segment', segmented)
        cv2.imshow('result', frame)
        video_result.write(frame)
        if cv2.waitKey(1) == ord('q'): break  # 0xff十六位全1，接收到按键时ord('q')判断为q退出

    # 处理完成
    video.release()
    video_result.release()
    cv2.destroyAllWindows()
    Store(chain_code_list, location_list)
