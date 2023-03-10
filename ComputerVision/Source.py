import cv2
import numpy
from skimage import morphology

contourArea_threshold = 900  # 要求轮廓面积大于contourArea_threshold
holeArea_threshold = 100  # 填补小于holeArea_threshold的孔

target_color = (0, 255, 0)  # 跟踪目标的定位框颜色、轮廓颜色
other_color = (255, 0, 0)  # 其他轮廓颜色
location_color = (0, 0, 255)  # 跟踪目标中心点的运动轨迹颜色

# 链码方向映射
# cv2中坐标轴（x向右，y向下），因此转换到坐标原点在左下角的坐标系后，链码方向映射变化：顺时针->逆时针
chain_code_dict = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}


def Tracking(_frame, _tracker, _previous_bbox):
    """目标跟踪：更新追踪器，并返回定位框"""
    # 更新追踪器
    succeed, bbox = _tracker.update(_frame)
    # 跟踪成功
    if succeed:
        _previous_bbox = bbox
        return [int(i) for i in bbox]
    # 跟踪失败,返还给main函数处理
    else:
        return [None] * 4


def Segment(_frame, _background_model):
    """目标分割：更新背景模型，处理并返回分割出的前景图片"""
    # 计算视频的前景
    foreground_mask = cv2.GaussianBlur(_frame, (3, 3), 0)
    foreground_mask = _background_model.apply(foreground_mask)

    # 滤波去噪、二值化
    # binary = cv2.GaussianBlur(foreground_mask, (5, 5), 0)
    binary = cv2.medianBlur(foreground_mask, 5)  # medianBlur比GaussianBlur可更好地去孤立点
    # 图像阈值化（当背景模型不检测背景时，foreground_mask已为二值图像）
    # 返回值ret、dst分别为：阈值、二值图像
    binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)[1]

    # 进行形态学改变，遍历每点：腐蚀（置为核内最小值0）、膨胀（置为核内最大值255）
    # morphed = morphology.remove_small_objects(binary.astype(bool), min_size=64).astype(numpy.uint8)
    morphed = morphology.remove_small_holes(binary.astype(bool), area_threshold=holeArea_threshold).astype(numpy.uint8)
    # morphologyEx 只支持二值化图像：MORPH_OPEN 开运算（腐蚀-膨胀），MORPH_CLOSE 闭运算（膨胀-腐蚀）
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # 生成k*k矩阵。MORPH_ELLIPSE：椭圆形
    # 填补小于area_threshold的孔
    morphed = morphology.remove_small_holes(morphed.astype(bool), area_threshold=holeArea_threshold).astype(numpy.uint8)
    morphed[morphed == 1] = 255

    # 绘制非目标前景的轮廓
    contours, _ = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:  # contour：[N, 1, 2]，每点为绝对位置
        if cv2.contourArea(contour) > contourArea_threshold:
            cv2.drawContours(_frame, contour, -1, other_color, 2)

    return morphed


def Contour_inChainCode(_frame, _segmented):
    """根据分割图片计算轮廓，并以链码形式返回"""
    # contours（轮廓）hierarchy（轮廓间关系）
    # mode寻找模式：RETR_LIST（所有轮廓，无关系），RETR_EXTERNAL（只找外边界）
    # method表示方法：CHAIN_APPROX_NONE（储存所有点），CHAIN_APPROX_SIMPLE（储存转折点）
    contours, _ = cv2.findContours(_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours: return
    # 默认一个定位框内只有一个轮廓
    contour = max(contours, key=cv2.contourArea)
    # 绘制轮廓
    cv2.drawContours(_frame, contour, -1, target_color, 3)
    # 将坐标转为链码
    # 计算i与i+1个元素的差分（长度比原来少了1）
    difference = numpy.diff(numpy.squeeze(contour), axis=0)
    # 使用dic.get遍历difference（map表达式效率高于列表迭代）
    code = list(map(chain_code_dict.get, [tuple(i) for i in difference]))
    return code


def _distance(_previous_bbox):
    """返回一个计算与此时_previous_bbox棋盘距离的函数，用于排序的key"""

    def _distance1(_contour):
        bbox = cv2.boundingRect(_contour)
        distance = numpy.abs(numpy.array(_previous_bbox) - numpy.array(bbox)).sum() \
            if cv2.contourArea(_contour) > contourArea_threshold else float('inf')
        # print(f'上一个:{_previous_bbox},找到一个contour:{bbox},距离为:{distance}')
        return distance

    return _distance1


def Store(chain_code_list, location_list):
    with open('链码.txt', 'w') as f:
        print('链码：')
        print(chain_code_list)
        for i in chain_code_list:
            f.writelines(' '.join(f'{_}' for _ in i) + '\n')
    with open('运动轨迹.txt', 'w') as f:
        print('\n运动轨迹：')
        print(location_list)
        print(location_list, file=f)
