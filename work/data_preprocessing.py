import os
import cv2

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
             "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
             "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]

if not os.path.exists('./img'):
    os.mkdir('./img')

## 分为车牌检测和车牌识别
# 转换检测数据
train_det = open('./train_det.txt', 'w', encoding='UTF-8')
dev_det = open('./dev_det.txt', 'w', encoding='UTF-8')
# 转换识别数据
train_rec = open('./train_rec.txt', 'w', encoding='UTF-8')
dev_rec = open('./dev_rec.txt', 'w', encoding='UTF-8')

# 总样本数
total_num = len(os.listdir('D:\ML\Dataset\CCPD2019\ccpd_base'))
# 训练样本数
train_num = int(total_num * 0.8)

count = 0

for item in os.listdir('D:\ML\Dataset\CCPD2019\ccpd_base'):
    path = 'D:/ML/Dataset/CCPD2019/ccpd_base/' + item

    # 01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24

    # 获取左上顶点和右下顶点的坐标
    _, _, bboxs, points, labels, _, _ = item.split('-')
    bboxs = bboxs.split('_')
    x1, y1 = bboxs[0].split('&')
    x2, y2 = bboxs[1].split('&')
    bboxs = [int(coord) for coord in [x1, y1, x2, y2]]

    # 获取车牌的四个顶点的坐标
    points = points.split('_')
    points = [point.split('&') for point in points]
    points_ = points[-2:] + points[:2]

    points = []
    for point in points_:
        points.append([int(_) for _ in point])

    labels = labels.split('_')
    prov = provinces[int(labels[0])]
    plate_number = [ads[int(label)] for label in labels[1:]]
    labels = prov + ''.join(plate_number)

    # 获取检测训练检测框位置
    line_det = path + '\t' + '[{"transcription": "%s", "points": %s}]' % (labels, str(points))
    line_det = line_det[:] + '\n'

    # 获取识别训练图片及标签
    img = cv2.imread(path)
    crop = img[bboxs[1]:bboxs[3], bboxs[0]:bboxs[2]]
    cv2.imwrite('D:/ML/Project/PaddleOCR-release-2.3/work/img/%06d.jpg' % count, crop)
    line_rec = 'D:/ML/Project/PaddleOCR-release-2.3/work/img/%06d.jpg\t%s\n' % (count, labels)

    # 写入txt
    if count <= train_num:
        train_det.write(line_det)
        train_rec.write(line_rec)
    else:
        dev_det.write(line_det)
        dev_rec.write(line_rec)
    count += 1
    print('\r' + str(round(count / total_num * 100, 2)) + '%', end='')

train_det.close()
train_rec.close()
dev_det.close()
dev_rec.close()

# 保存省份等信息
with open('./dict.txt', 'w', encoding='UTF-8') as file:
    for key in ads + provinces:
        file.write(key + '\n')
