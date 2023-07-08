import os
from PIL import Image
import numpy as np
from utils import iou

origin_face_bbx_file = "D:\list_bbox_celeba.txt" # 原来的样本数据（在生成样本时使用）
origin_face_landmarks_file = "D:\list_landmarks_celeba.txt" # 原来的landmark样本数据（在生成样本时使用）
origin_face_img_dir = "D:\celeba\img_celeba"  # 源图片（用于生成新样本）

save_path = "data/MTCNN"  # 生成样本的总的保存路径

float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]  # 控制正负样本比例，（控制比例？）


def gen_sample(face_size, stop_value):
    print("gen size:{} image".format(face_size))
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")  # 仅仅生成路径名
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")  # 连接存储各类图片数据的路径
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:  # 生成路径
        if not os.path.exists(dir_path):  # 判断改路径是否存在
            os.makedirs(dir_path)  # 创建文件夹

    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")  # 标签文件路径
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0  # 定义变量，并初始化
    negative_count = 0
    part_count = 0

    positive_anno_file = open(positive_anno_filename, "w")  # 打开文件
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")

    landmarks_list = []
    for i, line in enumerate(open(origin_face_landmarks_file)):  # 按行读取landmarks.txt文件
        if i < 2:
            continue
        strs = line.split()  # 切割，列表，包含路径和坐标值
        line = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % (strs[1], strs[2], strs[3], strs[4], strs[5],
                                                  strs[6], strs[7], strs[8], strs[9], strs[10])

        landmarks_list.append(line)  # 添加数据line到列表
        # landmarks_list.append('\n')

    for i, line in enumerate(open(origin_face_bbx_file)):  # 按行读取bbox.txt文件
        if i < 2:
            continue
        i = i - 2
        strs = line.split()  # 列表，包含路径和坐标值
        # 置信度   #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        image_filename = strs[0].strip()
        image_file = os.path.join(origin_face_img_dir, image_filename)

        with Image.open(image_file) as img:
            img_w, img_h = img.size  # 原图
            x1 = float(strs[1].strip())
            y1 = float(strs[2].strip())
            w = float(strs[3].strip())  # 人脸框
            h = float(strs[4].strip())
            x2 = float(x1 + w)
            y2 = float(y1 + h)

            if x1 < 0 or y1 < 0 or w < 0 or h < 0:  # 跳过坐标值为负数的
                continue
            boxes = [[x1, y1, x2, y2]]  # 当前真实框四个坐标（根据中心点偏移）， 二维数组便于IOU计算
            # 求中心点坐标
            cx = x1 + w / 2
            cy = y1 + h / 2
            side_len = max(w, h)
            seed = float_num[np.random.randint(0, len(float_num))]  # 取0到9之间的随机数作为索引
            count = 0

            '''
            随机获得一定大小的选框，这一步暂时不用关系选框大小，下一步会暴力缩放
            _side_len是新框的 长宽（正方形）
            _cx,_cy是它中心点坐标
            其他容易理解
            '''

            for _ in range(4):
                _side_len = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))  # 生成框
                _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))  # 中心点作偏移
                _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))
                _x1 = _cx - _side_len / 2  # 左上角
                _y1 = _cy - _side_len / 2
                _x2 = _x1 + _side_len  # 右下角
                _y2 = _y1 + _side_len
                if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:  # 左上角的点是否偏移到了框外边，右下角的点大于图像的宽和高
                    continue

                '''
                offset表示
                当前输入图片和真实选框的偏移量
                '''

                offset_x1 = (x1 - _x1) / _side_len  # 得到四个偏移量
                offset_y1 = (y1 - _y1) / _side_len
                offset_x2 = (x2 - _x2) / _side_len
                offset_y2 = (y2 - _y2) / _side_len

                px1, py1, px2, py2, px3, py3, px4, py4, px5, py5 = landmarks_list[i][:].split(',')
                offset_px1 = (int(px1) - _x1) / _side_len  # offset偏移量
                offset_py1 = (int(py1) - _y1) / _side_len
                offset_px2 = (int(px2) - _x1) / _side_len
                offset_py2 = (int(py2) - _y1) / _side_len
                offset_px3 = (int(px3) - _x1) / _side_len
                offset_py3 = (int(py3) - _y1) / _side_len
                offset_px4 = (int(px4) - _x1) / _side_len
                offset_py4 = (int(py4) - _y1) / _side_len
                offset_px5 = (int(px5) - _x1) / _side_len
                offset_py5 = (int(py5) - _y1) / _side_len

                crop_box = [_x1, _y1, _x2, _y2]
                face_crop = img.crop(crop_box)  # 图片裁剪

                face_resize = face_crop.resize((face_size, face_size))  # 对裁剪后的图片缩放

                '''
                0负样本
                1正样本
                2部分样本
                '''

                ious = iou(crop_box, np.array(boxes))[0]
                if ious > 0.65:  # 可以自己修改
                    positive_anno_file.write(
                        "{0}.png {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            os.path.join(positive_image_dir, str(positive_count)), 1, offset_x1, offset_y1,
                            offset_x2, offset_y2,
                            offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3,
                            offset_px4, offset_py4, offset_px5, offset_py5))
                    positive_anno_file.flush()  # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区
                    face_resize.save(os.path.join(positive_image_dir, "{0}.png".format(positive_count)))
                    # print("positive_count",positive_count)
                    positive_count += 1
                elif 0.65 > ious > 0.4:
                    part_anno_file.write(
                        "{0}.png {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            os.path.join(part_image_dir, str(part_count)), 2, offset_x1, offset_y1,
                            offset_x2, offset_y2,
                            offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3,
                            offset_px4, offset_py4, offset_px5, offset_py5))
                    part_anno_file.flush()
                    face_resize.save(os.path.join(part_image_dir, "{0}.png".format(part_count)))
                    # print("part_count", part_count)
                    part_count += 1
                elif ious < 0.1:
                    negative_anno_file.write(
                        "{0}.png {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                            os.path.join(negative_image_dir, str(negative_count)), 0))
                    negative_anno_file.flush()
                    face_resize.save(os.path.join(negative_image_dir, "{0}.png".format(negative_count)))
                    # print("negative_count", negative_count)
                    negative_count += 1
                count = positive_count + part_count + negative_count
                print(count)
            if count >= stop_value:
                break

    positive_anno_file.close()
    negative_anno_file.close()
    part_anno_file.close()


if __name__ == '__main__':
    gen_sample(12, 10000)
    gen_sample(24, 10000)
    gen_sample(48, 10000)
