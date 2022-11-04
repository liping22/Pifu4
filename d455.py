from yolo import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import os
import torch
from PIL import Image
from PIL import ImageFont
from utils.utils import yolo_correct_boxes
import random

###0415修改版，可以在框框的旁边，除了显示类别，置信度，还显示了深度值。但是深度值是否准确呢？随后就要开始阅读文献了。

yolo = YOLO()

def get_depth(frame,box,depth):
    distance_list = []
    mid_pos = [(box[1] + box[3])//2, (box[0] + box[2])//2] #确定索引深度的中心像素位置
    height = abs(box[2] - box[0])
    width = abs(box[3] - box[1])
    min_val = min(height, width) #框框的高和宽取较小值。
    #在框的中心点处画一个实心圆，颜色为黑色，半径参考min_val来确定，保证框框里能看到圆，而框不会被圆所覆盖。
    cv2.circle(frame, (int(mid_pos[0]), int(mid_pos[1])), int(min_val//6), (0, 0, 0), -1)
    for i in range(int(abs(box[1])), int(abs(box[3])-1)):
        for j in range(int(abs(box[0])), int(abs(box[2])-1)):
            # dist = depth[int(mid_pos[1]), int(mid_pos[0])]
            dist = depth[j-1, i-1]
            if dist:
                distance_list.append([dist, i, j])
    # 可以观察dist为0的数据多不多。(3002, 3) 3029.396548267456。3002和3029接近。说明dist为0的不太多。
    print(np.shape(distance_list), (height * width))

        #bias = random.randint(-min_val//4, min_val//4)
        #dist = depth[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]

        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
    distance_list = np.array(distance_list)
    try:
        np.mean(distance_list[:, 0])
        # 这个出问题，说明dist全部为0，返回空值
    except:
        return 0
    #randnum = 24
    #distance_list = np.sort(distance_list[:, 0])[(randnum//2-randnum//4:randnum//2+randnum//4), 0] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    return np.mean(distance_list[:, 0])

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


#显示深度值的函数
def dectshow(org_img, results, depth):
    img = org_img.copy()
    top_index = results[:, 4] * results[:, 5] > 0.2
    top_conf = results[top_index, 4] * results[top_index, 5]
    top_label = np.array(results[top_index, -1], np.int32)
    top_bboxes = np.array(results[top_index, :4])
    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
    #去掉灰条
    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array([640, 640]), np.array(np.shape(img)[0:2]))
    #boxes = top_ymax, top_xmin, top_ymin, top_xmax

    for i, c in enumerate(top_label):
        classes_path = 'model_data/my_classes.txt'
        class_names = get_class(classes_path)
        predicted_class = class_names[c]
        score = top_conf[i]
        #top, left, bottom, right = top_xmin[i][c], top_ymin[i][c], top_xmax[i][c], top_ymax[i][c]
        #top, left, bottom, right = top_xmin[i, :], top_ymin[i, :], top_xmax[i, :], top_ymax[i, :]
        #top, left, bottom, right = boxes[i]
        bottom, left, top, right = boxes[i]

        # top = top - 5
        # left = left - 5
        # bottom = bottom + 5
        # right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))#ymax
        left = max(0, np.floor(left + 0.5).astype('int32'))#xmin
        bottom = min(np.shape(img)[0], np.floor(bottom + 0.5).astype('int32'))#ymin
        right = min(np.shape(img)[1], np.floor(right + 0.5).astype('int32'))#xmax
        # 画框框
        label = '{} {:.2f}'.format(predicted_class, score)
        #label = label.encode('utf-8')
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        dist = get_depth(org_img, boxes[i], depth)
        cv2.putText(img, label + 'dep' + str(dist / 1000) + 'm', (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        #cv2.putText(img, label, (int(right), int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)#字体大小1字体粗细2
        #print(i)
    cv2.imshow('yolov4_img', img)

# #不显示深度值的函数
# def dectshow(org_img, results):
#     img = org_img.copy()
#     top_index = results[:, 4] * results[:, 5] > 0.2
#     top_conf = results[top_index, 4] * results[top_index, 5]
#     top_label = np.array(results[top_index, -1], np.int32)
#     top_bboxes = np.array(results[top_index, :4])
#     top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
#     #去掉灰条
#     boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array([640, 640]), np.array(np.shape(img)[0:2]))
#     #boxes = top_ymax, top_xmin, top_ymin, top_xmax
#
#     for i, c in enumerate(top_label):
#         classes_path = 'model_data/my_classes.txt'
#         class_names = get_class(classes_path)
#         predicted_class = class_names[c]
#         score = top_conf[i]
#         #top, left, bottom, right = top_xmin[i][c], top_ymin[i][c], top_xmax[i][c], top_ymax[i][c]
#         #top, left, bottom, right = top_xmin[i, :], top_ymin[i, :], top_xmax[i, :], top_ymax[i, :]
#         #top, left, bottom, right = boxes[i]
#         bottom, left, top, right = boxes[i]
#
#         # top = top - 5
#         # left = left - 5
#         # bottom = bottom + 5
#         # right = right + 5
#
#         top = max(0, np.floor(top + 0.5).astype('int32'))#ymax
#         left = max(0, np.floor(left + 0.5).astype('int32'))#xmin
#         bottom = min(np.shape(img)[0], np.floor(bottom + 0.5).astype('int32'))#ymin
#         right = min(np.shape(img)[1], np.floor(right + 0.5).astype('int32'))#xmax
#         # 画框框
#         label = '{} {:.2f}'.format(predicted_class, score)
#         #label = label.encode('utf-8')
#         cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
#         #dist = get_mid_pos(org_img, box, depth_data, 24)
#         #cv2.putText(img, label + str(dist / 1000)[:4] + 'm', (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
#         cv2.putText(img, label, (int(right), int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)#字体大小1字体粗细0.8
#         print(i)
#     cv2.imshow('yolov4_img', img)


while True:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            #print(np.shape(color_image)[0:2])
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            #ndarray，480，640，3
            results = yolo.detect_image(color_image)
            #print(results)
            try:
                LI, PING, xliping = np.shape(results)
                #这个出问题，说明results返回的不是原始的RGB图像，那肯定就是检测结果。是检测结果就执行dectshow(color_image, results)
            except:
                dectshow(color_image, results, depth_image)
                # cv2.imshow('yolov4_img', results)
                # 不显示深度值的函数调用
                # dectshow(color_image, results)
                # 显示深度值的函数调用
                #dectshow(color_image, results, depth_image)

            #print(np.shape(results))
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', images)
            #dectshow(color_image, results)
            #
            #results = torch.from_numpy(results)
            #results.show()
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


#下面代码是可以实现不停止检测直到检测到皮肤。但问题很明显，检测出来的框不在它该在的位置。
# def dectshow(org_img, results):
#     img = org_img.copy()
#     top_index = results[:, 4] * results[:, 5] > 0.3
#     top_conf = results[top_index, 4] * results[top_index, 5]
#     top_label = np.array(results[top_index, -1], np.int32)
#     top_bboxes = np.array(results[top_index, :4])
#     top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
#
#
#
#     for i, c in enumerate(top_label):
#         classes_path = 'model_data/my_classes.txt'
#         class_names = get_class(classes_path)
#         predicted_class = class_names[c]
#         score = top_conf[i]
#         #top, left, bottom, right = top_xmin[i][c], top_ymin[i][c], top_xmax[i][c], top_ymax[i][c]
#         top, left, bottom, right = top_xmin[i, :], top_ymin[i, :], top_xmax[i, :], top_ymax[i, :]
#         #top, left, bottom, right = boxes[i]
#         top = top - 5
#         left = left - 5
#         bottom = bottom + 5
#         right = right + 5
#
#         top = max(0, np.floor(top + 0.5).astype('int32'))
#         left = max(0, np.floor(left + 0.5).astype('int32'))
#         bottom = min(np.shape(img)[0], np.floor(bottom + 0.5).astype('int32'))
#         right = min(np.shape(img)[1], np.floor(right + 0.5).astype('int32'))
#         # 画框框
#         label = '{} {:.2f}'.format(predicted_class, score)
#         #label = label.encode('utf-8')
#         cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)
#         #dist = get_mid_pos(org_img, box, depth_data, 24)
#         #cv2.putText(img, label + str(dist / 1000)[:4] + 'm', (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
#         cv2.putText(img, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
#         print('i')
#     cv2.imshow('yolov4_img', img)
#
#
# while True:
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
#     # Start streaming
#     pipeline.start(config)
#     try:
#         while True:
#             frames = pipeline.wait_for_frames()
#             depth_frame = frames.get_depth_frame()
#             color_frame = frames.get_color_frame()
#             if not depth_frame or not color_frame:
#                 continue
#             depth_image = np.asanyarray(depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())
#             depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#             images = np.hstack((color_image, depth_colormap))
#             #ndarray，480，640，3
#             results = yolo.detect_image(color_image)
#             try:
#                 LI, PING, xliping = np.shape(results)
#             except:
#                 #cv2.imshow('yolov4_img', results)
#                 dectshow(color_image, results)
#
#             #print(np.shape(results))
#             #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#             #cv2.imshow('RealSense', images)
#
#             #dectshow(color_image, results)
#             #
#             #results = torch.from_numpy(results)
#
#             #results.show()
#             key = cv2.waitKey(10)
#             #if key & 0xFF == ord('q') or key == 27:
#                 #cv2.destroyAllWindows()
#                 #break
#     finally:
#         pipeline.stop()
