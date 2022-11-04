import cv2
import pyrealsense2 as rs
import numpy as np
import random

##20220415修改版，可运行通，在所拍摄图像上，能看到一直变化位置和大小的一个实心黑色实心圆。


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
            #(480, 640)
            depth_image = np.asanyarray(depth_frame.get_data())
            #(480,640,3)
            color_image = np.asanyarray(color_frame.get_data())
            #(480,640,3)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            #(480,1280,3)
            images = np.hstack((color_image, depth_colormap))

            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            #randnum = 24
            distance_list = []
            #mid_pos = [(box[1] + box[3]) // 2, (box[0] + box[2]) // 2]  # 确定索引深度的中心像素位置
            #min_val = min(np.shape(color_image)[0], np.shape(color_image)[1])  # 确定深度搜索范围
            for j in range(int(np.shape(color_image)[0])):
                for i in range(int(np.shape(color_image)[1])):
                     #bias = random.randint(-min_val // 48, min_val // 48)
                     dist = depth_image[j, i]
                     #加下面语句，imshow出来后，被黑圈圈给覆盖了哈哈哈。黑圈圈太密集了，成全黑色了。
                     # (255, 255, 255)白色(0, 0, 0)黑色
                     #cv2.circle(color_image, (int(j), int(i)), 4, (0, 0, 0), 1)
                     if dist:
                         distance_list.append([dist, i, j])#（i，j）像素坐标及其对应深度值。结合相机内参，即可计算其三维坐标。
                         #distance_list.append(dist)
                     #distance_list.append([dist, j, i])
                     #distance_list.append(dist)
            distance_list = np.array(distance_list)
            #distance_list = np.sort(distance_list)
            #distance_list =distance_list[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]#12-6=6  :  12+6=18

            #cv2.circle(color_image, (int(j), int(i)), 6000, (0, 0, 0), 1)
            #i639,x坐标。j479，y坐标。
            # print(distance_list[40000][1])
            # print(distance_list[40000][2])
            # print(distance_list[40000][0])
            cv2.circle(color_image, (int(distance_list[40000][1]), int(distance_list[40000][2])), distance_list[40000][0]//100, (0, 0, 0), -1)
            cv2.imshow('test_img', color_image)
            distance_list = distance_list[:, 0]
            # 如果用下面的代码取出来几个值，可以确定得到其均值。那么不用这行代码，就得到的是所有深度值的均值了。代码验证无误。
            # 因为上一句和这一句所得到的distance_list的shape和size值类似（仅仅大小不同），且数据类型也一样。
            # distance_list = distance_list[0:5]
            print(np.mean(distance_list))
            #distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
            #print(distance_list, np.mean(distance_list))

            # print(np.shape(distance_list))
            # print(np.shape(distance_list[:, 0]))
            # print(np.mean(distance_list[:, 0]))

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()