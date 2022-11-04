import numpy as np
a = np.array([[123.45351,117.810165,145.15724,163.69823,0.47149807,0.9988047,4], [123.45351,117.810165,145.15724,163.69823,0.47149807,0.9988047,4]])
#print(np.shape(a))
top_index = a[:, 4] * a[:, 5] > 0.3
top_bboxes = np.array(a[top_index, :4])
top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
top_ymin, top_xmin, top_ymax, top_xmax
b = np.array([640, 640])
c = np.array([480, 640])
new_shape = c *np.min(b / c)

offset = (b -new_shape)/2./b
scale = b/new_shape

box_yx = np.concatenate(((top_ymin+top_ymax)/2,(top_xmin+top_xmax)/2),axis=-1)/b
box_hw = np.concatenate((top_ymax-top_ymin,top_xmax-top_xmin),axis=-1)/b

box_yx = (box_yx - offset) * scale
box_hw *= scale

box_mins = box_yx - (box_hw / 2.)
box_maxes = box_yx + (box_hw / 2.)
# print(box_mins[:, 0:1])
# print(box_mins[:, 1:2])
# print(box_maxes[:, 0:1])
# print(np.shape(box_maxes[:, 1:2]))
boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
print(np.concatenate([c, c],axis=-1))
print(np.shape(np.concatenate([c, c],axis=-1)))
boxes *= np.concatenate([c, c],axis=-1)
print(np.shape(boxes))