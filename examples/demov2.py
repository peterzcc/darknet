"""Detector functions with different imread methods"""

import ctypes
from pydarknet import darknet_libwrapper as dn
from scipy.misc import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DetectedObject(object):
    def __init__(self, name, prob, x, y, w, h):
        try:
            name = name.decode()
        except AttributeError:
            pass
        self.name = name
        self.prob = prob
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(ctypes.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im


def vert_from_box(x,y,w,h):
    return [(x, y), (x+w,y), (x+w, y+h), (x,y+h)]


def _detector(net, meta, image, thresh=.5, hier=.5, nms=.45):
    dn.cuda_set_device(0)
    num = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num)
    dn.network_predict_image(net, image)
    dets = dn.get_network_boxes(net, image.w, image.h, thresh, hier, None, 0, num_ptr)
    num = num_ptr[0]
    if (nms):
        dn.do_nms_sort(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    # res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    dn.free_detections(dets, num)
    return res

# Darknet
# net = load_network("cfg/yolov3.cfg", "models/yolov3.weights", 0)
dn.cuda_set_device(0)
net = dn.load_network("cfg/yolov3-tiny.cfg", "models/yolov3-tiny.weights", 0)
meta = dn.get_metadata("cfg/coco.data")

# im = load_image_color('data/dog.jpg', 0, 0)
# result = _detector(net, meta, im)
# print 'Darknet:\n', result

# # scipy
# arr= imread('data/eagle.jpg')
# im = array_to_image(arr)
# result = _detector(net, meta, im)
# print('Scipy:\n', result)

# OpenCV
im_ori = cv2.imread("images/test.jpg")
im = array_to_image(im_ori)
dn.rgbgr_image(im)
result = _detector(net, meta, im)
print('OpenCV:\n', result)
for det_prediction in result:
    name, prob, box = det_prediction
    x, y, w, h = box
    npround = lambda x: tuple(np.round(x).astype(int))
    up_left = npround((x-w/2, y-h/2))
    down_right = npround((x + w/2, y + h/2))
    if name == b"person":
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)
    cv2.rectangle(im_ori, up_left, down_right, color, thickness=3)
cv2.putText(im_ori, "Num={}".format(len(result)),
            (int(im_ori.shape[1]/2), 120), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)

# cv2.imshow("detection", im_ori)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(cv2.cvtColor(im_ori, cv2.COLOR_BGR2RGB))
plt.show()
pass

