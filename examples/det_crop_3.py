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


def imcrop(img, bbox):
   x1, y1, x2, y2 = bbox
   if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
   return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def resize_image(img, target_width=1920):
    im = cv2.resize(img, dsize=(target_width, int(img.shape[0] * target_width / img.shape[1])),
                    interpolation=cv2.INTER_NEAREST)
    return im


def generate_patches_from_image(im_ori, patch_width=608,
                                offsets=np.array(((-1,0), (0,0), (1,0)))):
    center = np.array([im_ori.shape[1]/2, im_ori.shape[0]/2], dtype=np.int)
    pw = patch_width
    real_offsets = np.tile(offsets * pw, (1,2))
    center_box = np.concatenate([center-pw/2, center+pw/2]).astype(np.int)
    patch_boxes = np.tile(center_box, (offsets.shape[0], 1)) + real_offsets
    return patch_boxes


def get_detection_tuple(this_detection, meta):
    probs = np.array([this_detection.prob[i] for i in range(meta.classes)])
    pred_class = np.argmax(probs)
    b = this_detection.bbox
    this_result = (pred_class, this_detection.prob[pred_class], (b.x, b.y, b.w, b.h))
    return this_result

def _detector(net, meta, image, thresh=.5, hier=.5, nms=.45):
    # dn.cuda_set_device(0)
    num_det = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num_det)
    dn.network_predict_image(net, image)
    dets = dn.get_network_boxes(net, image.w, image.h, thresh, hier, None, 0, num_ptr)
    num_det = num_ptr[0]
    if (nms):
        dn.do_nms_sort(dets, num_det, meta.classes, nms)

    res = []
    # for j in range(num_det):
    #     for i in range(meta.classes):
    #         if dets[j].prob[i] > 0:
    #             b = dets[j].bbox
    #             res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))

    res = list(map(lambda i: get_detection_tuple(dets[i], meta), range(num_det)))

    # res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    dn.free_detections(dets, num_det)
    return res


def count_people_and_draw_detections(im_ori, patch_boxes, net, meta, PERSON_LABEL=0):
    num_pred = 0
    im_proc = im_ori.copy()
    results = []
    for i, box in enumerate(patch_boxes):

        this_patch = imcrop(im_ori, box)
        im_dn = array_to_image(this_patch)
        dn.rgbgr_image(im_dn)
        result = _detector(net, meta, im_dn)
        results.append(result)
        # print('Results:\n', result)

    for i, box in enumerate(patch_boxes):
        coord_offset = box[0:2]
        result = results[i]
        for det_prediction in result:
            label, prob, box = det_prediction
            x, y, w, h = box
            npround = lambda x: tuple((np.round(x) + coord_offset).astype(int))
            up_left = npround((x - w / 2, y - h / 2))
            down_right = npround((x + w / 2, y + h / 2))
            if label == PERSON_LABEL:
                color = (0, 255, 0)
                num_pred += 1
            else:
                color = (255, 0, 0)
            cv2.rectangle(im_proc, up_left, down_right, color, thickness=2)
    cv2.putText(im_proc, "Num={}".format(num_pred),
                (int(im_proc.shape[1] / 2), 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
    return im_proc, num_pred

def main():
    # Darknet
    # net = dn.load_network("cfg/yolov3.cfg", "models/yolov3.weights", 0)
    dn.cuda_set_device(0)
    net = dn.load_network("cfg/yolov3-tiny.cfg", "models/yolov3-tiny.weights", 0)
    meta = dn.get_metadata("cfg/coco.data")
    PERSON_LABEL = None
    for i in range(meta.classes):
        if meta.names[i] == b"person":
            PERSON_LABEL = i
    patch_boxes = None

    # OpenCV
    im_ori = cv2.imread("images/test.jpg")
    im_ori = resize_image(im_ori)
    if patch_boxes is None:
        patch_boxes = generate_patches_from_image(im_ori)

    im_proc, num_pred = \
        count_people_and_draw_detections(im_ori, patch_boxes, net, meta, PERSON_LABEL)

    should_show_image = False
    if should_show_image:
        plt.imshow(cv2.cvtColor(im_proc, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        cv2.imwrite("outputs/output.jpg", im_proc)


if __name__ == '__main__':
    main()
