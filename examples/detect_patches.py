"""Detector functions with different imread methods"""

import ctypes
from pydarknet import darknet_libwrapper as dn
from scipy.misc import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
import multiprocessing as mp
import queue


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


def resize_image(img, target_width=3840):
    im = cv2.resize(img, dsize=(target_width, int(img.shape[0] * target_width / img.shape[1])),
                    interpolation=cv2.INTER_NEAREST)
    return im


def generate_patches_from_image(im_ori, patch_width=608,
                                offsets=np.array(((-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0))),
                                default_offset=np.array([-392, 360], dtype=np.int)):
    center = np.array([im_ori.shape[1]/2, im_ori.shape[0]/2], dtype=np.int)+default_offset
    pw = patch_width
    real_offsets = np.tile(offsets * pw, (1,2))
    center_box = np.concatenate([center-pw/2, center+pw/2]).astype(np.int)
    patch_boxes = np.tile(center_box, (offsets.shape[0], 1)) + real_offsets
    return patch_boxes


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
    for j in range(num_det):
        dj = dets[j]
        preds = [(i, dj.prob[i],(dj.bbox.x,dj.bbox.y,dj.bbox.w,dj.bbox.h))
                 for i in range(meta.classes) if dj.prob[i] > 0]
        res.extend(preds)

    # res = sorted(res, key=lambda x: -x[1])
    dn.free_detections(dets, num_det)
    return res


def draw_detections(im_ori, results, patch_boxes, PERSON_LABEL=0):
    im_proc = im_ori.copy()
    num_pred = 0
    for result, box in zip(results, patch_boxes):
        coord_offset = box[0:2]
        cv2.rectangle(im_proc, tuple(box[0:2]), tuple(box[2:4]), (0, 0, 255), thickness=2)
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


def detect_patches(im_ori, patch_boxes, net, meta, log_time=False):
    results = []
    start_time = time.time()
    for i, box in enumerate(patch_boxes):
        this_patch = imcrop(im_ori, box)
        im_dn = array_to_image(this_patch)
        dn.rgbgr_image(im_dn)
        result = _detector(net, meta, im_dn)
        # dn.free_image(im_dn)
        results.append(result)
    compute_time = time.time() - start_time
    if log_time:
        print("Computation time: {}".format(compute_time))

    return results


def run_test_image(net, meta, show=0, PERSON_LABEL=0):
    im_ori = cv2.imread("images/test.jpg")
    im_ori = resize_image(im_ori)
    patch_boxes = None
    if patch_boxes is None:
        patch_boxes = generate_patches_from_image(
            im_ori)

    det_results = detect_patches(im_ori, patch_boxes, net, meta,log_time=True)
    im_proc, num_pred = draw_detections(im_ori, det_results, patch_boxes, PERSON_LABEL)

    should_show_image = show == 1
    if should_show_image:
        plt.imshow(cv2.cvtColor(im_proc, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        cv2.imwrite("outputs/output.jpg", im_proc)


def run_video_detection(args, datadir, outputdir, video_name, net, meta):
    PERSON_LABEL = None
    for i in range(meta.classes):
        if meta.names[i] == b"person":
            PERSON_LABEL = i
    patch_boxes = None
    print("Processing video: {}".format(video_name))
    out_frames_path = os.path.join(outputdir, video_name)
    assert not os.path.isdir(out_frames_path)
    os.mkdir(out_frames_path)
    in_frames_path = os.path.join(datadir, video_name)
    det_results = None
    for i, frame_name in enumerate(os.listdir(in_frames_path)):
        in_img_path = os.path.join(in_frames_path, frame_name)
        out_img_path = os.path.join(out_frames_path, frame_name)

        im_ori = cv2.imread(in_img_path)
        im_ori = resize_image(im_ori)
        if patch_boxes is None:
            patch_boxes = generate_patches_from_image(
                im_ori)
        if det_results is None or i % args.fskip == 0:
            det_results = detect_patches(im_ori, patch_boxes, net, meta)
        im_proc, num_pred = draw_detections(im_ori, det_results, patch_boxes, PERSON_LABEL)
        if i % 15 == 0:
            print("outputing: {}".format(out_img_path))
        cv2.imwrite(out_img_path, im_proc)


def run_full_detections(args):
    outputdir = args.outputdir
    if os.path.exists(outputdir):
        import shutil
        shutil.rmtree(outputdir, ignore_errors=True)
    os.mkdir(outputdir)
    datadir = args.datadir
    assert os.path.isdir(datadir)
    processes = []
    queues = []
    for gpu_id in args.gpu_ids:
        this_queue = mp.Queue()
        queues.append(this_queue)
        this_process = mp.Process(
            target=detection_process,
            args=(args, datadir, outputdir, gpu_id, this_queue))
        processes.append(this_process)
        this_process.start()
    i = 0
    for video_name in os.listdir(datadir):
        queues[i].put(video_name)
        i = (i + 1) % len(queues)

    for q in queues:
        q.put(None)

    for process in processes:
        process.join()


def detection_process(args, datadir, outputdir, gpu_id, q: mp.Queue):
    dn.cuda_set_device(gpu_id)
    net = dn.load_network("cfg/{}.cfg".format(args.model), "models/{}.weights".format(args.model), 0)
    meta = dn.get_metadata("cfg/coco.data")
    finished = False
    while not finished:
        try:
            video_name = q.get(block=True, timeout=10.0)
            if video_name is not None:
                run_video_detection(args, datadir, outputdir, video_name,
                                    net, meta)
            else:
                finished = True
        except queue.Empty:
            print("empty que")
        else:
            print("uncaught exception")


def main():
    # Darknet
    parser = argparse.ArgumentParser(description='people counting')
    parser.add_argument('--model', type=str, default="yolov3-tiny", help='model name')
    parser.add_argument('--show', type=int, default=0, help='whether to show image')
    parser.add_argument('--fstart', type=int, default=0, help='')
    parser.add_argument('--fend', type=int, default=-1, help='')
    parser.add_argument('--fskip', type=int, default=1, help='')
    parser.add_argument('--datadir', type=str, default="frames_southgate", help='')
    parser.add_argument('--outputdir', type=str, default="frames_output", help='')
    parser.add_argument('--test', type=int, default=1, help='')
    args = parser.parse_args()
    args.gpu_ids = [0, 1, 2, 3]

    if args.test == 1:
        dn.cuda_set_device(0)
        net = dn.load_network("cfg/{}.cfg".format(args.model), "models/{}.weights".format(args.model), 0)
        meta = dn.get_metadata("cfg/coco.data")
        PERSON_LABEL = None
        for i in range(meta.classes):
            if meta.names[i] == b"person":
                PERSON_LABEL = i

        run_test_image(net, meta, args.show, PERSON_LABEL=PERSON_LABEL)
    else:
        run_full_detections(args)


if __name__ == '__main__':
    main()
