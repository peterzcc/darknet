import argparse
import numpy as np
import cv2
import os

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


def main():
    parser = argparse.ArgumentParser(description='mcnn worldexp.')
    parser.add_argument('--data', type=str, default="./data_minibus/image001.png", help='source img')
    parser.add_argument('--output', type=str, default="./cropped_img", help='output dir')
    args = parser.parse_args()
    base_name = args.data.split('/')[-1][:-4]
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    img = cv2.imread(args.data)

    if img.shape[1] != 1920:
        img = cv2.resize(img, dsize=(1920, img.shape[0]*1920/img.shape[1]), interpolation=cv2.INTER_NEAREST)
    center = np.array([960, 540], dtype=np.int)
    box_c = 608
    box_c_2 = box_c/2
    box0 = np.concatenate([center-box_c_2, center+box_c_2]).astype(np.int)
    fullname_center = "{}_{}.jpg".format(base_name, "center")
    cv2.imwrite(os.path.join(args.output,fullname_center), imcrop(img, box0))
    box1_center = center
    box1_center[0] -= box_c
    box1 = np.concatenate([box1_center-box_c_2, box1_center+box_c_2]).astype(np.int)
    fullname_left = "{}_{}.jpg".format(base_name, "left")
    cv2.imwrite(os.path.join(args.output, fullname_left), imcrop(img, box1))
    box2_center = center
    box2_center[0] += box_c
    box2 = np.concatenate([box2_center-box_c_2, box2_center+box_c_2]).astype(np.int)
    fullname_right = "{}_{}.jpg".format(base_name, "right")
    cv2.imwrite(os.path.join(args.output, fullname_right), imcrop(img, box2))




if __name__ == '__main__':
    main()