import os, glob
from PIL import Image
from tqdm import tqdm
import numpy as np

max_length = 513
mapping = [14,14,7,0,0,0,1,6,6,7,6,7,14,6,6,7,1,1,1,8,8,8,8,6,6,14,11,10,14,3,12,13,14,14,14,14,7,14,14,14,14,6,14,6,9,9,14,9,9,14,14,14,5,14,4,2,14,5,14,14,4,4,14,14,14,11,255]

def get_target_size(img):
    width, height = img.size
    resize_ratio = 1.0 * max_length / max(width, height)
    return (int(resize_ratio * width), int(resize_ratio * height))

def resize_and_save(img, size, sample, filename):
    resized_img = img.resize(size, sample)
    resized_img.save(filename)
    return

gt_files = sorted(glob.glob('./gtFine/*/*/*.png'))
for gt_file in tqdm(gt_files):
    img_file = gt_file.replace('gtFine', 'leftImg8bit')
    gt = np.array(Image.open(gt_file))
    for org, now in enumerate(mapping):
        gt[gt == org] = now
    gt = Image.fromarray(gt)
    img = Image.open(img_file)
    size = get_target_size(img)
    resize_and_save(img, size, Image.ANTIALIAS, img_file)
    resize_and_save(gt, size, Image.NEAREST, gt_file)
