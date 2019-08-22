import os
import sys
import glob
import time
import tarfile
import argparse
import numpy as np
import tensorflow as tf 
from tqdm import tqdm
from PIL import Image 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str)
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--mask', type=str, default=None)
parser.add_argument('--size', type=int, default=513)

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path, input_size, mask_file = None):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    self.INPUT_SIZE = input_size

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

    if mask_file is not None:
      mask = Image.open(mask_file)
      width, height = mask.size
      resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
      target_size = (int(resize_ratio * width), int(resize_ratio * height))
      resized_mask = mask.resize(target_size, Image.NEAREST)
      self.mask = np.array(resized_mask)
      if len(self.mask.shape) == 3:
          self.mask = self.mask[..., 0]
      assert(len(self.mask.shape) == 2)
      self.mask = self.mask == 0
    else:
      self.mask = None

    return

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    t = time.time()
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)[..., np.newaxis]]})
    # print(target_size, ': %lfs' % (time.time() - t))
    seg_map = batch_seg_map[0]

    if self.mask is not None:
      seg_map[self.mask] = 255

    res = Image.fromarray(seg_map).resize(image.size, Image.NEAREST)
    return res


def create_autovision_label_colormap():
  """Creates a label colormap used in AutoVision.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  
  colormap = np.zeros((256, 3), dtype=np.uint8)
  _PALETTE = np.array(
    [[165, 42, 42], # Bird
    [0, 192, 0], # Ground Animal
    [196, 196, 196], # Curb
    [190, 153, 153], # Fence
    [180, 165, 180], # Guard Rail
    [90, 120, 150], # Barrier
    [102, 102, 156], # Wall
    [128, 64, 255], # Bike Lane
    [140, 140, 200], # Crosswalk - Plain
    [170, 170, 170], # Curb Cut
    [250, 170, 160], # Parking
    [96, 96, 96], # Pedestrian Area
    [230, 150, 140], # Rail Track
    [128, 64, 128], # Road
    [110, 110, 110], # Service Lane
    [244, 35, 232], # Sidewalk
    [150, 100, 100], # Bridge
    [70, 70, 70], # Building
    [150, 120, 90], # Tunnel
    [220, 20, 60], # Person
    [255, 0, 0], # Bicyclist
    [255, 0, 100], # Motorcyclist
    [255, 0, 200], # Other Rider
    [200, 128, 128], # Lane Marking - Crosswalk
    [255, 255, 255], # Lane Marking - General
    [64, 170, 64], # Mountain
    [230, 160, 50], # Sand
    [70, 130, 180], # Sky
    [190, 255, 255], # Snow
    [152, 251, 152], # Terrain
    [107, 142, 35], # Vegetation
    [0, 170, 30], # Water
    [255, 255, 128], # Banner
    [250, 0, 30], # Bench
    [100, 140, 180], # Bike Rack
    [220, 220, 220], # Billboard
    [220, 128, 128], # Catch Basin
    [222, 40, 40], # CCTV Camera
    [100, 170, 30], # Fire Hydrant
    [40, 40, 40], # Junction Box
    [33, 33, 33], # Mailbox
    [100, 128, 160], # Manhole
    [142, 0, 0], # Phone Booth
    [70, 100, 150], # Pothole
    [210, 170, 100], # Street Light
    [153, 153, 153], # Pole
    [128, 128, 128], # Traffic Sign Frame
    [0, 0, 80], # Utility Pole
    [250, 170, 30], # Traffic Light
    [192, 192, 192], # Traffic Sign (Back)
    [220, 220, 0], # Traffic Sign (Front)
    [140, 140, 20], # Trash Can
    [119, 11, 32], # Bicycle
    [150, 0, 255], # Boat
    [0, 60, 100], # Bus
    [0, 0, 142], # Car
    [0, 0, 90], # Caravan
    [0, 0, 230], # Motorcycle
    [0, 80, 100], # On Rails
    [128, 64, 64], # Other Vehicle
    [0, 0, 110], # Trailer
    [0, 0, 70], # Truck
    [0, 0, 192], # Wheeled Slow
    [32, 32, 32], # Car Mount
    [120, 10, 10], # Ego Vehicle
    [220, 180, 50], # Unpaved Road
    [0, 0, 0]], # Unlabeled
    dtype=np.uint8)
  colormap[:67,:] = _PALETTE
  return colormap.flatten()

def create_autovision_simple_label_colormap():
  """Creates a label colormap used in AutoVision.

  Returns:
    A Colormap for visualizing segmentation results.
  """

  colormap = np.zeros((256, 3), dtype=np.uint8)
  _PALETTE = np.array(
    [[90, 120, 150], # Barrier
    [70, 70, 70], # Building
    [0, 0, 142], # Car
    [152, 251, 152], # Terrain
    [0, 60, 100], # Heavy Vehicle
    [119, 11, 32], # Motorcycle
    [128, 64, 128], # Paved Road
    [170, 170, 170], # Pedestrian Area
    [220, 20, 60], # Person
    [250, 170, 30], # Pole Object
    [70, 130, 180], # Sky
    [220, 180, 50], # Unpaved Road
    [107, 142, 35], # Vegetation
    [0, 170, 30], # Water
    [255, 255, 255]], # Ignored Object
    dtype=np.uint8)
  colormap[:15,:] = _PALETTE
  return colormap.flatten()

if __name__ == '__main__':
  args = parser.parse_args()
  color_map = create_autovision_simple_label_colormap()
  os.popen('mkdir -p ' + args.output)

  MODEL = DeepLabModel(args.model, args.size, args.mask)
  print('Model loaded successfully!')

  im_files = glob.glob(args.input + '/*.jpg') + glob.glob(args.input + '/*.png')
  for im_file in tqdm(sorted(im_files)):
    basename = os.path.basename(im_file).replace('.jpg', '.png')
    im = Image.open(im_file)
    res = MODEL.run(im)
    res = res.convert('L')
    res.putpalette(color_map)
    res.save(args.output + '/' + basename)
