from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Detection.tiny_face_model import Model
import Detection.util as util
from argparse import ArgumentParser
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import pandas as pd
import pylab as pl
import time
import os
import sys
from scipy.special import expit
import glob

MAX_INPUT_DIM = 5000.0
img_crop = 0
class TinyFace(object):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.evaluate(data_dir)

  def overlay_bounding_boxes(self, raw_img, refined_bboxes):
    i = 1
    global img_crop
    # Overlay bounding boxes on an image with the color based on the confidence.
    for r in refined_bboxes:
      _score = expit(r[4])
      cm_idx = int(np.ceil(_score * 255))
      rect_color = [int(np.ceil(x * 255)) for x in util.cm_data[cm_idx]]  # parula

      _r = [int(x) for x in r[:4]]
      cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), (0, 255 , 255), 3)
      #img_crop = raw_img[ _r[1] - 5 : _r[3] + 5,  _r[0] -5 : _r[2] + 5 ,:]
      #img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
      #imgPath.append(img_crop)
      # img_crop = cv2.imwrite(os.path.join("detection", str(i) + ".png"), img_crop)
      # i = i + 1  
  def getArr(self):
    return imgPath

  def evaluate(self, data_dir):
    prob_thresh=0.5
    nms_thresh=0.1

    weight_file_path = "./Detection/pickle_file/mat2tf.pkl"
    # placeholder of input images. Currently batch size of one is supported.
    x = tf.placeholder(tf.float32, [1, None, None, 3]) # n, h, w, c

    # Create the tiny face model which weights are loaded from a pretrained model.
    model = Model(weight_file_path)
    score_final = model.tiny_face(x)

    with open(weight_file_path, "rb") as f:
      _, mat_params_dict = pickle.load(f)

    average_image = model.get_data_by_key("average_image")
    clusters = model.get_data_by_key("clusters")
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    normal_idx = np.where(clusters[:, 4] == 1)

    # main
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      fname = data_dir.split(os.sep)[-1]
      raw_img = cv2.imread(data_dir)
      raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
      raw_img_f = raw_img.astype(np.float32)

      def _calc_scales():
        raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
        min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                        np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
        max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
        scales_down = pl.frange(min_scale, 0, 1.)
        scales_up = pl.frange(0.5, max_scale, 0.5)
        scales_pow = np.hstack((scales_down, scales_up))
        scales = np.power(2.0, scales_pow)
        return scales

      scales = _calc_scales()
      start = time.time()

      # initialize output
      bboxes = np.empty(shape=(0, 5))

      # process input at different scales
      for s in scales:
        print("Processing {} at scale {:.4f}".format(fname, s))
        img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        img = img - average_image
        img = img[np.newaxis, :]
        # we don't run every template on every scale ids of templates to ignore
        tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
        ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

        # run through the net
        score_final_tf = sess.run(score_final, feed_dict={x: img})

        # collect scores
        score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
        prob_cls_tf = expit(score_cls_tf)
        prob_cls_tf[0, :, :, ignoredTids] = 0.0

        def _calc_bounding_boxes():
          # threshold for detection
          _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)

          # interpret heatmap into bounding boxes
          cy = fy * 8 - 1
          cx = fx * 8 - 1
          ch = clusters[fc, 3] - clusters[fc, 1] + 1
          cw = clusters[fc, 2] - clusters[fc, 0] + 1

          # extract bounding box refinement
          Nt = clusters.shape[0]
          tx = score_reg_tf[0, :, :, 0:Nt]
          ty = score_reg_tf[0, :, :, Nt:2*Nt]
          tw = score_reg_tf[0, :, :, 2*Nt:3*Nt]
          th = score_reg_tf[0, :, :, 3*Nt:4*Nt]

          # refine bounding boxes
          dcx = cw * tx[fy, fx, fc]
          dcy = ch * ty[fy, fx, fc]
          rcx = cx + dcx
          rcy = cy + dcy
          rcw = cw * np.exp(tw[fy, fx, fc])
          rch = ch * np.exp(th[fy, fx, fc])

          scores = score_cls_tf[0, fy, fx, fc]
          tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
          tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
          tmp_bboxes = tmp_bboxes.transpose()
          return tmp_bboxes

        tmp_bboxes = _calc_bounding_boxes()
        bboxes = np.vstack((bboxes, tmp_bboxes)) # <class 'tuple'>: (5265, 5)


      print("time {:.2f} secs for {}".format(time.time() - start, fname))

      refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                max_output_size=bboxes.shape[0], iou_threshold=nms_thresh)
      refind_idx = sess.run(refind_idx)
      refined_bboxes = bboxes[refind_idx]
      self.overlay_bounding_boxes(raw_img, refined_bboxes)
      plt.imshow(raw_img)
      plt.show()
   