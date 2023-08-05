import tensorflow as tf
import json
import matplotlib.pyplot as plt 
import cv2
import numpy as np

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


def load_labels(label_path):
  with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
    label = json.load(f)
  return [label['class']], label['bbox']


def visualise(res):
    fig, ax = plt.subplots(ncols = 4, figsize = (20,20))
    for idx in range(4):
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]

        cv2.rectangle(sample_image,
                        tuple(np.multiply(sample_coords[:2],[120,120]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                        (255, 0, 0), 2
                        )
        ax[idx].imshow(sample_image)
    plt.show()