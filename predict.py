import cv2
import matplotlib.pyplot as plt
import numpy as np

def predict(test_sample, y_pred):
    fig, ax = plt.subplots(ncols = 4 , figsize = (20, 20))
    for idx in range(4):
        sample_image = test_sample[0][idx]
        sample_coords = y_pred[1][idx]

        if y_pred[0][idx] > 0.5:
            cv2.rectangle(sample_image,
                        tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                        (255, 0, 0), 2
                        )
            ax[idx].imshow(sample_image)
    plt.show()