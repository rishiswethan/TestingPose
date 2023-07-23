import time

import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib.pyplot as plt

import cv2


model = 'models/selfie_multiclass_256x256.tflite'  # To segment face, hair, and background
# model = 'models/hair_segmenter.tflite'  # To segment hair


BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a image segmenter instance with the image mode:
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model),
    running_mode=VisionRunningMode.IMAGE,
    output_confidence_masks=True)

with ImageSegmenter.create_from_options(options) as segmenter:
    image = cv2.imread('input/stock-photo-causal-businessman-standing-with-hands-at-sides-isolated-on-white-background-497478265.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Run the image segmenter and calculate the mask
    segmented_masks = segmenter.segment(mp_image).confidence_masks

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(len(segmented_masks)):
        print("segmented_masks[i].shape: ", segmented_masks[i].numpy_view().shape)
        mask[segmented_masks[i].numpy_view() > 0.5] = i

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()

    print("segmented_masks: ", segmented_masks)


    # benchmark by running the model 100 times
    print("Benchmarking...")
    start_time = time.time()
    for _ in range(100):
        segmenter.segment(mp_image)
    end_time = time.time()

    print("Time taken to run 100 times: ", round(end_time - start_time, 1))
    print("Time taken to run 1 time: ", round((end_time - start_time) / 100, 3))