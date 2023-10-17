import time

from fastsam import FastSAM, FastSAMPrompt

import os.path

from torch import cuda
import cv2
from matplotlib import pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import mediapipe as mp

IMAGE_PATH = 'path-to-image'


class SegmentPartSam:
    def __init__(
            self,
            model_path="models/FastSAM-x.pt",
            mediapipe_model_path="models/pose_landmarker_heavy.task",
            device_name="cuda" if cuda.is_available() else "cpu",
            imgsz=1024,
    ):
        self.model = FastSAM(model_path)
        self.device_name = device_name
        self.imgsz = imgsz

        print("Device name: ", device_name)

        self.everything_results = self.model(IMAGE_PATH, device=device_name, retina_masks=True, imgsz=imgsz, conf=0.4, iou=0.9)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options_image = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mediapipe_model_path),
            running_mode=VisionRunningMode.IMAGE
        )

        self.landmarker = PoseLandmarker.create_from_options(options_image)

    def segment_part(self, image, part_idxs, return_perspective_numbers=None, show=False):
        """
        :param image: image to segment
        :param part_idxs: index of parts to segment in mediapipe landmark format. Must be a list.
                Example:
                {0 - nose, 1 - left eye (inner), 2 - left eye, 3 - left eye (outer), 4 - right eye (inner), 5 - right eye,
                6 - right eye (outer), 7 - left ear, 8 - right ear, 9 - mouth (left), 10 - mouth (right), 11 - left shoulder,
                12 - right shoulder, 13 - left elbow, 14 - right elbow, 15 - left wrist, 16 - right wrist, 17 - left pinky,
                18 - right pinky, 19 - left index, 20 - right index, 21 - left thumb, 22 - right thumb, 23 - left hip,
                24 - right hip, 25 - left knee, 26 - right knee, 27 - left ankle, 28 - right ankle, 29 - left heel,
                30 - right heel, 31 - left foot index, 32 - right foot index}
        :param show: whether to show the mask
        :param return_perspective_numbers: which perspective to return.
         0 is the default, and will return the mask of the smallest perspective. Example: wrist
         1 will return the mask of the second smallest perspective. Example: full hand
         2 will return the mask of the full object. Example: full person

        :return: mask of part
        """
        assert isinstance(part_idxs, list), "part_idxs must be a list of mediapipe landmark indices (integers). See docstring for more info."
        assert (type(image) == np.ndarray) or (type(image) == str), "image must be a numpy array or a path to an image."

        if type(image) == str:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        results = self.landmarker.detect(mp_image)
        lands = results.pose_landmarks[0]

        idx_to_norm_landmark = {}
        idx_to_actual_landmark = {}
        for idx in range(len(lands)):
            # print(f"{idx} - x {lands[idx].x} y {lands[idx].y} z {lands[idx].z}")
            idx_to_norm_landmark[idx] = (lands[idx].x, lands[idx].y, lands[idx].z)

            idx_to_actual_landmark[idx] = (lands[idx].x * image.shape[1], lands[idx].y * image.shape[0], lands[idx].z)

        mask_comb = None
        image_disp = image.copy()
        for i, part_id in enumerate(part_idxs):
            coords = [round(idx_to_actual_landmark[part_id][0]), round(idx_to_actual_landmark[part_id][1])]
            coords = np.array([coords])

            if return_perspective_numbers is None:
                perspective_number = 0
            else:
                perspective_number = return_perspective_numbers[i]

            # draw circle on image
            if show:
                cv2.circle(image_disp, (coords[0][0], coords[0][1]), 15, (0, 255, 0), -1)

            mask = self.get_mask_from_prompt(image=image, prompt_pt=coords, perspective_number=perspective_number)

            mask_comb = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8) if mask_comb is None else mask_comb

            mask_comb[mask > 0] = i + 1

        if show:
            plt.figure(figsize=(10, 10))

            plt.subplot(1, 2, 1)
            plt.imshow(image_disp)

            plt.subplot(1, 2, 2)
            plt.imshow(mask_comb)
            plt.show()

        return mask_comb

    def get_mask_from_prompt(self, image, prompt_pt, perspective_number=0):
        image = cv2.resize(image, (self.imgsz, self.imgsz))

        prompt_process = FastSAMPrompt(image, self.everything_results, device=self.device_name)

        ann = prompt_process.point_prompt(points=prompt_pt, pointlabel=[1])

        return ann[0]


SegmentPartSam().segment_part(image=IMAGE_PATH, part_idxs=[0], show=True)

sam_cls = SegmentPartSam()

img_arr = cv2.imread(IMAGE_PATH)
img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

start_time = time.time()
for i in range(100):
    sam_cls.segment_part(image=img_arr, part_idxs=[0], show=False)
end_time = time.time()

print("Time taken to run 100 times: ", round(end_time - start_time, 1))
print("Time taken to run 1 time: ", round((end_time - start_time) / 100, 3))
