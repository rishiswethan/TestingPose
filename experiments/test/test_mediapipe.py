import time

import mediapipe as mp
import utils
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "models/pose_landmarker_heavy.task"
# model_path = "models/pose_landmarker_full.task"
# model_path = "models/pose_landmarker_lite.task"

video_path = '/mnt/Extra/downloads/complex_dance_HD.mp4'

# Create a pose landmarker instance with the video mode:
options_video = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

options_image = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

fps = 30

frames = utils.convert_video_to_x_fps(cv2.VideoCapture(video_path), fps_out=fps, print_flag=True)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        print("landmark idx: ", idx)
        print(pose_landmarks)

    return annotated_image


with PoseLandmarker.create_from_options(options_video) as landmarker:
    delay_ms = 1000 / fps

    ms = 0
    total_infer_time = 0
    total_infers = 0
    for frame in frames:
        start_time = time.time()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        pose_landmarker_result = landmarker.detect_for_video(mp_image, round(ms))
        # pose_landmarker_result = landmarker.detect(mp_image)

        ms += delay_ms
        total_infer_time += time.time() - start_time
        total_infers += 1

        print("pose_landmarker_result: ", pose_landmarker_result)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        cv2.imshow("", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # delay for the next frame
        time_to_sleep = delay_ms - (time.time() - start_time)
        if time_to_sleep > 0:
            cv2.waitKey(round(time_to_sleep))

print("total_infer_time: ", total_infer_time)
print("total_infers: ", total_infers)
print("per frame: ", total_infer_time / total_infers)
