import csv
import os
import shutil
import threading
import time
import traceback

import cv2
import librosa
import matplotlib.pyplot as plt
import moviepy.editor as moviepy
import numpy as np


def convert_video_to_x_fps(vidcap, fps_out, print_flag=True):
    fps_in = vidcap.get(cv2.CAP_PROP_FPS)
    if print_flag:
        print("fps_in: ", fps_in)
        print("fps_out: ", fps_out)

    index_in = -1
    index_out = -1

    frames = []
    while True:
        success = vidcap.grab()
        if not success: break
        index_in += 1

        out_due = int(index_in / fps_in * fps_out)
        if out_due > index_out:
            success, frame = vidcap.retrieve()
            if not success: break
            index_out += 1

            frames.append(frame)

    return frames


def convert_video_to_frames(cv2_capture, frame_rate, crop_video_length, return_consistent_length=False, print_flag=True):
    """
    If the video is shorter than frame_rate frames, it will be padded with images of zeros
    If the video is longer than frame_rate frames, it will be split into multiple sets of video_length.
    """

    frames = convert_video_to_x_fps(cv2_capture, frame_rate, print_flag=print_flag)

    frame_rate_crop_video_length = round(frame_rate * crop_video_length)  # Target length of each clip expressed in frames

    # split into multiple clips, with the smallest clip at the beginning
    frames_clips = []
    for i in range(len(frames), -1, -frame_rate_crop_video_length):
        if i - frame_rate_crop_video_length - 1 < 0:
            frames_clips.append(frames[0:i])
        else:
            frames_clips.append(frames[i - frame_rate_crop_video_length:i])
    frames_clips.reverse()

    if return_consistent_length:
        # make the smallest clip the same length
        for i in range(len(frames_clips[0]), frame_rate_crop_video_length):
            frames_clips[0].append(np.zeros(frames_clips[0][0].shape, dtype=type(frames_clips[0][0][0][0])))

    if print_flag:
        print("\n__convert_video_to_frames__")
        print("original clips len(frames): ", len(frames))
        print("len(frames_clips): ", len(frames_clips))
        len_of_clips = [len(frames_clips[i]) for i in range(len(frames_clips))]
        print("Small clip has length: ", min(len_of_clips), " at index: ", len_of_clips.index(min(len_of_clips)))
        print("Big clip has length: ", max(len_of_clips), " at index: ", len_of_clips.index(max(len_of_clips)))
        # print("Median clip has length: ", np.median(len_of_clips), " at index: ", len_of_clips.index(np.median(len_of_clips)))

    return frames_clips
