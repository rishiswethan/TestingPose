import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from mmpose.apis import MMPoseInferencer
import utils


def visualize_keypoints(img_path, keypoints, pairs):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize keypoints to the size of the image
    height, width, _ = img.shape
    keypoints = [(int(x*width), int(y*height)) for x, y, _ in keypoints]

    # Draw keypoints and lines on the image
    for point in keypoints:
        cv2.circle(img, point, 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    for pair in pairs:
        cv2.line(img, keypoints[pair[0]], keypoints[pair[1]], (0, 255, 0), 2)

    return img


class Infer3D:
    def __init__(self, device='cuda'):
        self.inferencer = MMPoseInferencer(pose3d='human3d', device=device)

    def infer(self, img_path, return_vis=False, save_vis=False):
        output_dir = "vis" if save_vis else None

        result_generator = self.inferencer(img_path, show=False, out_dir=output_dir, return_vis=return_vis)
        result = next(result_generator)

        preds = result['predictions'][0]

        if return_vis:
            result_vis = result['visualization'][0]

            return result_vis, preds
        else:
            return preds

    def infer_video(self, video_path, return_vis=True, show_vis=False):

        frames = utils.convert_video_to_x_fps(cv2.VideoCapture(video_path), fps_out=3, print_flag=True)
        print("len(frames): ", len(frames))
        result_generator = self.inferencer(frames, show=False, out_dir=None, return_vis=return_vis)

        # results = next(result_generator)
        # print("results: ", results)

        visualisations = []
        preds = []
        start_time = time.time()
        for result in result_generator:
            plt.clf()  # Clear the plot

            preds.append(result['predictions'][0])

            if return_vis:
                vis = result['visualization'][0]
                visualisations.append(vis)

                if show_vis:
                    plt.imshow(vis)
                    plt.pause(0.01)  # Pause for 50 ms

                print("len(visualisations): ", len(visualisations))
                print("len(preds): ", len(preds))

        plt.clf()  # Clear the plot
        time_taken = time.time() - start_time

        print("num of frames: ", len(frames))
        print("time_taken: ", time_taken, " seconds")
        print("per frame: ", time_taken / len(frames), " seconds")



# img_path = '/mnt/nvme0n1p3/MySSD/Programming/AI/ClientProjects/develop/InstaMike/bkp/img.png'   # replace this with your own image path
img_path = '/mnt/Extra/downloads/complex_dance_HD.mp4'   # replace this with your own image path
# img_path = "/mnt/Extra/downloads/slow_dance.mp4"

infer3d = Infer3D()
results = infer3d.infer_video(img_path, return_vis=True, show_vis=True)
