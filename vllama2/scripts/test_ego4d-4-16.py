import os
import os.path as osp
from tqdm import trange

def with_opencv(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return fps, frame_count


def compare_frame_loading(filename):
    import cv2
    cap = cv2.VideoCapture(filename)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frame_count)

    from time import time

    start = time()
    for i in trange(100):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * 10)
        ret, frame = cap.read()
    end = time()
    print('setting', end - start)

    cap = cv2.VideoCapture(filename)
    start = time()
    for i in trange(99 * 10):
        ret, frame = cap.read()
    end = time()
    print('iterating', end - start)




if __name__ == '__main__':
    folder = '/grogu/datasets/ego4d/v1/full_scale'
    files = os.listdir(folder)
    files = sorted([f for f in files if f.endswith('.mp4')])

    for f in files[:10]:
        fps, frame_count = with_opencv(osp.join(folder, f))
        print(f, fps, frame_count)

    compare_frame_loading(osp.join(folder, '02d1f024-1470-4ce6-acde-b938f7847eb6.mp4'))