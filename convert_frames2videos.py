import cv2
import numpy as np
import os

from os.path import isfile, join

def convert_frames_to_video(rgb_pathIn, ann_pathIn, pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(rgb_pathIn) if isfile(join(rgb_pathIn, f))] # list of file-name under the path
    files = sorted(files)

    for fname in files:
        #reading each files
        rgb_img = cv2.imread(join(rgb_pathIn,fname))
        if ann_pathIn is not None:
            ann_img = cv2.imread(join(ann_pathIn,fname).split('.')[0] + '.png.png')
            img = np.concatenate([rgb_img, ann_img],axis=1)
        else:
            img = rgb_img

        height, width, layers = img.shape
        size = (width, height)
        #inserting the frames into an image array
        frame_array.append(img)

    # writing to a image array
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

def main():
    rgb_pathIn = '/nfs/hpc/share/yuanjial/Code/omni3d/output/demo/DLA34_pickup_examples'
    ann_pathIn = None
    info_file = None
    fps = 1.0

    pathOut = 'output/demo'
    os.makedirs(pathOut, exist_ok=True)

    if info_file is not None:
        with open(info_file) as f:
            fdir_list = [x.strip() for x in f.readlines()]
    else:
        fdir_list = os.listdir(rgb_pathIn)

    for video in fdir_list:
        rgb_src_path = join(rgb_pathIn, video)
        ann_src_path = join(ann_pathIn, video) if ann_pathIn is not None else None
        print(rgb_src_path, ' | ', ann_src_path)
        if os.path.exists(rgb_src_path) and (ann_src_path is None or os.path.exists(ann_src_path)):
            dst_path = os.path.join(pathOut, video + '.mp4')
            convert_frames_to_video(rgb_src_path, ann_src_path, dst_path, fps)
        else:
            print('One path not exists.')

if __name__=="__main__":
    main()
