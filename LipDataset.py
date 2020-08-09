import os
import cv2
import numpy as np

main_set_path = "dataset/pretrain"
vocab_path = "vocab.txt"
training_align_path = "training/unseen_speakers/datasets/align"
training_path = "training/unseen_speakers/datasets/train"
MAX_FRAME_COUNT = 154


class LipDataset():
    def write(output_path,text_path):
        text_file = open(text_path, "r")
        text = text_file.readlines()[4:]
        output = ""
        for line in text:
            w = line.split()
            output += str(int(float(w[1]) * 10000)) + " " + str(int(float(w[2]) * 10000)) + " " + w[0] + "\n"
        text[-1] = text[-1][:-1]
        align_file = open(os.path.join(output_path, text_path[-9:-4] + ".align"), "w")
        align_file.write(output)
        align_file.close()
        text_file.close()

    dirs = os.listdir(main_set_path)
    for index, dir in enumerate(dirs):
        output_path = os.path.join(training_align_path, "s" + str(index + 1))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        path = os.path.join(main_set_path,dir)
        text_files = [name for name in os.listdir(path) if name.endswith(".txt")]
        video_files = [name for name in os.listdir(path) if name.endswith(".mp4")]
        if len(text_files) != len(video_files):
            print ("WARNING !! check dataset")
            continue
        output_video_path = os.path.join(training_path, "s" + str(index + 1))
        if not os.path.exists(output_video_path):
            os.mkdir(output_video_path)
        call = "py scripts/extract_mouth_batch.py " + os.path.join(path," *.mp4 ") + output_video_path + " common/predictors/shape_predictor_68_face_landmarks.dat"
        os.system(call)
        for text_path in text_files:
            write(output_path, os.path.join(path,text_path))
        