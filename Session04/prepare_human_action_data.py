import os
import subprocess
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import re
import cv2 as cv

def prepare_human_action_data():
    """
    One Time usage after unziping  all needed files into /data/videos.
    Files: https://www.csc.kth.se/cvap/actions/
    Code from: https://github.com/tejaskhot/KTH-Dataset
    Prepares a split of Train, Test, Validation data of the human actions dataset.
    """
    data_path = os.getcwd()
    print(data_path)
    # sequences_list = ''
    video_files=os.listdir(data_path + '/data/videos/')
    video_files.sort()

    # extract frames from video clips
    args=['ffmpeg', '-i']
    for video in video_files:
        # print video
        video_name = video[:-11]    # remove '_uncomp.avi' from name
        # print 'video name is: ', video_name
        frame_name = 'frame_%d.jpg'    # count starts from 1 by default
        os.makedirs(data_path + '/data/frames/'+video_name)
        args.append(data_path + '/data/videos/'+video)
        args.append(data_path + '/data/frames/'+video_name+'/'+frame_name)
        ffmpeg_call = ' '.join(args)
        # print ffmpeg_call
        # print args
        subprocess.call(ffmpeg_call, shell=True)        # execute the system call
        args=['ffmpeg', '-i']
        if (video_files.index(video) + 1) % 50 == 0:
            print('Completed till video : ', (video_files.index(video) + 1))
                

    print('[MESSAGE]    Frames extracted from all videos')

    os.makedirs(data_path + '/data/' + 'TRAIN')
    os.makedirs(data_path + '/data/' + 'VALIDATION')
    os.makedirs(data_path + '/data/' + 'TEST')

    train = [11, 12, 13, 14, 15, 16, 17, 18]
    validation =[19, 20, 21, 23, 24, 25, 1, 4]
    test = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    # read file line by line and strip new lines
    lines = [line.rstrip('\n').rstrip('\r') for line in open('sequences_list.txt')]
    # remove blank entries i.e. empty lines
    lines = filter(None, lines)
    # split by tabs and remove blank entries
    lines = [filter(None, line.split('\t')) for line in lines]

    # code is in python 2.7 and returns filter object, convert to list with following:
    l_temp=[]
    for l in lines:
        l_temp.append(list(l))
    l_temp.sort()
    lines=l_temp
    del l_temp

    lines.sort()

    success_count=0
    error_count=0
    for line in lines:
        vid = line[0].strip(' ')
        subsequences = line[-1].split(',')
        person = int(vid[6:8])
        seq_list = []
        if person in train:
            move_to = 'TRAIN'
        elif person in validation:
            move_to = 'VALIDATION'
        else:
            move_to = 'TEST'
        try:
            for seq in subsequences:
                limits=seq.strip(' ').split('-')
                seq_list.append(limits[0])
                seq_list.append(limits[1])
            seq_path=data_path + '/data/' + move_to + '/' + vid + '_frame_' + seq_list[0] + '_' + seq_list[-1]
            os.makedirs(seq_path)
        except:
            print('-----------------------------------------------------------')
            print('[ERROR MESSAGE]: ')
            print('limits : ', limits) 
            print('seq_path : ', seq_path)
            print('-----------------------------------------------------------')
            continue  
            
        error_flag=False
        for i in range(int(seq_list[0]), int(seq_list[-1])+1):
            src = data_path + '/data' + '/frames/' + vid + '/frame_' + str(i) + '.jpg'
            # print i, src, limits
            dst = seq_path
            try:
                shutil.copy(src, dst)
            except:
                error_flag = True
        if error_flag:
            print("[ERROR]: ", seq_path)
            error_count+=1
       

        if (lines.index(line) + 1) % 50 == 0:
            print('Completed till video : ', (lines.index(line) + 1))
        success_count+=1

    print('[ALERT]        Total error count is : ', error_count)
    print('[MESSAGE]    Data split into train, validation and test')

def load_human_action_data(mode: str, seq_length: int):
    """
    Loads a dataset from the given prepare dataset human action path.
    Input:
        mode: 
            Mode of data, either train, valid or test mode
        seq_length:
            length of frames given for the dataset
    """

    # for each mode the data hence the path changes
    assert mode == "test" or mode == "train" or mode == "valid", "Error: give correct mode. Mode must be either test, train, valid."
    match mode:
        case "train":
            path = "./data/TRAIN"
        case "test":
            path = "./data/TEST"
        case "valid":
            path = "./data/VALIDATION"


    data = []
    labels = []
    # labeling according to index
    actions = ["walking", "running", "jogging", "handwaving", "handclapping", "boxing"]
    j = 0
    for root, dirs, files in os.walk(path, topdown=True):
        j += 1
        # sorts according to true integer value
        action_frame_list = []
        action_label_list = []
        for i, name in enumerate(sorted(files, key=lambda f: int(re.sub('\D', '', f)))):
            # save sequence to data, append new sequences afterwards
            # in case empty directiories are walked through test for empty list
            if (i % seq_length) == 0 and i != 0 and action_frame_list != [] :
                data.append(action_frame_list)
                labels.append(action_label_list)
                action_frame_list = []
                action_label_list = []
            # load image and save to list
            image = cv.imread(root + "/" + name, cv.IMREAD_GRAYSCALE)
            action_frame_list.append(image)
            # set full path name to find label according to file name
            full_name = os.path.join(root, name)[12:]
            # search for action label in string
            for act in actions:
                if re.search(act, full_name) != None:
                    action_label_list.append(actions.index(act))
                    break            

    data = torch.tensor(np.array(data))
    labels = torch.tensor(np.array(labels))
    return data, labels, actions


class HumanActionDataset(Dataset):
    def __init__(self, sequence_length: int=20, train: bool=True, transform: T=None):
        # concat valid and train data
        if train == True:
            train_data, train_labels, actions = load_human_action_data(mode="train", seq_length=sequence_length)
            valid_data, valid_labels, actions = load_human_action_data(mode="valid", seq_length=sequence_length)
            data = torch.concat((train_data, valid_data), dim=0)
            labels = torch.concat((train_labels, valid_labels), dim=0)
        elif train == False:
            data, labels, actions = load_human_action_data(mode="test", seq_length=sequence_length)
        self.seq_data = data
        self.seq_labels = labels
        self.action_labels = actions
        self.transform = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        # convert to float as well
        seq = self.seq_data[idx, :, :, :].float()
        # as a sequence contains 20 frames with each frame a label, we can get one label from each frame
        label = self.seq_labels[idx, -1]

        if self.transform:
            seq = self.transform(seq)

        return seq, label