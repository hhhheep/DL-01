import cv2 as cv
import numpy
import pandas as pd
import os

path = os.getcwd()
path = path + "\\images\\"

def read_local_txt(file,path = path):

    train_path = path + str(file) +'.txt'
    file_inf = pd.read_csv(train_path, header=None, sep=" ")
    file_inf.columns = ["image", "label"]
    return file_inf

def train_batch(file_inf,batch_size = 20):

    if len(file_inf) < batch_size:
        batch_size = len(file_inf)
    batch_inf = file_inf.sample(n=batch_size,replace=False,random_state=123)
    file_inf = file_inf.drop(index = batch_inf.index)

    batch = batch_inf.iloc[:,0]
    label = batch_inf.iloc[:,1]

    return batch,label,file_inf,batch_size

def read_image(batch,path = path):
    batch = list(batch)
    if len(batch)>1:

        images = []
        for img_path in batch:
            img = cv.imread(path + str(img_path))#???修改 路徑格式
            images.append(img)
        return images
    else:
        imag = cv.imread(path + str(batch[0]))

        return imag



if __name__ == '__main__':
    pass

