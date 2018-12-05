import cv2
import time
import math
import os
from skimage import io
import numpy as np
from PIL import Image
import copy

# tfScoreMaps_path="./output_files_tf_test/"
# tfScoreMap_path="./output_files_tf_test/im_resized.txt"
# tfScoreMap_path="./output_files_tf_test/1_(92).jpg_score_map.txt"
tfScoreMap_path="./output/logits/0_bn.txt"
# tfScoreMaps_path="./output/logits/0.txt"
# tfScoreMap_path="./output_files_tf_test/1_(88).jpg_scoreMap.txt"
rtScoreMap_path="./output/logits_pb/0_bn.txt"
# rtScoreMaps_path="./output/logits_pb/0.txt"

# rtScoreMap_path="./output_files_rtPy_test/1_(12).jpg_h_output.txt"
# rtScoreMap_path="output_files_rtC++_test/1_(12).jpg_h_output_ckpt-537321.txt"
# rtScoreMap_path="./output_files_rtC++_test/1_(12).jpg_h_output.txt"
tfScoreMapList=[]
rtScoreMapList=[]

def get_files(files_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['txt']
    for filename in os.listdir( files_path ):
        for ext in exts:
            if filename.endswith(ext):
                print (filename)
                files.append(os.path.join(files_path+filename))
                break
    print('Find {} files'.format(len(files)))
    return files

# def getImgs(imgs_path):
#
#     scoreMapList=[]
#     im_fn_list = get_images(imgs_path)
#     for im_fn in im_fn_list:
#         scoreMapList.append(cv2.imread(im_fn, cv2.IMREAD_GRAYSCALE))
#     return scoreMapList

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    print ("~~~~~numberOfLines:{}".format(numberOfLines))
    returnMat = np.zeros((numberOfLines,1024))        #prepare matrix to return
    value_list=[]
    index = 0
    for line in arrayOLines:
        line = line.strip()
        # print line.split(' ')[:]
        returnMat[index,:] = line.split(' ')[:]
        index += 1
    return returnMat.astype(np.float)

def getScoreMaps(files_path):

    scoreMapList=[]
    file_list = get_files(files_path)
    for filename in file_list:
        scoreMapList.append(file2matrix(filename))
    return file_list, scoreMapList

def getScoreMap(file_path):
    return file2matrix(file_path)

def compareScoreMaps(tfScoreMaps_path, rtScoreMaps_path):

    tfScoreMap_fileList, tfScoreMapList=getScoreMaps(tfScoreMaps_path)
    rtScoreMap_fileList=[]
    for item in tfScoreMap_fileList:
        print(item.replace('./output_files_tf_test/', './output_files_pb_test/'))
        rtScoreMap_fileList.append(item.replace('./output_files_tf_test/', './output_files_pb_test/'))

    rtScoreMapList=[]
    for filename in rtScoreMap_fileList:
        rtScoreMapList.append(file2matrix(filename))

    # rtScoreMap_fileList, rtScoreMapList=getScoreMaps(rtScoreMaps_path)
    print (rtScoreMap_fileList)
    assert(len(tfScoreMapList)==len(rtScoreMapList))
    for index in range(len(tfScoreMapList)):
        print("~~~~~~~~~index:{}".format(index))
        print("~~~~~~~~~tfScoreMap filename:{}".format(tfScoreMap_fileList[index].split("/")[-1]))
        print("~~~~~~~~~rtScoreMap filename:{}".format(rtScoreMap_fileList[index].split("/")[-1]))
        assert(tfScoreMap_fileList[index].split("/")[-1]==rtScoreMap_fileList[index].split("/")[-1])
        print("~~~~~~~~~tfScoreMap.shape:{}".format(tfScoreMapList[index].shape))
        print("~~~~~~~~~rtScoreMap.shape:{}".format(rtScoreMapList[index].shape))
        # diff=np.subtract(tfScoreMapList[index], rtScoreMapList[index])
        diff=tfScoreMapList[index]-rtScoreMapList[index]
        print ("diff.shape: {}".format(diff.shape))
        count=0
        for id in range(diff.shape[0]):
            if np.abs(diff[id])>0.1:
                # print ("id:{}".format(id))
                # print ("diff[id]:{}".format(diff[id]))
                count=count+1
        print ("max(diff): {}".format(np.max(diff)))
        print ("count: {}".format(count))

def compareScoreMap(tfScoreMap_path, rtScoreMap_path):

    tfScoreMap=getScoreMap(tfScoreMap_path)
    # tfScoreMap=tfScoreMap>0.8
    rtScoreMap=getScoreMap(rtScoreMap_path)
    # rtScoreMap=rtScoreMap>0.8
    assert(len(tfScoreMap)==len(rtScoreMap))
    # print(np.sum(tfScoreMap == rtScoreMap))

    print("~~~~~~~~~tfScoreMap.shape:{}".format(tfScoreMap.shape))
    print("~~~~~~~~~rtScoreMap.shape:{}".format(rtScoreMap.shape))
    # diff=np.subtract(tfScoreMapList[index], rtScoreMapList[index])
    diff=tfScoreMap-rtScoreMap
    print ("diff.shape: {}".format(diff.shape))
    count=0
    for row in range(diff.shape[0]):
        for col in range(diff.shape[1]):
            if np.abs(diff[row][col])>0.01:
                # print ("row:{}".format(row))
                # print ("col:{}".format(col))
                # print ("diff[row][col]:{}".format(diff[row][col]))
                count=count+1
    print ("max(diff): {}".format(np.max(diff)))
    print ("count: {}".format(count))


# def compareScoreMap(rtScoreMap_path):
#
#     tfScoreMapList=getImgs(tfScoreMap_path)
#     rtScoreMapList=getImgs(rtScoreMap_path)
#     assert(len(tfScoreMapList)==len(rtScoreMapList))
#     for index in range(len(tfScoreMapList)):
#         diff=tfScoreMapList[index]-rtScoreMapList[index]
#         print ("diff.shape: {}".format(diff.shape))
#         print ("max(diff): {}".format(np.max(diff)))


# def checkZeroValue():
#     rtScoreMapList=getImgs(rtScoreMap_path)
#     print (rtScoreMapList[0])
#     for index in range(len(rtScoreMapList)):
#         print(rtScoreMapList[index].shape)
#         print(np.sum(rtScoreMapList[index]==0))
#         io.imsave('./output_test/'+im_fn.split('/')[2]+'_scoreMap.jpg',score_map);

def checkZeroValue():

    im_fn_list = get_images(rtScoreMap_path)
    for im_fn in im_fn_list:
        io.imsave('./output_test/'+im_fn.split('/')[2]+'_scoreMap.jpg', cv2.imread(im_fn, cv2.IMREAD_GRAYSCALE))

if __name__ == '__main__':
    compareScoreMap(tfScoreMap_path, rtScoreMap_path)
    # compareScoreMaps(tfScoreMaps_path, rtScoreMaps_path)
    # checkZeroValue()
