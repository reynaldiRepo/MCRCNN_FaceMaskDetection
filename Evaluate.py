
import io
import os
from re import T
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from numpy.core.defchararray import index
from numpy.lib.function_base import append
from tensorflow.python.keras.layers.merge import average
from Iou import iou
from scipy.interpolate import interp1d

np.set_printoptions(suppress=True)

def evaluate(model="MCRCNN"):
    
    test_class_0,test_class_1,test_class_2 = None,None,None
    with open(os.path.join(os.getcwd(), "npy", "test_result_RCNN_0.npy"), "rb") as f:
        test_class_0 = np.load(f)
    f.close()
    with open(os.path.join(os.getcwd(), "npy", "test_result_RCNN_1.npy"), "rb") as f:
        test_class_1 = np.load(f)
    f.close()
    with open(os.path.join(os.getcwd(), "npy", "test_result_RCNN_2.npy"), "rb") as f:
        test_class_2 = np.load(f)
    f.close()
    print(test_class_0.shape, test_class_1.shape, test_class_2.shape)

    # //load groundtruth
    fileTest = open(os.path.join(os.getcwd(), "JsonData", "TEST_DATA_LOCAL.json"), "r")
    dataTest = json.load(fileTest)
    fileTest.close()
    # print(dataTest)
    
    # sort by descending confidence
    test_class_0 =  test_class_0[test_class_0[:, 2].argsort()[::-1]]
    test_class_1 =  test_class_1[test_class_1[:, 2].argsort()[::-1]]
    test_class_2 =  test_class_2[test_class_2[:, 2].argsort()[::-1]]

    #calculate map:
    treshold = 0.5
    numTPall = 1000;#define on create test case
    index_class = 0
    for test in (test_class_0,test_class_1,test_class_2):
        csv = open(os.path.join(os.getcwd(), "evaluate", "RCNN_class_"+str(index_class)+"_map.csv"), "w")
        csv.write("rank,image,confidence,tp,fp,acc_tp,acc_fp,precision,recall\n")
        acc_tp = 0
        acc_fp = 0
        rank = 1
        for box in test:
            print(box)
            print("get image on index", int(box[1]))
            gt = dataTest[int(box[1])]
            print(gt)
            # //test
            best_iou = 0
            class_gt = 0 
            for bboxes in gt['bboxes']:
                test_box = (int(box[3]),int(box[4]),int(box[5]),int(box[6]))
                gt_box = (bboxes['x1'],bboxes['y1'],bboxes['x2'],bboxes['y2'])
                cur_iou = iou(test_box, gt_box)
                print(cur_iou)
                if best_iou < cur_iou :
                    best_iou = cur_iou;
                    class_gt = bboxes['class']
            TP = 0
            FP = 0
            print("map rank ", rank)
            print ("iou = ", best_iou)
            print ("class test = ", int(box[0]))
            print ("class gt = ", class_gt)
            if best_iou >= treshold and class_gt == int(box[0]):
                TP = 1
                acc_tp +=1
            else:
                FP = 1
                acc_fp +=1

            precision = acc_tp / (acc_tp + acc_fp)
            recall = acc_tp / numTPall
            print("precision = ", precision)
            print("recall = ", recall)
            print("===========================================================")
            # csv.write("rank,image,confidence,tp,fp,acc_tp,acc_fp,precision,recall\n")
            csv.write("%d,%d,%f,%d,%d,%d,%d,%f,%f\n" %(rank, box[1], box[2], TP, FP, acc_tp, acc_fp, precision, recall))
            # break
            rank += 1
        csv.close()
        index_class += 1
        
def showcurve():
    eval_clas_0 = open(os.path.join(os.getcwd(), "evaluate", "RCNN_class_0_map.csv"))
    eval_clas_1 = open(os.path.join(os.getcwd(), "evaluate", "RCNN_class_1_map.csv"))
    eval_clas_2 = open(os.path.join(os.getcwd(), "evaluate", "RCNN_class_2_map.csv"))
    
    index_class = 0
    mAP = 0
    for csv in (eval_clas_0, eval_clas_1, eval_clas_2):
        row = csv.readlines()[1:]
        arrayprec = []
        arrayrecc = []
        for lines in row:
            lines = lines.split(",")
            arrayprec.append(float(lines[7]))
            arrayrecc.append(float(lines[8]))
        
        arrayrecc = np.array(arrayrecc)
        arrayprec = np.array(arrayprec)
        # //smooth precision
        below_index = 0
        decreasing_max_precision = np.maximum.accumulate(arrayprec[::-1])[::-1]
        print(decreasing_max_precision)

        # //mapping array ap
        array_preccision_smooth = []
        array_recall = []
        for point in range(len(decreasing_max_precision)):
            if decreasing_max_precision[point] not in array_preccision_smooth:
                array_preccision_smooth.append(decreasing_max_precision[point])
                if (point == 0):
                    array_recall.append(0)    
                else:
                    array_recall.append(arrayrecc[point])
        print(array_preccision_smooth)
        # add 0 point for start point
        print(array_recall)
        # calculate ap
        # AP = AP1 + .... + APN
        AP = []
        for N in range (len(array_preccision_smooth)):
            # A -> N
            if N+1 < len(array_preccision_smooth):
                APval = (array_recall[N+1] - array_recall[N]) * array_preccision_smooth[N]
                print("AP", N+1 ,"= ("+str(array_recall[N+1])+" - "+str(array_recall[N])+")"+" * "+str(array_preccision_smooth[N]),"=",APval)
                AP.append(APval)
        print("AP = ", sum(AP))
        mAP += sum(AP);

        classMapping = {0:"No mask", 1:"Correct mask", 2:"incorrect mask"}    
        plt.plot(arrayrecc, arrayprec, '--b')  
        plt.plot(arrayrecc, decreasing_max_precision, '-r')  
        plt.title("Curva Class "+ classMapping[index_class])
        plt.xlabel("reccal")
        plt.ylabel("precision")
        plt.show()
        # break;
        index_class += 1
    print("mAP = " ,mAP/index_class)


def timelinelog():
    test_class_0,test_class_1,test_class_2 = None,None,None
    with open(os.path.join(os.getcwd(), "npy", "test_result_RCNN_0.npy"), "rb") as f:
        test_class_0 = np.load(f)
    f.close()
    with open(os.path.join(os.getcwd(), "npy", "test_result_RCNN_1.npy"), "rb") as f:
        test_class_1 = np.load(f)
    f.close()
    with open(os.path.join(os.getcwd(), "npy", "test_result_RCNN_2.npy"), "rb") as f:
        test_class_2 = np.load(f)
    f.close()
    
    x = [];
    y = [];
    y1 = [];
    index = 0;
    for test in (test_class_0,test_class_1,test_class_2):
        for r in test:
            if index > 2731:
                continue;
            if index % 2 != 0 or index < 2:
                index += 1
                continue;
            else :
                x.append(index)
                y.append(r[7])
                y1.append(r[8])
                index += 1

    print((sum(y)) / len(y))
    print((sum(y1)) / len(y1))

    x = np.array(x)
    y = np.array(y)
    y1= np.array(y1)
    plt.plot(x, y1, '--b', label = 'Classifier & Regression Proccess (fc)')  
    plt.plot(x, y, '--r', label="RPN")  
    plt.title("Timeline waktu proses testing Faster R-CNN")
    plt.xlabel("image-n")
    plt.ylabel("Time(s)")
    plt.grid()
    plt.legend()
    plt.show()

# evaluate()

showcurve()
    
# timelinelog()