import sqlite3
import scipy.io
import os
import json

def getMafa(NFile = 100):
    matFile = os.path.join(os.getcwd(), "DATASET", "LABEL", "LabelTrainAll.mat")
    trainMatMafa = scipy.io.loadmat(matFile)

    DictJson = {}
    MaffaRootArray = trainMatMafa['label_train'][0]
    numFile = 0
    for data in MaffaRootArray :
        filename = data[1][0]
        anotasi = data[2]
        fname = os.path.join(os.getcwd(), 'DATASET', 'MAFA', 'train-images', 'images', data[1][0])
        if os.path.isfile(fname):
            DictJson[data[1][0]] = {
                'file' : fname,
                'ground_truth' : []
            }
            #retrive all groundTruth
            # DictJson[data[1][0]]['gt']
            for anot in anotasi:
                #identify class 
                typeClass = 1
                if anot[13] == 3:
                    typeClass = 1 #correct mask face
                if anot[13] == 2 or anot[13] == 1:
                    typeClass = 2 #incorrect mask face
                #save bounding box
                DictJson[data[1][0]]['ground_truth'].append({
                    'BOX' : { #save bounding box
                        'X' : int(anot[0]),
                        'Y' : int(anot[1]),
                        'W' : int(anot[2]),
                        'H' : int(anot[3])
                    },
                    'CLASS' : typeClass,
                    'OCC_TYPE' : int(anot[13])
                })
                #save occludder bounding box
            numFile += 1
            if numFile == NFile:
                if NFile != 0:
                    break
        else:
            continue
    with open(os.path.join(os.getcwd(), 'JsonData', 'MafaTrain.json'), 'w') as fp:
        JsonString = json.dump(DictJson, fp ,indent=4)
    print("Success Saving Mafa")
    return 0

def getAFLW(NFile = 100):
    trainSqlAFLW = os.path.join(os.getcwd(),"DATASET", "LABEL", "aflw.sqlite")
    conn = sqlite3.connect(trainSqlAFLW)    
    cur = conn.cursor()
    sql = "select x, y, w, h, file_id as filename from FaceRect, Faces where FaceRect.face_id = Faces.face_id Order by file_id ASC"
    cur.execute(sql)
    pathDir = os.path.join(os.getcwd(), 'DATASET', 'AFLW')
    rows = cur.fetchall()
    DictJson = {}
    numFile = 0
    for row in rows:
        file = os.path.join(pathDir, row[4])
        filename = row[4]
        if os.path.isfile(file):
            print(row)
            # check if file is exist on dict
            if DictJson.get(filename, None) == None:
                DictJson[filename] = {
                    'file':file,
                    'ground_truth':[]
                }
            DictJson[filename]['ground_truth'].append({
                'BOX' : {
                    'X':int(row[0]),
                    'Y':int(row[1]),
                    'W':int(row[2]),
                    'H':int(row[3]),
                }, 
                'CLASS' : 0,
                'OCC_TYPE' : 0
            })
            numFile += 1
            if numFile == NFile :
                if NFile != 0:
                    break
        else:
            continue
    cur.close()
    with open(os.path.join(os.getcwd(), 'JsonData', 'AFLW.json'), 'w') as fp:
        JsonString = json.dump(DictJson, fp ,indent=4)
    print("Success Saving AFLW")
    return 0


if __name__ == '__main__':
    getMafa(0)
    getAFLW(0)