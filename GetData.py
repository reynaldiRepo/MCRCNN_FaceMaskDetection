
import os
import json
import cv2
import matplotlib.pyplot as plt

# """Parse the data from annotation file

# Args:
#     input_path: annotation file path

# Returns:
#     all_data: list(filepath, width, height, list(bboxes))
#     classes_count: dict{key:class_name, value:count_num} 
#         e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
#     class_mapping: dict{key:class_name, value: idx}
#         e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
# """

def createDataset(max_per_class = 0):
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    
    # call json anotation file
    fs = open(os.path.join(os.getcwd(), "JsonData", "MafaTrain.json"), "r");
    Mafa = json.load(fs)
    fs.close()
    fs = open(os.path.join(os.getcwd(), "JsonData", "AFLW.json"), "r")
    Aflw = json.load(fs)
    fs.close()
    
    # Make sure the info saved in annotation file matching the format (path_filename, x1, y1, x2, y2, class_name)
    # Note:
    #	One path_filename might has several classes (class_name)
    #	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
    #	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
    #   x1,y1-------------------
    #	|						|
    #	|						|
    #	|						|
    #	|						|
    #	---------------------x2,y2
    
    # class note :
    # 1 - maskedface
    # 2 - incorrect maskedface
    # 3 - non maske
    
    indexClass = 0;
    
    state = "MAFA"
    
    for dataset in [Mafa, Aflw]:
        for data in dataset:
            
            curdata = dataset[data] #current data
            if len(curdata['ground_truth']) == 0:
                print("Doesnt has ground truth")
                continue
            
            image = cv2.imread(curdata['file'])
            size = image.shape
            
            filename = curdata['file']
            all_imgs[filename] = {}
            
            all_imgs[filename]['filepath'] = filename
            all_imgs[filename]['width'] = size[1]
            all_imgs[filename]['height'] = size[0]
            all_imgs[filename]['bboxes'] = []
            
            del image #clean memory
            del size #clean memory
            
            print("Finding ground truth")
            for gt in curdata['ground_truth']:
                try:
                        
                    # initiate new data on dictionary 
                    if class_mapping.get(gt['CLASS'], None) == None :
                        class_mapping[gt['CLASS']] = indexClass
                        indexClass += 1
                    if classes_count.get(gt['CLASS'], None) == None :
                        classes_count[gt['CLASS']] = 0
                        
                    #countionue loop on maximum file per class
                    if max_per_class != 0:
                        if classes_count[gt['CLASS']] == max_per_class :
                            print("max_per_class = ", max_per_class)
                            print("Skip, quantity ", gt['CLASS'], " = ", classes_count[gt['CLASS']])
                            continue
                                        
                    classes_count[gt['CLASS']] += 1
                    
                    x1 = int(gt['BOX']['X'])
                    x2 = int(gt['BOX']['X']) + int(gt['BOX']['W'])
                    y1 = int(gt['BOX']['Y'])
                    y2 = int(gt['BOX']['Y']) + int(gt['BOX']['H'])
                    
                    all_imgs[filename]['bboxes'].append({'class': gt['CLASS'], 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

                    print("write image ", all_imgs[filename]['filepath'])
                    print("Class Mapping ", class_mapping)
                    print("Class count", classes_count)
                    print("=======================================================")
                    print()
                    
                except Exception (e):
                    print(e)
            
            if max_per_class != 0:
                if state == "MAFA" :                     
                    if classes_count[1] == max_per_class and classes_count[2] == max_per_class:
                        print("Run for other dataset")
                        state = "AFLW"
                        break
                    
                if state == "AFLW" :                     
                    if classes_count[0] == max_per_class:
                        print("Run for other dataset")
                        state = "AFLW"
                        break
        
    all_data = []
    for key in all_imgs:
        if len(all_imgs[key]['bboxes']) != 0:
            all_data.append(all_imgs[key])
        
    classes_count['bg'] = 0
    class_mapping['bg'] = indexClass 

    print("classess count = \t",classes_count)
    print("classess mapping = \t",class_mapping)
    print("All image = ", all_data)
    
    
    allDataFile = os.path.join(os.getcwd(), "JsonData", "ALL_DATA.json")
    with open(allDataFile, 'w') as json_file:
        json.dump(all_data, json_file, indent=4)
    json_file.close()
    
    classMaping = os.path.join(os.getcwd(), "JsonData", "CLASS_MAPING.json")
    with open(classMaping, 'w') as json_file:
        json.dump(class_mapping, json_file, indent=4)
    json_file.close()
    
    ClassCount = os.path.join(os.getcwd(), "JsonData", "CLASS_COUNT.json")
    with open(ClassCount, 'w') as json_file:
        json.dump(classes_count, json_file, indent=4)
    json_file.close()
    
    return all_data, classes_count, class_mapping

# createDataset(2000)

def get_data():
    allDataFile = os.path.join(os.getcwd(), "JsonData", "ALL_DATA.json")
    allDataFile = open(allDataFile, "r")
    all_data = json.load(allDataFile)
    classMaping = os.path.join(os.getcwd(), "JsonData", "CLASS_MAPING.json")
    classMaping = open(classMaping, "r")
    class_mapping = json.load(classMaping)
    ClassCount = os.path.join(os.getcwd(), "JsonData", "CLASS_COUNT.json")
    ClassCount = open(ClassCount, "r")
    classes_count = json.load(ClassCount)
    return all_data, classes_count, class_mapping
    

def test():
    data = get_data()
    #make preview
    fig = plt.figure(figsize=(3,10))
    index = 1
    for im in data[0]:
        image = cv2.imread(im['filepath'])
        for b in im['bboxes']:
            cv2.rectangle(image,(b['x1'], b['y1']), (b['x2'], b['y2']), (255,255,0), 5)
            fig.add_subplot(3, 10, index)          
            # showing image
            plt.imshow(image)
            index += 1
            
    plt.show()
    
    
def GetImageByIndex(index=0):
    file = open(os.path.join(os.getcwd(), "JsonData", "ALL_DATA.json"), "r")
    data = json.load(file)
    file.close()
    return data[index]
    
# test()
# print(GetImageByIndex())