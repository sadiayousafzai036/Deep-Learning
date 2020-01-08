import json
import csv
import pandas as pd

import os

path = '/home/hassan/anaconda3/lib/python3.7/site-packages/tensorflow/models/research/object_detection/Human_Graffiti/images_annotations/'
path1 = '/home/hassan/anaconda3/lib/python3.7/site-packages/tensorflow/models/research/object_detection/Human_Graffiti/neg_images_annotations/'
files_list=[]
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.json' in file:
            files.append(os.path.join(r, file))
for r, d, f in os.walk(path1):
    for file in f:
        if '.json' in file:
            files.append(os.path.join(r, file))
json_final=[]
for f in files:
    files_list.append(f)
for i in range(0,6):
    with open(files_list[i]) as json_file:
        json_list=[]
        data = json.load(json_file)
        for points in data['shapes']:
            filename = ""
            filename = filename.join(data['imagePath'])
            if points['label']=="Human Graffiti":
                if filename.split('/')[2]=="Human":
                    json_list.append(("images/"+filename.split('/')[3]))
                else:
                    json_list.append(("images/"+filename.split('/')[2]))
            else:
                if filename.split('/')[2]=="Not Human":
                    json_list.append(("neg_images/"+filename.split('/')[3]))
                else:
                    json_list.append(("neg_images/"+filename.split('/')[2]))
            json_list.append(round(points['points'][0][0], 0))
            json_list.append(round(points['points'][0][1],0))
            json_list.append(round(points['points'][1][0],0))
            json_list.append(round(points['points'][1][1],0))
            json_list.append(points['label'])
            json_final.append(json_list)
            column_name = ['filename','xmin', 'ymin', 'xmax', 'ymax','label']
            json_df = pd.DataFrame(json_final,columns=column_name)

            json_df.to_csv('/home/hassan/anaconda3/lib/python3.7/site-packages/tensorflow/models/research/object_detection/Human_Graffiti/human_labels.csv',index=False)
            json_list = []
            json_file.close()







#json_df = pd.DataFrame(json_final,index=None)
#json_df.to_csv('/home/hassan/Desktop/human_labels.csv')
print('Successfully converted json to csv.')
