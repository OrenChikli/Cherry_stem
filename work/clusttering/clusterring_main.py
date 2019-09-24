from work.preprocess.data_functions import load_json,create_path, copy_images
from sklearn.cluster import KMeans
import numpy as np
import os
def kmeans():
    pass

def main():
    src_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20\color\sharpened\color_labels.json'
    pred_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20\clustered'
    orig_path =r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    labels_dict = load_json(src_path)
    x= list(labels_dict.values())
    x= np.array(x)
    k_means_model =KMeans(n_clusters=5, random_state=0)
    y=k_means_model.fit_predict(x)
    color2label_dict= {tuple(x[i]): y[i] for i in range(x.shape[0])}
    img_label_dict = {key: color2label_dict[tuple(value)] for (key,value) in labels_dict.items()}
    labels = np.unique(y)
    for label in labels:
        curr_path = create_path(pred_path,str(label))
        for img_name,img_label in img_label_dict.items():
            if img_label == label:

                copy_images([img_name],orig_path,curr_path)


    print("hello")

if __name__ == '__main__':
    main()