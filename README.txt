CNN for Classification of Tomato Plant Disease
Stephen Pachucki, Aldrin Padua


I . For data splitting:

Dependencies: splitfolders

1. Extract "Dataset.zip" in the same folder as the python scripts.
2. Run "dataset_splitter.py". A folder named "dataset_split" will be created containing three folders namely, training (65%), validation (15%), and test (20%). Each set contains separate folder for each class. 

Note: It is either you follow the steps above, or just extract the dataset_split zip file. They are just the same.

===========================================================================================

II. For data reading and loading:

Dependencies: Pillow, NumPy, glob

1. Run "cnn_v1.py".
2. Each of the class is numerically represented/labeled as follows:
    "Bacterial_spot" = 0
    "Early_blight" = 1
    "Healthy" = 2
    "Late_blight" = 3
    "Leaf_mold" = 4
    "Septoria_leaf_spot" = 5
    "Spider_mites" = 6
    "Target_spot" = 7
    "Tomato_mosaic_virus" = 8
    "Tomato_yellow_leaf_curl_virus" = 9

===========================================================================================

III. For training:

Dependencies: TensorFlow, Keras, Matplotlib