# **General information**
Semantic segmentation model with UNet architecture using Keras.


# **Quickstart**
  To run quickstart, you'll need:
  * Anaconda3 or greater (conda).
  * Run run_me.sh,that will install requirement libraries and virtual env
  * Ensure you are in conda env 'lobanova_test' and run :   
    - Predict :
      python application_entry_point.py   ***full_path_to_image*** ***destination_folder*** ***path_to_model***
    - Train :
      python train.py ***path_to_train_image***
      
  ## **Args for startup**   
   * Predict :
  ***full_path_to_image.png*** ***destination_folder*** ***path_to_model***
  
   1. ***full_path_to_image***: name of the folder that would be used as a source folder with image,that will be needed to predict mask     (exstension  .png)
   2. ***destination_folder***: name of the folder that would be used as a destination folder (where to store predicted result)
   3. ***path_to_model***  : full path to trained model with extension  .h5
   
   * Train :
  ***path_to_train_image*** 
  
   1. ***path_to_train_image***: name of the folder that would be used as a source folder with training images(data structure is below)
     
  ## **Example** 
  ```sh
  python application_entry_point.py  /segmentation/image.png  /Tools/segmentation  /segmentation/segmentation_model.h5
   ``` 
      
# **Data structure**
    Dataset was downloaded from https://www.kaggle.com/c/data-science-bowl-2018
    
    ├── stage1_test                   
    │   ├── {image_guid}
    |        ├──images
    |           ├──some_image.png   
    │   ├── {image_guid}
    |        ├──images
    |           ├──some_image.png   
    │   ├── ...        
    └─
    ├── stage1_train                  
    │   ├── {image_guid}
    |        ├──images
    |           ├──some_image.png 
    |        ├──masks
    |            ├──some_masks.png
    |            ├──some_masks.png
    |            ├──...
    │    
    │   ├── ...        
    └─
   

  # **Main steps to create model**
  1. Loading images from train path
  2. Loading and concatenating masks
  3. Check results from steps 1,2
  4. Definite custom metric and custom loss function
  5. Create Unet model 
  6. Data augmentation
  7. Fit and Train model
  8. Predict test images 
  9. Check results from step 8
  
