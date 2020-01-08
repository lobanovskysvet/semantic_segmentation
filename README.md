# **General information**
Semantic segmentation model with UNet architecture using Keras.


# **Quickstart**
  To run quickstart, you'll need:
  * Anaconda3 or greater (conda).
  * Run run_me.sh,that will install requirement libraries and virtual env
  * Ensure you are in env_name and run :
    - Predict :
      python application_entry_point.py  ***full_path_to_image*** ***destination_folder***
    - Train :
      python train.py

# **Data structure**
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
   

  
