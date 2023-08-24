# native_binary_classification
First exposure to deep learning library (keras/tensorflow)

Objective: determine if images are native or non-native plants to Hawaii

### About the data:

For the native plant dataset, I used images from from nativeplants.hawaii.edu. For the non-native plants I used a dandelion dataset from kaggle

Scraping the images online we get a small dataset: 555 pictures of native plants and 635 non-native pictures (dandelion)

Because the dataset is so small, I created new data by mirroring along the y axis. Went from 555 native pictures to 1110 and used the rest of dandelion dataset (1265)



### Dataset sources:

https://www.kaggle.com/datasets/coloradokb/dandelionimages

http://nativeplants.hawaii.edu/index/
