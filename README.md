# Anime-Face-Recognition
This repository collects a dataset of anime faces from Danbooru (mainly HoloEN members). The dataset contains about 11,700 images of anime faces with 20 characters. We attempt to use different learning strategies and feature extraction methods to observe their impact.

## How to run the code
```
python main.py --feature clip --model adaboost --result_csv clip_adaboost.csv
```
In features, you can use hog, color, or clip. In models, you can choose kmeans, adaboost, and resnet.

In addition, there are other configs like --use_pca, --n_fold, --train_subset_ratio and so on. Please refer to the main.py.