# TF2 utils

Utils to load and use the tf2 model.

## Setup

Download `model.tf2.zip` from the [release page](https://github.com/bgshih/aster/releases) and extract it under `aster/tf2_utils/tf2_weights/`.

## Running

```
python3 -m tf2_utils.main --images_path PATH_TO_IMAGE_1 PATH_TO_IMAGE_2 PATH_TO_IMAGE_3
```

Where PATH_TO_IMAGE_{1/2/3/...} are the paths to local text boxes.

## Details

* The model is saved in the SavedModel format and hence, the utils of this directory work independently of the rest of the code.

* All the code provided here was taken from https://github.com/NoAchache/TextBoxGan
