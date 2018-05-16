The code of this repository was used for the following publication. If you find this code useful please cite our paper:

```
@article{Gallego2017138,
title = "Staff-line removal with selectional auto-encoders",
author = "Antonio-Javier Gallego and Jorge Calvo-Zaragoza",
journal = "Expert Systems with Applications",
volume = "89",
pages = "138 - 148",
year = "2017",
issn = "0957-4174",
doi = "https://doi.org/10.1016/j.eswa.2017.07.002"
}
```

Below we include instructions to reproduce the experiments.



## Selectional Auto-Encoder (SAE)

The `sae.py` script performs the training and evaluation of the proposed algorithm. The parameters of this script are the following:


| Parameter    | Default | Description                      |
| ------------ | ------- | -------------------------------- |
| `-path`      |         | Path to the dataset              |
| `--gray`     |         | Use gray pictures                |
| `-modelpath` |         | Path to the model to load        |
| `--ptest`    |         | Test after each page of training |
| `--only_test`|         | Only evaluate. Do not train      |
| `-layers`    |  3      | Number of layers [1, 2, 3]       |
| `-window`    |  256    | Input window size                |
| `-step`      |  -1     | Step size. -1 to use window size |
| `-filters`   |  96     | Number of filters                |
| `-ksize`     |  5      | Kernel size                      |
| `-th`        |  0.3    | Selectional threshold. -1 to evaluate the range [0,1]     |
| `-super`     |  1      | Number of super epochs           |
| `-epoch`     |  200    | Number of epochs                 |
| `-batch`     |  8      | Batch size                       |
| `-page`      |  25     | Number of images to load per page for the training set    |
| `-page_test` |  -1     | Page size used for the test set. -1 to use the train size |
| `-esmode`    |  g      | Early stopping mode. g='global', p'per page'              |
| `-train_from`| -1      | Train from this image. -1 to deactivate offset   |
| `-train_to`  | -1      | Train up to this image. -1 to use the entire set |
| `-test_from` | -1      | Test from this image. -1 to deactivate offset    |
| `-test_to`   | -1      | Test up to this image. -1 to use the entire set  |


The only mandatory parameter is `-path`, the rest are optional. This parameter indicates the path to the dataset to be evaluated, which must have the following structure: it must have the folders `TrainingData` and` Test`. Each of these must have three subfolders: `BW`,` GR`, and `GT`, for the binary, grayscale, and ground-truth images, respectively. The images must be in PNG format and have the following name pattern: `TT_XXXX.png`, where` TT` is the image type (`BW`,` GR`, or `GT`) and` XXXX` is the identifier of the image.

The `--gray` parameter activates the use of grayscale images (` GR`) for training and evaluation. If this parameter is not indicated, the binary images (`BW`) will be used by default.

The `-modelpath` parameter indicates the name of the file with the network weights. This option allows to initialize the network either to evaluate a network model or to perform a fine-tuning process. This parameter can be used in combination with the option `--only_test` to only run the evaluation. 

The options `-layers`, `-window`, `-step`, `-filters`, `-ksize`, and `-th` allow to configure the network topology. If an external weight file is loaded, it has to match this configuration.

The parameters `-super`, `-epoch`, `-batch` configure the training stage. The options `-page` and `-page_test` allow modifying the number of images loaded in each super-epoch. The option `--ptest` performs an evaluation after each training stage.

By means of `-train_from`, `-train_to`, `-test_from`, `-test_to` you can configure the number of images to be evaluated in order to resume the training from a given point or to evaluate only a subset of the dataset.



#### Examples of use 

For example, to train a network model for the gray images of the CVC-MUSCIMA dataset with the parameters specified in the paper, you may run the following command:

```
$ python sae.py -path datasets/cvcmuscima/ --gray -layers 3 -window 256 -filters 96 -ksize 5 -th 0.3
```

To remove the staff-lines using the model provided for the BW images and the parameters specified in the paper, you may run the following command:


```
$ python sae.py -path datasets/cvcmuscima/ -modelpath MODELS/model_weights_BW_256x256_s256_l3_f96_k5_se1_e200_b8_p25_esg.h5 -layers 3 -window 256 -filters 96 -ksize 5 -th 0.3 --only_test
```

* _Note: to use the trained models it is necessary to set Theano as backend (see Trained models section)._



## Demo

The `demo.py` script allows to test the algorithm for a single image. This script has the following parameters: 


| Parameter    | Default | Description                      |
| ------------ | ------- | -------------------------------- |
| `-imgpath`   |         | Path to the image to process     |
| `-modelpath` |         | Path to the model to load        |
| `--demo`     |         | Activate demo mode               |
| `-save`      |  None   | Save the output image            |
| `-layers`    |  3      | Number of layers [1, 2, 3]       |
| `-window`    |  256    | Input window size                |
| `-step`      |  -1     | Step size. -1 to use window size |
| `-filters`   |  96     | Number of filters                |
| `-ksize`     |  5      | Kernel size                      |
| `-th`        |  0.3    | Selectional threshold            |


The `-imgpath` and` -modelpath` parameters are required. These parameters allow to indicate the image to be processed and the network model to be used. The `--demo` parameter shows an animation of the staff-lines removal process. The parameter `-save` indicates the name of the file to save the resulting image. The rest of parameters, as for the `sae.py` script, allow to configure the topology of the network model.

For example, to process the image _image001.png_ with the parameters specified in the paper, you have to run the following command:

```
KERAS_BACKEND=theano python demo.py --demo -imgpath image001.png -modelpath MODELS/model_weights_BW_256x256_s256_l3_f96_k5_se1_e200_b8_p25_esg.h5 -layers 3 -window 256 -filters 96 -ksize 5 -th 0.3
```


* _Note: to use the trained models it is necessary to set Theano as backend (see Trained models section)._




## Trained models

The `MODELS` folder includes the following trained models for the CVC-MUSCIMA dataset: 

* `model_weights_BW_256x256_s256_l3_f96_k5_se1_e200_b8_p25_esg.h5`
* `model_weights_GR_256x256_s256_l3_f96_k5_se1_e200_b8_p25_esg.h5`

These are the models used in the experimentation sections of the article, for the black and white images (`_BW_`) and the grayscale images (`_GR_`).

These models were trained using **Theano**, therefore to use it is necessary to install and activate this library.

You can set the environment variable `KERAS_BACKEND` to indicate the use of _Theano_ as follows: 

```
KERAS_BACKEND=theano python sae.py ...
```


## Dataset

The CVC-MUSCIMA dataset can be downloaded from the following link:

http://www.cvc.uab.es/cvcmuscima/index_database.html

This dataset is divided into three subsets: 

| Subset  | From    | To      | Deformations   |
| ------- | ------- | ------- | -------------- |
| TS1     | 1       | 500     | 3D distortions |
| TS2     | 501     | 1000    | Local noise    |
| TS3     | 1001    | 2000    | 3D distortions + local noise |



