# <p style="text-align:center;"> 🌳 Agronomic Characteristics Extraction and Prediction from Agrocam Vineyard Images 🌳 </p>

## 📘 Context

This project is the result of my end-of-year internship which consist in **extracting agronomic characteristics of vineyard images** taken with an open-source and low-cost camera system (named Agrocam) and **predicting them for a 15 day period**.

The extraction and prediction have been done on **images taken by an [Agrocam](https://agrocam.agrotic.org)**. We focused on camera installed on three type of vineyard :
- **[TVITI](https://agrocam.agrotic.org/data/79bt3wkh/)** => Plot managed sustainably by the winegrower and without ground cover
- **[AVITI](https://agrocam.agrotic.org/data/7s3a5abm/)** => Plot with an interrow vegetal cover composed of green fertilizer (sow by the winegrower)
- **[DVITI](https://agrocam.agrotic.org/data/4j7g2wk9/)** => Plot with an interrow vegetal cover composed of spontaneous vegetation (with no human intervention)

The **agronomic characteristics** extracted from the images and forecasted for 15 days are : 
- Canopy height
- Canopy porosity
- Leaf hue
- Inter-row hue.

## 📜 Table of Contents

1. [Pipeline Overview](#project-overview)  
2. [Pipeline Workflow](#pipeline-workflow)  
3. [Features](#features)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Input Data](#input-data)  
7. [Model Training and Evaluation](#model-training-and-evaluation)  
8. [Results Summary](#results-summary)  
9. [Project Structure](#project-structure)  
10. [Troubleshooting](#troubleshooting)  
11. [Contributors](#contributors)  
12. [License](#license)  

## ⚙️ Pipeline Overview

This pipeline aims at **extracting and predicting agronomic parameters from vineyard images in real condition**. For that, it has been seperated into 4 main steps :
- **Segmentation** of vineyard images into semantic zones (leaves, trunk, inter-row, irrigation sheath)
- **Extraction** of agronomics characteristics using those semantic zones
- **Validation** of those characteristics so that it's coherent with scientific knowledge
- **Prediction** of characteristics using temporal series of those vineyard images

## 🔎 Repository Organization

This pipeline is composed into 4 *almost* self-sufficient block :

| Folder | Content |
|:-:|:-:|
| [Core](https://github.com/nicolasgeffroy/agrocam_agro_chara/Core) | Contains an **example of an image database and intermediate results** which can be directly used by pipeline blocks. It is also where **output of each blocks as stored**. |
| [Segmentation](https://github.com/nicolasgeffroy/agrocam_agro_chara/Segmentation) | Contains all the function to summarize the **images into a dataset**, use them to **train a model (MobileNetv3 or DeepLabv3) for segmentation** and use this trained model. It also contains the function used to **determine the image format used for learning**. |
| [Extraction](https://github.com/nicolasgeffroy/agrocam_agro_chara/Extraction) | Contains all the functions which **uses the mask generated** by the [Segmentation](https://github.com/nicolasgeffroy/agrocam_agro_chara/Segmentation) (highlighting different ZOI) to **extract different agronomic characteristics** of images. |
| [Selection](https://github.com/nicolasgeffroy/agrocam_agro_chara/Selection) | Contains all the function which **uses the [extracted](https://github.com/nicolasgeffroy/agrocam_agro_chara/Extraction) agronomic characteristics** of each images to select the characteristics which **best represent agronomic reality**. |
| [Prediction](https://github.com/nicolasgeffroy/agrocam_agro_chara/Prediction) | Contains all the function which **trains a model (LSTM or CNN-LSTM hybrid) to predict**, using temporal series of vineyards images, **vineyard's futur characteristics** as well as function using the trained model. |

## 💻 Installation and usage

### 1️⃣ Initializing

<details><summary><b> 1. Clone the repository </b></summary>

To get this repository on your computer (to then use it), you can use those lines of codes with either the URL of this repository...

```bash
git clone https://github.com/nicolasgeffroy/agrocam_agro_chara
cd agrocam_agro_chara
```

or the SHH key (after [connecting to GitHub with SSH](https://decodementor.medium.com/connect-git-to-github-using-ssh-68ab338f4523)). 

```bash
git clone git@github.com:nicolasgeffroy/agrocam_agro_chara.git
cd agrocam_agro_chara
```

</details>

<details><summary><b> 2. Create a virtual environment (and use it) </b></summary>

Creating a virtual environment ensures that there is no dependency problem with other packages (downloaded for other project) and helps with reproductibility (keeps the package in a state where all the repository worked).

```bash
bash python -m venv .venv 
.venv/bin/activate # On Windows: venv\Scripts\activate
```

</details>

<details><summary><b> 3. Download the required package for this repository </b></summary>

After creating it, we download all the required package for this repository (in the "requirements.txt") in this environment.

```bash
python -m pip install -r requirements.txt
```

**All the package installed:**

| Package  | Version | Keywords                                         | For information                                                       |
| :-:      | :-:     | :-:                                              | :-:                                                                   |
| pillow | >=9.0 | Image processing                                     | [link](https://pillow.readthedocs.io/en/stable)                       |
| requests | >=2.0 | Make QGIS like request on the internet             | [link](https://requests.readthedocs.io/en/latest)                     |
| numpy | >=1.0 | Adding Array format and function to exploit them      | [link](https://numpy.org)                                             |
| pandas | >=1.0 | Adding DataFrame format and function to exploit them | [link](https://pandas.pydata.org/docs/getting_started/overview.html)  |
| tqdm | >=4.0 | Adds a progress bar to loops                           | [link](https://tqdm.github.io)                                        |
| torch | >=2.0 | Base for Deep Learning application                    | [link](https://docs.pytorch.org/docs/stable/index.html)               |
| torchvision | >=0.15 | Image Deep Learning framework                  | [link](https://docs.pytorch.org/vision/stable/index.html)             |
| scikit-learn | >=1.0 | Adds various model and Preprocessing tools     | [link](https://scikit-learn.org/stable)                               |
| tensorboard | >=2.0 | Adds a vizualisation kit for machine learning   | [link](https://www.tensorflow.org/tensorboard?hl=fr)                  |
| matplotlib | >=3.0 | Adds various plot to vizualise any data          | [link](https://matplotlib.org/cheatsheets)                            |
| scipy | >=1.0 | Statistic and other algorithm for scientific purpose  | [link](https://docs.scipy.org/doc/scipy/index.html)                   |

</details>

**Summary**

```bash
# 1. Clone the repository
git clone https://github.com/nicolasgeffroy/agrocam_agro_chara
# git clone git@github.com:nicolasgeffroy/agrocam_agro_chara.git
cd agrocam_agro_chara
# 2. Create a virtual environment (and use it)
bash python -m venv .venv 
.venv/bin/activate # On Windows: venv\Scripts\activate
# 3. Download the required package for this repository
python -m pip install -r requirements.txt
```

### 2️⃣ How to use it

**Each blocks can be called using this following example of line of code :**

```bash
python <name_folder>_function.py --<argument> ...
```

---

Just below, is displayed, for each block :
- **Purpose** => What do they do when called ?
- *Input =* Which default input do they use ?
- *Output =* Which output do they provide ?

| Arguments | description | Input |
| :-: |:-:| :-:|
|  --\<argument> | Description of the argument | Which input does it take |

<details> <summary><b> Examples of lines of code to use. </b></summary> 

```bash
python <name_folder>/<name_folder>_function.py --<argument> ...
```
==> Description of what does this line of code

</details>

---

### Segmentation
- **Purpose** = Trains a MobileNetv3 model to segment an image into 4 zone of interest : leaf, inter-row, trunc and sheath (**train**) or use a trained model to segment a set of images (**segment**).
- *Input =* 
    - (**train**) Image and ground truth mask chosen for training (*Core/image_train*) 
    - (**segment**) A set of vineyard images (*Core/all_image*) 
- *Output =* 
    - A database representing all the images inputed (*Core/Results/Image_chara_train.csv* for **train** or *Core/Results/Image_chara_all.csv* for **segment**)
    - (**train**) Weight of the trained model (*Segmentation/checkpoint*)
    - (**segment**) A mask per images (with all the classes) inputed (*Core/Results/Image_mask*)

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<folder_url_train_img> | URL of the folder containing input images | string | "Core/Images/image_train/image" |
|  --\<train_or_segment> | Choose to train an algorithm or use a trained one to segment images | "train" or "segment" | "segment" |
|  --\<folder_url_train_mask> | **training** = Ground-truth mask // **segment** = URL of the folder containing mask for calcultating the distance between trunc and sheath | string | "Core/Images/ image_train/masque_final" |
|  --\<weight_url> | Import weight of the model. If "No_weight" used, [pretrained weights for MobileNetV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.lraspp_mobilenet_v3_large) are used | string | "No_weight" |
|  --\<epochs> | **train** Number of epochs for training | int | 10 |

<details> <summary><b> Examples </b></summary>

```bash
python Segmentation/segmentation_function.py 
    --weight_url Segmentation/checkpoint/mobileNetv3_checkpoint_focal_HSV.pth 
    --folder_url_train_img Core/Images/all_image
```
==> **Generate for each images** in *Core/Images/all_image* the **segmentation masks** (in *Core/Results/Image_mask*) using a **MobileNetv3 model trained** with the specified weight (*mobileNetv3_checkpoint_focal_HSV.pth*). It also generate a database (*Core/Results/Image_chara_all.csv*) with all the images used and information about them.

```bash
python Segmentation/segmentation_function.py 
    --<train_or_segment> train 
    --<epochs> 100
```
==> **Train a pretrained MobileNetv3 model** for 100 epocks using training images (*Core/Images/image_train/image*) and their ground truth mask (*Core/Images/image_train/masque_final*). It also generate a database (*Core/Results/Image_chara_train.csv*) with all the images used and information about them.

</details>

### Extraction
- **Purpose** = Extracts the different agronomic characteristics from each vineyard images using the different mask generated by the [segmentation](#segmentation) block.
- *Input =* The database representing each images inputed during [segmentation](#segmentation) (*Core/Results/Image_chara_all.csv*).
- *Output =* A database with, for each image, all the vineyard's agronomic characteristics (*Core/Results/Agro_chara_vine.csv*)

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<name_of_database_used> | URL of the csv file containing images, mask and information about it | string | "Core/Results/Image_chara_all.csv" |
|  --\<name_of_mask_used> | Name of the entity used to determine the upper part of the image (used for the porosity characteristics) | "trunc" or "sheath" | "sheath" |

<details> <summary><b> Examples </b></summary>

```bash
python Extraction\extraction_function.py
```
==> Generate a database with for each image in *Core/Results/Image_chara_all.csv* we have their agronomic parameters associated. To determine the porosity, it used the sheath mask to determine the upper zone of the image.

```bash
python Extraction\extraction_function.py 
    --<name_of_database_used> "Core/Results/Image_chara_train.csv"
    --<name_of_mask_used> "trunc"
```
==> Generate a database with for each image in *Core/Results/Image_chara_train.csv* we have their agronomic parameters associated. To determine the porosity, it used the trunc mask to determine the upper zone of the image.
</details>

### Selection 

There are no function to call for now. You can go in this folder and look at the notebook to have an idea of the methodology.

**For now, all the agronomic characteristcs will be used for [prediction](#Prediction)**.

### Prediction 

- **Purpose** => Trains a LSTM or CNN-LSTM model to predict, using 15 last vineyard images, the next 15 days of vineyard agronomic characteristics : canopy porosity and height as well as leaf and interrow hue (**train**) or use a trained model to predict vineyard agronomic characteristics (**predict**).
- *Input =* Database with images of which we have extracted the vineyards characteristics during [extraction](#extraction) (*Core/Results/Agro_chara_vine.csv*) 
- *Output =* 
    - (**train**) Weight of the trained model (*Prediction/checkpoint*)
    - (**predict**) Prints the characteristics of the vineyard for the next 15 days.

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<lstm_model> | Class (in the designated package) of the LSTM model used as a prediction model | string | "model.cnn_lstm.CNN_LSTM" |
|  --\<weight_url_cnn> | Import weight of the cnn model used when a CNN-LSTM model is used. If given "No_weight", [pretrained weights for MobileNetV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.lraspp_mobilenet_v3_large) are used. | string | "No_weight" |
|  --\<weight_url_lstm> | Import weight of the lstm model. | string | "Prediction/checkpoint/MobileNet3_LSTM _checkpoint_final_hsv_notbi_norm.pth" |
|  --\<train_or_predict> | Choose to train an algorithm or use it to predict agronomic characteristics | "train" or "predict" | "predict" |
|  --\<time_start> | **predict** Start date of the 15 image used for prediction | string | "2024-04-20" |
|  --\<treatment> | Treatment of vine where the images were taken | string | "AVITI" |
|  --\<epochs> | **train** Number of epochs for training | int | 10 |

<details> <summary><b> Examples </b></summary>

```bash
python Prediction/prediction_function.py 
    --lstm_model model.cnn_lstm.CNN_LSTM 
    --weight_url_cnn Segmentation/checkpoint/mobileNetv3_checkpoint_focal_HSV.pth
    --weight_url_lstm Prediction/checkpoint/MobileNet3_LSTM_checkpoint_final_hsv_notbi_norm.pth
```
==> Using the 15 images described in *Core/Results/Agro_chara_vine.csv* that are AVITI vineyards and with the first taken the 2024-04-20, it prints the next vineyard characteristics (lasting from 2024-05-06 to 2024-05-20) predicted by a CNN-LSTM model with the cnn weight ("mobileNetv3_checkpoint_focal_HSV.pth") and the lstm weight ("MobileNet3_LSTM_checkpoint_final_hsv_notbi_norm.pth") given.

```bash
python Prediction/prediction_function.py 
    --lstm_model model.cnn_lstm.no_CNN_LSTM 
    --weight_url_lstm Prediction/checkpoint/MobileNet3_LSTM_test_nocnn_checkpoint.pth
```
==> Using the 15 images described in *Core/Results/Agro_chara_vine.csv* that are the AVITI vineyards and with the first taken the 2024-04-20, it prints the next vineyard characteristics (lasting from 2024-05-06 to 2024-05-20) predicted by a LSTM model with the lstm weight ("MobileNet3_LSTM_test_nocnn_checkpoint.pth") given.

```bash
python Prediction/prediction_function.py 
    --lstm_model model.cnn_lstm.CNN_LSTM 
    --train_or_predict train
    --epochs 100
```
==> Train a CNN-LSTM model (with a CNN pretrained to COCO) to predict the next 15 vineyard characteristics with 15 prior vineyard images retrieved in *Core/Results/Agro_chara_vine.csv* for 100 epochs.

</details>

## 🎁 Pretrained model weight

You can contact me at nico.geffroy.pro@gmail.com for the checkpoint file (.pth) with weight of the model trained for segmentation and the one trained for prediction. 

## ⭐ Model Evaluation

<center>

| **Segmentation** | MobileNetv3 | Deeplabv3 | \|\|\|\| | MobileNetv3 (details) | Leaf | Interrow | Sheath | Trunc
| :-: |:-:| :-:| -------- | :-: |:-:| :-:| :-: |:-:|
|  IoU             | 0.72        |   0.53    | \|\|\|\| | ==> | 0.87 | 0.92 | 0.42 | 0.58
| Sensibility      | 0.82        |   0.63    | \|\|\|\| | ==> | 0.95 | 0.96 | 0.60 | 0.75
| Specificity      | 0.98        |   0.97    | \|\|\|\| | ==> |  0.97 | 0.98 | 1.00 | 1.00

| **Prediction** | ES-LSTM | PE-LSTM | (ES-EP)-LSTM | LSTM |
| :-: |:-:| :-:| :-: |:-:|
|  MSE           | 0.27    | 0.07    | 0.06         | 0.07 |

</center>

<p style="text-align:center;">
(ES = MobileNetV3 entraîné sur la segmentation, EP = MobileNetV3 entraîné sur la
prédiction, PE = MobileNetV3 seulement pré-entraîné avec COCO)
</p>

## 💪 How to contribute ?

You can find what's can/have to be done for this repository (you can also check out the [**Issues**](https://github.com/nicolasgeffroy/agrocam_agro_chara/issues) tab) : 

| To be done        | Details           | How can you contribute ? |
| :-: |:-:| :-:|
|  Fill the Model Evaluation      | Adds the performance of models for the segmentation and prediction task in the main README file. | You can't sorry |
| Intermediate README files      | Adds all the README in the different blocks to detail what does each block (and their functions)  |  You can't sorry |
| Automate the Selection block      | In my work the selection part were done manualy and for the pipeline to be operationnal it needs to be automatic  |  You can either propose ideas with what you saw in the notebook and/or code the function making it automatic. |
| Automate the choice of image format      | In my work this part (in the segmentation block) were done manualy and for the pipeline to be operationnal it needs to be automatic  |  You can either propose ideas with what you saw in the notebook and/or code the function making it automatic. |

Make sure the stick as much as possible to the style in which the repository has been written.

Feel also free to signal bugs, highlight something you don't understand (sorry in advance for syntax errors 😅), where I made a mistake or other issues linked to the code or its documentation in the [**Issues**](https://github.com/nicolasgeffroy/agrocam_agro_chara/issues) tab.

## 📄 License

Released under the **MIT License**.  
You are free to use, modify, and distribute this project with attribution.

## 🌱 Citation

If you use this repository in your research or publication, please cite:

```latex
@misc{NicoGeff2025,
  author = {Nicolas, Geffroy},
  title = {Agronomic Characteristics Extraction and Prediction from Agrocam Vineyard Images},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nicolasgeffroy/agrocam_agro_chara}}
}
```
