# What is this?
This repository contains an implementation of [PairedCycleGAN](https://gfx.cs.princeton.edu/pubs/Chang_2018_PAS/Chang-CVPR-2018.pdf) plus its components (GAN, CycleGAN, Residual blocks, etc.) GANs could also be trained with Wasserstein loss and gradient penalty. The codebase is designed in a way such that it can be adapted easily to other future projects in deep learning. The code starts from the `train.py` script. First we have to define three things, which are:
* Dataset
* Model
* Model trainer (+ optimizer)

These three things should be defined separately, but we have to make sure that their points of interactions, if any, are compatible. For example, the dataset has to generate images with compatible sizes to the model. The trainer is, of course, deeply intertwined with the model itself, but the philosophy of this design is that direct access into the inner parts of the model should be minimized. The trainer is designed in a way such that it separates data pre-processing, logging, reporting, loading, and saving from the training algorithm as much as possible. Therefore, we should try to design our methods in a way that emphasizes algorithmic clarity over efficiency.

For this project, I created two datasets of before-and-after makeup images. One dataset is paired (pairs of before-and-after images of the same person), which is in the order of 1000 pairs, and another unpaired dataset, which is in the order of 5000 (total 10,000). Of course, curating these datasets cost me some painstakingly long periods of boring time, in addition to short bursts of depression from having to watch all of the hot girls that I will never get to hang out with. After that, I created a few models with their corresponding trainers, ending up with a trained PairedCycleGAN model. Algorithmically speaking, it's the same, but the engineering part is different. I didn't bother extracting face parts and just went ahead with the whole face, but I did do some face-morphing stuff. Anyway, my main motivation for creating a project like this was to understand the whole pipeline of creating a deep learning project from scratch (minus the coming-up-with-the-idea-in-the-first-place part).

The code in this project is missing a few important things. One of them is logging. Another is good docs everywhere. One more is good design choices and software engineering stuff. This is all doable but I'm not feeling motivated enough to fix any of it. The prettiest scenario that could happen is for someone to code this stuff for me.

## Creating the dataset

I will explain here the dataset creation pipeline, which is pretty boring.

- Search using `dataset/search/searcher.py` to generate `image_urls.csv`.
  - Optional: extract from Pinterest html sources with `pinterest/extract_pinterest_urls.py`, then run `cat pinterest/pinterest_urls.csv >> image_urls.csv`.
- Download using `dataset/download_images.py`.
- Clean dataset manually (a little tedious but unavoidable).
- Split images to **before** and **after** makeup (just split vertically, fix the rest manually). Use `dataset/data/split_images.py` to split vertically.
- Extract faces with `dataset/data/extract_faces.py`.
- Now you can call use your dataset by importing `MakeupDataset` from `dataset/dataset.py`, and then calling `MakeupDataset(dataset_dir)`, where `dataset_dir` is the path to the directory containing the processed images.

## Training

Just choose the model you want (in our case, it is the PairedCycleGAN), and then train it using its corresponding trainer. I created these trainers with rapid experimentation and debugging in mind.

I will add more details here later...

## Requirements
First, you need conda. Then do this:
```
conda create -n automakeup -c conda-forge -c pytorch python=3.7 pip pyyaml pillow=6.1 ffmpeg matplotlib opencv pytorch torchvision tensorboard
conda activate automakeup
python -m pip install cmake 
python -m pip install face_recognition
```
The last step takes some time because it installs dlib.
When running pip, make sure you're running the one you installed in `automakeup` env.
To ensure that, I activate `automakeup` env and use `python -m pip install` instead of simply `pip install`.
Also, `cmake` need to be installed in a separate step before `face_recognition` for some reason. I'm thinking about using another lightweight face recognition library at the moment (this one is lightweight and simple in terms of API, but installing dlib is annoying). If you still face an error from dlib, check whether you have gcc, g++, and make installed. If you are using a fresh Ubuntu container, for example, do this:
```
apt-get install gcc g++ make
```
And you'll be good to go.

## Running the program

To run the training experiment, do this:
```
conda activate automakeup
cd src
python train.py
```
