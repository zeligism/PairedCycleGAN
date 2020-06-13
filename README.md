# AutoMakeup
Automatic makeup generation.

## Creating the dataset

- Search using `dataset/search/searcher.py` to generate `image_urls.csv`.
  - Optional: extract from Pinterest html sources with `pinterest/extract_pinterest_urls.py`, then run `cat pinterest/pinterest_urls.csv >> image_urls.csv`.
- Download using `dataset/download_images.py`.
- Clean dataset manually (a little tedious but unavoidable).
- Split images to **before** and **after** makeup (just split vertically, fix the rest manually). Use `dataset/data/split_images.py` to split vertically.
- Extract faces with `dataset/data/extract_faces.py`.
- Now you can call use your dataset by importing `MakeupDataset` from `dataset/dataset.py`, and then calling `MakeupDataset(dataset_dir)`, where `dataset_dir` is the path to the directory containing the processed images.

## Training

Just choose the model you want (in our case, it is the PairedCycleGAN), and then train it using its corresponding trainer. I created these trainers for rapid testing purposes.

## Requirements
First, you need conda. Then do this:
```
conda create -n automakeup -c conda-forge -c pytorch python=3.7 pip pyyaml pillow=6.1 ffmpeg matplotlib opencv pytorch torchvision tensorboard
conda activate automakeup
python -m pip install cmake 
python -m pip install face_recognition
conda deactivate
```
The last step takes some time because it installs dlib.
When running pip, make sure you're running the one you installed in `automakeup` env.
To ensure that, I activate `automakeup` env and use `python -m pip install` instead of simply `pip install`.
Also, `cmake` need to be installed in a separate step before `face_recognition` for some reason. I'm thinking of using another lightweight face recognition library at the moment (this one is lightweight and simple in terms of API, but installing dlib can be non-straightforward sometimes).

To run the training process, use:
```
conda activate automakeup
cd src
python train.py
```

