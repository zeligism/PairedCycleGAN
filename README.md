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
conda create -n automakeup
conda install -n automakeup pip pyyaml pytorch torchvision -c pytorch
pip install cmake 
pip install face_recognition
```
When running pip, make sure you're running the one you installed with conda inside `automakeup` env.

Now run the training process using:
```
conda activate automakeup
cd src
python train.py
```

