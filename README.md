# AutoMakeup
Automatic makeup generation.

## Creating the dataset

- Search using `dataset/search/searcher.py` to generate `image_urls.csv`.
  - Optional: extract from Pinterest html sources with `pinterest/extract_pinterest_urls.py`, then run `cat pinterest/pinterest_urls.csv >> image_urls.csv`.
- Download using `dataset/download_images.py`.
- Clean dataset manually (a little tedious but ultimately unavoidable).
- Split images to **before** and **after** makeup (for the majority, just split vertically). Use `dataset/data/split_images.py` to split vertically.
- Extract faces with `dataset/data/extract_faces.py`. 
- Run `dataset/data/make_dataset.py`.
- Optional: upload images (dataset) if you want using `dataset/upload_images.py`.
- Test your dataset with `dataset/dataset.py`. If it's working, then congratulations! You just created a functioning makeup dataset.

## Creating the generator

- ...