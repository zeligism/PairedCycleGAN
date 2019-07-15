# AutoMakeup
Automatic makeup generation.

## Steps

- Install requirements.
- Search using `dataset/search/searcher.py` to generate `image_urls.csv`.
  - Optional: extract from Pinterest html sources, if you have any, then run `cat pinterest_urls.csv >> image_urls.csv`.
- Download using `dataset/data/download.py`. Clean dataset manually (a little tedious but ultimately unavoidable).
- Split images to **before** and **after** makeup (just split the qualified images by half vertically). Save remaining images for unsupervised training.
- Process images by extracting faces. Standardize image sizes.