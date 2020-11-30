# Crawling posts from instagram hashtag feed

Go to hastag crawler folder:
    cd third_party/instagram-hashtag-crawler

Assuming the folder "hashtags" contains no feeds (if so, move them into a separate folder or delete them), we start crawling and group rawfeeds in a folder:
```
    python __init__.py -u <USER> -p <PASSWORD> -f hashtag_files/<query>.txt
    mkdir hashtags/<query> && mv hashtags/*.json hashtags/<query>
```
where <query> could be either 'makeup' or 'nomakeup'.

Now we go back and simply extract the urls from these rawfeeds (note that I edited the crawler in instagram-hastag-crawler to simply stop after getting the rawfeed without beautifying). The argument --hashtag-dir is "third_party/instagram-hashtag-crawler/hashtags" by default, change it as necessary. The command is:
```
    python extract_instagram_urls.py -o "<query>_urls.csv"
```

Now go back two directories and download the images using `download_images.sh` as follows:
```
    ./download_images.sh "search/instagram/test_urls.csv"
```