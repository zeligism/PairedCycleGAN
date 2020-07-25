
import os
import glob
import json
import argparse

HASHTAG_DIR="third_party/instagram-hashtag-crawler/hashtags"
IMAGE_URL_CSV="instagram_urls.csv"
LOW_RES=1
HIGH_RES=0

def get_post_image_urls(post, res=HIGH_RES):
    if "image_versions2" in post:
        return [post["image_versions2"]["candidates"][res]["url"]]
    elif "carousel_media" in post:
        return [subpost["image_versions2"]["candidates"][res]["url"]
                for subpost in post["carousel_media"]]
    else:
        return []

def main(args):
    # Get JSON files in hashtag-crawling results directory
    json_files = os.path.join(args.hashtag_dir, "*.json")
    hashtag_json_fs = [f for f in glob.glob(json_files) if "rawfeed" in f]

    # Get the posts from each hashtag crawl file
    hashtag_posts = []
    for hashtag_json_f in hashtag_json_fs:
        with open(hashtag_json_f, "r") as json_f:
            posts = json.load(json_f)
            hashtag_posts.append(posts)
    
    # Get the image url of each post
    image_urls = [get_post_image_urls(post) for posts in hashtag_posts for post in posts]
    image_urls = [url for post_urls in image_urls for url in post_urls]  # flatten

    # Write the urls in the output file
    with open(args.out, "w") as f:
        f.writelines(image_url+"\n" for image_url in image_urls)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Extract URLs of images from the JSON crawl files of instagram-hashtag-crawler.")
    
    parser.add_argument("--hashtag-dir", type=str, default=HASHTAG_DIR,
        help="directory containing hashtag crawling results.")
    parser.add_argument("-o", "--out", type=str, default=IMAGE_URL_CSV,
        help="text file containing image urls.")
    
    args = parser.parse_args()

    main(args)

