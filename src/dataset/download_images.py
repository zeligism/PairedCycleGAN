
import time
import requests
import os
import argparse

# The file where image_urls were exported to
DOWNLOAD_DIR = os.path.join("data", "downloaded")
IMAGE_URLS = os.path.join("search", "image_urls.csv")

# Get absolute path of this file and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DOWNLOAD_DIR = os.path.join(FILE_DIR, DOWNLOAD_DIR)
IMAGE_URLS = os.path.join(FILE_DIR, IMAGE_URLS)

# Variables to deal with errors occuring during download
TRY_AGAIN = False  # retry previously failed requests (for a second run of download.py)
ERROR_TAG = b"(error)"  # The error tag is always prepended to an error file
IS_ERROR_FILE = lambda f: f.read()[:len(ERROR_TAG)] == ERROR_TAG
IMAGE_NAME_FORMAT = lambda index: "{:05d}".format(index)  # The format of image names


def download_image(image_url, image_path="untitled"):
    """
    Download an image from `image_url` and save it to `image_path`.

    Args:
        image_url: The url of the image to be downloaded.
        image_path: The path where the image will be saved.

    Returns:
        "Success" or the exception in case of an error.
    """

    try:
        # Download image in chunks
        with requests.get(image_url, stream=True, timeout=30) as image_response:
            image_response.raise_for_status()
            with open(image_path, 'wb') as f:
                chunk_size = 1 << 10
                for chunk in image_response.iter_content(chunk_size):
                    f.write(chunk)
            return "Success"
    except Exception as e:
        # Image will be text describing the error message
        with open(image_path, 'w') as f:
            f.write("(error) {}".format(e))
        return e


def download_images(image_urls, download_dir):
    """
    Download the images from `image_urls` and save them in `download_dir`.

    Args:
        image_urls: The urls of the images to be downloaded.
        download_dir: The directory where the images will be saved.
    """

    # Download images
    for index, image_url in enumerate(image_urls):

        # Create image name and path
        image_name = IMAGE_NAME_FORMAT(index)
        image_path = os.path.join(download_dir, image_name)

        # If a file called 'image_name' already exists, open it and find whether
        # it has an '(error)' tag in it. If it doesn't, then we already downloaded
        # it successfully, so we skip it. If it does, then we skip it only if we
        # don't want to try downloading it again.
        if os.path.exists(image_path):
            with open(image_path, "rb") as image:
                if not IS_ERROR_FILE(image):
                    continue  # skip because we already downloaded this image
                if not TRY_AGAIN:
                    continue  # skip because we don't want to try again

        # Download image and check download success
        print("[{:05d}]  Downloading {} ... ".format(index, image_url), end="")
        status = download_image(image_url, image_path)
        print(status)


def delete_error_files(download_dir):
    """
    Delete error files, i.e. images that failed to download.

    Args:
        download_dir: The directory where the images are saved.
    """
    
    num_errors_files = 0
    index = 0
    notexist_tally = 0

    while notexist_tally < 10:  # XXX: bad heuristic check
        # Create image name and path
        image_name = IMAGE_NAME_FORMAT(index)
        image_path = os.path.join(download_dir, image_name)

        # Delete error file, if any
        if os.path.exists(image_path):
            notexist_tally = 0
            with open(image_path, "rb") as image:
                if IS_ERROR_FILE(image):
                    print("Removing %s" % image_name)
                    os.remove(image_path)
                    num_errors_files += 1
        else:
            notexist_tally += 1

        index += 1

    print("Deleted %d error files." % num_errors_files)
    return num_errors_files


def main(args):

    start_time = time.time()

    # Create dataset directory if it doesn't exist
    if not os.path.isdir(args.download_dir):
        os.mkdir(args.download_dir)

    # Download images
    with open(args.image_urls, "r") as f:
        image_urls = (line.rstrip() for line in f)
        download_images(image_urls, args.download_dir)

    delete_error_files(args.download_dir)

    print("Time elapsed = {:.3f}".format(time.time() - start_time))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Download images from a file of image urls.")
    
    parser.add_argument("-o", "--download_dir", type=str, default=DOWNLOAD_DIR,
        help="the directory where the images will be downloaded.")
    parser.add_argument("-i", "--image_urls", type=str, default=IMAGE_URLS,
        help="the output file where the urls of the images are saved.")
    
    args = parser.parse_args()

    main(args)

