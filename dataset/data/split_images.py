
import time
import os
from PIL import Image


### NOTE: we assume that all visible files in source dir are images ###
# Source directory of images to be split (before makeup and after makeup)
SOURCE_DIR = os.path.join("processing", "cleaned")
# Target directory where split images will be saved
TARGET_DIR = os.path.join(SOURCE_DIR, "split")

# Get absolute path of this file and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(FILE_DIR, SOURCE_DIR)
TARGET_DIR = os.path.join(FILE_DIR, TARGET_DIR)

# Create directories if they don't exist
if not os.path.isdir(SOURCE_DIR): os.mkdir(SOURCE_DIR)
if not os.path.isdir(TARGET_DIR): os.mkdir(TARGET_DIR)


def split_image_vertically(fname):

    with Image.open(os.path.join(SOURCE_DIR, fname)) as img:

        # Remove extension from file name and rename split images
        img_name = fname.split(".")[0]
        ext = "." + img.format.lower()
        img_path_left = os.path.join(TARGET_DIR, img_name + "-before" + ext)
        img_path_right = os.path.join(TARGET_DIR, img_name + "-after" + ext)

        if os.path.exists(img_path_left) or os.path.exists(img_path_right):
            return

        # Create left and right crops to split image vertically
        (left, upper, right, lower) = img.getbbox()
        mid_x = left + (right - left) // 2
        left_box = (left, upper, mid_x, lower)
        right_box = (mid_x, upper, right, lower)

        # Save left and right crops of image
        img.crop(left_box).save(img_path_left, format=img.format)
        img.crop(right_box).save(img_path_right, format=img.format)


def main():

    start_time = time.time()

    for fname in os.listdir(SOURCE_DIR):
        # Skip directories or files starting with `.`
        if fname[0] == ".": continue
        if os.path.isdir(os.path.join(SOURCE_DIR, fname)): continue

        print("Splitting image {}... ".format(fname), end="")
        try:
            split_image_vertically(fname)
            print("Done.")
        except Exception:
            print("Failed.")

    print("Time elapsed = {:.3f}".format(time.time() - start_time))


if __name__ == '__main__':
    main()

