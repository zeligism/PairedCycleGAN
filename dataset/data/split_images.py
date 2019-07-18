
import os
import argparse
from PIL import Image

from utility import files_iter


# Get absolute path of this file and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

### NOTE: we assume that all visible files in source dir are images ###
SOURCE_DIR = os.path.join(FILE_DIR, "processing", "cleaned")
TARGET_DIR = os.path.join(FILE_DIR, "processing", "split")


def split_image(fname, source_dir, target_dir):
    """
    Splits the image `fname` (left and right) and save the splits.

    Args:
        fname: The name of the file (image) to be split.
        source_dir: Directory of source images.
        target_dir: Directory where processed images will be saved.
    """

    with Image.open(os.path.join(source_dir, fname)) as img:

        # Remove extension from file name and rename split images
        img_name = fname.split(".")[0]
        ext = "." + img.format.lower()
        img_path_left = os.path.join(target_dir, img_name + "-before" + ext)
        img_path_right = os.path.join(target_dir, img_name + "-after" + ext)

        if os.path.exists(img_path_left) or os.path.exists(img_path_right):
            return  # this checks if the images was already split

        # Create left and right crops for splitting the image
        (left, upper, right, lower) = img.getbbox()
        mid_x = left + (right - left) // 2
        left_box = (left, upper, mid_x, lower)
        right_box = (mid_x, upper, right, lower)

        # Save left and right crops of image
        img.crop(left_box).save(img_path_left, format=img.format)
        img.crop(right_box).save(img_path_right, format=img.format)


def split_images(source_dir, target_dir):
    """
    Try to split the images in source_dir and save them to target_dir.

    Args:
        source_dir: Directory of source images.
        target_dir: Directory where processed images will be saved.
    """

    # Create target directory if it doesn't exist
    if not os.path.isdir(target_dir): os.mkdir(target_dir)

    for fname in files_iter(source_dir):
        # Try to split the image
        try:
            # We assume that fname has no dots except the one before its extension
            print("Splitting image {}... ".format(fname.split(".")[0]), end="")
            split_image(fname, source_dir, target_dir)
            print("Done.")
        except Exception:
            print("Failed.")


def main(args):
    split_images(args.source_dir, args.target_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Split makeup images into before and after.")
    
    parser.add_argument('--source_dir', type=str, default=SOURCE_DIR,
        help="Source directory of images to be split in half.")
    parser.add_argument('--target_dir', type=str, default=TARGET_DIR,
        help="Target directory where split images will be saved.")
    
    args = parser.parse_args()

    main(args)


