
import os
import argparse
from PIL import Image
from utility import files_iter


# Get absolute path of this file and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

### NOTE: we assume that all visible files in source dir are images ###
SOURCE_DIR = os.path.join(FILE_DIR, "processing", "cleaned")
DEST_DIR = os.path.join(FILE_DIR, "processing", "splits")


def split_image(file_name, source_dir, dest_dir):
    """
    Splits the image `file_name` (to left and right) and save the splits.

    Args:
        file_name: The name of the file (image) to be split.
        source_dir: Directory of source images.
        dest_dir: Directory where split images will be saved.
    """

    with Image.open(os.path.join(source_dir, file_name)) as img:

        # Remove extension from file name and rename split images
        img_name = file_name.split(".")[0]
        ext = img.format.lower()
        img_path_left = os.path.join(dest_dir, "{}-before.{}".format(img_name, ext))
        img_path_right = os.path.join(dest_dir, "{}-after.{}".format(img_name, ext))

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


def split_images(source_dir, dest_dir):
    """
    Try to split the images in source_dir and save them to dest_dir.

    Args:
        source_dir: Directory of source images.
        dest_dir: Directory where processed images will be saved.
    """

    # Create destination directory if it doesn't exist
    if not os.path.isdir(dest_dir): os.mkdir(dest_dir)

    for file_name in files_iter(source_dir):
        try:
            # We assume that file_name has no dots except the one before its extension
            print("Splitting image {}... ".format(file_name.split(".")[0]), end="")
            split_image(file_name, source_dir, dest_dir)
            print("Done.")

        except Exception:
            print("Failed.")


def main(args):
    split_images(args.source_dir, args.dest_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="split makeup images into before and after.")
    
    parser.add_argument('--source_dir', type=str, default=SOURCE_DIR,
        help="source directory of images to be split in half.")
    parser.add_argument('--dest_dir', type=str, default=DEST_DIR,
        help="destination directory where split images will be saved.")
    
    args = parser.parse_args()

    main(args)


