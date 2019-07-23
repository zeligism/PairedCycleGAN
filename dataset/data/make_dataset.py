
import os
import argparse
from shutil import copyfile
from utility import files_iter


# Get absolute path and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

### NOTE: we assume that all visible files in source dir are images ###
SOURCE_DIR = os.path.join(FILE_DIR, "processing", "faces")
DEST_DIR = os.path.join(FILE_DIR, "processed")


def process_image(fname, index, source_dir, dest_dir):
    """
    Process image for dataset.

    Args:
        fname: The name of the file (image).
        index: index of image in the dataset.
        source_dir: Directory of source images.
        dest_dir: Directory where processed images will be saved.
    """
    
    img_full_name, ext = fname.split(".")
    img_name, when = img_full_name.split("-")

    # Path of source images
    src_before = os.path.join(source_dir, "{}-before.{}".format(img_name, ext))
    src_after  = os.path.join(source_dir, "{}-after.{}".format(img_name, ext))
    # Path of destination (renamed) images
    dest_before = os.path.join(dest_dir, "{:05}-before.{}".format(index, ext))
    dest_after  = os.path.join(dest_dir, "{:05}-after.{}".format(index, ext))

    if os.path.exists(dest_before) or os.path.exists(dest_after):
        return  # skip because it seems like we already copied the images

    # Copy source to dest
    print("Copying... ", end="")
    copyfile(src_before, dest_before)
    copyfile(src_after, dest_after)


def make_dataset(source_dir, dest_dir):
    """
    Make dataset in dest_dir from images in source_dir.

    Args:
        source_dir: Directory of source images.
        dest_dir: Directory where processed images will be saved.
    """

    # Create destination directory if it doesn't exist
    if not os.path.isdir(dest_dir): os.mkdir(dest_dir)

    # Look at all the files in the source_dir
    index = 0
    for fname in files_iter(source_dir):
        try:
            print("Processing image {}... ".format(fname), end="")
            process_image(fname, index, source_dir, dest_dir)
            print("Done.")
            index += 1
        except Exception as e:
            print("Failed.")
            print(e)
            raise e

    print("Total images in dataset = {}".format(index))


def main(args):
    make_dataset(args.source_dir, args.dest_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process images to make dataset.")
    
    parser.add_argument('--source_dir', type=str, default=SOURCE_DIR,
        help="Source directory of images to be processed.")
    parser.add_argument('--dest_dir', type=str, default=DEST_DIR,
        help="Destination directory of dataset.")
    
    args = parser.parse_args()

    main(args)

