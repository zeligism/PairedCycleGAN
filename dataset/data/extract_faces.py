
import os
import argparse
from PIL import Image
import face_recognition
from utility import files_iter


# Get absolute path and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

### NOTE: we assume that all visible files in source dir are images ###
SOURCE_DIR = os.path.join(FILE_DIR, "processing", "split")
DEST_DIR = os.path.join(FILE_DIR, "processing", "faces")


def extract_face(fname, source_dir, dest_dir):
    """
    Extract the first detected face from the image in `fname` and save it.

    Args:
        fname: The name of the file (image).
        source_dir: Directory of source images.
        dest_dir: Directory where processed images will be saved.
    """

    # Check if destination image already exists (i.e. processed previously)
    face_image_path = os.path.join(dest_dir, fname)
    if os.path.exists(face_image_path): return

    # load image and extract faces from it
    image = face_recognition.load_image_file(os.path.join(source_dir, fname))
    face_locations = face_recognition.face_locations(image)
    print("Extracted {} face(s)... ".format(len(face_locations)), end="")
    if len(face_locations) == 0:
        raise Exception(" Couldn't extract any faces.")

    # Crop face and save
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.save(face_image_path)


def extract_faces(source_dir, dest_dir):
    """
    Try to extract faces from the images in source_dir and save them to dest_dir.

    Args:
        source_dir: Directory of source images.
        dest_dir: Directory where processed images will be saved.
    """

    # Create destination directory if it doesn't exist
    if not os.path.isdir(dest_dir): os.mkdir(dest_dir)

    for fname in files_iter(source_dir):
        # Try to extract face from file (image)
        try:
            print("Extracting face from {}... ".format(fname), end="")
            extract_face(fname, source_dir, dest_dir)
            print("Done.")
        except Exception as e:
            print("Failed.")
            print(e)


def main(args):
    extract_faces(args.source_dir, args.dest_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract faces and save them.")
    
    parser.add_argument('--source_dir', type=str, default=SOURCE_DIR,
        help="Source directory of images from which faces will be extracted.")
    parser.add_argument('--dest_dir', type=str, default=DEST_DIR,
        help="Destination directory where face images will be saved.")
    
    args = parser.parse_args()

    main(args)

