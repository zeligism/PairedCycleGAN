
import time
import os
from PIL import Image
import face_recognition



### NOTE: we assume that all visible files in source dir are images ###
# Source directory of images to be split (before makeup and after makeup)
SOURCE_DIR = os.path.join("processing", "split")
# Target directory where split images will be saved
TARGET_DIR = os.path.join("processing", "faces")

# Get absolute path of this file and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(FILE_DIR, SOURCE_DIR)
TARGET_DIR = os.path.join(FILE_DIR, TARGET_DIR)

# Create directories if they don't exist
if not os.path.isdir(SOURCE_DIR): os.mkdir(SOURCE_DIR)
if not os.path.isdir(TARGET_DIR): os.mkdir(TARGET_DIR)


def extract_face(fname):
    """
    Extract the first detected face from the image in `fname` and save it.

    Args:
        fname: The name of the file holding the image.
    """

    name, ext = fname.split(".")
    face_image_path = os.path.join(TARGET_DIR, name + "-face" + "." + ext)
    if os.path.exists(face_image_path): return

    image = face_recognition.load_image_file(os.path.join(SOURCE_DIR, fname))
    face_locations = face_recognition.face_locations(image)
    print("Extracted {} face(s)... ".format(len(face_locations)), end="")

    # Extract face from image
    if len(face_locations) == 0:
        raise Exception(" Couldn't extract any face.")
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.save(face_image_path)


def main():

    start_time = time.time()

    for fname in os.listdir(SOURCE_DIR):
        # Skip directories or files starting with `.`
        if fname[0] == ".": continue
        if os.path.isdir(os.path.join(SOURCE_DIR, fname)): continue

        print("Extracting face from {}... ".format(fname), end="")
        try:
            extract_face(fname)
            print("Done.")
        except Exception as e:
            print("Failed.")
            print(e)

    print("Time elapsed = {:.3f}".format(time.time() - start_time))


if __name__ == '__main__':
    main()

