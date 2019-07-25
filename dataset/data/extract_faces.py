
import os
import argparse
import pickle
from PIL import Image
import face_recognition
from utility import files_iter


# Get absolute path and force relative-to-file paths
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

### NOTE: we assume that all visible files in source dir are images ###
SOURCE_DIR = os.path.join(FILE_DIR, "processing", "splits")
DEST_DIR = os.path.join(FILE_DIR, "processing", "faces")


def extract_faces(source_dir, faces_dir, with_landmarks=True):
    """
    Try to extract faces from the images in source_dir and save them to faces_dir.

    Args:
        source_dir: Directory of source images.
        faces_dir: Directory where face images will be saved.
        with_landmarks: Extract faces landmarks as well
    """

    landmarks_dir = os.path.join(faces_dir, "landmarks")

    # Create destination directory if it doesn't exist
    if not os.path.isdir(faces_dir): os.mkdir(faces_dir)
    if with_landmarks and not os.path.isdir(landmarks_dir): os.mkdir(landmarks_dir)

    for file_name in files_iter(source_dir):
        break
        # Try to extract face from file (image)
        try:
            print("Extracting face from {}... ".format(file_name), end="")
            face_image_name = extract_face(file_name, source_dir, faces_dir)
            
            # Extract landmarks if needed
            if with_landmarks:
                extract_landmarks(face_image_name, faces_dir, landmarks_dir)
            
            print("Done.")

        except Exception as e:
            print("Failed."); print(e)

    clean_faces(faces_dir)
    if with_landmarks: clean_landmarks(faces_dir, landmarks_dir)


def extract_face(file_name, source_dir, dest_dir):
    """
    Extract the first detected face from the image in `file_name` and save it.

    Args:
        file_name: The name of the file (image).
        source_dir: Directory of source images.
        dest_dir: Directory where processed images will be saved.

    Returns:
        The name of the face image.
    """

    face_image_name = file_name

    # Check if destination image already exists (i.e. processed previously)
    face_image_path = os.path.join(dest_dir, face_image_name)
    if not os.path.exists(face_image_path):
        # load image and extract faces from it
        source_path = os.path.join(source_dir, file_name)
        image = face_recognition.load_image_file(source_path)
        face_locations = face_recognition.face_locations(image)

        print("Extracted {} face(s)... ".format(len(face_locations)), end="")
        if len(face_locations) == 0:
            raise Exception(" Couldn't extract any faces.")

        # Crop face and save
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(face_image_path)

    return face_image_name


def extract_landmarks(file_name, source_dir, dest_dir):
    """
    Extract the first detected face from the image in `file_name` and save it.

    Args:
        file_name: The path to the face image
        source_dir: Directory of images.
        dest_dir: Directory where processed images will be saved.

    Returns:
        The name of the landmarks file.
    """

    landmarks_name = file_name.split(".")[0] + ".pickle"

    # Check if landmarks already exists
    landmarks_path = os.path.join(dest_dir, landmarks_name)
    if not os.path.exists(landmarks_path):
        # load image and extract landmarks from it
        source_path = os.path.join(source_dir, file_name)
        image = face_recognition.load_image_file(source_path)
        face_landmarks = face_recognition.face_landmarks(image)

        print("Extracted {} face landmarks... ".format(len(face_landmarks)), end="")
        if len(face_landmarks) == 0:
            raise Exception(" Couldn't extract any landmarks.")

        # Save landmarks
        with open(landmarks_path, "wb") as f:
            pickle.dump(face_landmarks[0], f)

    return landmarks_name


def clean_faces(faces_dir):
    """
    Clean incomplete face pairs (either before or after image is missing)

    Args:
        source_dir: Directory of the examples.
    """

    for file_name in files_iter(faces_dir):

        image_name, ext = file_name.split(".")
        index, which = image_name.split("-")

        other_which = "after" if which == "before" else "before"
        other_file = "{}-{}.{}".format(index, other_which, ext)

        if not os.path.exists(os.path.join(faces_dir, other_file)):
            # Remove this file if the other does not exist
            os.remove(os.path.join(faces_dir, file_name))
            print("Removed {}".format(file_name))


def clean_landmarks(faces_dir, landmarks_dir):
    """
    Clean landmarks not associated to any images in faces_dir.

    Args:
        faces_dir: Directory of the faces.
        landmarks_dir: Directory of the landmarks.
    """

    faces_set = set(f.split(".")[0] for f in files_iter(faces_dir))
    for landmarks in files_iter(landmarks_dir):
        landmarks_name = landmarks.split(".")[0]
        if landmarks_name not in faces_set:
            os.remove(os.path.join(landmarks_dir, landmarks))
            print("Removed {}".format(landmarks))


def main(args):
    extract_faces(args.source_dir, args.dest_dir, args.with_landmarks)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="extract faces and save them.")
    
    parser.add_argument('--source_dir', type=str, default=SOURCE_DIR,
        help="source directory of images from which faces will be extracted.")
    parser.add_argument('--dest_dir', type=str, default=DEST_DIR,
        help="destination directory where face images will be saved.")
    parser.add_argument("--with_landmarks", action="store_true",
        help="extract faces landmarks as well")
    
    args = parser.parse_args()

    main(args)

