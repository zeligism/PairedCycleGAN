
import argparse
import json
import face_recognition
import glob


def main(args):
    ...


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="...")
    
    parser.add_argument('--faces_dir', type=str,
        help="source directory of faces to be filtered.")
    parser.add_argument('--by', type=str, default="image", choices=("image", "identity"),
        help="filter by image or by identity.")
    
    args = parser.parse_args()

    main(args)

