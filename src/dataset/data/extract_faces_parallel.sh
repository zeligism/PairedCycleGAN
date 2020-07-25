SOURCE_DIR=${$1:-"instagram/test_images"}
DEST_DIR={$2:-"instagram/test_faces"}
mkdir -p "$DEST_DIR"
find "$SOURCE_DIR" -maxdepth 1 -type f -exec basename {} \; \
| parallel "python extract_faces.py --source_dir $SOURCE_DIR --dest_dir $DEST_DIR --image {}"