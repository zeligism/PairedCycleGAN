SOURCE_DIR=${1:-"instagram/test_images"}
DEST_DIR=${2:-"instagram/test_faces"}
LOG="extract_faces.log"

mkdir -p "$DEST_DIR"
rm -f "$LOG"

time (find "$SOURCE_DIR" -maxdepth 1 -type f -exec basename {} \; \
| parallel --bar "python extract_faces.py --source_dir $SOURCE_DIR --dest_dir $DEST_DIR --image {} 1>>$LOG 2>&1")