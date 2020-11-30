# I'm kinda embarrassed that the following shell script employing wget and
# parallel blows my download_images.py out of the water.
# Not sure why I even thought that writing that python script would be a good idea.

# Download
DOWNLOAD_DIR="downloaded_images"
echo "Downloading images in $1"
cat "$1" | parallel --gnu "wget {} -P ${DOWNLOAD_DIR}/"
echo "Done."
echo ""

# Rename
index=1
for file in ${DOWNLOAD_DIR}/*; do
    index_name=$(printf "${DOWNLOAD_DIR}/%05d.jpg" $index)
    echo "Renaming $file to $index_name"
    mv "$file" "$index_name"
    ((index++))
done
