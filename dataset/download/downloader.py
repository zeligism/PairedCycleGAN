import requests
import os


def download_image(image_url, image_path="untitled"):

    # Download image in chunks
    image_response = requests.get(image_url, stream=True)
    if image_response.status_code == 200:
        with open(image_path, 'wb') as f:
            for chunk in image_response:
                f.write(chunk)
            return image_response.status_code

    return -1


def download_images(image_urls, dataset_dir="."):

    assert os.path.isdir(dataset_dir)

    # Download images
    print("Downloading {} images:".format(len(image_urls)))
    for index, image_url in enumerate(image_urls):

        # Create image name and path
        image_name = "{:05d}".format(index)
        image_path = dataset_dir + "/" + image_name

        # Download image and check download success
        print("[{:05d}]  Downloading {} ... ".format(index, image_url), end="")
        status_code = download_image(image_url, image_path)

        # Add image url to dataset metadata (can add more metadata later)
        print("Success." if status_code == 200 else "Fail. ({})".format(status_code))


def main():
    pass


if __name__ == '__main__':
    main()