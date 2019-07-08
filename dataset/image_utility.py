
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def imshow_from_url(url):
    image_data = requests.get(url)
    image_data.raise_for_status()
    image = Image.open(BytesIO(image_data.content))
    fig = plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.show()
