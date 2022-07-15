import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

Path("../champion_images_cropped/").mkdir(parents=True, exist_ok=True)
directory = "../champion_images"

for filename in os.listdir(directory):
    print(filename)
    f = os.path.join(directory, filename)

    # Open the input image as numpy array, convert to RGB
    img = Image.open(f).convert("RGB")
    img = img.crop((6, 6, 114, 114))


    npImage = np.array(img)
    h, w = img.size
    # Create same size alpha layer with circle
    alpha = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0, 0, h, w], 0, 360, fill=255)
    # Convert alpha Image to numpy array
    npAlpha = np.array(alpha)
    # Add alpha layer to RGB
    npImage = np.dstack((npImage, npAlpha))

    # Save with alpha
    Image.fromarray(npImage).save(os.path.join("../champion_images_cropped", filename))