from PIL import Image
from resizeimage import resizeimage
import FaceRecognition

def run(image, count):
    FaceRecognition.run(image)

    with open('ROI.png', 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [400, 175], validate=False)
            cover.save('resizedROI%d.png' % count, image.format)
