"""
Extract all images from a directory,
and save them to another directory after changing the size
"""

from PIL import Image
import os.path
import glob

def convertpng(pngfile,outdir,origin_point=5,width=256,height=256):
    img=Image.open(pngfile)

    try:
        # The position of the four corners of the cropped images
        box = (origin_point, origin_point, origin_point+width, origin_point+height)
        new_img = img.crop(box)
        new_img.save(os.path.join(outdir,os.path.basename(pngfile)))
    except Exception as e:
        print(e)


for pngfile in glob.glob("E:\\Codes\\ThinCloudRemove\\dataset\\original_cloud\\*.png"):
        convertpng(pngfile,"E:\\Codes\\ThinCloudRemove\\dataset\\test\\cloud")