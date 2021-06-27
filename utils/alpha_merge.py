import numpy as np
import cv2
import random
import os

class BatchRename():

    def __init__(self):
        self.foreGroundImage_path = r'E:\\Codes\\GenerateCloudImgProcessing\\generate_cloud_data\\mask700' # generated cloud layers

        self.background_path = r'E:\\Codes\\GenerateCloudImgProcessing\\sentineL1C_references'

        self.merged_path = r'E:\\Codes\\GenerateCloudImgProcessing\\sentineL1C_cloudy'

    def alphamerge(self, foreGroundImage, background):
        ## split channels of the image
        b, g, r, a = cv2.split(foreGroundImage)

        # Get the foreground part of the image,
        # which in this case is the part with the alpha channel removed
        foreground = cv2.merge((b, g, r))

        # Get alpha mask of the image
        alpha = cv2.merge((a, a, a))
        # alpha = foreground * (0.2 * random.uniform(0.5,1)+1)

        foreground = foreground.astype(float)
        background = background.astype(float)

        alpha = alpha.astype(float) / 255

        cv2.waitKey(0)

        # The foreground and background are weighted,
        # and the weighting coefficient of each pixel is the pixel value at the corresponding position of alpha mask.
        # The foreground part is 1, and the background part is 0
        foreground = cv2.multiply(0.5 * alpha, foreground)  # 0.5 * alpha
        background = cv2.multiply(0.5 * alpha, background)

        outImage = foreground + background

        return outImage

    def rename(self):
        filelist = os.listdir(self.foreGroundImage_path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.foreGroundImage_path), item)
                dst = os.path.join(os.path.abspath(self.merged_path),format(str(i))  + '.png')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()



