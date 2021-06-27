#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/2/15/015 15:15

import os

class BatchRename():

    """Name the images in the folder by a specific serial number"""

    def __init__(self):
        self.path = r'E:\JB23\codes\SRCNN-master\GF-1'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path),format(str(i)) + '.png')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d pngs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
