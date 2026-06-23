from imwatermark import WatermarkDecoder
import cv2
from sympy.testing.runtests import method

img = cv2.imread("/Users/kevin/Desktop/aaa.png")

for method in ['dwtDct', 'dwtDctSvd', 'rivaGan']:
    decoder = WatermarkDecoder('bytes', 32)
    try:
        wm = decoder.decode(img, method=method)
        print(f"{method}: {wm}")
    except:
        print(f"{method}: no watermark")