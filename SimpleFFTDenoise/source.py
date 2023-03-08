import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
def fft(img, show=True,save=True):
    fft_sh = np.fft.fftshift(np.fft.fft2(img))  # pomeranje koordinatnog pocetka u centar slike
    img_fft_log = np.log(np.abs(fft_sh))
    if show:
        cv2.imshow("FFT",img_fft_log*5)
    if save:
        cv2.imwrite("fft_mag.png",img_fft_log*5)
    return fft_sh

def ifft(mag, cmod):
    return np.abs(np.fft.ifft2(cmod * np.exp(mag)))

def filter(img, center,show=True,save=True):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_norm = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    max_dist = sqrt(center[0]*center[0]+center[1]*center[1])
    steps = 1.7
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            #if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > radius*radius:
            dist = sqrt((x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]))
            percentage = dist/max_dist #od 0 do 1
            dist=percentage*steps+1.1
            img_fft_log[x,y]=img_fft_log[x,y]*(1/dist)
                    
    if show:
        cv2.imshow("Filtered FFT",img_fft_log*5)
    if save:
        cv2.imwrite("fft_mag_filtered.png",img_fft_log*5)

    img_filtered = ifft(img_fft_log, img_mag_norm) * 5

    return img_filtered

if __name__ == '__main__':
    img = cv2.imread("input.png", cv2.COLOR_BGR2GRAY)
    center = (256, 256)
    cv2.imshow("Image",img)
    img_filtered = filter(img, center)
    cv2.imshow("Filtered image",img_filtered)
    cv2.imwrite("output.png",img_filtered)
    cv2.waitKey(0)