import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def get_rgb_strcture_tensor(img, sigma=1.0):
    r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 0]

    Ix_R, Iy_R, Ixy_R = get_structure_tensor(r, sigma)
    Ix_G, Iy_G, Ixy_G = get_structure_tensor(g, sigma)
    Ix_B, Iy_B, Ixy_B = get_structure_tensor(b, sigma)

    Ix = (Ix_R + Ix_G + Ix_B)
    Iy = (Iy_R + Iy_G + Iy_B)
    Ixy = (Ixy_R + Ixy_G + Ixy_B)

    return Ix, Iy, Ixy


def get_structure_tensor(img, sigma=1.0):
    dx = convolve(img, np.array([[-1, 0, 1]]), mode='nearest')
    dy = convolve(img, np.array([[-1, 0, 1]]).T, mode='nearest')

    Ix = dx**2
    Iy = dy**2
    Ixy = dx*dy

    ksize = int(6 * sigma) | 1
    outer = np.outer(np.exp(-np.arange(-(ksize - 1)//2, (ksize - 1)//2 + 1)**2 / (2 * sigma**2)), np.ones(ksize))

    Ix = convolve(Ix, outer, mode='nearest')
    Iy = convolve(Iy, outer, mode='nearest')
    Ixy = convolve(Ixy, outer, mode='nearest')

    return Ix, Iy, Ixy


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val) * 255
    return normalized_image.astype(np.uint8)


def experiment(sig):
    img = cv2.imread('images/cells3.jpg', cv2.IMREAD_UNCHANGED)
    image = img.astype(float) / 255.0

    Ix, Iy, Ixy = get_rgb_strcture_tensor(image, sigma=sig)

    structure_tensor = np.array([[Ix, Ixy], [Ixy, Iy]])
    structure_trace = np.trace(structure_tensor)

    plt.imshow(Ix, cmap='gray')
    plt.title('Ix Sig='+str(sig))
    plt.savefig('results/Ix_sig='+str(sig)+'.png', dpi=900)
    plt.show()

    plt.imshow(Iy, cmap='gray')
    plt.title('Iy Sig='+str(sig))
    plt.savefig('results/Iy_sig='+str(sig)+'.png', dpi=900)
    plt.show()

    plt.imshow(Ixy, cmap='gray')
    plt.title('Ixy Sig='+str(sig))
    plt.savefig('results/Ixy_sig='+str(sig)+'.png', dpi=900)
    plt.show()

    plt.imshow(structure_trace, cmap='gray')
    plt.title('Structure Tensor Trace Sigma='+str(sig))
    plt.savefig('results/tensor_trace_sig='+str(sig)+'.png', dpi=900)
    plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.show()

    Ix, Iy, Ixy = get_structure_tensor(gray, sigma=sig)
    gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Grayscale Gradient Magnitude Sig='+str(sig))
    plt.savefig('results/gradient_magnitude_sig='+str(sig)+'.png', dpi=900)
    plt.show()


if __name__ == '__main__':
    experiment(sig=1.0)
    experiment(sig=5.0)
