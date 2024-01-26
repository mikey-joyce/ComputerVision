import numpy
import cv2
import matplotlib.pyplot as plt


def compare_orig_vs_new(orig, new):
    plt.subplot(121), plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(122), plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB)), plt.title('New')
    plt.show()


def plot_rgb_hist(image, title='Histogram'):
    hist_red = cv2.calcHist([image], [2], None, [256], [0, 256])
    hist_green = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_blue = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Plot the histograms
    plt.plot(hist_red, color='red', label='Red')
    plt.plot(hist_green, color='green', label='Green')
    plt.plot(hist_blue, color='blue', label='Blue')

    # Customize the plot
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Load in the test images !
    barack = cv2.imread('images/bobama.jpg', cv2.IMREAD_UNCHANGED)
    plot_rgb_hist(barack, title='Barack Histogram')

    michelle = cv2.imread('images/mobama.jpg', cv2.IMREAD_UNCHANGED)
    plot_rgb_hist(michelle, title='Michelle Histogram')

    # Gaussian blur
    kernel_shape = (21, 21)
    barack_lowpass = cv2.GaussianBlur(barack, kernel_shape, 0)
    plot_rgb_hist(barack_lowpass, title='Barack Blur Histogram')

    michelle_lowpass = cv2.GaussianBlur(michelle, kernel_shape, 0)
    plot_rgb_hist(michelle_lowpass, title='Michelle Blur Histogram')

    # Compare images
    compare_orig_vs_new(barack, barack_lowpass)
    compare_orig_vs_new(michelle, michelle_lowpass)

    # Sharpen images
    barack_unsharp = cv2.addWeighted(barack, 2.0, barack_lowpass, -1.0, 0)
    plot_rgb_hist(barack_unsharp, title='Barack Sharp Histogram')

    michelle_unsharp = cv2.addWeighted(michelle, 2.0, michelle_lowpass, -1.0, 0)
    plot_rgb_hist(michelle_unsharp, title='Michelle Sharp Histogram')

    # Compare images
    compare_orig_vs_new(barack, barack_unsharp)
    compare_orig_vs_new(michelle, michelle_unsharp)

