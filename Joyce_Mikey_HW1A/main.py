import numpy as np
import cv2
import matplotlib.pyplot as plt


def compare_orig_vs_new(orig, new):
    plt.subplot(121), plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(122), plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB)), plt.title('New')
    plt.show()


def plot_rgb_hist(image, title='Histogram', fig_path=None):
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

    if fig_path:
        plt.savefig(fig_path, dpi=900)

    plt.show()


def hybrid_img(blur_img, sharp_img, sigma=5.0):
    blur_img = blur_img.astype(np.float32) / 255.0
    sharp_img = sharp_img.astype(np.float32) / 255.0

    # Gaussian blur
    kernel_size = int(6 * sigma) + 1
    gaussian = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian = gaussian @ gaussian.T  # 2D Convolution

    lowpass_blur = cv2.filter2D(blur_img, -1, gaussian)
    highpass_blur = blur_img - lowpass_blur

    lowpass_sharp = cv2.filter2D(sharp_img, -1, gaussian)
    highpass_sharp = sharp_img - lowpass_sharp

    hybrid_1 = highpass_sharp + lowpass_blur
    hybrid_2 = highpass_blur + lowpass_sharp

    hybrid_1 = np.clip(hybrid_1, 0, 1)
    hybrid_2 = np.clip(hybrid_2, 0, 1)

    return hybrid_1, hybrid_2, lowpass_blur, highpass_blur, lowpass_sharp, highpass_sharp


def denormalize(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


if __name__ == '__main__':
    plots_dir = 'results/plots/'
    imgs_dir = 'results/images/'
    # Load in the test images!
    cat = cv2.imread('images/cat_crp.jpg', cv2.IMREAD_UNCHANGED)
    plot_rgb_hist(cat, title='Cat Histogram', fig_path=plots_dir+'cat_orig_hist.png')

    dog = cv2.imread('images/dog_crp.jpg', cv2.IMREAD_UNCHANGED)
    plot_rgb_hist(dog, title='Dog Histogram', fig_path=plots_dir+'dog_orig_hist.png')

    sig = 5.0
    h1_img, h2_img, low_dog, high_dog, low_cat, high_cat = hybrid_img(dog, cat, sigma=sig)

    h1_img = denormalize(h1_img)
    h2_img = denormalize(h2_img)
    low_dog = denormalize(low_dog)
    high_dog = denormalize(high_dog)
    low_cat = denormalize(low_cat)
    high_cat = denormalize(high_cat)

    cv2.imwrite(imgs_dir + 'hybrid_dog_cat_sig=' + str(sig) + '.png', h1_img)
    cv2.imwrite(imgs_dir + 'hybrid_cat_dog_sig=' + str(sig) + '.png', h2_img)
    cv2.imwrite(imgs_dir + 'low_dog.png', low_dog)
    cv2.imwrite(imgs_dir + 'low_cat.png', low_cat)
    cv2.imwrite(imgs_dir + 'high_dog.png', high_dog)
    cv2.imwrite(imgs_dir + 'high_cat.png', high_cat)

    plot_rgb_hist(low_cat, title='Cat Smoothed', fig_path=plots_dir+'cat_smooth_hist.png')
    plot_rgb_hist(low_dog, title='Dog Smoothed', fig_path=plots_dir+'dog_smooth_hist.png')
    plot_rgb_hist(high_cat, title='Cat Sharpened', fig_path=plots_dir+'cat_sharp_hist.png')
    plot_rgb_hist(high_dog, title='Dog Sharpened', fig_path=plots_dir+'dog_sharp_hist.png')

    plt.imshow(cv2.cvtColor(h1_img, cv2.COLOR_BGR2RGB))
    plt.title('Hybrid; sigma=' + str(sig))
    plt.show()

    sig = 20.0
    h1_img, h2_img, low_dog, high_dog, low_cat, high_cat = hybrid_img(dog, cat, sigma=sig)

    h1_img = denormalize(h1_img)
    h2_img = denormalize(h2_img)

    cv2.imwrite(imgs_dir + 'hybrid_dog_cat_sig=' + str(sig) + '.png', h1_img)
    cv2.imwrite(imgs_dir + 'hybrid_cat_dog_sig=' + str(sig) + '.png', h2_img)

    plt.imshow(cv2.cvtColor(h2_img, cv2.COLOR_BGR2RGB))
    plt.title('Hybrid; sigma=' + str(sig))
    plt.show()
