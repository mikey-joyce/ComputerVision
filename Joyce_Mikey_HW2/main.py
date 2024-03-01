import cv2
import numpy as np
from skimage.feature import peak_local_max as plm
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def computeFeatureMeasure(I, k=0.1, window=3):
    blur = cv2.GaussianBlur(I, (3, 3), 0)
    R = np.zeros((I.shape[0], I.shape[1]))

    dx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    dxx = dx**2
    dyy = dy**2
    dxy = dx * dy

    scale = int(window/2)
    for y in range(scale, I.shape[0]-scale):
        for x in range(scale, I.shape[1]-scale):
            y_low, y_hi = y-scale, y+1+scale
            x_low, x_hi = x-scale, x+1+scale

            Sxx = np.sum(dxx[y_low:y_hi, x_low:x_hi])
            Syy = np.sum(dyy[y_low:y_hi, x_low:x_hi])
            Sxy = np.sum(dxy[y_low:y_hi, x_low:x_hi])

            H = np.array([[Sxx, Sxy], [Sxy, Syy]])

            R[y_low, x_low] = np.linalg.det(H) - k * (np.matrix.trace(H)**2)

    R = (R - np.min(R)) * (255.0/(np.max(R) - np.min(R)))

    return np.max(R)-R


def FeatureMeasure2Points(R, npoints=500):
    mask = np.zeros_like(R)

    maxima = plm(image=R, num_peaks=npoints, p_norm=2)

    for x, y in maxima:
        mask[x][y] = 1

    return maxima, mask


def generateFeatureDescriptor(I, coords, size_patch=3):
    # simple feature descriptor, just grabbing patches around each coordinate
    Dlist, official_coords = [], []
    for x, y in coords:
        beg_x = x - size_patch // 2
        end_x = beg_x + size_patch

        beg_y = y - size_patch // 2
        end_y = beg_y + size_patch

        if beg_x >= 0 and beg_y >= 0 and end_x <= I.shape[1] and end_y <= I.shape[0]:
            Dlist.append(I[beg_y:end_y, beg_x:end_x])
            official_coords.append([x, y])
        else:
            # print(f"Skipping patch ({x},{y}): out of image bounds")
            z = None    # this is just here to ensure that this "else" statement runs
                        # during testing i used the print statement, but needed to
                        # omit it so that my TPR and FPRs were able to be viewed for reporting

    return Dlist, official_coords


def computeDescriptorDistances(Dlist1, Dlist2):
    Dist = np.zeros((len(Dlist1), len(Dlist2)))

    i = 0
    for f1 in Dlist1:
        j = 0
        for f2 in Dlist2:
            d = np.linalg.norm(f1 - f2)
            Dist[i][j] = d
            j += 1
        i += 1

    return Dist


def Distance2Matches_DistThresh(Dist, Th1):
    MatchList = []
    for i in range(Dist.shape[0]):
        for j in range(Dist.shape[1]):
            if Dist[i][j] < Th1:  # just match everything if it is below the thresh
               MatchList.append([i, j])

    return np.array(MatchList)


def Distance2Matches_NearestMatch(Dist, Th2):
    MatchList = []
    for i in range(Dist.shape[0]):
        last, index = None, None
        for j in range(Dist.shape[1]):
            if last is None:
                last = Dist[i][j]
                index = j
            else:
                if Dist[i][j] < last:
                    last = Dist[i][j]
                    index = j
        if last < Th2:  # nearest match
            MatchList.append([i, index])

    return np.array(MatchList)


def Distance2Matches_NearestRatio(Dist, Th3):
    MatchList = []
    for i in range(Dist.shape[0]):
        last, index, second = None, None, None
        for j in range(Dist.shape[1]):
            if last is None:
                last = Dist[i][j]
                index = j
            else:
                if Dist[i][j] < last:
                    second = last
                    last = Dist[i][j]
                    index = j
        if (last/second) < Th3:  # nearest ratio
            MatchList.append([i, index])

    return np.array(MatchList)


def transformation(coords, transform):
    homogeneous = np.hstack((coords, np.ones((len(coords), 1))))
    points = np.dot(transform, homogeneous.T).T
    points = points[:, :2]/points[:, 2:]
    return np.round(points[:, :2]).astype(int)


def compute_metrics(matches, kp2, truth, k=100):
    tpr, fpr = 0, 0
    for i, j in matches:
        d = np.linalg.norm(kp2[j] - truth[i])
        if abs(d) < k:
            tpr += 1
        else:
            fpr += 1

    return tpr, fpr


def visualize_matches(img1, img2, matches, kp1, kp2, name):
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(wspace=0)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    for i, j in matches:
        p1, p2 = kp1[i], kp2[j]
        connection = ConnectionPatch(xyA=p1, xyB=p2, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="yellow")
        ax2.add_artist(connection)

    plt.title(name)
    plt.savefig('results/' + str(name) + '.png', dpi=900)
    plt.show()


def experiment(tmat, img1, img2, name, k=0.5, Th1=10, Th2=10, Th3=0.5):
    # do harris detection
    img1_corners = computeFeatureMeasure(bike1, k=k)
    img2_corners = computeFeatureMeasure(bike2, k=k)

    img1_kp, img1_mask = FeatureMeasure2Points(img1_corners)
    img2_kp, img2_mask = FeatureMeasure2Points(img2_corners)

    plt.imshow(img1, cmap='gray')
    plt.contour(img1_mask)
    plt.title('Img 1 Mask: ' + name)
    plt.savefig('results/' + str(name) + 'img1mask' + '.png', dpi=900)
    plt.show()

    plt.imshow(img2, cmap='gray')
    plt.contour(img2_mask)
    plt.title('Img 2 Mask: ' + name)
    plt.savefig('results/' + str(name) + 'img2mask' +'.png', dpi=900)
    plt.show()

    img1_descriptors, img1_kp = generateFeatureDescriptor(img1, img1_kp)
    img2_descriptors, img2_kp = generateFeatureDescriptor(img2, img2_kp)

    img1_and_2_dist = computeDescriptorDistances(img1_descriptors, img2_descriptors)

    dist_matches = Distance2Matches_DistThresh(img1_and_2_dist, Th1=Th1)
    nn_matches = Distance2Matches_NearestMatch(img1_and_2_dist, Th2=Th2)
    nnr_matches = Distance2Matches_NearestRatio(img1_and_2_dist, Th3=Th3)

    H1to2p = np.loadtxt(tmat)
    truth = transformation(img1_kp, H1to2p)

    print('# Detected points: ', min(len(img1_kp), len(img2_kp)))

    tpr, fpr = compute_metrics(dist_matches, img2_kp, truth)
    print('True positive rate: ', tpr)
    print('False positive rate: ', fpr)
    print('# matched points: ', len(dist_matches))

    tpr, fpr = compute_metrics(nn_matches, img2_kp, truth)
    print('True positive rate: ', tpr)
    print('False positive rate: ', fpr)
    print('# matched points: ', len(nn_matches))


    tpr, fpr = compute_metrics(nnr_matches, img2_kp, truth)
    print('True positive rate: ', tpr)
    print('False positive rate: ', fpr)
    print('# matched points: ', len(nnr_matches))

    visualize_matches(bike1, bike2, dist_matches, img1_kp, img2_kp, name + '_distance')
    visualize_matches(bike1, bike2, nn_matches, img1_kp, img2_kp, name + '_nn')
    visualize_matches(bike1, bike2, nnr_matches, img1_kp, img2_kp, name + '_nnr')


if __name__ == '__main__':
    img_dir = 'images/'
    bike_dir = img_dir + 'bikes/'

    bike1 = cv2.imread(bike_dir + 'img1.ppm')
    bike2 = cv2.imread(bike_dir + 'img2.ppm')

    bike1 = cv2.cvtColor(bike1, cv2.COLOR_BGR2GRAY)
    bike2 = cv2.cvtColor(bike2, cv2.COLOR_BGR2GRAY)

    print('EXPERIMENT #1')
    experiment(bike_dir + 'H1to2p', bike1, bike2, 'bike1_and_bike2')

    bike5 = cv2.imread(bike_dir + 'img5.ppm')
    bike5 = cv2.cvtColor(bike5, cv2.COLOR_BGR2GRAY)

    print('EXPERIMENT #2')
    experiment(bike_dir + 'H1to5p', bike1, bike5, 'bike1_and_bike5')
