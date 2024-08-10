import cv2
import os
import numpy as np

# Parameters
ALPHA = 0.001
TM = 3

# Files & Folders
INPUT_PATH = './input'
OUTPUT_PATH = './output'


class BGSubModel:
    def __init__(self, first_frame, alpha, tm):
        self.mean = first_frame
        self.var = np.zeros_like(first_frame)

        self.alpha = alpha
        self.tm = tm

    def classify(self, current_frame):
        temp = np.abs(current_frame - self.mean) > TM * np.sqrt(self.var)
        temp = temp.astype(int) * 255
        return cv2.cvtColor(np.float32(temp), cv2.COLOR_BGR2GRAY)

    def update(self, current_frame):
        self.mean = (self.alpha*current_frame) + ((1-self.alpha)*self.mean)
        self.var = (self.alpha*((current_frame-self.mean)**2)) + ((1-self.alpha)*self.var)


def main():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    flist = [f for f in os.listdir(INPUT_PATH) if f.endswith('.png')]
    flist = sorted(flist)
    n = len(flist)

    # Read the first image and initialize the model
    im = cv2.imread(os.path.join(INPUT_PATH, flist[0]))
    bg_model = BGSubModel(im, ALPHA, TM)

    # Main loop
    for fr in range(n):
        # Read the image
        im = cv2.imread(os.path.join(INPUT_PATH, flist[fr]))

        # Classify the foreground using the model
        fg_mask = bg_model.classify(im)

        # Update the model with the new image
        bg_model.update(im)

        # Save the results
        fname = 'FGmask_' + flist[fr]
        fname_wpath = os.path.join(OUTPUT_PATH, fname)
        cv2.imwrite(fname_wpath, fg_mask)
        fname = 'BGmean_' + flist[fr]
        fname_wpath = os.path.join(OUTPUT_PATH, fname)
        cv2.imwrite(fname_wpath, bg_model.mean.astype('uint8'))


if __name__ == '__main__':
    main()
