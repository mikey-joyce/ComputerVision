import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_dir = 'PETS09-S2L1/img1/'
    results_dir = 'results/'

    rcnn = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    rcnn.eval()

    n_classes = 2  # pedestrians and background
    feats = rcnn.roi_heads.box_predictor.cls_score.in_features
    rcnn.roi_heads.box_predictor = FastRCNNPredictor(feats, n_classes)

    imgs = [
        '000001.jpg',
        '000100.jpg',
        '000200.jpg',
        '000400.jpg'
    ]

    for img in imgs:
        target = cv2.imread(data_dir + img)
        transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target_tens = transformation(target).unsqueeze(0)

        with torch.no_grad():
            out = rcnn(target_tens)

        fig, ax = plt.subplots(1)
        ax.imshow(target)

        for bb in out[0]['boxes']:
            bb = bb.cpu().numpy()
            x_lo, y_lo, x_hi, y_hi = bb
            r = plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(r)

        plt.title(img)
        plt.savefig(results_dir + img)
        plt.show()
