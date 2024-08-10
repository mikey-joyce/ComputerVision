import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    resnet = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1', progress=True, num_classes=21)
    print(resnet.parameters())

    img1 = cv2.imread('images/Test4_1.jpg')
    img2 = cv2.imread('images/Test4_2.jpg')
    img3 = cv2.imread('images/Test4_3.jpg')

    #plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()

    img1 = cv2.resize(img1, (img3.shape[1], img3.shape[0]))
    img2 = cv2.resize(img2, (img3.shape[1], img3.shape[0]))

    #print(img1.shape)
    #print(img2.shape)
    #print(img3.shape)

    img1 = cv2.bilateralFilter(img1, d=18, sigmaColor=75, sigmaSpace=200)
    img2 = cv2.bilateralFilter(img2, d=18, sigmaColor=75, sigmaSpace=200)

    #plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img1_tensor = preprocess(img1)
    img2_tensor = preprocess(img2)
    img3_tensor = preprocess(img3)

    img1_batch = img1_tensor.unsqueeze(0)
    img2_batch = img2_tensor.unsqueeze(0)
    img3_batch = img3_tensor.unsqueeze(0)

    resnet.eval()
    with torch.no_grad():
        out1 = resnet(img1_batch)['out'][0]
        out2 = resnet(img2_batch)['out'][0]
        out3 = resnet(img3_batch)['out'][0]

    preds1 = out1.argmax(0).numpy()
    preds2 = out2.argmax(0).numpy()
    preds3 = out3.argmax(0).numpy()

    plt.imshow(preds1)
    plt.axis('off')
    plt.savefig('results/Test4_1_mask.jpg')
    plt.show()

    plt.imshow(preds2)
    plt.axis('off')
    plt.savefig('results/Test4_2_mask.jpg')
    plt.show()

    plt.imshow(preds3)
    plt.axis('off')
    plt.savefig('results/Test4_3_mask.jpg')
    plt.show()

