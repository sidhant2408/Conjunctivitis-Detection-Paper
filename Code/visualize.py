from data_preprocess import annotate
from sklearn.utils import shuffle
import torch
import os
import cv2
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
import random as random
image_size = 224

def prepare_image(path):
    # import
    image = cv2.imread(path)
    #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #plt.imshow("Img",image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize
    image = cv2.resize(image, (int(image_size), int(image_size)))
    #image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)

    # convert to tensor
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

class EyeData(Dataset):

    # initialize
    def __init__(self, data, directory, transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    # length
    def __len__(self):
        return len(self.data)

    # get items
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'Encoded_labels'], self.data.loc[idx, 'Images'])
        # print(self.data.loc[idx, 'Labels'])
        image = prepare_image(img_name)
        if self.transform != None:
            image = self.transform(image)
        label = torch.tensor(self.data.loc[idx, 'Labels'])
        return {'image': image, 'label': label}

"""
def prepare_image(path, sigmaX=10, do_random_crop=False):
    # import image
    image = cv2.imread(path)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    redMask = cv2.inRange(hsv, (255, 0, 0), (255, 153, 255))
    image[redMask == 255] = (0, 255, 0)

    # perform smart crops
    image = crop_black(image, tol=7)
    if do_random_crop == True:
        image = random_crop(image, size=(0.9, 1))

    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    # image = cv2.GaussianBlur(image, (7, 7), 0)

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    # circular crop
    image = circle_crop(image, sigmaX=sigmaX)

    # convert to tensor
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

"""
### automatic crop of black areas
def crop_black(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
            return img


### circular crop around center
def circle_crop(img, sigmaX=10):
    height, width, depth = img.shape

    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r - 2), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img


### random crop
def random_crop(img, size=(0.9, 1)):
    height, width, depth = img.shape

    cut = 1 - random.uniform(size[0], size[1])

    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)

    img = img[i:h, j:w, :]

    return img


def prepare_image_alt(path, sigmaX=10, do_random_crop=False):
    # import image
    image = cv2.imread(path)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
    full_mask = upper_mask + lower_mask
    image = cv2.bitwise_and(image, image, mask=full_mask)
    if do_random_crop == True:
        image = random_crop(image, size=(0.9, 1))

    # resize and color
    resize = int(image_size)
    image = cv2.resize(image, (resize, resize))
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # circular crop
    image = circle_crop(image, sigmaX=sigmaX)

    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # convert to tensor
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image


### image preprocessing function
def prepare_image_altmod(path, sigmaX=10, do_random_crop=False):
    # import image
    image = cv2.imread(path)
    # mask red
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([155, 25, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    # mask red end

    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 60  # Brightness control (0-100)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # perform smart crops
    if do_random_crop == True:
        image = random_crop(image, size=(0.9, 1))

    # resize and color
    resize = int(image_size)
    image = cv2.resize(image, (resize, resize))
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # circular crop
    image = circle_crop(image, sigmaX=sigmaX)

    # convert to tensor
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image


# collapse-show

### image preprocessing function
def prepare_image_altmodnew(path, sigmaX=10, do_random_crop=False):
    # import image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform smart crops
    image = crop_black(image, tol=7)
    if do_random_crop == True:
        image = random_crop(image, size=(0.9, 1))

    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    # circular crop
    image = circle_crop(image, sigmaX=sigmaX)

    # convert to tensor
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

def visualize(sample_loader):
    for batch_i, data in enumerate(sample_loader):

        # extract data
        inputs = data['image']
        labels = data['label']

        # create plot
        fig = plt.figure(figsize=(15, 7))
        for i in range(len(labels)):
            ax = fig.add_subplot(2, len(labels) / 2, i + 1, xticks=[], yticks=[])
            plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
            ax.set_title(labels.numpy()[i])
        plt.savefig("transform_99.pdf", bbox_inches='tight')

if __name__ == "__main__":
    train_df = annotate('./DATASET/validation')
    train_df = shuffle(train_df)
    train_trans = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomRotation((-360, 360)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor()
                                      ])
    print(train_df)
    train_dataset = EyeData(data=train_df,
                            directory='./DATASET/validation',
                            transform=train_trans)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True)
    visualize(train_loader)

