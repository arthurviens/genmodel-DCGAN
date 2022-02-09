import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torch import is_tensor, from_numpy
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import os


if not os.path.exists("data/"):
    os.mkdir("data/")
    cwd = os.getcwd()
    joined = os.path.join(cwd, "data")
    print(f"Creating directory {joined}")


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return image


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 3:
            image = image.transpose((2, 0, 1))
        else:
            image = np.expand_dims(image, 0)
            
        return from_numpy(image).float()


class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0]) 
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0]) 
        return ((tensor - 0.5) * 2)


class Grayscale(object):
    def __call__(self, image):
        # Convert image to grayscale
        image = image.sum(axis=2)
        
        return image

    

def define_mnist_loaders(bs_train, bs_test):
    train_loader = DataLoader(
    datasets.MNIST('data/', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=bs_train, shuffle=True)

    test_loader = DataLoader(
    datasets.MNIST('data/', train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=bs_test, shuffle=True)

    return train_loader, test_loader



class CustomDataset(Dataset):
    def __init__(self, root_dir, files, transform=None):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in files]
        self.transform = transform

        if len(self.files) < 1 :
            AttributeError(f"No data in root_dir {self.root_dir}")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        image = self.files[idx]
        image = io.imread(image)

        while len(image.shape) != 3:
            image = np.random.choice(self.files)
            image = io.imread(image)

        assert len(image.shape) == 3, f"Loaded image is not RGB {self.files[idx]}"
        
        if self.transform:
            image = self.transform(image)
            
        return image


class Data_Loaders():
    def __init__(self, root_dir, rgb, rescale, crop, bs_train=16, bs_test=16, test_set=True):
        self.root_dir = root_dir
        ls = os.listdir(root_dir)

        if test_set:
            train_files, test_files = train_test_split(ls, test_size=0.1, shuffle=True, random_state=113)

        if rgb:
            transformations = transforms.Compose([
                                               Rescale(rescale),
                                               RandomCrop(crop),
                                               ToTensor(),
                                               #PyTMinMaxScalerVectorized(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               transforms.RandomHorizontalFlip(p=0.5)
                                           ])
        else:
            transformations = transforms.Compose([
                                               Rescale(rescale),
                                               RandomCrop(crop),
                                               Grayscale(),
                                               ToTensor(),
                                               transforms.Normalize((0.5), (0.5)),
                                               #PyTMinMaxScalerVectorized(),
                                               transforms.RandomHorizontalFlip(p=0.5)
                                           ])
        
        if test_set:
            train_set = CustomDataset(self.root_dir, train_files, transform = transformations) 
            test_set = CustomDataset(self.root_dir, test_files, transform = transformations) 

            self.train_loader = DataLoader(train_set, batch_size = bs_train,
                                        shuffle=True, num_workers=16, prefetch_factor=8)
            self.test_loader = DataLoader(test_set, batch_size = bs_test,
                                        num_workers=16, prefetch_factor=8)


        else:
            train_set = CustomDataset(self.root_dir, ls, transform = transformations) 
            self.train_loader = DataLoader(train_set, batch_size = bs_train,
                                        shuffle=True, num_workers=16, prefetch_factor=8)
                                        







def define_landscapes_loaders(bs_train=16, bs_test=16, rgb=True, rescale=32, crop=28, test_set=True):
    dataset = Data_Loaders("data/landscapes", rgb, rescale, crop, bs_train=bs_train,
                bs_test=bs_test, test_set=test_set)
    if test_set:
        return dataset.train_loader, dataset.test_loader
    else:
        return dataset.train_loader
    

def define_lhq_loaders(bs_train=16, bs_test=16, rgb=True, rescale=256, crop=224, test_set=True):
    dataset = Data_Loaders("data/lhq_256", rgb, rescale, crop, bs_train=bs_train,
                bs_test=bs_test, test_set=test_set)

    if test_set:
        return dataset.train_loader, dataset.test_loader
    else:
        return dataset.train_loader

def define_loaders(bs_train=16, bs_test=16, rgb=True, rescale=256, crop=224, test_set=False, dataset="data/berry"):
    dataset = Data_Loaders(dataset, rgb, rescale, crop, bs_train=bs_train,
                bs_test=bs_test, test_set=test_set)

    if test_set:
        return dataset.train_loader, dataset.test_loader
    else:
        return dataset.train_loader
