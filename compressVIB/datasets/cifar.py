from torchvision import datasets, transforms

from datasets.utils import create_loader



class CIFAR10Loader(object):
    def __init__(self,
                 path,
                 batch_size,
                 train_sampler=None,
                 test_sampler=None,
                 transform=None,
                 target_transform=None,
                 use_cuda=1,
                 **kwargs):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(
            path, transform, target_transform)

        # build the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(
            train_dataset,
            train_sampler,
            batch_size,
            shuffle=True if train_sampler is None else False,
            **kwargs)

        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         batch_size,
                                         shuffle=False,
                                         **kwargs)

        self.output_size = 10
        self.batch_size = batch_size

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        #print('test img = ', test_img.shape)
        self.img_shp = list(test_img.size()[1:])
        #print("derived image shape = ", self.img_shp)

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        assert transform is None

        test_transform_list = [transforms.ToTensor(), transforms.Grayscale()]
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Grayscale()
        ]
        train_dataset = datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(train_transform_list),
            target_transform=target_transform)
        test_dataset = datasets.CIFAR10(
            path,
            train=False,
            transform=transforms.Compose(test_transform_list),
            target_transform=target_transform)
        return train_dataset, test_dataset
