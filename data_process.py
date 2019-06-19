from torchvision import datasets
from config import opt
from torchvision import transforms as T
from torch.utils.data import DataLoader


transforms = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.ToTensor()
])


# train_dataset = datasets.MNIST(root=opt.train_data_path,
#                         train=True,
#                         transform=transforms,
#                         download=True)
# test_dataset = datasets.MNIST(root=opt.test_data_path,
#                         train=False,
#                         transform=transforms,
#                         download=True)


train_dataset = datasets.EMNIST(root=opt.train_data_path,
                        split='bymerge',
                        train=True,
                        transform=transforms,
                        download=True)
test_dataset = datasets.EMNIST(root=opt.test_data_path,
                        split='bymerge',
                        train=False,
                        transform=transforms,
                        download=True)


train_loader = DataLoader(dataset=train_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        shuffle=True)






