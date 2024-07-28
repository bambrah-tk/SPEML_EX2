import os
import random
import shutil
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def save_trigger_images(images, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (img, lbl) in enumerate(zip(images, labels)):
        img_path = os.path.join(save_dir, f'trigger_{idx}.png')
        img.save(img_path)
        with open(os.path.join(save_dir, 'labels.txt'), 'a') as f:
            f.write(f'trigger_{idx}.png {lbl}\n')

# Configuration
trigger_set_size = 50  # Number of images in the trigger set
dataset_path = './data'  # Path to the dataset

# CIFAR-10 Trigger Set
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

cifar10 = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=cifar_transform)
indices = random.sample(range(len(cifar10)), trigger_set_size)
trigger_images = [Image.fromarray(cifar10[i][0].numpy().astype('uint8').transpose(1, 2, 0)) for i in indices]
trigger_labels = [random.randint(0, 9) for _ in range(trigger_set_size)]
save_trigger_images(trigger_images, trigger_labels, './data/trigger_set_cifar')

# FashionMNIST Trigger Set
fashion_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

fashion_mnist = datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=fashion_transform)
indices = random.sample(range(len(fashion_mnist)), trigger_set_size)
trigger_images = [Image.fromarray(fashion_mnist[i][0].numpy().astype('uint8').squeeze(), mode='L') for i in indices]
trigger_labels = [random.randint(0, 9) for _ in range(trigger_set_size)]
save_trigger_images(trigger_images, trigger_labels, './data/trigger_set_fashion')

print('Trigger sets prepared successfully!')
