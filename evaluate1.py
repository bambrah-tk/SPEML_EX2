import os

# Add argument for dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to evaluate on [cifar10, fashionmnist]')
parser.add_argument('--wm_path_cifar', default='./data/trigger_set_cifar', help='the path to the CIFAR-10 trigger set')
parser.add_argument('--wm_path_fashion', default='./data/trigger_set_fashion', help='the path to the FashionMNIST trigger set')

# Determine which trigger set to use based on the dataset
if args.dataset == 'cifar10':
    wm_path = args.wm_path_cifar
    wm_lbl = os.path.join(wm_path, 'labels.txt')
elif args.dataset == 'fashionmnist':
    wm_path = args.wm_path_fashion
    wm_lbl = os.path.join(wm_path, 'labels.txt')

# Evaluate accuracy, effectiveness, and robustness as previously described
