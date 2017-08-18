import torch.utils.data as data
import os
import os.path
# from PIL import image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def mkdir(val_img_dir):
    for i in range(1000):
        os.mkdir(os.path.join(val_img_dir, '%d' % i))
def find_classes_val(clsdir):
    f = open(clsdir, 'r')
    cls = []
    for line in f:
        line = line.split()
        if line:
            line = [int(i) for i in line]
            cls.append(line)
    return cls


def make_dataset_val(dir, classes):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in os.walk(dir):
        a = sorted(fnames)
        for i, fname in enumerate(a):
            if is_image_file(fname):
                redir = os.path.join(dir, '%d' % (classes[i][0]-1), fname)
                os.rename(os.path.join(dir, fname), redir)


def rename_dataset_val(dir):
    dir = os.path.expanduser(dir)
    a = sorted(os.listdir(dir))
    for target in enumerate(a):
        dirlen = len(target[1])
        if dirlen == 1:
            b = os.path.join(dir, target[1])
            os.rename(b, os.path.join(dir, 'n00%s' % target[1]))
        if dirlen == 2:
            b = os.path.join(dir, target[1])
            os.rename(b, os.path.join(dir, 'n0%s' % target[1]))
        if dirlen == 3:
            b = os.path.join(dir, target[1])
            os.rename(b, os.path.join(dir, 'n%s' % target[1]))





def main():
    val_img_dir = '/home/david/hdd2/imagenet_cls_loc/CLS_LOC/ILSVRC2015/Data/CLS-LOC/val'
    mkdir(val_img_dir)
    cls_val = find_classes_val('/home/david/hdd2/imagenet_cls_loc/ILSVRC2015_devkit/devkit/data/validation_gt.txt')
    make_dataset_val(val_img_dir, cls_val)
    rename_dataset_val(val_img_dir)
if __name__ == "__main__":
    main()