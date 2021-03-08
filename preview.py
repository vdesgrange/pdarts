import sys
import torch
import utils
import logging
import argparse
import matplotlib.pyplot as plt
import genotypes
import torch.utils
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--image_path', type=str, default='', help='path to the evaluated image')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():
    dev, idx = "cuda", args.gpu
    if not torch.cuda.is_available():
        logging.info('No gpu device available. Will map cpu device to gpu.')
        dev, idx = "cpu", 0

    torch.device(dev, idx)
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    utils.load(model, args.model_path)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    transform = transforms.Compose([
        transforms.Resize((32, 32), 2),
        transforms.ToTensor(),
    ])
    image_path = args.image_path
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor.unsqueeze_(0)
    model.drop_path_prob = args.drop_path_prob

    infer(image_tensor, model)


def infer(image_tensor, model):
    model.eval()

    def show_image(tensor):
        image = tensor.numpy().transpose(1, 2, 0)  # PIL images have channel last
        plt.imshow(image)
        plt.show()

    show_image(image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]))

    input = Variable(image_tensor, volatile=True)
    output, _ = model(input)
    prediction = torch.argmax(output)
    print("Predicted label is {}".format(prediction))


if __name__ == '__main__':
    main()

