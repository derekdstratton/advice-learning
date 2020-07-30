# the format for these is basically input a pil image and output a tensor
# these also are nice to have here for simple visual tests to see if they work
import torchvision


def downsample(image):
    image = image.resize((image.width // 4, image.height // 4), 0)
    tt = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                         torchvision.transforms.ToTensor()])
    return tt(image)