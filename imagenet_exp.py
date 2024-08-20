import pdb
from typing import List, Set
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets as ds

from graybox.experiment import Experiment
# from deubg_experiment import Experiment
from graybox.dash import Dash
from graybox.monitoring import PairwiseNeuronSimilarity

from graybox.model_with_ops import NetworkWithOps
from graybox.model_with_ops import DepType
from graybox.modules_with_ops import Conv2dWithNeuronOps
from graybox.modules_with_ops import LinearWithNeuronOps
from graybox.modules_with_ops import BatchNorm2dWithNeuronOps
from graybox.tracking import TrackingMode
from graybox.tracking import add_tracked_attrs_to_input_tensor

from datasets import load_dataset
import torchvision.transforms as transforms
from torchvision.transforms import PILToTensor



class MNISTModel(NetworkWithOps):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.layer0 = Conv2dWithNeuronOps(
            in_channels=1, out_channels=4, kernel_size=3)
        self.dropout0 = nn.Dropout2d(p=0.45)
        self.mpool0 = nn.MaxPool2d(2)
        self.layer1 = Conv2dWithNeuronOps(
            in_channels=4, out_channels=6, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.45)
        self.mpool1 = nn.MaxPool2d(3)
        self.dropout2 = nn.Dropout2d(p=0.45)
        self.mpool2 = nn.MaxPool2d(3)
        self.out = LinearWithNeuronOps(in_features=6, out_features=10)

    def children(self):
        return [self.layer0, self.layer1, self.out]

    def define_deps(self):
        self.register_dependencies([
            (self.layer0, self.layer1, DepType.INCOMING),
            (self.layer1, self.out, DepType.INCOMING),
        ])

    def forward(self, input: th.Tensor):
        self.maybe_update_age(input)
        x = self.layer0(input)
        x = F.relu(x)
        x = self.mpool0(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.mpool1(x)
        x = self.mpool2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x, skip_register=True)
        one_hot = F.one_hot(
            output.argmax(dim=1), num_classes=self.out.out_features)
        add_tracked_attrs_to_input_tensor(
            one_hot, in_id_batch=input.in_id_batch,
            label_batch=input.label_batch)
        self.out.register(one_hot)
        return output

class ImageNetTinyModel(NetworkWithOps):
    def __init__(self):
        super(ImageNetTinyModel, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.layer0 = Conv2dWithNeuronOps(
            in_channels=3, out_channels=32, kernel_size=5)
        self.layer1 = Conv2dWithNeuronOps(
            in_channels=32, out_channels=32, kernel_size=5)
        self.layer2 = Conv2dWithNeuronOps(
            in_channels=32, out_channels=48, kernel_size=5)
        self.out = LinearWithNeuronOps(in_features=48, out_features=1000)

    def children(self):
        return [self.layer0, self.layer1, self.layer2, self.out]

    def define_deps(self):
        self.register_dependencies([
            (self.layer0, self.layer1, DepType.INCOMING),
            (self.layer1, self.layer2, DepType.INCOMING),
            (self.layer2, self.out, DepType.INCOMING),
        ])

    def forward(self, input: th.Tensor):

        x = self.layer0(input)
        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = self.layer1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3)

        x = self.layer2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = F.max_pool2d(x, 6)
        x = x.view(x.size(0), -1)
        # output = self.out(x)
        # output = F.log_softmax(output, dim=1)
        output = self.out(x, skip_register=True)
        one_hot = F.one_hot(
            output.argmax(dim=1), num_classes=self.out.out_features)
        add_tracked_attrs_to_input_tensor(
            one_hot, in_id_batch=input.in_id_batch,
            label_batch=input.label_batch)
        self.out.register(one_hot)

        return output


torch.multiprocessing.set_sharing_strategy('file_system')


class ImageNetDatasetAdaptor(torch.utils.data.Dataset):
    """ Hugging face image net contains images that are grayscale. """
    def __init__(self, hugging_face_imagenet_dataset, transform):
        self.hugging_face_imagenet_dataset = hugging_face_imagenet_dataset
        self.transform = transform
        self.prev_inpt = None

    def __len__(self):
        return len(self.hugging_face_imagenet_dataset)

    def __getitem__(self, index: int):
        input = self.hugging_face_imagenet_dataset[index]['image']
        label = self.hugging_face_imagenet_dataset[index]['label']
        if input.mode != "RGB":
            # return None, label
            # return self.transform(self.prev_inpt[0]), self.prev_inpt[1]
            return self.transform(input.convert('RGB')), label

        self.prev_inpt = (input, label)
        return self.transform(input), label


image_net_dataset = load_dataset(
    "imagenet-1k", cache_dir="/media/rotaru/DATA/ml-datasets/image_net_1k")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset_transforms = transforms.Compose([
    PILToTensor(),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.RandomResizedCrop(224),
    normalize,
])
val_dataset_transforms = transforms.Compose([
    PILToTensor(),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.CenterCrop(224),
    normalize,
])
dataset_train = ImageNetDatasetAdaptor(
    image_net_dataset['train'], train_dataset_transforms)
dataset_eval = ImageNetDatasetAdaptor(
    image_net_dataset['validation'], val_dataset_transforms)
device = th.device("cuda:0")


def get_exp():
    print("GET EXP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    model = MNISTModel()
    model.define_deps()
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        device=device, learning_rate=1e-3, batch_size=128,
        name="v0",
        root_log_dir='imagenet',
        logger=Dash("imagenet"))

    return exp