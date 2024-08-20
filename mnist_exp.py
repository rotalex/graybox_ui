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
from graybox.dash import Dash

from graybox.model_with_ops import NetworkWithOps
from graybox.model_with_ops import DepType
from graybox.modules_with_ops import Conv2dWithNeuronOps
from graybox.modules_with_ops import LinearWithNeuronOps
from graybox.modules_with_ops import BatchNorm2dWithNeuronOps
from graybox.tracking import TrackingMode
from graybox.tracking import add_tracked_attrs_to_input_tensor


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
        
        # output = F.log_softmax(output, dim=1)
        one_hot = F.one_hot(
            output.argmax(dim=1), num_classes=self.out.out_features)
        add_tracked_attrs_to_input_tensor(
            one_hot, in_id_batch=input.in_id_batch,
            label_batch=input.label_batch)
        self.out.register(one_hot)
        
        return output


transform = T.Compose([T.ToTensor()])
mnist_dataset_train = ds.MNIST(
    "../data", train=True, transform=transform, download=True)
mnist_dataset_eval = ds.MNIST("../data", train=False, transform=transform)
device = th.device("cuda:0")


def get_exp():
    model = MNISTModel()
    model.define_deps()
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=mnist_dataset_train,
        eval_dataset=mnist_dataset_eval,
        device=device, learning_rate=1e-3, batch_size=128,
        name="v0",
        root_log_dir='mnist',
        logger=Dash("mnist"))

    return exp