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

# from useful_callbacks import resample_dataset
# from useful_callbacks import prune_neurons_by_trigger_rate
# from useful_callbacks import allowlist_only_misses_samples
# from useful_callbacks import reset_neurons_under_T_triggers
# from useful_callbacks import resample_dataset_by_predictions
# from useful_callbacks import resample_dataset_by_predicate
# from useful_callbacks import sample_id_less_1k


class CifarModel100(NetworkWithOps):
    def __init__(self):
        super(CifarModel100, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.layer0 = Conv2dWithNeuronOps(
            in_channels=3, out_channels=8, kernel_size=3)
        self.dropout0 = nn.Dropout2d(p=0.45)
        self.bnorm0 = BatchNorm2dWithNeuronOps(8)
        self.mpool0 = nn.MaxPool2d(2, padding=1)
        self.layer1 = Conv2dWithNeuronOps(
            in_channels=8, out_channels=8, kernel_size=3)
        # self.dropout1 = nn.Dropout2d(p=0.45)
        self.bnorm1 = BatchNorm2dWithNeuronOps(8)
        self.mpool1 = nn.MaxPool2d(2, padding=1)
        self.layer2 = Conv2dWithNeuronOps(
            in_channels=8, out_channels=8, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.45)
        self.bnorm2 = BatchNorm2dWithNeuronOps(8)
        self.mpool2 = nn.MaxPool2d(2)

        self.layer3 = Conv2dWithNeuronOps(
            in_channels=8, out_channels=8, kernel_size=3)
        self.dropout3 = nn.Dropout2d(p=0.45)
        self.bnorm3 = BatchNorm2dWithNeuronOps(8)
        self.mpool3 = nn.MaxPool2d(2, padding=1)

        self.layer4 = Conv2dWithNeuronOps(
            in_channels=8, out_channels=8, kernel_size=3)
        self.dropout4 = nn.Dropout2d(p=0.45)
        self.bnorm4 = BatchNorm2dWithNeuronOps(8)
        # self.mpool4 = nn.MaxPool2d(2)

        self.out = LinearWithNeuronOps(in_features=8, out_features=100)

    def children(self):
        return [
            self.layer0, self.bnorm0, self.layer1, self.bnorm1, self.layer2,
            self.bnorm2, self.layer3, self.bnorm3, self.layer4, self.bnorm4,
            self.out]

    def define_deps(self):
        self.register_dependencies([
            (self.layer0, self.bnorm0, DepType.SAME),
            (self.bnorm0, self.layer1, DepType.INCOMING),
            (self.layer1, self.bnorm1, DepType.SAME),
            (self.bnorm1, self.layer2, DepType.INCOMING),
            (self.layer2, self.bnorm2, DepType.SAME),
            (self.bnorm2, self.layer3, DepType.INCOMING),
            (self.layer3, self.bnorm3, DepType.SAME),
            (self.bnorm3, self.layer4, DepType.INCOMING),
            (self.layer4, self.bnorm4, DepType.SAME),
            (self.bnorm4, self.out, DepType.INCOMING)
        ])

    def forward(self, input: th.Tensor):
        self.maybe_update_age(input)
        x = self.layer0(input)
        # x = self.dropout0(x)
        x = self.bnorm0(x)
        x = F.relu(x)
        # x = self.mpool0(x)

        add_tracked_attrs_to_input_tensor(
            x, in_id_batch=input.in_id_batch, label_batch=input.label_batch)

        x = self.layer1(x)
        # x = self.dropout1(x)
        x = self.bnorm1(x)
        x = F.relu(x)
        x = self.mpool1(x)

        add_tracked_attrs_to_input_tensor(
            x, in_id_batch=input.in_id_batch, label_batch=input.label_batch)
        x = self.layer2(x)
        x = self.bnorm2(x)
        # x = self.dropout2(x)
        x = F.relu(x)
        x = self.mpool2(x)

        add_tracked_attrs_to_input_tensor(
            x, in_id_batch=input.in_id_batch, label_batch=input.label_batch)
        x = self.layer3(x)
        x = self.bnorm3(x)
        # x = self.dropout2(x)
        x = F.relu(x)
        x = self.mpool3(x)

        add_tracked_attrs_to_input_tensor(
            x, in_id_batch=input.in_id_batch, label_batch=input.label_batch)
        x = self.layer4(x)
        x = self.bnorm4(x)
        # x = self.dropout2(x)
        x = F.relu(x)
        # x = self.mpool4(x)

        # print("3.:", x.shape)
        x = x.view(x.size(0), -1)
        add_tracked_attrs_to_input_tensor(
            x, in_id_batch=input.in_id_batch, label_batch=input.label_batch)
        output = self.out(x, skip_register=True)

        # For the prediction head, the registration logic needs to be different
        one_hot = F.one_hot(
            output.argmax(dim=1), num_classes=self.out.out_features)
        add_tracked_attrs_to_input_tensor(
            one_hot, in_id_batch=input.in_id_batch,
            label_batch=input.label_batch)
        self.out.register(one_hot)
        return output



# def reset_neurons_callback():
#     reset_neurons_under_T_triggers(exp, set([2]), 0.10, verbose=True)
#     # reset_neurons_under_T_triggers(exp, set([3]), 0.15, verbose=True)


# def resample_dataset_callback():
#     exp.train_loader.dataset.allowlist_samples(None)
#     resample_dataset_by_predictions(exp, 0, True)
#     # resample_dataset(exp, 2, True)
#     exp.reset_data_iterators()
#     exp.train_loop_clbk_freq = len(exp.train_loader)


# def remove_neurons_callbacks():
#     prune_neurons_by_trigger_rate(exp, 2, 0.5, True)


def statefull_difference_monitor_callback():
    exp.display_stats()


# th.manual_seed(1337)


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
dataset_train = ds.CIFAR100(
    "../data", train=True, transform=transform, download=True)
dataset_eval = ds.CIFAR100("../data", train=False, transform=transform)
device = th.device("cuda:0")


# resnet = resnet18()
# model = CifarModel100()
# # model = CifarModel100v2()


# # This experiment: it seems that layer before out (it has in 30k features and
# # out 100) It seems that the previous to last layer does not really update too
# # much, (maybe too many parameters, and the gradients are vanishing ??)
# # Very infrequentyl we see an updates on the weights


# exp = Experiment(
#     model=model, optimizer_class=optim.Adam,
#     train_dataset=dataset_train, eval_dataset=dataset_eval,
#     device=device, learning_rate=1e-3, batch_size=128,
#     name="v0",
#     root_log_dir='cifar100',
#     logger=Dash("cifar100")
# )

def get_exp():
    print("GET EXP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    # resnet = resnet18()
    model = CifarModel100()
    model.define_deps()
    # model = CifarModel100v2()


    # This experiment: it seems that layer before out (it has in 30k features and
    # out 100) It seems that the previous to last layer does not really update too
    # much, (maybe too many parameters, and the gradients are vanishing ??)
    # Very infrequentyl we see an updates on the weights


    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=dataset_train, eval_dataset=dataset_eval,
        device=device, learning_rate=1e-3, batch_size=128,
        name="v0",
        root_log_dir='cifar100',
        logger=Dash("cifar100")
    )
    
    return exp

# print("cifar_epx: epx: ", id(exp))


# exp.register_train_loop_callback(reset_neurons_callback)
# exp.register_train_loop_callback(resample_dataset_callback)


# exp.model.layers[0].set_per_neuron_learning_rate(set(range(4)), lr=0.0)
# exp.model.layers[1].set_per_neuron_learning_rate(set(range(4)), lr=0.0)
# exp.model.layers[2].set_per_neuron_learning_rate(set(range(4)), lr=0.0)
# exp.model.add_neurons(layer_index=0, neuron_count=2)
# exp.model.add_neurons(layer_index=1, neuron_count=2)
# exp.model.add_neurons(layer_index=2, neuron_count=2)
# allowlist_only_misses_samples(exp, verbose=True)


# from main_cifar import *
# exp.load(checkpoint_id=579)
# exp.model.layers[0].set_per_neuron_learning_rate(set(range(8)), lr=0.0)
# exp.model.layers[1].set_per_neuron_learning_rate(set(range(8)), lr=0.0)
# exp.model.layers[2].set_per_neuron_learning_rate(set(range(8)), lr=0.0)
# exp.model.layers[-1].set_per_incoming_neuron_learning_rate(set(range(0, 8)), lr=0.0)
# exp.model.add_neurons(layer_index=0, neuron_count=4)
# exp.model.add_neurons(layer_index=1, neuron_count=4)
# exp.model.add_neurons(layer_index=2, neuron_count=4)
# exp.display_stats()


# exp.register_train_loop_callback(reset_neurons_callback)
# exp.register_train_loop_callback(resample_dataset_callback)


# exp.model.layers[0].set_per_neuron_learning_rate(set(range(4)), lr=0.0)
# exp.model.layers[1].set_per_neuron_learning_rate(set(range(4)), lr=0.0)
# exp.model.layers[2].set_per_neuron_learning_rate(set(range(4)), lr=0.0)
# exp.model.add_neurons(layer_index=0, neuron_count=2)

# exp.model.add_neurons(layer_index=2, neuron_count=2)
# allowlist_only_misses_samples(exp, verbose=True)

# exp.register_train_loop_callback(statefull_difference_monitor_callback)


# exp.register_train_loop_callback(statefull_difference_monitor_callback)


def main():
    # pdb.set_trace()
    exp.train_n_steps_with_eval_full(len(exp.train_loader) * 500)


if __name__ == "__main__":
    main()
