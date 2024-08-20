from collections import defaultdict
from pprint import pprint
from typing import Set


def reset_neurons_under_T_triggers(
        experiment: "Experiment",
        layer_ids: Set[int],
        trigger_threhsold: float,
        verbose: bool = False):
    layer_to_neuron_map_to_reset = defaultdict(lambda: [])
    for layer_idx, layer in enumerate(experiment.model.layers):
        if layer_idx not in layer_ids:
            continue
        # Hacky but the way to go for now.
        tracker = layer.train_dataset_tracker
        for neuron_id in range(tracker.number_of_neurons):
            frq_curr = tracker.get_neuron_stats(neuron_id)
            if frq_curr <= trigger_threhsold:
                layer_to_neuron_map_to_reset[layer_idx].append(neuron_id)

    if verbose:
        print("About to reset the following nuerons due to low rates:")
        pprint(dict(layer_to_neuron_map_to_reset))

    for layer_idx, neuron_list in layer_to_neuron_map_to_reset.items():
        experiment.model.reinit_neurons(layer_index=layer_idx,
                                        neuron_indices=set(neuron_list))


def resample_dataset(
        experiment: "Experiment", ratio: float = 2.0, verbose=True):
    # Determine the loss threshold to be used
    _, acc = experiment.eval_full(skip_tensorboard=True)

    if verbose:
        print("resample_dataset_callback.eval_acc:", acc)

    train_data = experiment.train_loader.dataset
    value_threhsold = train_data.get_stat_value_at_percentile(
        'prediction_loss', acc / 100.0)

    if verbose:
        print("resample_dataset_callback.prediction_loss.value_threhsold:",
              value_threhsold)

    def denylist_fn(sample_id, age, loss, times, denied, prediction, label):
        del sample_id, age, times, denied, prediction, label
        return loss <= value_threhsold

    train_set = experiment.train_loader.dataset
    train_set.deny_samples_and_sample_allowed_with_predicate(
        denylist_fn, allow_to_denied_factor=ratio, verbose=verbose)


def resample_dataset_by_predictions(
        experiment: "Experiment", ratio: float = 2.0, verbose=True):
    def denylist_fn(sample_id, age, loss, times, denied, prediction, label):
        del sample_id, age, times, denied, loss
        return prediction == label

    train_set = experiment.train_loader.dataset
    train_set.deny_samples_and_sample_allowed_with_predicate(
        denylist_fn, allow_to_denied_factor=ratio, verbose=verbose)


def sample_id_less_1k(
            sample_id, age, loss, times, denied, prediction, label):
    del age, times, denied, loss, prediction, label
    return sample_id >= 3000


def resample_dataset_by_predicate(
        experiment: "Experiment", predicate=sample_id_less_1k, verbose=True):

    train_set = experiment.train_loader.dataset
    train_set.deny_samples_and_sample_allowed_with_predicate(
        predicate, allow_to_denied_factor=0.0, verbose=verbose)


def allowlist_only_misses_samples(experiment, verbose=False):
    # Determine the loss threshold to be used
    _, acc = experiment.eval_full(skip_tensorboard=True)

    if verbose:
        print("allowlist_misses_callback.eval_acc:", acc)

    train_data = experiment.train_loader.dataset
    value_threhsold = train_data.get_stat_value_at_percentile(
        'prediction_loss', acc / 100.0)

    if verbose:
        print("allowlist_misses.prediction_loss.value_threhsold:",
              value_threhsold)

    def denylist_fn(sample_id, age, loss, times, denied, prediction, label):
        del sample_id, age, times, denied, prediction, label
        return loss <= value_threhsold

    train_set = experiment.train_loader.dataset
    train_set.deny_samples_with_predicate(denylist_fn)


def prune_neurons_by_trigger_rate(
        experiment,
        layer_idx: int,
        trigger_threshold: float,
        verbose: bool = False):
    neurons_to_prune = []
    layer = experiment.model.layers[layer_idx]
    # Hacky but the way to go for now.
    tracker = layer.train_dataset_tracker
    for neuron_id in range(tracker.number_of_neurons):
        frq_curr = tracker.get_neuron_stats(neuron_id)
        if frq_curr <= trigger_threshold:
            neurons_to_prune.append(neuron_id)

    if verbose:
        print("About to prune the following nuerons due to low rates:")
        pprint(neurons_to_prune)
        experiment.model.prune(
            layer_index=layer_idx,
            neuron_indices=set(neurons_to_prune))