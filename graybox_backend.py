import grpc
import time
import io

import numpy as np

from concurrent import futures
from threading import Thread
from PIL import Image
from typing import List, Tuple, Iterable

import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from collections import defaultdict

from scope_timer import ScopeTimer

from tiny_image_net_exp import get_exp

experiment = get_exp()


def training_thread_callback():
    while True:
        print("Training thread callback ", str(experiment), end="\r")
        if experiment.get_is_training():
            experiment.train_step_or_eval_full()

        time.sleep(0.1)


training_thread = Thread(target=training_thread_callback)
training_thread.start()


HYPER_PARAMETERS = {
    ("Experiment Name", "experiment_name", "text", lambda: experiment.name),
    ("Left Training Steps", "training_left", "number", lambda: experiment.training_steps_to_do),
    ("Learning Rate", "learning_rate", "number", lambda: experiment.learning_rate),
    ("Batch Size", "batch_size", "number", lambda: experiment.batch_size),
    ("Eval Frequency", "eval_frequency", "number", lambda: experiment.eval_full_to_train_steps_ratio),
    ("Checkpoint Frequency", "checkpooint_frequency", "number", lambda: experiment.experiment_dump_to_train_steps_ratio),
}


def get_hyper_parameters_pb(
        hype_parameters_desc_tuple: Tuple) -> List[pb2.HyperParameterDesc]:

    hyper_parameters_pb2 = []
    for (label, name, type_, getter) in hype_parameters_desc_tuple:
        hyper_parameter_pb2 = None
        if type_ == "text":
            hyper_parameter_pb2 = pb2.HyperParameterDesc(
                label=label,
                name=name,
                type=type_,
                string_value=getter()
            )
        else:
            hyper_parameter_pb2 = pb2.HyperParameterDesc(
                label=label,
                name=name,
                type=type_,
                numerical_value=getter()
            )
        hyper_parameters_pb2.append(hyper_parameter_pb2)

    return hyper_parameters_pb2


def get_neuron_representations(layer) -> Iterable[pb2.NeuronStatistics]:

    layer_id = layer.get_module_id()
    neuron_representations = []
    for neuron_idx in range(layer.neuron_count):
        age = int(layer.train_dataset_tracker.get_neuron_age(neuron_idx))
        trate = layer.train_dataset_tracker.get_neuron_triggers(neuron_idx)
        erate = layer.eval_dataset_tracker.get_neuron_triggers(neuron_idx)
        evage = layer.eval_dataset_tracker.get_neuron_age(neuron_idx)

        trate = trate/age if age > 0 else 0
        erate = erate/evage if evage > 0 else 0

        neuron_lr = layer.get_per_neuron_learning_rate(neuron_idx)

        neuron_representation = pb2.NeuronStatistics(
            neuron_id=pb2.NeuronId(layer_id=layer_id, neuron_id=neuron_idx),
            neuron_age=age,
            train_trigger_rate=trate,
            eval_trigger_rate=erate,
            learning_rate=neuron_lr,
        )
        for incoming_id, incoming_lr in  layer.incoming_neuron_2_lr.items():
            neuron_representation.incoming_neurons_lr[incoming_id] = incoming_lr

        neuron_representations.append(neuron_representation)

    return neuron_representations


def get_layer_representation(layer) -> pb2.LayerRepresentation:
    layer_representation = None
    if "Conv2d" in layer.__class__.__name__:
        layer_representation = pb2.LayerRepresentation(
            layer_id=layer.get_module_id(),
            layer_name=layer.__class__.__name__,
            layer_type="Conv2d",
            incoming_neurons_count=layer.incoming_neuron_count,
            neurons_count=layer.neuron_count,
            kernel_size=layer.kernel_size[0],
            stride=layer.stride[0],
        )
    elif "Linear" in layer.__class__.__name__:
        layer_representation = pb2.LayerRepresentation(
            layer_id=layer.get_module_id(),
            layer_name=layer.__class__.__name__,
            layer_type="Linear",
            incoming_neurons_count=layer.incoming_neuron_count,
            neurons_count=layer.neuron_count,
        )
    if layer_representation is None:
        return None

    layer_representation.neurons_statistics.extend(
        get_neuron_representations(layer))
    return layer_representation


def get_layer_representations(model):
    layer_representations = []
    for layer in model.layers:
        layer_representation = get_layer_representation(layer)
        if layer_representation is None:
            continue
        layer_representations.append(layer_representation)
    return layer_representations


def get_data_set_representation(dataset) -> pb2.SampleStatistics:
    print("[BACKEND].get_data_set_representation")
    data_records = ScopeTimer("records.train")
    from tqdm import tqdm
    with data_records:
        sample_stats = pb2.SampleStatistics()
        sample_stats.origin = "train"
        sample_stats.sample_count = len(dataset.wrapped_dataset)

        for sample_id, row in tqdm(enumerate(dataset.as_records())):
            sample_stats.sample_label[sample_id] = row['label']
            sample_stats.sample_prediction[sample_id] = row['predicted_class']
            sample_stats.sample_last_loss[sample_id] = row['prediction_loss']
            sample_stats.sample_encounters[sample_id] = row['exposure_amount']
            sample_stats.sample_discarded[sample_id] = row['deny_listed']

    print(data_records)
    return sample_stats


def tensor_to_bytes(tensor):
    # Convert tensor to numpy array and transpose to (H, W, C) format
    if tensor.shape[0] > 1:
        np_img = tensor.numpy().transpose(1, 2, 0)
        np_img = (np_img * 255).astype(np.uint8)
        mode = "RGB"
    else:
        np_img = tensor.squeeze(0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        mode = "L"

    img = Image.fromarray(np_img, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format='png')
    return buf.getvalue()


class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    def StreamStatus(self, request_iterator, context):
        global experiment
        while True:
            log = experiment.logger.queue.get()
            if log is None:
                break
            metrics_status, annotat_status = None, None
            if "metric_name" in log:
                metrics_status = pb2.MetricsStatus(
                    name=log["metric_name"],
                    value=log["metric_value"],
                )
            elif "annotation" in log:
                annotat_status = pb2.AnnotatStatus(
                    name=log["annotation"])
                for key, value in log["metadata"].items():
                    annotat_status.metadata[key] = value

            training_status = pb2.TrainingStatusEx(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                experiment_name=log["experiment_name"],
                model_age=log["model_age"],
            )

            if metrics_status:
                training_status.metrics_status.CopyFrom(metrics_status)
            if annotat_status:
                training_status.annotat_status.CopyFrom(annotat_status)

            experiment.logger.queue.task_done()

            yield training_status

    def ExperimentCommand(self, request, context):
        print("ExperimentServiceServicer.ExperimentCommand", request)
        if request.HasField('hyper_parameter_change'):
            # TODO(rotaru): handle this request
            hyper_parameters = request.hyper_parameter_change.hyper_parameters

            if hyper_parameters.HasField('is_training'):
                experiment.set_is_training(hyper_parameters.is_training)

            if hyper_parameters.HasField('learning_rate'):
                experiment.set_learning_rate(
                    hyper_parameters.learning_rate)

            if hyper_parameters.HasField('batch_size'):
                experiment.set_batch_size(hyper_parameters.batch_size)

            if hyper_parameters.HasField('training_steps_to_do'):
                experiment.set_training_steps_to_do(
                    hyper_parameters.training_steps_to_do)

            if hyper_parameters.HasField('experiment_name'):
                experiment.set_name(hyper_parameters.experiment_name)
            return pb2.CommandResponse(
                success=True, message="Hyper parameter changed")
        if request.HasField('deny_samples_operation'):
            denied_cnt = len(request.deny_samples_operation.sample_ids)
            experiment.train_loader.dataset.denylist_samples(
                set(request.deny_samples_operation.sample_ids))
            return pb2.CommandResponse(
                success=True, message=f"Denied {denied_cnt} samples")
        if request.HasField('load_checkpoint_operation'):
            checkpoint_id = request.load_checkpoint_operation.checkpoint_id
            experiment.load(checkpoint_id)

        response = pb2.CommandResponse(success=True, message="")
        # print("Experiment.command get state commnad")
        if request.get_hyper_parameters:
            response.hyper_parameters_descs.extend(
                get_hyper_parameters_pb(HYPER_PARAMETERS))
        # print("constructed response: ", response)
        if request.get_interactive_layers:
            if request.HasField('get_single_layer_info_id'):
                response.layer_representations.extend([
                    get_layer_representation(
                        experiment.model.get_layer_by_id(
                            request.get_single_layer_info_id))])
            else:
                response.layer_representations.extend(
                    get_layer_representations(experiment.model))
        # print("constructed response: ", response)
        if request.get_data_records:
            if request.get_data_records == "train":
                response.sample_statistics.CopyFrom(
                    get_data_set_representation(
                        experiment.train_loader.dataset))

        return response

    def GetSample(self, request, context):
        print(f"ExperimentServiceServicer.GetSample({request})")

        if not request.HasField('sample_id') or not request.HasField('origin'):
            return pb2.SampleRequestResponse(
                error_message="Invalid request. Provide sample_id & origin.")

        if request.origin not in ["train", "eval"]:
            return pb2.SampleRequestResponse(
                error_message=f"Invalid origin {request.origin}")

        if request.sample_id < 0:
            return pb2.SampleRequestResponse(
                error_message=f"Invalid sample_id {request.sample_id}")

        dataset = None
        if request.origin == "train":
            dataset = experiment.train_loader.dataset
        elif request.origin == "eval":
            dataset = experiment.eval_loader.dataset

        if dataset is None:
            return pb2.SampleRequestResponse(
                error_message=f"Dataset {request.origin} not found.")

        if request.sample_id >= len(dataset):
            return pb2.SampleRequestResponse(
                error_message=f"Sample {request.sample_id} not found.")

        data, _, label = dataset._getitem_raw(request.sample_id)
        #TODO: apply transform too
        image_bytes = tensor_to_bytes(data)

        response = pb2.SampleRequestResponse(
            sample_id=request.sample_id,
            origin=request.origin,
            label=label,
            data=image_bytes,
        )

        return response

    def ManipulateWeights(self, request, context):
        print(f"ExperimentServiceServicer.ManipulateWeights({request})")
        answer = pb2.WeightsOperationResponse(
            success=False, message="Unknown error")
        weight_operations = request.weight_operation

        if weight_operations.op_type == pb2.WeightOperationType.REMOVE_NEURONS:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id = neuron_id.layer_id
                layer_id_to_neuron_ids_list[layer_id].append(
                    neuron_id.neuron_id)

            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                experiment.model.prune(
                    layer_id=layer_id,
                    neuron_indices=set(neuron_ids))

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Pruned {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.ADD_NEURONS:
            experiment.model.add_neurons(
                layer_id=weight_operations.layer_id,
                neuron_count=weight_operations.neurons_to_add)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=\
                    f"Added {weight_operations.neurons_to_add} "
                    f"neurons to layer {weight_operations.layer_id}")
        elif weight_operations.op_type == pb2.WeightOperationType.FREEZE:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id = neuron_id.layer_id
                layer_id_to_neuron_ids_list[layer_id].append(
                    neuron_id.neuron_id)
            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                experiment.model.freeze(
                    layer_id=layer_id,
                    neuron_ids=neuron_ids)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Frozen {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.REINITIALIZE:
            for neuron_id in weight_operations.neuron_ids:
                experiment.model.reinit_neurons(
                    layer_id=neuron_id.layer_id,
                    neuron_indices={neuron_id.neuron_id})

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Reinitialized {weight_operations.neuron_ids}")
        return answer

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=6))
    servicer = ExperimentServiceServicer()
    pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    # experiment.toggle_training_status()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
