import time
from threading import Condition
from threading import Event
from threading import Lock


class Experiment:
    def __init__(
            self,
            model,
            optimizer_class,
            train_dataset,
            eval_dataset,
            device,
            learning_rate: float,
            batch_size: int,
            training_steps_to_do: int = 64,
            name: str = "baseline",
            root_log_dir: str = "root_experiment",
            logger=None,
            train_shuffle: bool = True,
            tqdm_display: bool = True,
            get_train_data_loader: None = None,
            get_eval_data_loader: None = None):
        
        self.name = name
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = False
        self.training_steps_to_do = training_steps_to_do
        self.lock = Lock()
        self.pause_event = Event()

        self.eval_full_to_train_steps_ratio = 256
        self.experiment_dump_to_train_steps_ratio = 1024
        self.occured_train_steps = 0
        self.occured_eval__steps = 0

    def __repr__(self):
        with self.lock:
            return f"[exp] is_train: {self.is_training} " + \
                f"steps: {self.training_steps_to_do}"

    def toggle_training_status(self):
        """Toggle the training status. If the model is training, it will stop.
        """
        with self.lock:
            self.is_training = not self.is_training

    def set_training_steps_to_do(self, steps: int):
        """Set the number of training steps to be performed.
        Args:
            steps (int): the number of training steps to be performed
        """
        with self.lock:
            self.training_steps_to_do = steps

    def get_is_training(self) -> bool:
        """Returns whether the model is training."""
        with self.lock:
            return self.is_training

    def set_is_training(self, is_training: bool):
        """Set whether the model is training."""
        with self.lock:
            print("[exp] Setting is_training to: ", is_training)
            self.is_training = is_training                

    def get_training_steps_to_do(self) -> int:
        """"Get the number of training steps to be performed."""
        with self.lock:
            return self.training_steps_to_do

    def train_step_or_eval_full(self):
        """Train the model for one step or evaluate the model on the full."""
        time.sleep(.1)
        # print("[exp].train_one_step",
        #       self.get_is_training(),
        #       self.get_training_steps_to_do())
        print("[exp].train_one_step", ...)
        with self.lock:
            if not self.is_training:
                print("[exp].train_one_step: not training")
                return
            self.training_steps_to_do -= 1

            if self.training_steps_to_do <= 0:
                self.is_training = False
                self.pause_event.clear()
        time.sleep(0.5)