"""Interruptible Thread to run the different function separately with the
possibility of stopping the thread at any time."""
import threading


class InterruptibleThread:
    """"Threa wrapper, that permit the function to be stopped at any time."""
    def __init__(self, unit_work_fn):
        self.thread = None
        self.unit_work_fn = unit_work_fn
        self.stop_event = threading.Event()

    def _run(self):
        while not self.stop_event.is_set():
            self.unit_work_fn()

    def start(self):
        """Starts the thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run)
            self.thread.start()
        else:
            print("Thread is already running.")

    def stop(self):
        """Stops the thread."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.stop_event.clear()

    def __repr__(self):
        return f"IntThread: {self.thread} {self.stop_event}"

