from .trainer import Trainer

class SimpleTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)

    def _variable_weights_init(self):
        pass

    def _optimizer_update(self):
        self.optimizer.update()   