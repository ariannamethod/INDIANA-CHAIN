from typing import List


class TrainerCallback:
    pass


class TrainerControl:
    pass


class TrainerState:
    def __init__(self):
        self.is_world_process_zero = True
        self.global_step = 0


class TrainingArguments:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    return []
