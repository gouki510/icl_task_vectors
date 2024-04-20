import itertools
import random
import string
from typing import List, Literal, Iterable

from core.data.tasks.task import Task
import numpy as np

class CopyTask(Task):
    def __init__(
        self,
        tokenizer,
        birustiness : int,
        classes: List[str],
        num_labels: int,
        epsilon: float,
    ):
        super().__init__(tokenizer)
        self.birustiness = birustiness # < num_inputs
        self.classes = classes
        self.num_classes = len(classes) 
        self.num_labels = num_labels    
        self.epsilon = epsilon
        self.labels = np.random.randint(self.num_labels, size=(self.num_classes))
        
    def build_input(self, class_id: int) -> str:
        return f"{self.classes[class_id]} {self.labels[class_id]}"
        
    def _encode_list(self, inp: str) -> str:
        return self.separator.join(inp)

    def _decode_list(self, inp: str) -> str:
        return inp.split(self.separator)

    def _random_input(self) -> str:
        length = random.choice(self.list_lenghts)
        return self._encode_list(random.choices(self.elements_space, k=length))

    def sample_inputs(self, num_inputs: int, exclude: List[str] = ()) -> List[str]:
        # in this sample inculde the exclude list
        assert num_inputs <= self.num_classes, "Not enough inputs to sample from"
        if len(exclude) > 0:
            birst_class = exclude[0]
            inputs = [birst_class] * self.birustiness
        else:
            inputs = self.classes[random.choice(range(self.num_classes))]
            return inputs
        while len(inputs) < num_inputs:
            class_id = random.choice(range(self.num_classes))
            inputs.append(self.classes[class_id])
        ordering = list(range(num_inputs))
        random.shuffle(ordering)
        inputs = [inputs[i] for i in ordering]
        return inputs
        
    def calc_output(self, inp) -> str:
        class_id = self.classes.index(inp)
        return self.labels[class_id]

    def num_examples(self) -> int:
        num_examples = 0
        num_examples += self.num_classes
        return num_examples
