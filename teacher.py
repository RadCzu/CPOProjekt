import os
from generation import Generation

class KerasNertworkOptimizer:
    def __init__(self, candidates, output_dir, model, name):
        self.candidates.candidates = candidates
        self.output_dir = output_dir
        self.starting_model = model
        self.name = name

    def optimize(self):
        learning_dir = os.path.join(self.output_dir, self.name)
        os.makedirs(learning_dir, exist_ok=True)
        generation1 = Generation()