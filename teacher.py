import os
from generation import Generation
import shutil

class KerasNertworkOptimizer:
    def __init__(self, candidates, output_dir, model_path, name, generations, mutations_per_model):
        self.candidates = candidates
        self.output_dir = output_dir
        self.starting_model_path = model_path
        self.name = name
        self.generations = generations
        self.mutations_per_model = mutations_per_model

    def optimize(self):
        learning_dir = os.path.join(self.output_dir, self.name)
        os.makedirs(learning_dir, exist_ok=True)
        gen1_dir = os.path.join(learning_dir, "gen_1")
        os.makedirs(gen1_dir, exist_ok=True)

        model_name = os.path.basename(self.starting_model_path)
        dest_model_path = os.path.join(gen1_dir, model_name)
        shutil.copy(self.starting_model_path, dest_model_path)

        generation = Generation(candidates=self.candidates, directory=gen1_dir, number=1, mutability=100, mutations=self.mutations_per_model)
        current_generations = 0
        while current_generations <= self.generations:
            current_generations += 1
            generation.mutate()
            generation.train_models(1)
            generation = generation.create_successor()

    def resume_learning(self, generation):
        learning_dir = os.path.join(self.output_dir, self.name)
        gen_dir = os.path.join(learning_dir, "gen_" + str(generation))
        generation = Generation(candidates=self.candidates, directory=gen_dir, number=generation, mutability=100,
                                mutations=self.mutations_per_model)
        generation.remake_all()
        current_generations = 0

        current_generations += 1
        generation.train_models(1)
        generation = generation.create_successor()

        current_generations = 0
        while current_generations <= self.generations:
            current_generations += 1
            generation.mutate()
            generation.train_models(1)
            generation = generation.create_successor()

    def resume_generation(self, generation):
        learning_dir = os.path.join(self.output_dir, self.name)
        gen_dir = os.path.join(learning_dir, "gen_" + str(generation))
        generation = Generation(candidates=self.candidates, directory=gen_dir, number=generation, mutability=100,
                                mutations=self.mutations_per_model)
        current_generations = 0
        generation = generation.create_successor()

        current_generations = 0
        while current_generations <= self.generations:
            current_generations += 1
            generation.mutate()
            generation.train_models(1)
            generation = generation.create_successor()
