"""From-scratch Base + online shared Refiner; same CLI as train_neural_ffd.py.

Example: python train_population_ffd.py --config configs/neural_ffd_1k_refiner.yaml
         --device cuda:0 --memory-profile throughput --exact
Use this entry for its own --resume / --evaluate checkpoints as well.
"""
from utils.runtime_tmp import configure_local_temporary_directory

configure_local_temporary_directory()

from train_neural_ffd import main
from utils.population_training import PopulationWorkflow


if __name__ == '__main__':
    main(workflow=PopulationWorkflow())
