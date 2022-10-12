from src.features.optimisers.evolutionary_operators import crossovers, mutations, parent_selections, \
    population_selections
from src.features.optimisers.geneticAlgorithm import GeneticAlgorithm, EGAFS
from src.features.optimisers.differentialEvolution import DifferentialEvolution
from src.features.optimisers.particleSwarmOptimisation import ParticleSwarmOptimisation, ParticleSwarmOptimisationIniPG
from src.features.binarisation.bound_handlers import LimitBoundHandler
from src.features.binarisation.binarisers import FixedBinariser


fs_wrappers = [
    # EGAFS(200, 2000, 0.9, 0.05, 0.8,
    #                  crossovers.VariableSlicedCrossover(), mutations.BitFlipMutation(), parent_selections.Rank(),
    #                  population_selections.Generational()),
    # ParticleSwarmOptimisationIniPG(population_size=30, max_nfes=3000, c1=1.49618, c2=1.49618,
    #                                inertia=0.7298, bound_handler=LimitBoundHandler(0, 1), binarizer=FixedBinariser(alpha=0.6)),
    GeneticAlgorithm(50, 10000, 0.9, 0.1,
                     crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.RouletteWheel(),
                     population_selections.MuPlusLambda()),
    DifferentialEvolution(50, 10000, 0.9, 0.5,
                          LimitBoundHandler(0, 1), FixedBinariser()),
    ParticleSwarmOptimisation(population_size=30, max_nfes=10000, c1=1.496, c2=1.496, inertia=0.7298, bound_handler=LimitBoundHandler(0, 1), binarizer=FixedBinariser()),
]
