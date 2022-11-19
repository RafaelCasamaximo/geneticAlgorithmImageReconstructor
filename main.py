import os
import sys
import numpy
import itertools
import geneticAlgorithm
import plot
import imageio

def cli():
    targetImage = imageio.imread('test.jpg')
    targetChromosome = geneticAlgorithm.imageToChromosome(targetImage)
    shape = targetImage.shape

    solutionPerPopulation = 8
    numberOfParentsMating = 4
    mutationPercent = 0.01

    numberOfPossiblePermutations = len(list(itertools.permutations(iterable = numpy.arange(0, numberOfParentsMating), r = 2)))
    numberOfRequiredPermutations = solutionPerPopulation - numberOfPossiblePermutations

    if numberOfRequiredPermutations > numberOfPossiblePermutations:
        print('ERRO: Inconsistência na seleção do tamanho da população ou do número de pais.')
        print('ERRO: É impossivel executar o programa com esses critérios.')
        sys.exit(1)

    newPopulation = geneticAlgorithm.createInitialPopulation(shape, solutionPerPopulation)
    for iteration in range(50000):
        qualities = geneticAlgorithm.populationFitness(targetChromosome, newPopulation)
        print('INFO: Quality: ', numpy.max(qualities), ' Iteration: ', iteration)

        parents = geneticAlgorithm.selectMatingPool(newPopulation, qualities, numberOfParentsMating)
        newPopulation = geneticAlgorithm.crossover(parents, shape, solutionPerPopulation)
        newPopulation = geneticAlgorithm.mutation(newPopulation, numberOfParentsMating, mutationPercent)
        plot.saveImages(iteration, qualities, newPopulation, shape, 500, os.curdir + '//')
        
    plot.showIndividuals(newPopulation, shape)

if __name__ == '__main__':
    cli()