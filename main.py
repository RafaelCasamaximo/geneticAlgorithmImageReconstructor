import os
import sys
import numpy
import itertools
import geneticAlgorithm
import plot
import imageio
import click

@click.command()
@click.option('--figure', '-f', help='Figure relative path.')
@click.option('--output', '-o', default=os.curdir + '//', help='Output relative path.')
@click.option('--savepoint', '-s', default=500, help='Savepoint interval.')
@click.option('--population', '-p', default=8, help='Solutions per Population.')
@click.option('--mating', '-m', default=4, help='Number of Parents Mating.')
@click.option('--mutation', '-d', default=0.01, help='Mutation Percent.')
@click.option('--generations', '-g', default=50000, help='Number of generations.')
@click.option('--verbose', '-v', is_flag=True, help='Output log to the terminal.')
def cli(figure, output, savepoint, population, mating, mutation, generations, verbose):
    if figure == None:
        print('ERRO: É necessário uma imagem inicial.')
        sys.exit(1)

    targetImage = imageio.v2.imread(figure)
    targetChromosome = geneticAlgorithm.imageToChromosome(targetImage)
    shape = targetImage.shape

    solutionPerPopulation = population
    numberOfParentsMating = mating
    mutationPercent = mutation

    imageArray = []

    numberOfPossiblePermutations = len(list(itertools.permutations(iterable = numpy.arange(0, numberOfParentsMating), r = 2)))
    numberOfRequiredPermutations = solutionPerPopulation - numberOfPossiblePermutations

    if numberOfRequiredPermutations > numberOfPossiblePermutations:
        print('ERRO: Inconsistência na seleção do tamanho da população ou do número de pais.')
        print('ERRO: É impossivel executar o programa com esses critérios.')
        sys.exit(1)

    newPopulation = geneticAlgorithm.createInitialPopulation(shape, solutionPerPopulation)
    for iteration in range(generations):
        qualities = geneticAlgorithm.populationFitness(targetChromosome, newPopulation)
        if verbose:
            print('INFO: Quality: ', numpy.max(qualities), ' Iteration: ', iteration)

        parents = geneticAlgorithm.selectMatingPool(newPopulation, qualities, numberOfParentsMating)
        newPopulation = geneticAlgorithm.crossover(parents, shape, solutionPerPopulation)
        newPopulation = geneticAlgorithm.mutation(newPopulation, numberOfParentsMating, mutationPercent)
        plot.saveImages(iteration, qualities, newPopulation, shape, savepoint, output, imageArray)
        
    if verbose:
        plot.showIndividuals(newPopulation, shape)
    
    imageio.mimsave(output + 'solution.gif', imageArray)


if __name__ == '__main__':
    cli()