import os
import sys
import numpy
import itertools
import geneticAlgorithm
import plot
import imageio
import click
import matplotlib.pyplot
import pandas as pd

@click.command()
@click.option('--figure', '-f', help='Figure relative path.')
@click.option('--output', '-o', default=os.curdir + '//', help='Output relative path.')
@click.option('--savepoint', '-s', default=500, help='Gif savepoint interval.')
@click.option('--imageinterval', '-i', default=0, help='Savepoint interval to write image file. No images are created if not included.')
@click.option('--population', '-p', default=8, help='Solutions per Population.')
@click.option('--mating', '-m', default=4, help='Number of Parents Mating.')
@click.option('--mutation', '-d', default=0.01, help='Mutation Percent.')
@click.option('--generations', '-g', default=50000, help='Number of generations.')
@click.option('--verbose', '-v', is_flag=True, help='Output log to the terminal.')
@click.option('--report', '-r', is_flag=True, help='Generate a report with the data.')
def cli(figure, output, savepoint, imageinterval, population, mating, mutation, generations, verbose, report):
    if figure == None:
        print('ERRO: É necessário uma imagem inicial.')
        sys.exit(1)

    targetImage = imageio.v2.imread(figure)
    targetChromosome = geneticAlgorithm.imageToChromosome(targetImage)
    shape = targetImage.shape
    targetFitness = numpy.sum(targetChromosome)
    qualitiesArray = []

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
        qualitiesArray.append(numpy.max(qualities))
        if verbose:
            print('INFO: Quality: ', numpy.max(qualities), ' Iteration: ', iteration)

        parents = geneticAlgorithm.selectMatingPool(newPopulation, qualities, numberOfParentsMating)
        newPopulation = geneticAlgorithm.crossover(parents, shape, solutionPerPopulation)
        newPopulation = geneticAlgorithm.mutation(newPopulation, numberOfParentsMating, mutationPercent)
        plot.saveImages(iteration, qualities, newPopulation, shape, savepoint, imageinterval, output, imageArray)
        
    if verbose:
        plot.showIndividuals(newPopulation, shape)
    
    matplotlib.pyplot.imsave(output + 'solution' + '.png', imageArray[-1])
    imageio.mimsave(output + 'solution.gif', imageArray)

    data = {
        'targetFitness': targetFitness,
        'qualitiesArray': [x for x in qualitiesArray],
        'time': [i for i in range(generations)] 
    }

    df = pd.DataFrame(data)

    matplotlib.pyplot.plot(df['time'], df['targetFitness'], color='red', marker=',', label = 'Target')
    matplotlib.pyplot.plot(df['time'], df['qualitiesArray'], color='blue', marker=',', label = 'Current')
    # matplotlib.pyplot.ylim([min(data['qualitiesArray']), data['targetFitness']])
    matplotlib.pyplot.title('Quality vs Generation', fontsize=14)
    matplotlib.pyplot.xlabel('Generations', fontsize=14)
    matplotlib.pyplot.ylabel('Quality', fontsize=14)
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.show()


if __name__ == '__main__':
    cli()