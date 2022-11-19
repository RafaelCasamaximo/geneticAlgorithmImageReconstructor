import numpy
import matplotlib.pyplot
import geneticAlgorithm

def saveImages(currentIteration, qualities, newPopulation, shape, savePoint, saveDirectory):
    """
    Salva a melhor solução de acordo com a função fitness daquela geração como uma image no diretório especificado.
    """
    if numpy.mod(currentIteration, savePoint) == 0:
        bestSolutionChromosome = newPopulation[numpy.where(qualities == numpy.max(qualities))[0][0], :]
        bestSolutionImage = geneticAlgorithm.chromosomeToImage(bestSolutionChromosome, shape)
        matplotlib.pyplot.imsave(saveDirectory + 'solution_' + str(currentIteration) + '.png', bestSolutionImage)

def showIndividuals(individuals, shape):
    """
    Mostra todos os individuos em um único gráfico
    """
    numberOfIndividuals = individuals.shape[0]
    figRowCol = 1
    for k in range(1, numpy.uint16(individuals.shape[0] / 2)):
        if numpy.floor(numpy.power(k, 2) / numberOfIndividuals) == 1:
            figRowCol = k
            break
    fig, axis = matplotlib.pyplot.subplots(figRowCol, figRowCol)
    currentIndividual = 0
    for idRow in range(figRowCol):
        for idCol in range(figRowCol):
            if currentIndividual >= individuals.shape[0]:
                break
            else:
                currentImage = geneticAlgorithm.chromosomeToImage(individuals[currentIndividual, :], shape)
                axis[idRow, idCol].imshow(currentImage)
                currentIndividual = currentIndividual + 1