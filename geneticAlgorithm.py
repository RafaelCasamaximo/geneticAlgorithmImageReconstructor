import numpy
import itertools
import functools
import operator
import random

def imageToChromosome(image):
    """
    Faz a representação genética da imagem convertendo de um array tridimensional para um array unidimensional.
    """
    return numpy.reshape(a = image, newshape = (functools.reduce(operator.mul, image.shape)))

def chromosomeToImage(chromosome, shape):
    """
    Converte de um array unidimensional para um array tridimensional com base na forma da imagem original.
    """
    return numpy.reshape(a = chromosome, newshape = shape)

def createInitialPopulation(shape, numberOfIndividuals = 8):
    """
    Define uma imagem de pixels aleatórios do mesmo formato da imagem original como a população inicial.
    """
    population = numpy.empty(shape = (numberOfIndividuals, functools.reduce(operator.mul, shape)), dtype = numpy.uint8)
    for individualNumber in range(numberOfIndividuals):
        population[individualNumber, :] = numpy.random.random(functools.reduce(operator.mul, shape)) * 256
    return population

def fitness(targetChromosome, individualChromosome):
    """
    Calcula a diferença entre o pixel da evolução e do pixel original e inverte o valor pois a fitness é crescente.
    """
    quality = numpy.mean(numpy.abs(targetChromosome - individualChromosome))
    return numpy.sum(targetChromosome) - quality

def populationFitness(targetChromosome, population):
    """
    Essa função calcula a função fitness para todas as soluções na população
    """
    qualities = numpy.zeros(population.shape[0])
    for individualNumber in range(population.shape[0]):
        qualities[individualNumber] = fitness(targetChromosome, population[individualNumber, :])
    return qualities

def selectMatingPool(population, qualities, numberOfParents):
    """
    Seleciona os melhores individuos da geração atual para passar pelo crossover e formar a próxima geração.
    Depois que seleciona o melhor define a qualidade dele como -1 para ele não ser selecionado na próxima iteração.
    """
    parents = numpy.empty((numberOfParents, population.shape[1]), dtype = numpy.uint8)
    for parentNumber in range(numberOfParents):
        maxQualityId = numpy.where(qualities == numpy.max(qualities))
        maxQualityId = maxQualityId[0][0]
        parents[parentNumber, :] = population[maxQualityId, :]
        qualities[maxQualityId] = -1
    return parents

def crossover(parents, shape, numberOfIndividuals = 8):
    """
    Essa função cria uma nova população escolhendo genes aleatórios e realizando
     o crossover (troca de informação genética) entre esses elementos selecionados.
    """
    newPopulation = numpy.empty(shape = (numberOfIndividuals, functools.reduce(operator.mul, shape)), dtype = numpy.uint8)
    newPopulation[0 : parents.shape[0], :] = parents
    numberNewlyGenerated = numberOfIndividuals - parents.shape[0]
    parentsPermutations = list(itertools.permutations(iterable = numpy.arange(0, parents.shape[0]), r = 2))
    selectedPermutations = random.sample(range(len(parentsPermutations)), numberNewlyGenerated)
    combId = parents.shape[0]
    for comb in range(len(selectedPermutations)):
        selectedCombId = selectedPermutations[comb]
        selectedComb = parentsPermutations[selectedCombId]
        halfSize = numpy.int32(newPopulation.shape[1] / 2)
        newPopulation[combId + comb, 0 : halfSize] = parents[selectedComb[0], 0 : halfSize]
        newPopulation[combId + comb, halfSize :] = parents[selectedComb[1], halfSize :]
    return newPopulation

def mutation(population, numberOfParentsMating, mutationPercent):
    """
    Faz a mutação de uma amostra porcentual aleatória dos genes.
    Os genes da amostra também são alterados aleatoriamente.
    O percentual de mutação deve ser informado como um valor de 0 - 100.
    """
    for id in range(numberOfParentsMating, population.shape[0]):
        randomId = numpy.uint32(numpy.random.random(size = numpy.uint32(mutationPercent / 100 * population.shape[1])) * population.shape[1])
        newValues = numpy.uint8(numpy.random.random(size = randomId.shape[0]) * 256)
        population[id, randomId] = newValues
    return population