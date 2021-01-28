# coding=utf-8
from simpleai.search.utils import BoundedPriorityQueue, InverseTransformSampler
from simpleai.search.models import SearchNodeValueOrdered
import math
import random
import numpy as np
import random as rd
from random import randint


from random import Random  



class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

    def getWeight(self):
        return self.weight

    def getValue(self):
        return self.value

class Knapsack:
    def __init__(self, knapsackcapacity, items):
        self.currentSolution = [False]*len(items)
        self.neighbourSolutions = []
        self.knapsackcapacity = knapsackcapacity
        self.items = items
        self.iterationCount = 0

    def generateNeighbourSolutions(self):
        #Generate Hamming vectors as neighbourSolution and store them in neighbourSolutions
        self.neighbourSolutions = []
        numberofitems = len(self.currentSolution)
        for j in range(numberofitems):
            neighbourSolution = []
            for i in range(numberofitems):
                if i==j:
                    neighbourSolution.append(not self.currentSolution[i])
                else:
                    neighbourSolution.append(self.currentSolution[i])
            self.neighbourSolutions.append(neighbourSolution)

    def findNextSolution(self):
        #Find the weights and values of the neighbour solutions and store them in solutionWeight and solutionValue
        numberofitems = len(self.neighbourSolutions)
        solutionWeight = [0]*numberofitems
        solutionValue = [0]*numberofitems
        for i in range(numberofitems):
            solutionWeight[i] = self.getWeight(self.neighbourSolutions[i])
            solutionValue[i] = self.getValue(self.neighbourSolutions[i])

        #Find the highest value from the neighbour solutions and replace current solution if higher
        bestValue = self.getValue(self.currentSolution)
        indexBestValue = -1
        for j in range(numberofitems):
            if solutionWeight[j]<=knapsackcapacity and solutionValue[j]>bestValue:
                bestValue = solutionValue[j]
                indexBestValue = j
        if indexBestValue != -1:
            self.currentSolution = self.neighbourSolutions[indexBestValue]
            return True
        return False
    
    def printCurrentSolution(self):
        self.iterationCount += 1
        i = str(self.iterationCount)
        s = str(list(map(int, self.currentSolution)))
        v = str(self.getValue(self.currentSolution))
        w = str(self.getWeight(self.currentSolution))
        print('Iteration ' + i +': ' + s + ' Value: ' + v + ' Weight: ' + w)

    def getWeight(self, solution):
        numberofitems = len(solution)
        solutionWeight = 0
        for i in range(numberofitems):
            if solution[i]:
                solutionWeight += self.items[i].getWeight()
        return solutionWeight

    def getValue(self, solution):
        numberofitems = len(solution)
        solutionValue = 0
        for i in range(numberofitems):
            if solution[i]:
                solutionValue += self.items[i].getValue()
        return solutionValue

if __name__ == "__main__":

    numberofitems= int(input("Please enter number of items:"))
    knapsackcapacity= int(input("Please enter knapsack capacity:"))

    weightList = [] 
    valueList= []


    for i in range(0, numberofitems): 
        a = int(input("Please enter items weights:")) 
  
        weightList.append(a) # adding the element 
      

    for i in range(0, numberofitems): 
        b = int(input("Please enter items values:")) 
  
        valueList.append(b) # adding the element 

    

    print('Knapsack capacity: ', knapsackcapacity)
    print('Number of items: ',numberofitems)
    print('Weights: ', weightList)
    print('Values: ', valueList)
    print()

    items = []
    for i in range(numberofitems):
        item = Item(weightList[i], valueList[i])
        items.append(item)

    knapsack = Knapsack(knapsackcapacity, items)

    while True:
        knapsack.generateNeighbourSolutions()
        if knapsack.findNextSolution():
            knapsack.printCurrentSolution()
        else:
            print()
            break