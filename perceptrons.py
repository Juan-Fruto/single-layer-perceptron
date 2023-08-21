from abc import abstractmethod, ABCMeta
from tabulate import tabulate
import random

class Perceptron(metaclass = ABCMeta):

  def __init__(self):
    # protected attributes
    self._train_dataset = []
    self._test_dataset = []
    self.model = {}
    self._input = []

  def set_dataset(self, dataset, train_percent):
    for index, item in enumerate(dataset):

      if(index <= (round(len(dataset) * train_percent) - 1)):
        self._train_dataset.append(item)
      else:
        self._test_dataset.append(item)

    return self

  @abstractmethod
  def train(self):
    pass

  @abstractmethod  
  def predict(self, input):
    pass

class SingleLayerPerceptron(Perceptron):
  def __init__(self):
    super().__init__()
    self.axon = 0

  def set_dataset(self, dataset, train_percent):
    super().set_dataset(dataset, train_percent)
    return self

  def __sign_function(self, entry):
    if(entry >= 0):
      return 1
    else:
      return -1

  def train(self):
    ALPHA = 0.5 # learning rate
    error = 1
    verbose = []
    
    for axions in self._train_dataset: # axions = {'x1': 1,'x2': 10,'output': -1}
      sumation = 0
      _, train_output = axions.popitem() # axions = {'x1': 1,'x2': 10}

      for index, (key, value) in enumerate(axions.items()): # key = x1, value = 1
        
        weight_key = 'w' + str(index + 1)
        self.model[weight_key] = 0

        if(axions == self._train_dataset[0]):
          self.model[weight_key] = random.uniform(1, 5)
        else:
          self.model[weight_key] += (ALPHA * error * value) # W(k+1)= Wk + αEXk
        
        sumation += value * self.model[weight_key]
      
      output = self.__sign_function(sumation)
      
      error = train_output - output

      verbose.append([output, train_output, error])

      if(error == 0):
        headers = ["output","expected output", "error"]
        print(tabulate(verbose, headers, tablefmt="psql", showindex=True))
        print("training process finished")
        break

  def predict(self, inputs = []):
    output = []
    
    if(len(inputs) == 0):
      # inputs = self._test_dataset
      for i, v in enumerate(self._test_dataset): # i: 1, v: {'x1': 3.5,'x2': 9,'output': -1}
        input = []
        
        for index, value in enumerate(v.values()): # index: 1, value: 3.5
          input.append(value)
        
        del input[-1]
        inputs.append(input)
      
      print("Inputs:", inputs)

    for i in inputs:
      sumation = 0
      for index, value in enumerate(self.model.values()):
        sumation += value * i[index]

      label = self.__sign_function(sumation)
      output.append(label)

    return output