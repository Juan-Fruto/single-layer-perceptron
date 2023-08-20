from abc import abstractmethod, ABCMeta
import random

class Perceptron(metaclass = ABCMeta):

  def __init__(self):
    # protected attributes
    self._train_dataset = []
    self._test_dataset = []
    self.model = [{}]
    self._input = []

  def set_dataset(self, dataset, train_percent):
    for index, item in enumerate(dataset):

      if(index <= (round(len(dataset) * train_percent) - 1)):
        self._train_dataset.append(item)
      else:
        self._test_dataset.append(item)

    return self
  
  def set_input(self, input):
    self.input = input

  @abstractmethod
  def train(self):
    pass
  
  @abstractmethod
  def get_output(self):
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
    alpha = 0.5 # learning rate
    error = 1
    
    for axions in self._train_dataset: # axions = {'x1': 1,'x2': 10,'output': -1}
      sumation = 0
      _, train_output = axions.popitem() # axions = {'x1': 1,'x2': 10}

      for key, value in axions.items(): # key = x1, value = 1
        
        weight = 0
        if(axions == self._train_dataset[0]):
          weight = random.uniform(1, 5)
        else:
          weight = weight + (alpha * error * value) # W(k+1)= Wk + αEXk
        
        sumation += value * weight
      
      output = self.__sign_function(sumation)
      print(output, sumation)

      error = train_output - output

      if(error == 0):
        print("error=", error)
        """
        --> save the weights in the model <--
        """
        self.model = {"model": 3}
        break


  def set_input(self, input):
    # super().set_input(input)

    # for i in (self.input):
    #   self.axons.append({'x': i})

    # self.__random_weights()
    
    return self

  def get_axon(self):
    return self.axon
  
  def get_output(self):
    pass