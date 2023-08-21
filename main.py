from perceptrons import SingleLayerPerceptron 
import csv

ALFA = 0.5

def main():
  slp = SingleLayerPerceptron()

  slp\
    .set_dataset([
      {'x1': 1,'x2': 10,'output': -1},
      {'x1': 5,'x2': 2,'output': 1},
      {'x1': 7,'x2': 3,'output': 1},
      {'x1': 3,'x2': 7,'output': -1},
      {'x1': 3.5,'x2': 9,'output': -1},
    ], .8)\
    .train()
  
  print("--> Debugin test <--")
  print("Train dataset:", slp._train_dataset)
  print("Test dataset:", slp._test_dataset)
  print("--> <--")

  print("Model: ", slp.model)
  print("Predictions", slp.predict())

if __name__ == "__main__":
  main()