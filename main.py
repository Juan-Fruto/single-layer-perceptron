from perceptrons import SingleLayerPerceptron 

ALFA = 0.5

def main():
  pml = SingleLayerPerceptron()

  pml\
    .set_dataset([
      {'x1': 1,'x2': 10,'output': -1},
      {'x1': 5,'x2': 2,'output': 1},
      {'x1': 7,'x2': 3,'output': 1},
      {'x1': 3,'x2': 7,'output': -1},
      {'x1': 3.5,'x2': 9,'output': -1},
    ], .8)\
    .train()
  
  # print("axos\n", pml.get_axons())
  print("train\n", pml._train_dataset)
  print("test\n", pml._test_dataset)
  print(pml.model)

if __name__ == "__main__":
  main()