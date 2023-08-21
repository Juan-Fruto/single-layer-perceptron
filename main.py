from perceptrons import SingleLayerPerceptron 
import csv
import random

def dataset_parsing(csv_filename):
  with open(csv_filename) as f:
    reader = csv.DictReader(f)
    
    data = []

    for row in reader:
      num_type_row =  {key: float(value) for key, value in row.items()}
      data.append(num_type_row)

    random.shuffle(data)
    return data
   
def iris():
  dataset = dataset_parsing("./datasets/iris/Iris.csv")
  
  # removing the id field and the rows who have the class 2
  iris_dataframe = []
  for i in dataset:
    del i['Id']
    if(i['Species'] != 2):
      iris_dataframe.append(i)

  slp_iris = SingleLayerPerceptron()

  # training the dataframe
  slp_iris.set_dataset(iris_dataframe, .9).train()

  # showing the model
  print("Model: ", slp_iris.model)

  # show the predictions (outputs)
  print("Predictions", slp_iris.predict())

def bill_autentication():
  ba_dataframe = dataset_parsing("./datasets/bill_authentication/bill_authentication.csv")

  slp_ba = SingleLayerPerceptron()

  # training the dataframe
  slp_ba.set_dataset(ba_dataframe, .9).train()
  
  # print("--> Debugin test <--")
  # print("Train dataset:", slp_ba._train_dataset)
  # print("Test dataset:", slp_ba._test_dataset)
  # print("--> <--")

  # showing the model
  print("Model: ", slp_ba.model)

  # show the predictions (outputs)
  print("Predictions", slp_ba.predict())

def main():
  #iris()
  bill_autentication()

if __name__ == "__main__":
  main()