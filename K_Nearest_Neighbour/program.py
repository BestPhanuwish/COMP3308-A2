from typing import List
import math

def euclidean(a: list, b: list) -> float:
    """_summary_
    Args:
        a (list): normalised attribute from the training data
        b (list): normalised attribute from the test data

    Returns:
        float: euclidean distance
    """
    inside = 0
    for i in range(len(a)):
        inside += pow(a[i]-b[i], 2)
    return math.sqrt(inside)

class TrainingData():
    def __init__(self, data: list, result: str) -> None:
        """_summary_
        Args:
            data (list): list of pre train data
            result (str): yes or no result
        """
        self.data = data
        self.result = result
    
    def __repr__(self) -> str:
        return str(self.data) + ": " + str(self.result)

def classify_nn(training_filename, testing_filename, k):
    # initialise variable
    training_data: List[TrainingData] = []
    testing_data: List[float] = []
    result: List[str] = []
    
    # collect training data in object
    with open(training_filename, "r") as f:
        for line in f:
            data = line.split(",")
            training_data.append(TrainingData(
                [float(i) for i in data[:-1]],
                data[-1].strip()
            ))
    
    # collect testing data in object
    with open(testing_filename, "r") as f:
        for line in f:
            testing_data.append([float(i) for i in line.split(",")])
            
    # for each of testing data, classify the result
    for test_data in testing_data:
        
        # sort train data with respect to euclidean distance from smallest to largest
        sorted_training_data = sorted(training_data, key=lambda train_data: euclidean(train_data.data, test_data))
        
        # only select k number of data to classify result
        classifier = {}
        for train_data in sorted_training_data[:k]:
            classifier[train_data.result] = classifier.get(train_data.result, 0) + 1
        
        # Check if all values in the classifier dictionary are equal and not only had 1 class return "yes"
        if len(set(classifier.values())) == 1 and len(classifier) != 1:
            result.append("yes")
        else:
            # if not then return the result class that had the most data in it and save it
            result.append(max(classifier, key=classifier.get))
        
    return result

print(classify_nn("train.csv", "test.csv", 5))