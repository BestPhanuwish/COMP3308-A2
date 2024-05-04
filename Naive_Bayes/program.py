from typing import List, Dict
import math

def mean(numbers: List[float]) -> float:
    return sum(numbers)/len(numbers)

def std_dev(numbers: List[float]) -> float:
    return math.sqrt( sum([(x - mean(numbers)) ** 2 for x in numbers]) / len(numbers) )

class AttributeData():
    def __init__(self, result: str) -> None:
        """_summary_
        Args:
            mean (float): mean for all data of that attribute
            std_dev (float): standard deviation for all data of that attribute
            data (list): list of pre train data in number
            result (str): class that this attribute are in (yes or no)
        """
        self.mean = None
        self.std_dev = None
        self.data = []
        self.result = result
    
    def add_data(self, data: float) -> None:
        self.data.append(data)
    
    def get_mean(self) -> float:
        if self.mean == None:
            self.mean = mean(self.data)
        return self.mean
    
    def get_std_dev(self) -> float:
        if self.std_dev == None:
            self.std_dev = std_dev(self.data)
        return self.std_dev
        
    def __repr__(self) -> str:
        return "{Class: " + str(self.result) + ", Mean: " + str(self.mean) + ", Std: " + str(self.std_dev) + "}"

def pdf(x: float, mean: float, std_dev: float) -> float:
    # if standard deviation is 0 then it could cause error
    if std_dev == 0:
        return 1 # so we return 1 to get rid of the problem and potentially didn't change the value
    return (1 / (std_dev * math.sqrt(2 * math.pi))) * math.e ** (-(x - mean)**2 / (2 * std_dev**2))

def classify_nb(training_filename, testing_filename):
    # initialise variable
    attribute_data: Dict[str, List[AttributeData]] = {}
    testing_data: List[float] = []
    result: List[str] = []
    p_result: Dict[str, int] = {}
    
    # collect attribute data that will be useful from training file
    with open(training_filename, "r") as f:
        total_p = 0
        for line in f:
            data = line.split(",")
            classed = data[-1].strip()
            
            # classify the result by create a list of attribute on dict if not exist
            if classed not in attribute_data:
                attribute_data[classed] = []
                p_result[classed] = 0
                
                # initialise empty attribute data for this classifier
                for _ in range(len(data[:-1])):
                    attribute_data[classed].append(AttributeData(classed))
            
            # add number to each attribute
            for i, attribute in enumerate(attribute_data[classed]):
                num = float(data[i])
                attribute.add_data(num)
            
            # count all the data and how many data are yes or no
            p_result[classed] += 1
            total_p += 1
        
        # find p(yes) and p(no) after collect all training data
        for classed in p_result:
            p_result[classed] = p_result.get(classed) / total_p
                        
    
    # collect testing data in object
    with open(testing_filename, "r") as f:
        for line in f:
            testing_data.append([float(i) for i in line.split(",")])
            
            
    # for each of testing data, classify the result
    for test_data in testing_data:
        
        # calculate p(classed|E) for each result (in this case is yes or no)
        choice: Dict[str, float] = {}
        for classed, data in attribute_data.items():
            
            p_e = 1 # initialise starter probability
            
            # find pdf for each attribute and keep multiply them
            for i, x in enumerate(test_data):
                p_e *= pdf(x, data[i].get_mean(), data[i].get_std_dev())
                
            choice[classed] = p_e * p_result[classed]
            
        # Check if all values in the classifier dictionary are equal and not only had 1 class return "yes"
        if len(set(choice.values())) == 1 and len(choice) != 1:
            result.append("yes")
        else:
            # if not then return the result class that had the most data in it and save it
            result.append(max(choice, key=choice.get))
        
    return result