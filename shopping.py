import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Load the csv file

    evidence = []
    labels = []

    # List of values we want as int other than month and visitorType which require other considerations

    intValues = ["Administrative", "Informational", "ProductRelated", "OperatingSystems", "Browser", "Region", "TrafficType"]

    # List of values we want as floats

    floatValues = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

    # Dict of months of the year

    monthsToNum = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}
    
    # Simple dict linking BOOL string values to real Bool values

    stringBoolToBool = {
        "FALSE" : 0,
        "TRUE" : 1
    }

    with open(filename, newline='') as f:

        reader = csv.DictReader(f)
        for row in reader:
            # Create a current labels list to add to the labels list later
            currentEvidences = []
            # iterate on each rows which is a dict item
            for header, value in row.items():
                if header == "Revenue":
                    labels.append(stringBoolToBool[value])
                elif header in intValues:
                    currentEvidences.append(int(value))
                elif header in floatValues:
                    currentEvidences.append(float(value))
                elif header == "Month":
                    currentEvidences.append(monthsToNum[value])
                elif header == "VisitorType":
                    currentEvidences.append(1 if value == "Returning_Visitor" else 0)
                elif header == "Weekend":
                   currentEvidences.append(stringBoolToBool[value])
            # We add the current evidences to the evidence list
            evidence.append(currentEvidences)
                    
            # Add the current evidence to evidence list 
    #print(evidence, labels)
    # return the tupple
    return (evidence, labels)



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create a new model instance with the K Neighbors Classifier algo implemented in the scikit-learn lib with one neighbor as specified
    model = KNeighborsClassifier(n_neighbors=1)

    # Train the model with the data set previously loaded
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    sensitivity = 0.0
    specificity = 0.0

    predictPos = 0.0
    predictNeg = 0.0



    for index, label in enumerate(labels):
        # True positive
        if label == predictions[index] == 1:
            sensitivity += 1.0
            predictPos += 1.0
        # False positive
        elif label == 0 and predictions[index] == 1:
            predictPos += 1.0
        # True negative
        elif label == predictions[index] == 0:
            specificity += 1.0
            predictNeg += 1.0
        # False negative
        elif label == 1 and predictions[index] == 0:
            predictNeg += 1.0

    # Calculate the sensitivity and specificity
    sensitivity /= predictPos
    specificity /= predictNeg

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
