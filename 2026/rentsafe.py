import matplotlib.pyplot as plt # For plotting
import numpy as np              # Linear algebra library
import pandas as pd # to store the data in a data frame
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as treeViz
import graphviz
from IPython.display import display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

#see the corresponding Jupyter Notebook for more explanation
#of this code

def plot_confusion_matrix(X, t, bt, group = "Everyone"):
    """
    Use the sklearn model "bt" to make predictions for the data "X",
    then compare the prediction with the target "t" to plot the confusion matrix.

    Moreover, this function prints the accuracy, precision and recall
    """
    cm = confusion_matrix(t, bt.predict(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bt.classes_)
    disp.plot()
    tn, fp, fn, tp = cm.ravel()
    print("Accuracy: ", ((tp + tn) / (tp + fp + fn + tn)))
    print("Precision: ", (tp / (tp + fp)))
    print("Recall: ", (tp / (tp + fn)))
    plt.title(f"Confusion Matrix for {group}")

    plt.show()

def build_all_models(max_depths,
                     min_samples_split,
                     criterions,
                     X_train=None,
                     t_train=None,
                     X_valid=None,
                     t_valid=None):
    """
    Parameters:
        `max_depths` - A list of values representing the max_depth values to be
                       try as hyperparameter values
        `min_samples_split` - An list of values representing the min_samples_split
                       values to try as hyperpareameter values
        `criterion` -  A string; either "entropy" or "gini"

    Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
    the training and validation accuracies (via the `score()` method).
    In other words, out[(max_depth, min_samples_split)]['val'] = validation score and
                    out[(max_depth, min_samples_split)]['train'] = training score
    For that combination of (max_depth, min_samples_split) hyperparameters.
    """
    out = {}

    for c in criterions:
        for d in max_depths:
            for s in min_samples_split:
                out[(d, s, c)] = {}
                # Create a DecisionTreeClassifier based on the given hyperparameters and fit it to the data
                tree = DecisionTreeClassifier(criterion=c, max_depth=d, min_samples_split=s)
                tree.fit(X_train, t_train)

                # TODO: store the validation and training scores in the `out` dictionary
                out[(d, s, c)]['val'] = tree.score(X_valid, t_valid)  # TODO
                out[(d, s, c)]['train'] = tree.score(X_train, t_train)
    return out


def grid_search(X_train, t_train, X_valid, t_valid):
    # Hyperparameters values to try in our grid search
    criterions = ["entropy", "gini"]
    max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
    min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    res = build_all_models(max_depths,
                           min_samples_split,
                           criterions,
                           X_train=X_train,
                           t_train=t_train,
                           X_valid=X_valid,
                           t_valid=t_valid)

    maxaccuracy = 0
    bestvals = (-1, -1, "None")
    for d, s, c in res:
        if res[(d, s, c)]['val'] > maxaccuracy:
            maxaccuracy = res[(d, s, c)]['val']
            bestvals = (d, s, c)
    return bestvals

if __name__ == "__main__":

    # loading data
    data = pd.read_csv("data/Rentsafe.csv", low_memory=False)
    data.describe()

    plt.title("Box Plot Showing the Distribution of 'CURRENT BUILDING EVAL SCORE'")
    plt.boxplot(data["CURRENT BUILDING EVAL SCORE"])
    plt.show()

    plt.title("Box Plot Showing the Distribution of 'LONGITUDE'")
    plt.boxplot(data["LONGITUDE"])
    plt.show()

    data["FENCING"].value_counts()
    data["GRAFFITI"].value_counts()

    #show some features of the data
    data["GRAFFITI"].value_counts().plot(kind='bar', title='evalutions of the extent of Graffiti')
    plt.show()

    data["GRAFFITI"].value_counts().plot(kind='hist', title='evalutions of the extent of Graffiti')
    plt.show()

    data.plot(kind='scatter', x='CLEANING LOG', y ='CURRENT BUILDING EVAL SCORE', title='cleaning log v overall evaluations')
    plt.show()

    sns.scatterplot(x='CLEANING LOG', y ='CURRENT BUILDING EVAL SCORE', data = data).set_title("cleaning log v overall evaluations")
    plt.show()

    sns.pairplot(data, vars = ["CURRENT BUILDING EVAL SCORE", "GRAFFITI", "LONGITUDE"], hue = "BUILDING CLEANLINESS")
    plt.show()

    #show some crosstabs
    pd.crosstab(data["GRAFFITI"], data["BUILDING CLEANLINESS"])

    color_scale = [(0, 'blue'), (1,'red')]

    fig = px.scatter_map(data, lat="LATITUDE", lon="LONGITUDE",
                   color="BUILDING CLEANLINESS", color_continuous_scale=color_scale,
                   hover_name="BUILDING CLEANLINESS",
                   zoom=10)
    fig.show()

    # define "Good buildings"
    data['Good'] = data["CURRENT BUILDING EVAL SCORE"] >= data["CURRENT BUILDING EVAL SCORE"].median()
    data['Good'].value_counts()

    # show some more crosstabs.  What defines a good building?
    pd.crosstab(data["Good"], data["FENCING"])
    pd.crosstab(data["Good"], data["GRAFFITI"])
    pd.crosstab(data["Good"], data["BUILDING CLEANLINESS"])
    pd.crosstab(data["Good"], data["PEST CONTROL LOG"])

    data.boxplot(column='FENCING', by='Good')
    plt.show()

    #grab features that might inform goodness
    data_fets = np.stack([
        # Type of Establishment
        data["FENCING"],
        data["BUILDING CLEANLINESS"],
        data["INT. HANDRAIL / GUARD - MAINT."],
        data["GRAFFITI"],
        data["CLEANING LOG"],
        data["PEST CONTROL LOG"],
        data["MAINTENANCE LOG"],
        data["STATE OF GOOD REPAIR PLAN"],
        data["TENANT SERVICE REQUEST LOG"],
        data["INT. HALLWAY - WALLS / CEILING"],
        #Location
        data["LONGITUDE"],
        data["LATITUDE"]
    ], axis=1)

    print(data_fets.shape) # Should be (8000, 14)

    # name features that might inform goodness
    feature_names = [
        "FENCING",
        "BUILDING CLEANLINESS",
        "INT. HANDRAIL / GUARD - MAINT.",
        "GRAFFITI",
        "CLEANING LOG",
        "PEST CONTROL LOG",
        "MAINTENANCE LOG",
        "STATE OF GOOD REPAIR PLAN",
        "TENANT SERVICE REQUEST LOG",
        "INT. HALLWAY - WALLS / CEILING",
        #Location
        "LONGITUDE",
        "LATITUDE"]

    # Split the data into X (dependent variables) and t (response variable)
    X = data_fets
    t = np.array(data["Good"] == True)

    # First, we will use `train_test_split` to split the data set into training, test and validation sets.
    # The first split uses 80% of the data for training+validation and 20% for testing.
    # The second split starts with the 80% of training+validation and assigns 25% of it to a validation set
    # Overall, this gives 60%-20%-20% for train-validation-test split.
    X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=0.2, random_state=1)
    X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=0.25, random_state=1)

    # Creating a DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    tree.fit(X_train, t_train)

    # Print the training and validation scores (accuracy)
    print("Training Set Accuracy:", tree.score(X_train, t_train))
    print("Validation Set Accuracy:", tree.score(X_valid, t_valid))

    dot_data = treeViz.export_graphviz(tree, feature_names=feature_names, max_depth=5, class_names=["Good", "Bad"], filled=True, rounded=True)
    display(graphviz.Source(dot_data))

    # Creating a DecisionTreeClassifie with a shallower depth
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=1)

    tree.fit(X_train, t_train)

    # Print the training and validation accuracy
    # Shallow tree = worse fit!
    print("Training Accuracy:", tree.score(X_train, t_train))
    print("Validation Accuracy:", tree.score(X_valid, t_valid))

    # Creating a DecisionTreeClassifier that is deeper
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=200)

    tree.fit(X_train, t_train)

    # Print the training and validation accuracy
    # Deeper tree = better fit!
    print("Training Accuracy:", tree.score(X_train, t_train))
    print("Validation Accuracy:", tree.score(X_valid, t_valid))

    #let's find the BEST tree
    bestvals = grid_search(X_train, t_train, X_valid, t_valid)
    best_tree = DecisionTreeClassifier(criterion=bestvals[2], max_depth=bestvals[0], min_samples_split=bestvals[1])
    best_tree.fit(X_train, t_train)

    # Print the training and validation accuracy
    print("Training Accuracy:", best_tree.score(X_train, t_train))
    print("Validation Accuracy:", best_tree.score(X_valid, t_valid))

    #train a model using only cleanliness and location
    data_fets = np.stack([
        data["BUILDING CLEANLINESS"],
        #Location
        data["LONGITUDE"],
        data["LATITUDE"]
    ], axis=1)

    feature_names = [
        "BUILDING CLEANLINESS",
        #Location
        "LONGITUDE",
        "LATITUDE"]

    X = data_fets
    t = np.array(data["Good"] == True)
    X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=0.2, random_state=1)
    X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=0.25, random_state=1)
    bestvals = grid_search(X_train, t_train, X_valid, t_valid)
    criterion = "entropy"
    best_tree = DecisionTreeClassifier(criterion=criterion, max_depth=bestvals[0], min_samples_split=bestvals[1])
    best_tree.fit(X_train, t_train)

    east = X_valid[:, -2] <= -79.3923 # identify rows of buildings on the east
    plot_confusion_matrix(X_valid[east], t_valid[east], best_tree, group = "East")

    west = X_valid[:, -2] > -79.3923 # identify rows of buildings on the west
    plot_confusion_matrix(X_valid[west], t_valid[west], best_tree, group = "West")