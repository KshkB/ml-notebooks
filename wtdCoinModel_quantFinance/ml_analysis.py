import pandas as pd
import ml_generals as ml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from dataprep import PriceData

class mlModel(PriceData):

    models = [
        Perceptron(),
        LogisticRegression(),
        SVC(),
        GaussianNB(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier()
    ]

    def __init__(self, period, intv, depth, trainTestSplit):
        super().__init__(period, intv, depth)

        self.trainTestSplit = trainTestSplit

    def fitted(self, ticker):

        training_filename = f"datasets/{ticker}_training_{self.start}_{self.end}.csv"
        labels_filename = f"datasets/{ticker}_labels_{self.start}_{self.end}.csv"
        
        trainData = pd.read_csv(training_filename)
        trainData = trainData.to_numpy()
        trainDataNormalised = StandardScaler().fit_transform(trainData)

        labelsData = pd.read_csv(labels_filename)
        labelsData = labelsData.values.ravel()

        data_train, data_test, labels_train, labels_test = train_test_split(
            trainDataNormalised,
            labelsData,
            test_size = self.trainTestSplit
        )
        fitted = []
        for classifier in mlModel.models:
            fitted += [classifier.fit(data_train, labels_train)]

        return data_train, data_test, labels_train, labels_test, fitted

class mlResults:

    model_names = [
        Perceptron.__name__,
        LogisticRegression.__name__,
        SVC.__name__,
        GaussianNB.__name__,
        DecisionTreeClassifier.__name__,
        RandomForestClassifier.__name__,
        KNeighborsClassifier.__name__
    ]

    results_columns = [
        'Classifier',
        'Accuracy score',
        'ROC-AUC score',
        'Confusion Matrix',
        'Cross validation score', 
        'Aggregate score'
    ]

    def __init__(self, data_train, data_test, labels_train, labels_test, fitted_models):
        
        self.data_train = data_train
        self.data_test = data_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.fitted_models = fitted_models

        results_df = pd.DataFrame(columns = mlResults.results_columns)
        for i, clss in enumerate(self.fitted_models):
            name = [mlResults.model_names[i]]
            rsults = ml.modelValidationResults(
                clss,
                train = self.data_train,
                train_labels = self.labels_train,
                test = self.data_test,
                test_labels = self.labels_test
            )

            lst = [name] + [rsults]
            lst_flattened = [item for sublst in lst for item in sublst]
            results_df = results_df.append(pd.Series(
                lst_flattened, index = mlResults.results_columns
            ), ignore_index = True)

        self.results = results_df

    def show(self):
        return self.results.style.hide_index()

    def bestModel(self, metric):

        row = self.results.sort_values(metric).iloc[[-1]]
        modelName = row['Classifier'].values[0]
        modelNameIndex = mlResults.model_names.index(modelName)
        classifier = self.fitted_models[modelNameIndex]

        return classifier
