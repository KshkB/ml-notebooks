from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score

def modelValidationResults(classifier, train, train_labels, test, test_labels):

    predictions = classifier.predict(test)
    acc = accuracy_score(test_labels, predictions)
    rocAucScore = roc_auc_score(test_labels, predictions)
    cmtrx = confusion_matrix(test_labels, predictions)
    cval_mean = cross_val_score(
        classifier,
        train,
        train_labels, 
        cv=10,
        scoring='accuracy'
    ).mean()

    aggregate = (1/3)*(acc + rocAucScore + cval_mean)
    return acc, rocAucScore, cmtrx, cval_mean, aggregate

