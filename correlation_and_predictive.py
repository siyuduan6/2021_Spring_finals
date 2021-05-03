import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.api import OLS
from scipy import stats
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report


# GV_rate = the number of GV cases / the population of the region /1000)
# the number of case per 1000 people
# edu = the percentage of people over 25 years old with high school or higher degree
# pov = the percentage of people who are under poverty level
# un = the percentage of people over 16 years old who are unemployed

def linear(X, y):
    lr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    model = lr.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ols = OLS(y,X).fit()
    print(ols.summary())
    plot_model(X_train, X_test, y_train, y_pred)
    score = model.score(X_test, y_test)
    print(model, score)
    return model

def single_linear(X, y):
    model = LinearRegression()
    fu = model.fit(X, y)
    print(fu.coef_)
    plt.plot(X, fu.predict(X))
    return fu



def plot_model(X_train, X_test, y_train,pred):
    sns.relpolt(X_train, y_train, edgecolors=(0, 0, 0))
    sns.relplot(X_test, pred, lw=3, colors = "")
    plt.show()

def plot_correlation(x):
    corr = x.corr(method="spearman")
    plt.matshow(corr)


def penalty(X, y):
    lr = Ridge(alpha=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    model = lr.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_model(X_train, X_test, y_train, y_pred)
    score = model.score(X_test, y_test)
    print(model, score)
    scores = cross_val_score(lr, X=xv, y=yv, cv=5)
    print(scores)
    return model

def classifiers(X, y):
    #names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    #         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    #         "Naive Bayes", "QDA"]
    classifiers = [
        KNeighborsClassifier(25),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    np.random.seed(31415)
    i = 0
    while i < len(classifiers):
        model2=classifiers[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        model3 = model2.fit(X_train, y_train)
        y_pred = model3.predict(X_test)
        score = model3.score(X_test, y_test)
        print(model3, score)
        score1 = cross_val_score(model2, X=X, y=y, cv=10)
        scores = classification_report(y_test,y_pred,digits=3)
        print(scores)
        i += 1
        return score, score1

if __name__ == '__main__':
    # Multi Linear Regression
    db = pd.read_csv("Variables_for_analysis.csv")
    pov = db["poverty"] / 100
    un = db["unemployment"] / 100
    edu = db["High school or higher"] / 100
    edu2 = db["Bachelor or higher"] / 100
    xv = pd.DataFrame([edu2,pov, un]).T
    yv = pd.DataFrame(np.log10(db["rate"]))

    linear(xv, yv)
    classifiers(xv,yv)
    dba = xv.merge(yv, left_index = True, right_index=True)
    # Plot correlation metrix and pairwise metrix
    plot_correlation(dba)
    sns.pairplot(dba)


