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
def zip_split(file):
    """
    Split the zip code from address.
    :param file: dataframe containing variables
    :return: file with new column:zip

    >>> file1 = pd.DataFrame({"Geographic Area Name": ["ZCTA5 35004","ZCTA5 35005","ZCTA5 35006"]})
    >>> print(zip_split(file1))
      Geographic Area Name    zip
    0          ZCTA5 35004  35004
    1          ZCTA5 35005  35005
    2          ZCTA5 35006  35006

    """
    file["zip"] = file["Geographic Area Name"].str.split(" ", expand=True)[1].astype("str")
    return file

def linear(X, y):
    """
    Multi-Linear Regression
    :param X: x variables
    :param y: y variable
    :return: linear regression model

    """
    lr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    model = lr.fit(X_train, y_train)
    ols = OLS(y,X).fit()
    print(ols.summary())
    score = model.score(X_test, y_test)
    print(model, score)
    return model

def plot_correlation(x):
    """
    Plot correlation matrix
    :param x: dataframe containing x and y variables
    :return: correlation matrix
    >>> db = pd.read_csv("Variables.csv")
    >>> pov = db["poverty"] / 100
    >>> un = db["unemployment"] / 100
    >>> edu = db["High school or higher"] / 100
    >>> edu2 = db["Bachelor or higher"] / 100
    >>> xv1 = pd.DataFrame([edu, pov, un]).T
    >>> yv = pd.DataFrame(np.log10(db["rate"]))
    >>> dba = xv1.merge(yv, left_index=True, right_index=True)
    >>> plot_correlation(dba)
                           High school or higher   poverty  unemployment      rate
    High school or higher               1.000000 -0.699980     -0.557392 -0.314174
    poverty                            -0.699980  1.000000      0.625390  0.441839
    unemployment                       -0.557392  0.625390      1.000000  0.291978
    rate                               -0.314174  0.441839      0.291978  1.000000
    """
    corr = x.corr(method="spearman")
    plt.matshow(corr)
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    return corr


def penalty(X, y):
    """
    Ridge model
    :param X: x variables
    :param y: y variables
    :return: Ridge model

    """
    lr = Ridge(alpha=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    model = lr.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(model, score)
    scores = cross_val_score(lr, X=X, y=y, cv=5)
    print(scores)
    return model

def classifiers(X, y):
    """
    Try different classification models and output the report of prediction
    :param X: x variables
    :param y: y variables
    :return: accuracy score
    """
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
        model2 = classifiers[i]
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
    db = pd.read_csv("Variables.csv")
    pov = db["poverty"] / 100
    un = db["unemployment"] / 100
    edu = db["High school or higher"] / 100
    edu2 = db["Bachelor or higher"] / 100
    xv1 = pd.DataFrame([edu, pov, un]).T
    xv2 = pd.DataFrame([edu2, pov, un]).T
    xv3 = pd.DataFrame([edu, edu2, pov, un]).T
    yv = pd.DataFrame(np.log10(db["rate"]))

    model1 = linear(xv1, yv)
    model2 = linear(xv2, yv)
    dba = xv3.merge(yv, left_index=True, right_index=True)
    # Plot correlation metrix and pairwise metrix
    print(plot_correlation(dba))
    sns.pairplot(dba)

    # Further Analysis
    # import other variables age and sex ratio
    age_file = pd.read_csv("age-2017.csv", header=1, low_memory=False)
    age_new = zip_split(age_file)[
        ["zip", "Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over",
         "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over"]].rename(
        columns={"Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over": "age",
                 "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over": "sex"})
    age_new["zip"] = age_new["zip"].astype("float64")
    db5 = db.merge(age_new, on="zip")
    db5["age"] = db5["age"].astype("float64")
    db5["sex"] = db5["sex"].astype("float64")
    describe = db5.describe()
    print(describe)
    db = db5
    un = db["unemployment"] / 100
    pov = db["poverty"] / 100
    edu2 = db["Bachelor or higher"] / 100
    age = db["age"].astype("float64") / 100
    sex = db["sex"].astype("float64") / 100
    xv = pd.DataFrame([edu2, pov, un, sex, age]).T
    yv = pd.DataFrame(db["level"])

    classifiers(xv, yv)
    dba2 = xv.merge(yv, right_index=True, left_index=True)
    plot_correlation(dba2)


