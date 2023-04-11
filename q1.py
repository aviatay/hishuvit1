import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def create_dict_of_samples(num_of_samples, mean1, mean2, mean3, cov):
    samples = {}
    for i in range(num_of_samples):
        num = random.random()
        if (num <= 1 / 3):
            samples[tuple(np.random.multivariate_normal(mean1, cov, 1)[0])] = '1'
        if (1 / 3 < num <= 2 / 3):
            samples[tuple(np.random.multivariate_normal(mean2, cov, 1)[0])] = '2'
        if (2 / 3 < num <= 1):
            samples[tuple(np.random.multivariate_normal(mean3, cov, 1)[0])] = '3'
    return samples


def q1(num_of_samples, mean1, mean2, mean3, cov):
    samples = create_dict_of_samples(num_of_samples, mean1, mean2, mean3, cov)
    return samples


def q2(train, x_column='first random variable', y_columm='seconed random variable'):
    for key, value in train.items():
        if value == '1':
            plt.scatter(x=key[0], y=key[1], color='red', label=value, s=10)
        if value == '2':
            plt.scatter(x=key[0], y=key[1], color='yellow', label=value, s=10)
        if value == '3':
            plt.scatter(x=key[0], y=key[1], color='green', label=value, s=10)

    XLIM = (-10, 5)
    YLIM = (-5, 10)
    # Add titles, legend and grid
    plt.title(f'Samples - {x_column} vs. {y_columm} scatter plot')
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.xlabel(x_column)
    plt.ylabel(y_columm)
    plt.legend(['sample1', 'sample2', 'sample3'])
    plt.grid(True)
    plt.show()
    # plt.savefig('plot1.png')


def q3(num_of_samples, mean1, mean2, mean3, cov):
    test_set = create_dict_of_samples(num_of_samples, mean1, mean2, mean3, cov)
    for key, value in test_set.items():
        plt.scatter(x=key[0], y=key[1], color='blue', label=value, s=10)

    XLIM = (-10, 5)
    YLIM = (-5, 10)
    x_column = 'first random variable'
    y_columm = 'seconed random variable'
    # Add titles, legend and grid
    plt.title(f'Samples - {x_column} vs. {y_columm} scatter plot')
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.xlabel(x_column)
    plt.ylabel(y_columm)
    plt.legend(['test'])
    plt.grid(True)
    plt.show()
    return test_set


def q4(train1, test1, k):
    x_cols_train = []
    y_cols_train = []
    for key in train1.keys():
        x_cols_train.append(key)
    for value in train1.values():
        y_cols_train.append(value)
    # Instantiate the classifier
    model = KNeighborsClassifier(n_neighbors=k)
    # Fit on the training set
    model.fit(X=x_cols_train, y=y_cols_train)

    y_train_true = y_cols_train
    y_test_true = list(test1.values())
    y_train_pred = model.predict(X=x_cols_train)
    y_test_pred = model.predict(X=list(test1.keys()))

    classification_error_rate_train = np.sum(y_train_pred != y_train_true) / len(y_train_true)
    classification_error_rate_test = np.sum(y_test_pred != y_test_true) / len(y_test_true)
    ### יש פער של כ0.2 לטובת סט האימון. ניתן להסיק כי ישנה התאמת יתר על סט האימון
    # מכיוון שהוא מורץ עבור K=1 ובמצב זה כל נקודה השכנה הכי קרובה של עצמה ולכן תסווג באופן מדוייק. בשונה מסט הטסט.
    return classification_error_rate_train, classification_error_rate_test


def q5(train1, test1, k):
    k_arr = np.arange(1, k + 1)
    train_error = []
    test_error = []
    for i in range(1, k + 1):
        error_rate_train, error_rate_test = q4(train1, test1, i)
        train_error.append(error_rate_train)
        test_error.append(error_rate_test)

    plt.plot(k_arr, train_error, label="train_set")
    plt.plot(k_arr, test_error, label="test_set")
    plt.xlabel("K")
    plt.ylabel("Error")
    plt.grid(True)
    plt.xticks(k_arr)
    plt.legend()
    plt.show()

    ##לא. ניתן לקחת כדוגמה נגדית גרף בו מתקבל אותו אחוז דיוק לשני K שונים ######################


def q6(num_of_samples, mean1, mean2, mean3, cov):
    m_arr = np.arange(10, 41, 5)
    train_error = []
    test_error = []
    test_data = create_dict_of_samples(num_of_samples, mean1, mean2, mean3, cov)
    for i in range(10, 41, 5):
        train_data = create_dict_of_samples(i, mean1, mean2, mean3, cov)
        error_rate_train, error_rate_test = q4(train_data, test_data, 10)
        train_error.append(error_rate_train)
        test_error.append(error_rate_test)
    plt.plot(m_arr, train_error, label="train_set")
    plt.plot(m_arr, test_error, label="test_set")
    plt.xlabel("m train size")
    plt.ylabel("Error")
    plt.grid(True)
    plt.xticks(m_arr)
    plt.legend()
    plt.show()
    # #היינו מצפים כי כככל שסט האימון יגדל השגיאה תקטן במידה מסויימתת או תשאר מונוטונית,
    # אך אינה תעלה. להפתעתנו לפעמים קיימת עלייה באחוז השגיאה עם גדילת סט האימון.


if __name__ == '__main__':
    mean1 = [-1, 1]
    mean2 = [-2.5, 2.5]
    mean3 = [-4.5, 4.5]
    cov = np.identity(2)
    train = q1(700, mean1, mean2, mean3, cov)
    q2(train, x_column='first random variable', y_columm='seconed random variable')
    test = q3(300, mean1, mean2, mean3, cov)
    classification_error_rate_train, classification_error_rate_test = q4(train, test, 1)
    print(f"classification_error_rate_train: {classification_error_rate_train:.2f}")
    print(f"classification_error_rate_test: {classification_error_rate_test:.2f}")
    q5(train, test, 20)
    q6(100, mean1, mean2, mean3, cov)
    #q7: ישנם שינויים בגרף בין חזרה לחזרה. אפשר להסיק כי השינויים נובעים מכך שמדובר בסדרי גודל קטנים
    # של סטי האימון, ולכן ישנן תנודות ובכל הרצה המונוטוניות לה ציפינו אינה מתקיימת.

    #q8: נרצה לשפר את הסיווג לפי שכנים ע"י הגבלת המרחק בין הנקודה אותה רוצים לסווג לבין שכניה.
    # עבור הK הנבחר נרצה לסנן, אם קיים, את השכן אשר מרחקו מהנקודה הוא
    # פי שניים או יותר (ניתן לבחור כל יחס מרחק אחר)
    # ממרחק שאר השכנים מהנקודה.
    # בשיטה זו ניתן למנוע סיווג של נקודה מסויימת הנגרם עקב נקודות חריגות המשפיעות על הדאטה.

