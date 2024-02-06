import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def mean(data):
    """
    Calculate the mean of a list of numbers.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The mean of the numbers in the list.
    """
    dataSize = len(data)
    summ = 0
    for i in data:
        summ += i
    return summ / dataSize


def filter_data(data: pd.DataFrame):
    """
    Filter and preprocess the data by converting non-numeric values to numeric and filling missing values with the column mean.

    Args:
        data (pd.DataFrame): The input data to be filtered.

    Returns:
        pd.DataFrame: The filtered data.
    """
    for col in data.columns:
        data_to_numeric = pd.to_numeric(data[col], errors="coerce")
        data[col] = data_to_numeric.fillna(data_to_numeric.mean())
    return data


def median(data):
    """
    Calculate the median of a given list of numbers.

    Parameters:
    data (list): A list of numbers.

    Returns:
    float: The median value of the input list.
    """
    sortedData = np.sort(data)
    middle_elements = 0
    if len(sortedData) % 2 == 0:
        middle_elements = sortedData[
            int(len(sortedData) / 2) - 1 : int(len(sortedData) / 2) + 1
        ]
        middle_elements = (middle_elements[0] + middle_elements[1]) / 2
    else:
        middle_elements = sortedData[int((len(sortedData) + 1) / 2)]
    return middle_elements


def mode(data):
    """
    Calculate the mode of a given list of data.

    Parameters:
    data (list): The list of data.

    Returns:
    The mode of the data.
    """
    counts = []
    unique = np.unique(data)
    for i in range(len(unique)):
        count = 0
        for j in range(1, len(data)):
            if data[i] == data[j]:
                count += 1
        counts.append(count)
    occurence = 0
    max = 0
    mod = 0
    for count in counts:
        if max < count:
            max = count
            mod = data[counts.index(count)]
            occurence = 0
        elif max == count:
            occurence += 1
    return mod


def compare(mean, median, mod):
    """
    Compare the mean, median, and mode of a dataset and determine the symmetry.

    Parameters:
    mean (float): The mean value of the dataset.
    median (float): The median value of the dataset.
    mod (float): The mode value of the dataset.

    Returns:
    str: The symmetry of the dataset. Possible values are "Symmetric", "Positive", or "Negative".
    """
    if abs(mean - median) < 0.1 and abs(mean - mod) < 0.1:
        return "Symmetric"
    elif mod < median and median < mean:
        return "Positive"
    else:
        return "Negative"


def centralTrends(data: pd.DataFrame):
    """
    Calculate the central trends (mean, median, mode, symmetry) for each column in the given DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame.

    Returns:
    np.ndarray: A 2D array containing the central trends for each column.
    """
    trends = np.array([])
    trends = np.append(trends, np.array(["", "moyenne", "mediane", "mode", "symtrie"]))
    for key in data.columns:
        col = np.array([key])
        mn = mean(data[key])
        med = median(data[key])
        mod = mode(data[key])
        col = np.append(col, mn)
        col = np.append(col, med)
        col = np.append(col, mod)
        col = np.append(col, compare(mn, med, mod))
        trends = np.append(trends, col)
    return trends.reshape(-1, 5).T


def quartile(data):
    """
    Calculate the quartiles of a given dataset.

    Args:
        data (array-like): The dataset for which quartiles need to be calculated.

    Returns:
        tuple: A tuple containing the first quartile (Q1), second quartile (Q2 or median),
               third quartile (Q3), minimum value (Q0), and maximum value (Q4) of the dataset.
    """
    q0 = np.min(data)
    q2 = median(data)
    q4 = np.max(data)
    nq1 = int(0.25 * len(data))
    nq3 = int(0.75 * len(data))
    sorted = np.sort(data)
    q1 = sorted[nq1]
    q3 = sorted[nq3]
    return q0, q1, q2, q3, q4


def boxPlots(df: pd.DataFrame):
    """
    Remove outliers from the given DataFrame using the box plot method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    pd.DataFrame: The DataFrame containing the outliers.
    """
    outliers = df.copy()
    for col in df.columns:
        q0, q1, q2, q3, q4 = quartile(df[col])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        df[col] = df[col][~mask]
        outliers[col] = outliers[col][mask]
        # new_col, outlier = boxPlot(df, col)
    return df, outliers


def correlation_matrix(corr):
    """
    Calculates the correlation matrix and returns the matrix along with the keys of the maximum and minimum correlation values.

    Parameters:
    corr (dict): A dictionary representing the correlation matrix.

    Returns:
    tuple: A tuple containing the correlation matrix, the keys of the maximum correlation values, and the keys of the minimum correlation values.
    """
    max_corr = {}
    min_corr = {}
    min_val = 1
    max_val = -1

    for key in corr.keys():
        for key2 in corr.keys():
            if key != key2:
                if corr[key][key2] > max_val:
                    if len(max_corr.keys()) == 2:
                        min_key = min(max_corr, key=max_corr.get)
                        max_corr.pop(min_key)
                        max_corr[(key, key2)] = corr[key][key2]
                        max_val = corr[key][key2]
                    else:
                        max_corr[(key, key2)] = corr[key][key2]
                        max_val = corr[key][key2]
                if corr[key][key2] < min_val:
                    if len(min_corr.keys()) == 2:
                        max_key = max(min_corr, key=min_corr.get)
                        min_corr.pop(max_key)
                        min_corr[(key, key2)] = corr[key][key2]
                        min_val = corr[key][key2]
                    else:
                        min_corr[(key, key2)] = corr[key][key2]
                        min_val = corr[key][key2]

    return corr, max_corr, min_corr


def seperate_missing_values(data: pd.DataFrame):
    """
    Separates missing values in each column of a DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary where the keys are column names and the values are arrays of indexes
          corresponding to the missing values in each column.
    """
    missing_data = {}
    for col in data.columns:
        data_to_numeric = pd.to_numeric(data[col], errors="coerce")
        missing = data_to_numeric.isnull()
        # save only the indexes of the missing values
        missing_data[col] = missing[missing == True].index
    return missing_data


def replace_missing_values(data_col, labels, missing_data):
    """
    Replace missing values in a data column using a decision tree regressor.

    Args:
        data_col (numpy.ndarray): The data column containing missing values.
        labels (numpy.ndarray): The corresponding labels for the data column.
        missing_data (list): A list of indices indicating the positions of missing values.

    Returns:
        numpy.ndarray: An array containing the predicted values to replace the missing values.
    """
    # prepare the labels
    to_predict = np.array([])
    c = 0
    for i in missing_data:
        to_predict = np.append(to_predict, labels[i + c])
        labels = np.delete(labels, i + c)
        data_col = np.delete(data_col, i + c)
        c -= 1

    # applying the decision tree classifier
    tree = DecisionTreeRegressor()
    labels = labels.reshape(-1, 1)
    data_col = data_col.ravel()
    tree.fit(labels, data_col)
    to_predict = to_predict.reshape(-1, 1)

    res = tree.predict(to_predict)
    return res


def replace_missing_values_all(data, labels, missing_data):
    """
    Replaces missing values in the given data dictionary with corresponding values from the labels dictionary.

    Args:
        data (dict): A dictionary containing the data.
        labels (dict): A dictionary containing the labels.
        missing_data (dict): A dictionary containing the indices of missing values for each key in the data dictionary.

    Returns:
        dict: A new dictionary with missing values replaced.

    """
    new_data = data
    for key in data.keys():
        if len(missing_data[key]) > 0:
            x = 0
            for i in missing_data[key]:
                new_data[key][i] = replace_missing_values(
                    new_data[key], labels, missing_data[key]
                )[x]
                x += 1
    return new_data


def replace_aberrant_data(data, labels):
    """
    Replace aberrant data in a dataset with predicted values from a linear regression model.

    Args:
        data (pandas.DataFrame): The dataset containing the aberrant data.
        labels (pandas.Series): The corresponding labels for the dataset.

    Returns:
        tuple: A tuple containing the replaced dataset and a dictionary of aberrant data indexes for each column.
    """
    replaced_data = data.copy()
    all_aberrant_data = {}
    for column in data.columns:
        abberant_data = []
        aberrant_labels = []
        new_data = []
        new_labels = []
        llabels = labels.copy()

        q0, q1, q2, q3, q4 = quartile(data[column])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Save the indexes of the aberrant data
        abbr = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index
        abberant_data.extend(abbr)
        all_aberrant_data[column] = abberant_data
        aberrant_labels.extend(labels[abbr])
        # Remove the aberrant data from the dataset
        new_data.append(data[column].drop(abbr))
        new_labels.append(llabels.drop(abbr))
        # Train a linear regression model with the non-aberrant data
        model = LinearRegression()
        model.fit(
            pd.concat(new_labels, axis=0).values.reshape(-1, 1),
            pd.concat(new_data, axis=0),
        )

        # Replace the aberrant data with the predicted values from the linear regression model
        replaced_data.loc[abberant_data, column] = model.predict(
            np.array(aberrant_labels).reshape(-1, 1)
        )

    return replaced_data, all_aberrant_data


def reduce_data_horizontal(data):
    """
    Remove duplicate rows from the given data horizontally.

    Parameters:
    data (pandas.DataFrame): The input data.

    Returns:
    pandas.DataFrame: The data with duplicate rows removed.
    """
    unique_data = data.drop_duplicates()
    return unique_data


def theta(A, B):
    """
    Calculate the theta value between two arrays A and B.

    Parameters:
    A (array-like): First input array.
    B (array-like): Second input array.

    Returns:
    float: The theta value.

    """
    s = np.sum(A * B)
    n = len(A)
    abar = mean(A)
    bbar = mean(B)
    return (s - n * abar * bbar) / (n * np.std(A) * np.std(B))


def reduce_data_vertical(data):
    """
    Reduce the data vertically by removing columns that have a high correlation with each other.

    Args:
        data (pandas.DataFrame): The input data.

    Returns:
        tuple: A tuple containing two elements:
            - unique_columns (pandas.DataFrame): The modified data with correlated columns removed.
            - correlated_columns (list): The list of columns that were removed due to high correlation.
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr()
    # Find columns with a correlation of 1
    correlated_columns = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            # calculate the theta between the two columns and the columns must not be in the correlated_columns list and the columns must not be the same
            if (
                theta(data[corr_matrix.columns[i]], data[corr_matrix.columns[j]]) > 0.9
                and corr_matrix.columns[j] not in correlated_columns
                and corr_matrix.columns[i] not in correlated_columns
                and corr_matrix.columns[i] != corr_matrix.columns[j]
            ):
                correlated_columns.append(corr_matrix.columns[j])

    # Remove one of the correlated columns
    unique_columns = data.drop(correlated_columns, axis=1)
    return unique_columns, correlated_columns


def min_max(new_min, new_max, data):
    """
    Applies min-max scaling to the given data.

    Parameters:
    new_min (float): The desired minimum value for the scaled data.
    new_max (float): The desired maximum value for the scaled data.
    data (pandas.DataFrame): The input data to be scaled.

    Returns:
    pandas.DataFrame: The scaled data.
    """
    for col in data.columns:
        new_data = data[col]
        old_min = np.min(new_data)
        old_max = np.max(new_data)
        data[col] = new_min + ((new_data - old_min) * (new_max - new_min)) / (
            old_max - old_min
        )
    return data


def sigma(data):
    """
    Calculate the standard deviation of the given data.

    Parameters:
    data (array-like): Input data.

    Returns:
    float: The standard deviation of the data.
    """
    s = np.sum(data**2) / len(data)
    s -= mean(data) ** 2
    return np.sqrt(s)


def z_score(data):
    """
    Apply z-score normalization to the given data.

    Parameters:
    data (DataFrame): The input data to be normalized.

    Returns:
    DataFrame: The normalized data.
    """
    for col in data.columns:
        mn = mean(data[col])
        sig = sigma(data[col])
        data[col] = (data[col] - mn) / sig
    return data
