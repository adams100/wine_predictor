
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import random
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def data():
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")  
    our_data = red.to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return(our_data)

def Logistic_Regression_Type():

    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    X = wine.drop(columns=["type", "quality"])
    y = wine["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    #param
    model_score = model.score(X_test_scaled, y_test)
    predictions = model.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Red": [conf[0][0], conf[1][0]],
        "White": [conf[0][1], conf[1][1]]
    }
    #param
    conf_table = pd.DataFrame(data, index=["Red", "White"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report = pd.DataFrame(class_report).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    global ran
    ran = random.randrange(0, len(X_test))
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    current_params = ran_selection.drop(columns=["index", "Predictions", "Actual"])
    #param
    params_table = current_params.to_html(index=False).replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return (model_score, params_table, result_table, class_report, conf_table)



def Logistic_Regression_Quality():
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    bins = [2, 6, 9]
    labels = ["Bad Wine", "Good Wine"]
    wine["quality"] = pd.cut(wine["quality"], labels=labels, bins=bins)
    X = wine.drop(columns=["type", "quality"])
    y = wine["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    params = {
        "C": [0.75, 1, 0.25]
    }
    mygrid = GridSearchCV(LogisticRegression(max_iter=99999999), scoring='accuracy', param_grid=params)
    mygrid.fit(X_train_scaled, y_train)
    #param
    model_score_quality = mygrid.score(X_test_scaled, y_test)
    #param
    grid_search_quality = pd.DataFrame(mygrid.cv_results_).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    predictions = mygrid.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Bad Wine": [conf[0][0], conf[1][0]],
        "Good Wine": [conf[0][1], conf[1][1]]
    }
    conf_table_quality = pd.DataFrame(data, index=["Bad Wine", "Good Wine"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report_quality = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report_quality = pd.DataFrame(class_report_quality).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    #use same prediction as Wine Type
    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    #param
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table_quality = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return(grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality)


def Random_Forrest_Type():    
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    X = wine.drop(columns=["type", "quality"])
    y = wine["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    #param
    model_score = model.score(X_test_scaled, y_test)
    predictions = model.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Red": [conf[0][0], conf[1][0]],
        "White": [conf[0][1], conf[1][1]]
    }
    #param
    conf_table = pd.DataFrame(data, index=["Red", "White"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report = pd.DataFrame(class_report).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')

    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    global ran
    ran = random.randrange(0, len(X_test))
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    current_params = ran_selection.drop(columns=["index", "Predictions", "Actual"])
    #param
    params_table = current_params.to_html(index=False).replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return (model_score, params_table, result_table, class_report, conf_table)





def Random_Forrest_Quality():
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    bins = [2, 6, 9]
    labels = ["Bad Wine", "Good Wine"]
    wine["quality"] = pd.cut(wine["quality"], labels=labels, bins=bins)
    X = wine.drop(columns=["type", "quality"])
    y = wine["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    params = {
        "n_estimators": np.arange(90, 120, 10)
    }
    mygrid = GridSearchCV(RandomForestClassifier(), scoring='accuracy', param_grid=params)
    mygrid.fit(X_train_scaled, y_train)
    #param
    model_score_quality = mygrid.score(X_test_scaled, y_test)
    #param
    grid_search_quality = pd.DataFrame(mygrid.cv_results_).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    predictions = mygrid.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Bad Wine": [conf[0][0], conf[1][0]],
        "Good Wine": [conf[0][1], conf[1][1]]
    }
    conf_table_quality = pd.DataFrame(data, index=["Bad Wine", "Good Wine"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report_quality = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report_quality = pd.DataFrame(class_report_quality).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    #use same prediction as Wine Type
    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    #param
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table_quality = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return(grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality)

    


def KNN_Type():
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    X = wine.drop(columns=["type", "quality"])
    y = wine["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    #param
    model_score = model.score(X_test_scaled, y_test)
    predictions = model.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Red": [conf[0][0], conf[1][0]],
        "White": [conf[0][1], conf[1][1]]
    }
    #param
    conf_table = pd.DataFrame(data, index=["Red", "White"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report = pd.DataFrame(class_report).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    global ran
    ran = random.randrange(0, len(X_test))
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    current_params = ran_selection.drop(columns=["index", "Predictions", "Actual"])
    #param
    params_table = current_params.to_html(index=False).replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return (model_score, params_table, result_table, class_report, conf_table)



def KNN_Quality():
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    bins = [2, 6, 9]
    labels = ["Bad Wine", "Good Wine"]
    wine["quality"] = pd.cut(wine["quality"], labels=labels, bins=bins)
    X = wine.drop(columns=["type", "quality"])
    y = wine["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    params = {
        "n_neighbors": np.arange(3, 7, 1)
    }
    mygrid = GridSearchCV(KNeighborsClassifier(), scoring='accuracy', param_grid=params)
    mygrid.fit(X_train_scaled, y_train)
    #param
    model_score_quality = mygrid.score(X_test_scaled, y_test)
    
    #param
    grid_search_quality = pd.DataFrame(mygrid.cv_results_).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    predictions = mygrid.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Bad Wine": [conf[0][0], conf[1][0]],
        "Good Wine": [conf[0][1], conf[1][1]]
    }
    # model_score_quality = accuracy_score(y_test, predictions)
    conf_table_quality = pd.DataFrame(data, index=["Bad Wine", "Good Wine"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report_quality = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report_quality = pd.DataFrame(class_report_quality).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    #use same prediction as Wine Type
    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    #param
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table_quality = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return(grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality)








def SVC_Type():
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    X = wine.drop(columns=["type", "quality"])
    y = wine["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    model = SVC()
    model.fit(X_train_scaled, y_train)
    #param
    model_score = model.score(X_test_scaled, y_test)
    predictions = model.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Red": [conf[0][0], conf[1][0]],
        "White": [conf[0][1], conf[1][1]]
    }
    #param
    conf_table = pd.DataFrame(data, index=["Red", "White"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report = pd.DataFrame(class_report).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    global ran
    ran = random.randrange(0, len(X_test))
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    current_params = ran_selection.drop(columns=["index", "Predictions", "Actual"])
    #param
    params_table = current_params.to_html(index=False).replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return (model_score, params_table, result_table, class_report, conf_table)


def SVC_Quality():
    red = pd.read_csv("winequality-red.csv")
    white = pd.read_csv("winequality-white.csv")
    red = red.append(white)
    wine = red
    bins = [2, 6, 9]
    labels = ["Bad Wine", "Good Wine"]
    wine["quality"] = pd.cut(wine["quality"], labels=labels, bins=bins)
    X = wine.drop(columns=["type", "quality"])
    y = wine["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    params = {
        "C": np.arange(.90, 1.20, .1),
    }
    mygrid = GridSearchCV(SVC(), scoring='accuracy', param_grid=params)
    mygrid.fit(X_train_scaled, y_train)
    #param
    model_score_quality = mygrid.score(X_test_scaled, y_test)
    #param
    grid_search_quality = pd.DataFrame(mygrid.cv_results_).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    predictions = mygrid.predict(X_test_scaled)
    conf = confusion_matrix(y_test, predictions)
    data = {
        "Bad Wine": [conf[0][0], conf[1][0]],
        "Good Wine": [conf[0][1], conf[1][1]]
    }
    conf_table_quality = pd.DataFrame(data, index=["Bad Wine", "Good Wine"]).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    class_report_quality = classification_report(y_test, predictions, output_dict=True)
    #param
    class_report_quality = pd.DataFrame(class_report_quality).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    #use same prediction as Wine Type
    result_df = X_test
    result_df["Predictions"] = predictions
    result_df["Actual"] = y_test
    result_df = result_df.reset_index()
    ran_selection = pd.DataFrame(result_df.iloc[ran]).transpose()
    #param
    current_pred = ran_selection["Predictions"]
    actual = ran_selection["Actual"]
    #param
    result_table_quality = pd.DataFrame({"Predicted": current_pred, "Actual": actual}).to_html().replace("\n", "").replace('<tr style="text-align: right;">', '<tr>')
    return(grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality)