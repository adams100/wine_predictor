from flask import Flask, render_template, redirect, request
from log_reg import Logistic_Regression_Type, Logistic_Regression_Quality, Random_Forrest_Type, Random_Forrest_Quality, KNN_Type, KNN_Quality, SVC_Type, SVC_Quality
import log_reg
# import flask
app = Flask(__name__)

title_Features = "Features"
title_Score = "Accuracy: "
title_Classification_Report = "Classification Report"
title_Confusion_Matrix = "Confusion Matrix" 
title_Data = "Wine Data"
title_Wine_Type = "Wine Type"
title_Wine_Quality = "Wine Quality"
title_gridsearch = "Parameter Tuning with GridSearchCV"
title_LR = "Logistic Regression"
title_RFC = "Random Forrest Classifier"
title_KNN = "K Neighbors Classifier"
title_SVC = "Support Vector Classification"
@app.route("/")
def index():
    # model_score, params_table, result_table, class_report, conf_table = wine_type()
    # if request.method == 'POST':
    #     return render_template("index.html", result_table=result_table, model_score=model_score, params_table=params_table, conf_table=conf_table, class_report=class_report) 
    return render_template("index.html")


    # return render_template("index.html", title_Features=title_Features, title_Score=title_Score, title_Classification_Report=title_Classification_Report,\
        # title_Confusion_Matrix=title_Confusion_Matrix)

@app.route("/logistic_regression")
def logistic_regression():
    model_score, params_table, result_table, class_report, conf_table = Logistic_Regression_Type()
    grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality = Logistic_Regression_Quality()
    return render_template("index.html", title_Features=title_Features, title_Score=title_Score, title_Classification_Report=title_Classification_Report,\
        title_Confusion_Matrix=title_Confusion_Matrix, title_Wine_Quality=title_Wine_Quality, title_Wine_Type=title_Wine_Type, title_gridsearch=title_gridsearch, result_table=result_table, model_score=model_score, params_table=params_table, conf_table=conf_table,\
             class_report=class_report, grid_search_quality=grid_search_quality, conf_table_quality=conf_table_quality, class_report_quality=class_report_quality\
                 , result_table_quality=result_table_quality, model_score_quality=model_score_quality, model_title=title_LR)



@app.route("/random_forrest")
def random_forrest():
    model_score, params_table, result_table, class_report, conf_table = Random_Forrest_Type()
    grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality = Random_Forrest_Quality()
    return render_template("index.html", title_Features=title_Features, title_Score=title_Score, title_Classification_Report=title_Classification_Report,\
        title_Confusion_Matrix=title_Confusion_Matrix, title_Wine_Quality=title_Wine_Quality, title_Wine_Type=title_Wine_Type, title_gridsearch=title_gridsearch,\
             result_table=result_table, model_score=model_score, params_table=params_table, conf_table=conf_table,\
             class_report=class_report, grid_search_quality=grid_search_quality, conf_table_quality=conf_table_quality, class_report_quality=class_report_quality\
                 , result_table_quality=result_table_quality, model_score_quality=model_score_quality, model_title=title_RFC)

@app.route("/knn")
def knn():
    model_score, params_table, result_table, class_report, conf_table = KNN_Type()
    grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality = KNN_Quality()
    return render_template("index.html", title_Features=title_Features, title_Score=title_Score, title_Classification_Report=title_Classification_Report,\
        title_Confusion_Matrix=title_Confusion_Matrix, title_Wine_Quality=title_Wine_Quality, title_Wine_Type=title_Wine_Type, title_gridsearch=title_gridsearch,\
             result_table=result_table, model_score=model_score, params_table=params_table, conf_table=conf_table,\
             class_report=class_report, grid_search_quality=grid_search_quality, conf_table_quality=conf_table_quality, class_report_quality=class_report_quality\
                 , result_table_quality=result_table_quality, model_score_quality=model_score_quality, model_title=title_KNN)



@app.route("/svc")
def svc():
    model_score, params_table, result_table, class_report, conf_table = SVC_Type()
    grid_search_quality, conf_table_quality, class_report_quality, result_table_quality, model_score_quality = SVC_Quality() 
    return render_template("index.html", title_Features=title_Features, title_Score=title_Score, title_Classification_Report=title_Classification_Report,\
        title_Confusion_Matrix=title_Confusion_Matrix, title_Wine_Quality=title_Wine_Quality, title_Wine_Type=title_Wine_Type, title_gridsearch=title_gridsearch,\
             result_table=result_table, model_score=model_score, params_table=params_table, conf_table=conf_table,\
             class_report=class_report, grid_search_quality=grid_search_quality, conf_table_quality=conf_table_quality, class_report_quality=class_report_quality\
                 , result_table_quality=result_table_quality, model_score_quality=model_score_quality, model_title=title_SVC)

if __name__ == "__main__":
    app.run(debug=True)

