import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import SCORERS, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from itertools import count
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from features import *

from timeit import timeit
import collections
import sys
import os
import warnings
warnings.filterwarnings("ignore")

high_volume_days = {"2020-02-16", "2020-02-17", "2020-02-18"}

cat1 = Hotel = {"3586", "7011"}
cat1_IC_avg = 0.00392
cat1_SCHEME_FEE_avg = 0.000985
cat1_AMOUNT_SEK_avg = 3209.0

cat2 = Transport = {"4111", "4121", "4411", "4722"}
cat2_IC_avg = 0
cat2_SCHEME_FEE_avg = 0
cat2_AMOUNT_SEK_avg = 1923.0

cat3 = Stores = {"5399", "5411", "5499", "5661", "5732", "5812", "5814", "5921", "5941", "5977"}
cat3_IC_avg = 0.303
cat3_SCHEME_FEE_avg = 0.0545
cat3_AMOUNT_SEK_avg = 188.2

cat4 = Car = {"5511", "5533", "5541", "5542", "7531", "7538"}
cat4_IC_avg = 0.0729
cat4_SCHEME_FEE_avg = 0.0137
cat4_AMOUNT_SEK_avg = 382.0

cat5 = Miscellaneous = {"5200", "7832"}
cat5_IC_avg = 0.0939
cat5_SCHEME_FEE_avg = 0.0153
cat5_AMOUNT_SEK_avg = 516.0

baltic = {"LV", "EE"}
top = {"OSLO", "STOCKHOLM", "BROMMA", "GÖTEBORG", "HANDEN", "HELSINKI", "KUNGENS KURVA", "KUNGSÄNGEN", "LIDINGÖ", "MALMÖ", "MALMO", "NACKA", "SKARHOLMEN", "SOLLENTUNA", "SOLNA", "STAVANGER", "SUNDBYBERG", "UPPSALA", "VALLINGBY", "TALLINNESTONI"}

avg_SE_IC = 0.404
avg_SE_SCHEME_FEE = 0.0723
avg_SE_trans_amount = 169.867
avg_FI_IC = 0.0492
avg_FI_SCHEME_FEE = 0.0100
avg_FI_trans_amount = 259.305


def read_and_clean_DW_data():
    data = pd.read_csv("Data/combined.csv", delimiter=";")

    to_keep = ["COUNTRY_CD", "AMOUNT_SEK", "MCC", "CALCULATED_IC", "CALCULATED_SCHEME_FEE", "PROCESSING_DATE", "TRANSACTION_TIME", "CALCULATED_MSC", "SOURCE_CURRENCY", "DESTINATION_CURRENCY", "MERCHANT_CITY"]
    data["AMOUNT_SEK"] = data["AMOUNT_SEK"].astype(float)
    for i in data.columns:
        line = i.split(";")
        for x in line:
            if x not in to_keep:
                data = data.drop(x, axis=1)

    for i in range(len(data)):
        if data.iloc[i]["COUNTRY_CD"] == "SE":
            data.at[i, "COUNTRY_CD"] = 1
        else:
            data.at[i, "COUNTRY_CD"] = 2

        if data.iloc[i]["PROCESSING_DATE"] in high_volume_days:
            data.at[i, "PROCESSING_DATE"] = 1
        else:
            data.at[i, "PROCESSING_DATE"] = 0
        
        try:
            if int(data.iloc[i]["TRANSACTION_TIME"]) in range(80000, 180000):
                data.at[i, "TRANSACTION_TIME"] = 1
            else:
                data.at[i, "TRANSACTION_TIME"] = 0
        except ValueError:
            data.at[i, "TRANSACTION_TIME"] = 0

    _, numbers = np.unique(data["MERCHANT_CITY"], return_inverse=True)
    data["MERCHANT_CITY"] = numbers

    return data

def read_data():
    data = pd.read_csv("Data/combined_X.csv", delimiter=";")
    return data

def encode_row(row):
    encoded = np.zeros(8)
    IC = 0
    SCHEME_FEE = 0
    AMOUNT_SEK = 0
    catIC = 0
    catSCHEME = 0
    catAMOUNT = 0
    
    if row["COUNTRY_CD"] == "SE":
        IC = avg_SE_IC
        SCHEME_FEE = avg_SE_SCHEME_FEE
        AMOUNT_SEK = avg_SE_trans_amount

    else:
        IC = avg_FI_IC
        SCHEME_FEE = avg_FI_SCHEME_FEE
        AMOUNT_SEK = avg_FI_trans_amount

    if row["MCC"] in cat1:
        catIC = cat1_IC_avg
        catSCHEME = cat1_SCHEME_FEE_avg
        catAMOUNT = cat1_AMOUNT_SEK_avg
    elif row["MCC"] in cat2:
        catIC = cat2_IC_avg
        catSCHEME = cat2_SCHEME_FEE_avg
        catAMOUNT = cat2_AMOUNT_SEK_avg
    elif row["MCC"] in cat3:
        catIC = cat3_IC_avg
        catSCHEME = cat3_SCHEME_FEE_avg
        catAMOUNT = cat3_AMOUNT_SEK_avg
    elif row["MCC"] in cat4:
        catIC = cat4_IC_avg
        catSCHEME = cat4_SCHEME_FEE_avg
        catAMOUNT = cat4_AMOUNT_SEK_avg
    elif row["MCC"] in cat5:
        catIC = cat5_IC_avg
        catSCHEME = cat5_SCHEME_FEE_avg
        catAMOUNT = cat5_AMOUNT_SEK_avg
    
    if row["AMOUNT_SEK"] > AMOUNT_SEK:
        encoded[0] = 1
        if row["AMOUNT_SEK"] > catAMOUNT:
            encoded[1] = 1
    
    if row["CALCULATED_SCHEME_FEE"] < SCHEME_FEE:
        encoded[2] = 1
        if row["CALCULATED_SCHEME_FEE"] < catSCHEME:
            encoded[3] = 1
    
    if row["CALCULATED_IC"] < IC:
        encoded[4] = 1
        if row["CALCULATED_IC"] < catIC:
            encoded[5] = 1

    if row["PROCESSING_DATE"] == 1:
        encoded[6] = 1

    if row["TRANSACTION_TIME"] == 1:
        encoded[7] = 1

    return encoded

def make_encoded(data):
    encoded = []
    for i in range(len(data)):
        encoded.append(encode_row(data.iloc[i]))
    
    df_encoded = pd.DataFrame(encoded)
    df_encoded.to_csv("Data/combined_encoded.csv", sep=";")

def read_encoded():
    data = pd.read_csv("Data/combined_encoded.csv", delimiter=";")
    return data

def make_y(encoded, c):
    y = []
    for i in range(len(encoded)):
        row = encoded.iloc[i]
        
        count = 0
        for value in row:
            if value == 1:
                count += 1

        if count >= c:
            y.append(1)
        else:
            y.append(0)
    
    return y

def gaussian_naive_bayes(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    return y_pred

def AdaBoost_naive_bayes(X_train, y_train, X_test, y_test):
    ada = AdaBoostClassifier(n_estimators=100)
    y_pred = ada.fit(X_train, y_train).predict(X_test)
    
    return y_pred
    

def Descision_Tree(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier(max_depth=5)
    y_pred = dt.fit(X_train, y_train).predict(X_test)
    
    return y_pred

def Random_Forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100)
    y_pred = rf.fit(X_train, y_train).predict(X_test)
    
    return y_pred

def K_Nearest(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    y_pred = knn.fit(X_train, y_train).predict(X_test)
    
    return y_pred

def SGD(X_train, y_train, X_test, y_test):
    sgd = SGDClassifier(loss="hinge", penalty="l2")
    y_pred = sgd.fit(X_train, y_train).predict(X_test)
   
    return y_pred
    
def SGDReg(X_train, y_train, X_test, y_test):
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    reg.fit(X_train, y_train)
    
    return reg

def make_predictions(data, y, i):
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
        print("Making the models")
        time_y_pred_gnb = timeit(stmt=lambda: gaussian_naive_bayes(X_train, y_train, X_test, y_test), number=1)
        time_y_pred_ada = timeit(stmt=lambda: AdaBoost_naive_bayes(X_train, y_train, X_test, y_test), number=1)
        time_y_pred_dt = timeit(stmt=lambda: Descision_Tree(X_train, y_train, X_test, y_test), number=1)
        time_y_pred_rf = timeit(stmt=lambda: Random_Forest(X_train, y_train, X_test, y_test), number=1)
        time_y_pred_knn = timeit(stmt=lambda: K_Nearest(X_train, y_train, X_test, y_test), number=1)
        time_y_pred_sgd = timeit(stmt=lambda: SGD(X_train, y_train, X_test, y_test), number=1)

        times = [time_y_pred_gnb, time_y_pred_ada, time_y_pred_dt, time_y_pred_rf, time_y_pred_knn, time_y_pred_sgd]

        y_pred_gnb = gaussian_naive_bayes(X_train, y_train, X_test, y_test)
        y_pred_ada = AdaBoost_naive_bayes(X_train, y_train, X_test, y_test)
        y_pred_dt = Descision_Tree(X_train, y_train, X_test, y_test)
        y_pred_rf = Random_Forest(X_train, y_train, X_test, y_test)
        y_pred_knn = K_Nearest(X_train, y_train, X_test, y_test)
        y_pred_sgd = SGD(X_train, y_train, X_test, y_test)

        classification_gnb = classification_report(y_test, y_pred_gnb, output_dict=True)
        classification_ada = classification_report(y_test, y_pred_ada, output_dict=True)
        classification_dt = classification_report(y_test, y_pred_dt, output_dict=True)
        classification_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        classification_knn = classification_report(y_test, y_pred_knn, output_dict=True)
        classification_sgd = classification_report(y_test, y_pred_sgd, output_dict=True)
        classifications = [classification_gnb, classification_ada, classification_dt, classification_rf, classification_knn, classification_sgd]

        time_vs_score = []
        for k in range(len(classifications)):
            classification_time_vs_score = []
            for key in classifications[k]["weighted avg"]:
                classification_time_vs_score.append(float(classifications[k]["weighted avg"][key])/float(times[k]))
            time_vs_score.append(classification_time_vs_score)
        print("Classification report")
        print(time_vs_score)


        # Make dataframe of time vs score
        time_vs_score_df = pd.DataFrame(time_vs_score)
        time_vs_score_df.columns = ["Precision", "Recall", "F1", "Support"]
        time_vs_score_df.index = ["Gaussian Naive Bayes", "AdaBoost Naive Bayes", "Decision Tree", "Random Forest", "K Nearest", "SGD"]
        time_vs_score_df = time_vs_score_df.drop("Support", axis=1)
        print(time_vs_score_df)
        time_vs_score_df.to_csv("Data/time/time_vs_score_" + str(i) + ".csv", sep=";")
        
        #### Temporarily commented to prevent writing to files #####
        # print("Writing to file")
        # with open("Data/time/time_matrices_k_" + str(i) + ".txt", "w") as f:  
        #     f.write("Gaussian Naive Bayes: " + str(y_pred_gnb) + "\n")
        #     f.write("AdaBoost Naive Bayes: " + str(y_pred_ada) + "\n")
        #     f.write("Decision Tree: " + str(y_pred_dt) + "\n")
        #     f.write("Random Forest: " + str(y_pred_rf) + "\n")
        #     f.write("K Nearest: " + str(y_pred_knn) + "\n")
        #     f.write("SGD: " + str(y_pred_sgd) + "\n")
        # print("Done")
        # print("Writing to file")
        # with open("Data/confmatrix/confusion_matrices_k_" + str(i) + ".txt", "w") as f:  
        #     f.write("The label split: " + str(collections.Counter(y)) + "\n")
        #     f.write("The confusion matrix for GNB: \n" + str(confusion_matrix(y_test, y_pred_gnb)) + "\n")
        #     f.write("The classification report for GNB: \n" + str(classification_report(y_test, y_pred_gnb)) + "\n")
        #     f.write("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_gnb).sum()) + "\n")
        #     f.write("\n")
        #     f.write("The confusion matrix for ADA: \n" + str(confusion_matrix(y_test, y_pred_ada)) + "\n")
        #     f.write("The classification report for ADA: \n" + str(classification_report(y_test, y_pred_ada)) + "\n")
        #     f.write("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_ada).sum()) + "\n")
        #     f.write("\n")
        #     f.write("The confusion matrix for DT: \n" + str(confusion_matrix(y_test, y_pred_dt)) + "\n")
        #     f.write("The classification report for DT: \n" + str(classification_report(y_test, y_pred_dt)) + "\n")
        #     f.write("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_dt).sum()) + "\n")
        #     f.write("\n")
        #     f.write("The confusion matrix for RF: \n" + str(confusion_matrix(y_test, y_pred_rf)) + "\n")
        #     f.write("The classification report for RF: \n" + str(classification_report(y_test, y_pred_rf)) + "\n")
        #     f.write("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_rf).sum()) + "\n")
        #     f.write("\n")
        #     f.write("The confusion matrix for KNN5: \n" + str(confusion_matrix(y_test, y_pred_knn)) + "\n")
        #     f.write("The classification report for KNN5: \n" + str(classification_report(y_test, y_pred_knn)) + "\n")
        #     f.write("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_knn).sum()) + "\n")
        #     f.write("\n")
            # f.write("The confusion matrix for SGD: \n" + str(confusion_matrix(y_test, y_pred_sgd)) + "\n")
            # f.write("The classification report for SGD: \n" + str(classification_report(y_test, y_pred_sgd)) + "\n")
            # f.write("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_sgd).sum()) + "\n")
        #     f.write("\n")
        #     f.write("\n")
        # print("Done")
        # print()


def main():
    data = read_and_clean_DW_data()    
    data = data[~data["COUNTRY_CD"].isin(["NO", "DK", "EE", "LV"])]

    make_encoded(data)
    encoded = read_encoded()
    y = make_y(encoded, 4)

    for i in range(3, len(data.columns)):
        data2 = SelectKBest(chi2, k=i).fit_transform(data, y)
        make_predictions(data2, y, i)

main()