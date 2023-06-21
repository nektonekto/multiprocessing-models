from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import multiprocessing
from multiprocessing import Process
import os 
from sklearn.metrics import mean_absolute_error

# from sklearn.svm import SVC
# from sklearn.datasets import load_diabetes, load_wine
# from sklearn.metrics import RocCurveDisplay
# import matplotlib.pyplot as plt

# with open('diab-769.csv', 'r') as file:
#     df = pd.read_csv(file)
with open('diab-8000.csv', 'r') as file1:
    df = pd.read_csv(file1)

print(multiprocessing.cpu_count())


def fit(model, x, y, model_list, data):
    model.fit(x, y)
    model_list.append(model)
    print(f'Worker for model {model}')
    print(model.predict(data))
    # print(f'Модель: {mean_absolute_error(y_test, result)}')
    
    return model


def selection(df):
    if 'target' in df.columns.tolist():
        df_target = df[['target']].copy()
        df_value = df.drop(['target'], axis=1)
        return df_value, df_target

def model_svr_multiprocessing(df):
    start = time.time()
    model_list = []

    model = SVR()
    model_2 = SVR()
    
    model_list.append(model)
    model_list.append(model_2)

    dataframe = selection(df)
    df_regr, y_regr = dataframe[0], dataframe[1]
    

    X_train, X_test, y_train, y_test = train_test_split(df_regr, y_regr, test_size = 0.1, random_state = 41)
    # with multiprocessing.Pool() as process:
   
    # single
    # for i in model_list:
    #     i.fit(X_train, y_train)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # multiproccessing
    proc_list = []
    trained_models = []
    # with multiprocessing.Pool(processes=2) as proc:
    cycle_time = time.time()
    for i in model_list:
        proc = Process(target=fit, args=(i, X_train, y_train, trained_models, X_test))
        # proc = Process(target=i.fit(X_train, y_train), args=(i, X_train, y_train, trained_models))
        proc_list.append(proc)
        proc.start()
    end_cycle_time = time.time()
    print(f'Время запуска процессов: {end_cycle_time - cycle_time}')
    for i in proc_list:
        i.join()    

    

    # for model in model_list:
    #     result = model.predict(X_test)
    #     print(f'Модель: {mean_absolute_error(y_test, result)}')

    # pred_model_1 = model.predict(X_test)
    # pred_model_2 = model_2.predict(X_test)

    end = time.time()
    print(f'Время выполнения: {end-start}')
    # print(f'Модель 1: {mean_absolute_error(y_test, pred_model_1)}\n'
        #   f'Модель 2: {mean_absolute_error(y_test, pred_model_2)}')


model_svr_multiprocessing(df)