from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import multiprocessing
from multiprocessing import Process
import os 
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# from sklearn.svm import SVC
# from sklearn.datasets import load_diabetes, load_wine
# from sklearn.metrics import RocCurveDisplay
# import matplotlib.pyplot as plt

# with open('diab-769.csv', 'r') as file:
#     df = pd.read_csv(file)
# with open('diab-8000.csv', 'r') as file1:
#     df = pd.read_csv(file1)
with open('diab-769.csv', 'r') as file1:
    df = pd.read_csv(file1)
    print('Процесс считывания...')
print('Процесс считывания закончен.')


def fit_and_test(model, x, y, model_dict):
    print(f'Выполняет процесс обучения с использованием процесса {os.getpid()}')
    model.fit(x, y)
    model_dict[model] = model
    # print(f'Результат модели: {mean_absolute_error(test_data, predict_values)}')
    
    # return model


def selection(df):
    if 'target' in df.columns.tolist():
        df_target = df[['target']].copy()
        df_value = df.drop(['target'], axis=1)
        return df_value, df_target


def model_svr_multiprocessing(df, test):
    start = time.time()
    model_list = []

    model = SVR()
    model_2 = SVR(C=10)
    
    model_list.append(model)
    model_list.append(model_2)

    dataframe = selection(df)
    df_regr = dataframe[0]
    y_regr = dataframe[1]

    X_train, X_test, y_train, y_test = train_test_split(df_regr, y_regr, test_size = 0.1, random_state = 41)
    # with multiprocessing.Pool() as process:
   
    print('Начало процесса обучения...')
    if test == 'single':
        # start single --------------------------------------------------------------------------------------------
        print('----------------Выполнение в однопоточном режиме с использованием одного процесса.----------------')
        for i in model_list:
            i.fit(X_train, y_train)
        for i in model_list:
            print(f'Ошибка модели: {mean_absolute_error(y_test, i.predict(X_test))}')
        # end single ----------------------------------------------------------------------------------------------
    else:
        # start multiproccessing ----------------------------------------------------------------------------------
        print('------------------------------Выполнение в мультипроцессорном режиме.------------------------------')

        # Получение моделей через словарь =========================================================================
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        # =========================================================================================================
    
        # Получение обученных моделей через очередь ================================================================
        # model_queue = Queue()  # Создаем очередь
        # for i in model_list:
        #     new_process = Process(target=fit_and_test, args=(i, X_train, y_train, model_queue, X_test, y_test))
        #     new_process.start()
        
        # for _ in model_list:
        #     model = model_queue.get()  # Получаем модель из очереди
        #     model_list.append(model)
        #     # Делайте с моделью то, что вам необходимо
        # ==========================================================================================================

        processes_list = []

        for i in model_list:
            new_process = Process(target=fit_and_test, args=(i, X_train, y_train, return_dict))
            processes_list.append(new_process)
            new_process.start()
            if i == model_list[0]:
                print('Процесс 1 вызван')
            if i == model_list[1]:
                print('Процесс 2 вызван')
            print(f'Состояние процесса {new_process} -----> {new_process.is_alive()}')

        for i in processes_list:
            if i:
                print(f'Процесс {i} жив. Ожидание завершения.')
                i.join()  

        print('Процесс обучения закончен.')  

        trained_model_list = list(return_dict.values()) # Обученные модели
        result = test_trained_models(trained_model_list, X_test, y_test)
        # result = test_trained_models()
        # trained_model_list = list(return_dict.values()) # Обученные модели
        # for model in trained_model_list:
        #     predict_values = model.predict(X_test)
        #     print(f'Результат модели {model}: {mean_absolute_error(y_test, predict_values)}')

        # end multiprocessing ---------------------------------------------------------------------------------------

    end = time.time()
    print(f'Время выполнения: {end-start}')

    


def test_trained_models(model_list, data_for_predict, data_for_test):
    print('Начало тестирование обученных моделей...')
    for i, model in enumerate(model_list):
        predict_values = model.predict(data_for_predict)
        print(f'Результат модели {i}: {mean_absolute_error(data_for_test, predict_values)}')


model_svr_multiprocessing(df, 'mt')