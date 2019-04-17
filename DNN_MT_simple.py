from NBHighThroughput import *
from external_functions import *
from numpy import *
from random import choice
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, r2_score, precision_score, recall_score, \
    log_loss
import os

class DNN_MT_simple:
    def __init__(self, **kwargs):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.splitted = None
        self.feature_number = None
        # parameters of selected DNN model
        self.parameters = {
            'dropout': 0.5,
            'output_activation': 'sigmoid',
            'optimization': 'SGD',
            'learning_rate': 0.005,
            'units_in_input_layer': 5000,
            'units_in_hidden_layers': [2500, 250, 10],
            'nb_epoch': 100,
            'batch_size': 75,
            'early_stopping': True,
            'patience': 30
        }
        self.filename = None
        self.verbose = 1
        # model selection parameters
        self.parameters_batch = None
        # self.parameters_batch = {
        #     'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     'output_activation': ['sigmoid'],
        #     'optimization': ['SGD', 'Adam', 'RMSprop'],
        #     'learning_rate': [0.015, 0.010, 0.005, 0.001],
        #     'batch_size': [16, 32, 64, 128, 256],
        #     'nb_epoch': [200],
        #     'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
        #                                [2500, 100], [2500, 500]],
        #     'units_in_input_layer': [5000],
        #     'early_stopping': [True],
        #     'patience': [80]
        # }
        self.model_selection_history = []
        if len(kwargs.keys()) <= 5:
            self.X = kwargs['X']
            self.y = kwargs['y']
            self.splitted = False
        elif len(kwargs.keys()) >= 6:
            self.X_train = kwargs['X_train']
            self.X_test = kwargs['X_test']
            self.y_train = kwargs['y_train']
            self.y_test = kwargs['y_test']
            self.splitted = True
        if kwargs['cv']:
            assert type(kwargs['cv']) is int, 'cv value must be of type int.'
            assert kwargs['cv'] >= 3, 'cv value must be at least 3.'
            self.cv = kwargs['cv']
        if self.splitted == True:
            self.feature_number = self.X_train.shape[1]
        elif self.splitted == False:
            self.feature_number = self.X.shape[1]
        self.parameters_batch = kwargs["parameters_batch"]
        self.types = kwargs["types"]


    def multiple_y(self,endpoints):
        if self.splitted == True:
            self.y_train_list = []
            self.y_test_list = []
        elif self.splitted == False:
            self.y_list = []
        self.insert_endpoints(endpoints)
        for i in range(self.endpoints_number):
            if self.splitted == True:
                end_train = self.y_train.iloc[:,i]
                end_test = self.y_test.iloc[:,i]
                self.y_train_list.append(end_train)
                self.y_test_list.append(end_test)
            elif self.splitted == False:
                end = self.y.iloc[:,i]
                self.y_list.append(end)

    def insert_endpoints(self,endpoints):
        self.endpoints = endpoints
        self.endpoints_number = len(endpoints)

    def generate_y_dicts(self):
        if self.splitted == True:
            self.y_trains = {}
            self.y_tests = {}
            for output,y in self.output_list,self.y_train_list:
                self.y_trains[output] = y
            for output, y in self.output_list, self.y_test_list:
                self.y_tests[output] = y
        elif self.splitted == False:
            self.y_s = {}
            for output,y in zip(self.output_list,self.y_list):
                self.y_s[output] = y

    def create_DNN_model(self, print_model=True):
        print("Creating DNN model")
        fundamental_parameters = ['dropout', 'optimization', 'learning_rate',
                                  'units_in_input_layer',
                                  'units_in_hidden_layers', 'nb_epoch', 'batch_size']
        for param in fundamental_parameters:
            if self.parameters[param] == None:
                print("Parameter not set: " + param)
                return
        self.print_parameter_values()
        # Input layer
        # input = Input(shape=(self.parameters['units_in_input_layer'],),name = "inputs")
        input = Input(shape=(self.feature_number,),name = "inputs")
        # constructing all hidden layers
        for units,i in zip(self.parameters['units_in_hidden_layers'],range(len(self.parameters['units_in_hidden_layers']))):
            if i == 0:
                x = Dense(units,activation="relu")(input)
            else:
                x = Dense(units,activation="relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(self.parameters['dropout'])(x)
        self.output_dict = {}
        self.output_list = []

        for type in enumerate(self.types):
            if type[1] == "bin":
                units = 1
                output_activation = "sigmoid"
            elif type[1] == "multi":
                if self.splitted == True:
                    units = len(self.y_train[type[0]].unique())
                elif self.splitted == False:
                    units = len(self.y.iloc[:,type[0]].unique())
                output_activation = "softmax"
            elif type[1] == "reg":
                units = 1
                # output_activation = "linear"
                output_activation = None

            output_name = self.endpoints[type[0]]
            self.output_dict[output_name] = Dense(units, activation=output_activation, name=output_name)(x)
            self.output_list.append(output_name)

        #
        # for output in enumerate(self.parameters["output_activation"]):
        #     if self.parameters['types'][output[0]] == "bin":
        #         units = 1
        #     elif self.parameters['types'][output[0]] == "multi":
        #         if self.splitted == True:
        #             units = len(self.y_train[output[0]].unique())
        #         elif self.splitted == False:
        #             units = len(self.y[output[0]].unique())
        #     elif self.parameters['types'][output[0]] == "reg":
        #         units = 1
        #     output_name = self.endpoints[output[0]] + "_output"
        #     # constructing the final layer - Multi task DNN.
        #     self.output_dict[output_name] = Dense(units, activation= output[1], name = output_name)(x)
        #     self.output_list.append(output_name)

        self.loss_types = {"bin":"binary_crossentropy", "multi":"sparse_categorical_crossentropy","reg": "mean_squared_error"}
        self.metric_types = {"bin":"accuracy","multi": "accuracy","reg":r2_keras}
        self.metrics_dict = {}
        self.loss_dict = {}

        for output,type in zip(self.output_list,self.types):
            self.loss_dict[output] = self.loss_types[type]
            self.metrics_dict[output] = self.metric_types[type]


        if self.parameters['optimization'] == 'SGD':
            optim = SGD(lr=self.parameters['learning_rate'])
            # optim.lr.set_value(self.parameters['learning_rate'])
        elif self.parameters['optimization'] == 'RMSprop':
            optim = RMSprop(lr= self.parameters['learning_rate'])
            # optim.lr.set_value(self.parameters['learning_rate'])
        elif self.parameters['optimization'] == 'Adam':
            optim = Adam()
        elif self.parameters['optimization'] == 'Adadelta':
            optim = Adadelta()

        model = Model(outputs = list(self.output_dict.values()),input=[input])
        model.compile(optimizer=optim, loss = list(self.loss_dict.values()),metrics = list(self.metrics_dict.values()))
        if self.verbose == 1: str(model.summary())
        self.model = model
        self.multiple_y(self.endpoints)
        self.generate_y_dicts()
        print("DNN model sucessfully created")

    def print_parameter_values(self):
        print("Hyperparameters")
        for key in sorted(self.parameters):
            print(key + ": " + str(self.parameters[key]))


    def fit_model(self, X_train, X_test, y_train, y_test):
        print("Fitting DNN model")
        start_time = time.time()
        if self.parameters['nb_epoch'] and self.parameters['batch_size']:
            if self.parameters['early_stopping']:
                early_stopping = EarlyStopping(monitor='val_loss', patience=self.parameters['patience'])
                self.history = self.model.fit(X_train, y_train, epochs=self.parameters['nb_epoch'],
                                              batch_size=self.parameters['batch_size'],
                                              verbose=self.verbose, validation_data=(X_test, y_test),
                                              callbacks=[early_stopping])
            else:
                self.history = self.model.fit(X_train, y_train, epochs=self.parameters['nb_epoch'],
                                              batch_size=self.parameters['batch_size'],
                                              verbose=self.verbose, validation_data=(X_test, y_test))
        fit_time = time.time() - start_time
        print("DNN model successfully fit in ", timer(fit_time))
        return fit_time

    def print_fit_results(self, train_scores, val_scores):
        for score in range(self.endpoints_number):
            print("Metrics for endpoint" + " " + self.endpoints[score])
            print('val_accuracy: ', val_scores[self.endpoints[score]])
            # print('val_loss: ', val_scores[0])
            print('train_accuracy: ', train_scores[self.endpoints[score]])
            # print('train_loss: ', train_scores[0])
            print("train/val loss ratio: ", min(self.history.history['loss']) / min(self.history.history['val_loss']))

    def predict_values(self,X):
        y_pred = self.model.predict(X)
        y_pred = [float(np.round(x)) for x in y_pred]
        y_pred = np.ravel(y_pred)
        return y_pred

    def evaluate_model(self, X_test, y_test):
        print("Evaluating model with hold out test set.")
        y_pred = self.model.predict(X_test)
        # print(y_pred)
        # y_pred = [float(np.round(x)) for x in y_pred]
        # y_pred = np.ravel(y_pred)
        # print(y_pred)
        # print(y_test)
        scores = dict()
        for i in range(self.endpoints_number):
            if self.types[i] == "bin":
                # print(y_pred)
                # print(type(y_pred))
                y_pred_v1 = [float(np.round(x)) for x in y_pred[i]]
                y_pred_v2 = np.ravel(y_pred_v1)
                # print(y_test[i])
                # print(y_pred_v2)
                scores[str(self.endpoints[i])] = accuracy_score(y_test[i], y_pred_v2)
                # scores[str(self.endpoints[i]) + "_log_loss"] = log_loss(y_test[i], y_pred_v2)
            elif self.types[i]  == "multi":
                y_pred_v1 = [int(np.argmax(x)) for x in y_pred[i]]
                # print(y_pred)
                # y_pred = [float(np.round(x)) for x in y_pred]
                y_pred_v2 = np.ravel(y_pred_v1)
                # print(y_test[i])
                # print(y_pred_v2)
                # scores['roc_auc'] = roc_auc_score(y_test, y_pred)
                scores[str(self.endpoints[i])] = accuracy_score(y_test[i], y_pred_v2)
                # scores[str(self.endpoints[i]) + "_log_loss"] = log_loss(y_test[i], y_pred[i], labels = np.unique(y_test[i]))
                # scores['f1_score'] = f1_score(y_test, y_pred)
                # scores['mcc'] = matthews_corrcoef(y_test, y_pred)
                # scores['precision'] = precision_score(y_test, y_pred)
                # scores['recall'] = recall_score(y_test, y_pred)
                # scores['log_loss'] = log_loss(y_test, y_pred)
            elif self.types[i] == "reg":
                # print(y_test[i])
                # print(y_pred[i])
                scores[str(self.endpoints[i])] = r2_score(y_test[i], y_pred[i])
        for metric, score in scores.items():
            print(metric + ': ' + str(score))
        return scores

    def batch_parameter_shufller(self):
        chosen_param = {}
        for key in self.parameters_batch:
            chosen_param[key] = choice(self.parameters_batch[key])
        return chosen_param

    def set_new_parameters(self):
        new_parameters = self.batch_parameter_shufller()
        dnn_parameters = {}
        for key in new_parameters:
            dnn_parameters[key] = new_parameters[key]
        self.parameters = dnn_parameters


    def val_data_generator(self,X,y,test_size=0.25):
        X_train, X_val, y_train,y_val = train_test_split(X,y,test_size= test_size,shuffle= True)
        return X_train,X_val,y_train,y_val

    def y_splitter(self,y):
        y_list = []
        for key in y.columns:
            y_list.append(y.loc[:,key])
        return y_list

    def y_splitter_v2(self,y):
        y_list = []
        for i in range(y.shape[1]):
            y_list.append(y[:,i])
        return y_list


    def cv_fit(self,X,y,n_folds = 3,shuffle = True):
        self.set_new_parameters()
        self.create_DNN_model()
        init_weights = self.model.get_weights()
        skf = KFold(n_splits = n_folds,shuffle=shuffle)
        train_scores_kf = []
        valid_scores_kf = []
        cv_results = {}
        i = 0
        for train, valid in skf.split(X):
            print("Running Fold " + str(i + 1) + str("/") + str(n_folds))
            # print(X)
            # print(y)
            X_train, X_valid = X.values[train], X.values[valid]
            y_train, y_valid = y.values[train], y.values[valid]
            # print(y_train)
            # print(y_valid)
            y_train_splt = self.y_splitter_v2(y_train)
            y_valid_splt = self.y_splitter_v2(y_valid)
            # print(y_train_splt)
            self.model.set_weights(init_weights)
            self.fit_model(X_train, X_valid, y_train_splt, y_valid_splt)
            # print(self.model.loss_functions)
            # print(self.model.metrics)
            train_scores = self.evaluate_model(X_train, y_train_splt)
            valid_scores = self.evaluate_model(X_valid, y_valid_splt)
            train_scores_kf.append(train_scores)
            valid_scores_kf.append(valid_scores)
            i += 1
        # print(self.model.loss_functions)
        # print(self.model.metrics)
        train_scores_list = []
        valid_scores_list = []
        for endpoint in self.endpoints:
            for i in range(n_folds):
                train_scores_list.append(train_scores_kf[i][endpoint])
                valid_scores_list.append(valid_scores_kf[i][endpoint])
                cv_results[endpoint + '_train_score_mean'] = np.mean(train_scores_list)
                cv_results[endpoint + '_valid_score_mean'] = np.mean(valid_scores_list)
        return cv_results

    def model_selection(self,n_iter=2,cv=3):
        pass