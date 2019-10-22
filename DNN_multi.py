#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from external_functions import *
from numpy import *
from random import choice
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import os

class DNN_multi:
    """
    This is class is to be used for generating deep learning models for multi class classification problems.

    The number of folds to be applied in cross-validation should be inputed in the "cv" argument

    The X and y data can be provided or X_train,X_test,y_train,y_test instead.

    An argument with a batch of hyperparameters can be provided
    """
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
            # 'output_activation': 'softmax',
            'optimization': 'SGD',
            'learning_rate': 0.005,
            'units_in_input_layer': 5000,
            'units_in_hidden_layers': [2500, 250, 10],
            'nb_epoch': 100,
            'batch_size': 75,
            'early_stopping': True,
            'patience': 30,
            'batch_normalization': True
        }
        self.filename = None
        self.verbose = 1
        if "parameters_batch" in kwargs:
            self.parameters_batch = kwargs["parameters_batch"]
        else:
            # self.parameters_batch = None
            self.parameters_batch = {
                'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'optimization': ['Adadelta', 'Adam', 'RMSprop', 'SGD'],
                'learning_rate': [0.015, 0.010, 0.005, 0.001, 0.0001],
                'batch_size': [16, 32, 64, 128, 256],
                'nb_epoch': [100, 150, 200],
                'units_in_hidden_layers': [[2048, 1024, 512], [1024, 128], [2048, 1024, 512, 128], [2048, 128, 16],
                                           [2048, 128], [2048, 512]],
                'units_in_input_layer': [5000],
                'early_stopping': [True],
                'batch_normalization': [True, False],
                'patience': [80]
            }
        self.model_selection_history = []
        if "X" and "y" in kwargs:
            self.X = kwargs['X']
            self.y = kwargs['y']
            self.splitted = False
        # elif len(kwargs.keys()) >= 5:
        elif "X_train" in kwargs:
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
        # self.parameters_batch = kwargs['parameters_batch']


    def print_parameter_values(self):
        """
        Prints the hyperparameters used in the DNN
        :return: None
        """
        print("Hyperparameters")
        for key in sorted(self.parameters):
            print(key + ": " + str(self.parameters[key]))

    def create_DNN_model(self, verbose=True):
        """
        Creates a Keras DNN architecture and compiles it
        :param verbose: If 1 or True it shows the training process
        :return: None
        """
        print("Creating DNN model")
        fundamental_parameters = ['dropout', 'optimization', 'learning_rate',
                                  'units_in_input_layer',
                                  'units_in_hidden_layers', 'nb_epoch', 'batch_size', 'batch_normalization']
        for param in fundamental_parameters:
            if self.parameters[param] == None:
                print("Parameter not set: " + param)
                return
        self.print_parameter_values()
        model = Sequential()
        # Input layer
        model.add(Dense(self.parameters['units_in_input_layer'], input_dim=self.feature_number, activation='relu'))
        if self.parameters['batch_normalization'] == True:
            model.add(BatchNormalization())
        model.add(Dropout(self.parameters['dropout']))
        # constructing all hidden layers
        for layer in self.parameters['units_in_hidden_layers']:
            model.add(Dense(layer, activation='relu'))
            if self.parameters['batch_normalization'] == True:
                model.add(BatchNormalization())
            model.add(Dropout(self.parameters['dropout']))
        # constructing the final layer
        if self.y is not None:
            model.add(Dense(len(np.unique(self.y)), activation="softmax"))
        else:
            model.add(Dense(len(self.y_train.unique()), activation="softmax"))
        if self.parameters['optimization'] == 'SGD':
            optim = SGD(lr = self.parameters['learning_rate'] )
        elif self.parameters['optimization'] == 'RMSprop':
            optim = RMSprop(lr = self.parameters['learning_rate'])
        elif self.parameters['optimization'] == 'Adam':
            optim = Adam()
        elif self.parameters['optimization'] == 'Adadelta':
            optim = Adadelta()
        # model = multi_gpu_model(model, gpus=2)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
        if self.verbose == 1: str(model.summary())
        self.model = model
        print("DNN model sucessfully created")

    def cv_fit(self, X, y, cv=5):
        """
        Performs a cross validation
        :param X: Dataset with input data
        :param y: Dataset with output data (labels)
        :param cv: number of folds to be used in the cv
        :return: three dictionaries with fitting evaluation scores, fitting history and fitting time
        """
        if self.parameters['nb_epoch'] and self.parameters['batch_size']:
            self.create_DNN_model()
            init_weights = self.model.get_weights()
            cvscores = []
            cvhistory = []
            time_fit = []
            i = 0
            skf = StratifiedKFold(n_splits=cv, shuffle=True)
            for train, valid in skf.split(X, y):
                print("Running Fold " + str(i + 1) + str("/") + str(cv))
                X_train, X_valid = X[train], X[valid]
                y_train, y_valid = y[train], y[valid]
                self.model.set_weights(init_weights)
                time_fit.append(self.fit_model(X_train, X_valid, y_train, y_valid))
                cvscores.append(self.evaluate_model(X_valid, y_valid))
                cvhistory.append(self.history)
                i += 1
        return cvscores, cvhistory, time_fit

    def fit_model(self, X_train, X_test, y_train, y_test):
        """
        Model fitting using the given dataset
        :param X_train: Dataset with input train data
        :param X_test: Dataset with input test data
        :param y_train: Dataset with output train data (labels)
        :param y_test: Dataset with output test data (labels)
        :return: time spent in the fitting process
        """
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
        print("#" * 20)
        return fit_time

    def fit_model_noval(self, X_train, y_train):
        """
        Model fitting using the given dataset without validation data
        :param X_train: Dataset with input train data
        :param y_train: Dataset with output train data (labels)
        :return: time spent in the fitting process
        """
        print("Fitting DNN model")
        start_time = time.time()
        if self.parameters['nb_epoch'] and self.parameters['batch_size']:
            self.history = self.model.fit(X_train, y_train, epochs=self.parameters['nb_epoch'],
                                          batch_size=self.parameters['batch_size'],
                                          verbose=self.verbose)
        fit_time = time.time() - start_time
        print("DNN model successfully fit in ", timer(fit_time))
        print("#" * 20)
        return fit_time

    def print_fit_results(self, train_scores, val_scores):
        """
        Prints the training and evaluation scores of the model
        :param train_scores: training scores of the model
        :param val_scores: evaluation scores of the model
        :return: None:
        """
        print('val_accuracy: ', val_scores[1])
        print('val_loss: ', val_scores[0])
        print('train_accuracy: ', train_scores[1])
        print('train_loss: ', train_scores[0])
        print("train/val loss ratio: ", min(self.history.history['loss']) / min(self.history.history['val_loss']))

    def predict_values(self,X):
        """
        Predicts the output data for a given input data
        :param X: Dataset with input data
        :return: Predicted output data
        """
        y_pred = self.model.predict(X)
        y_pred = [int(np.argmax(x)) for x in y_pred]
        y_pred = np.ravel(y_pred)
        return y_pred

    def evaluate_model(self, X_test, y_test):
        """
        Performs model evaluation using different metrics using test data
        :param X_test: Dataset with test input data
        :param y_test: Dataset with test output data
        :return: dictionary with metric scores from the test datasets
        """
        print("Evaluating model with hold out test set.")
        y_pred = self.model.predict(X_test)
        y_pred = [int(np.argmax(x)) for x in y_pred]
        y_pred = np.ravel(y_pred)
        scores = dict()
        scores['accuracy'] = accuracy_score(y_test, y_pred)
        scores['f1_score'] = f1_score(y_test, y_pred, average = "weighted")
        scores['mcc'] = matthews_corrcoef(y_test, y_pred)
        scores['precision'] = precision_score(y_test, y_pred, average = "weighted")
        scores['recall'] = recall_score(y_test, y_pred, average = "weighted")
        for metric, score in scores.items():
            print(metric + ': ' + str(score))
        return scores

    def format_scores_cv(self, scores_cv_list):
        """
        Calculates the metric means and standard deviations from a dictionary
        :param scores_cv_list: dictionary with the metric results of a cross-validation
        :return: three dictionaries with metric means, standard deviations and raw scores
        """
        raw_scores = dict.fromkeys(list(scores_cv_list[0].keys()))
        for key, value in raw_scores.items():
            raw_scores[key] = []
        for score in scores_cv_list:
            for metric, value in score.items():
                raw_scores[metric].append(value)
        mean_scores = dict.fromkeys(list(scores_cv_list[0].keys()))
        sd_scores = dict.fromkeys(list(scores_cv_list[0].keys()))
        for metric in raw_scores.keys():
            mean_scores[metric] = np.mean(raw_scores[metric])
            sd_scores[metric] = np.std(raw_scores[metric])
        for metric in mean_scores.keys():
            print(metric, ': ', str(mean_scores[metric]), ' +/- ', sd_scores[metric])
        return mean_scores, sd_scores, raw_scores

    def model_selection(self, X, y, n_iter_search=2, n_folds=2):
        """
        Test multiple hyperparameter configurations evaluating using cross-validation

        Generates a dictionary with the results of each hyperparameter configuration

        :param X: Dataset with input data
        :param y: Dataset with output data
        :param n_iter_search: Number of iterations (number of hyperparameter configurations to test)
        :param n_folds: Number of folds to be used in the cross-validation process
        :return: None
        """
        print("Selecting best DNN model")
        old_parameters = self.parameters.copy()
        self.model_selection_history = []
        for iteration in range(n_iter_search):
            mean_train_accuracy = []
            mean_train_loss = []
            mean_val_accuracy = []
            mean_val_loss = []
            mean_time_fit = []
            print("Iteration no. " + str(iteration + 1))
            new_parameters = self.batch_parameter_shufller()
            temp_values = new_parameters.copy()
            for key in new_parameters:
                self.parameters[key] = new_parameters[key]
            i = 0
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False,)
            for train, valid in skf.split(X, y):
                print("Running Fold " + str(i + 1) + str("/") + str(n_folds))
                X_train, X_valid = X[train], X[valid]
                y_train, y_valid = y[train], y[valid]
                self.create_DNN_model()
                time_fit = self.fit_model(X_train, X_valid, y_train, y_valid)
                train_scores = self.model.evaluate(X_train, y_train)
                train_accuracy = train_scores[1]
                train_loss = train_scores[0]
                val_scores = self.model.evaluate(X_valid, y_valid)
                val_accuracy = val_scores[1]
                val_loss = val_scores[0]
                self.print_fit_results(train_scores, val_scores)
                mean_train_accuracy.append(train_accuracy)
                mean_train_loss.append(train_loss)
                mean_val_accuracy.append(val_accuracy)
                mean_val_loss.append(val_loss)
                mean_time_fit.append(time_fit)
                temp_values['train_accuracy_' + str(i + 1)] = train_accuracy
                temp_values['train_loss_' + str(i + 1)] = train_loss
                temp_values['val_accuracy_' + str(i + 1)] = val_accuracy
                temp_values['val_loss_' + str(i + 1)] = val_loss
                temp_values['time_fit_' + str(i + 1)] = time_fit
                i += 1
                K.clear_session()
                del self.history
                del self.model
            temp_values['mean_train_accuracy'] = np.mean(mean_train_accuracy)
            temp_values['mean_train_loss'] = np.mean(mean_train_loss)
            temp_values['mean_val_accuracy'] = np.mean(mean_val_accuracy)
            temp_values['mean_val_loss'] = np.mean(mean_val_loss)
            temp_values['mean_time_fit'] = np.mean(mean_time_fit)
            self.model_selection_history.append(temp_values)
        self.parameters = old_parameters.copy()
        print("Best DNN model successfully selected")

    def find_best_model(self):
        """
        Selects the model with best performance
        :return: dictionary with the hyperparameters of the model with best performance, value of the metric used for evaluating model performance
        """
        if self.model_selection_history:
            best_model = None
            max_val_accuracy = 0
            for dic in self.model_selection_history:
                if dic['mean_val_accuracy'] > max_val_accuracy:
                    best_model = dic
                    max_val_accuracy = dic['mean_val_accuracy']
        # If all models have score 0 assume the first one
        if best_model is None:
            best_model = self.model_selection_history[0]
        # print("Best model:")
        # self.print_parameter_values()
        # print("Accuracy: " + str(max_val_accuracy))
        return best_model,max_val_accuracy

    def select_best_model(self):
        """
        Selects the model with best performance and updates the hyperparameter configuration
        :return: None
        """
        best_model, max_val = self.find_best_model()
        for key in self.parameters:
            self.parameters[key] = best_model[key]
        print("Best model:")
        self.print_parameter_values()
        print("Accuracy: " + str(max_val))


    def batch_parameter_shufller(self):
        """
        Creates a dictionary with a random choice of hyperaparameters
        :return: dictionary with random hyperparameters
        """
        chosen_param = {}
        for key in self.parameters_batch:
            chosen_param[key] = choice(self.parameters_batch[key])
        return chosen_param

    def set_filename(self,filename):
        """
        Changes the name of the file
        :param filename: new name of the file
        :return: None
        """
        self.filename = filename

    def plot_model_performance(self, cv_history, root_dir, file_name, save_fig=True, show_plot=False):
        """
        Creates a plot showing the a metric value and the loss in different number of epoch during fitting
        :param cv_history: dictionary with metric scores from cross-validation
        :param root_dir: name of the directory where the results will be stored
        :param file_name: name of the file to be stored
        :param save_fig: boolean true if the plot is to be saved
        :param show_plot: boolean true if the plot is to be shown
        :return: None
        """
        # summarize history for loss
        ## Plotting the loss with the number of iterations
        fig = plt.figure(figsize=(20, 15))
        fig.add_subplot(121)
        for record in cv_history:
            plt.semilogy(record.history['loss'], color='blue')
            plt.semilogy(record.history['val_loss'], color='orange')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        ## Plotting the error with the number of iterations
        ## With each iteration the error reduces smoothly
        fig.add_subplot(122)
        for record in cv_history:
            plt.plot(record.history['acc'], color='blue')
            plt.plot(record.history['val_acc'], color='orange')
            plt.legend(['train', 'test'], loc='upper left')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if save_fig:
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            i = 0
            while os.path.exists(root_dir + '/' + file_name + '_graph_results_' + str(i) + '.png'):
                i += 1
            file_name = os.path.join(root_dir, file_name + '_graph_results_' + str(i) + '.png')
            print("Writing graph results file with path: ", file_name)
            plt.savefig(file_name)
        if show_plot: plt.show()

    def write_model_selection_results(self, root_dir, file_name):
        """
        Writes a csv with the model selection cross-validation results
        :param root_dir: name of the directory where the results will be stored
        :param file_name: name of the file to be stored
        :return: None
        """
        d = self.model_selection_history
        sequence = ['mean_val_accuracy', 'mean_val_loss', 'mean_train_accuracy',
                    'mean_train_loss', 'mean_time_fit']
        sequence_tv_scores = [key for key in d[0] if key.startswith(("train_", "val_", "time_"))]
        sequence_tv_scores.sort()
        sequence_parameters = [x for x in self.parameters]
        sequence_parameters.sort()
        sequence.extend(sequence_tv_scores + sequence_parameters)
        df = pd.DataFrame(d, columns=sequence).sort_values(['mean_val_accuracy'], ascending=[False])
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + '/' + file_name + '_model_selection_table_' + str(i) + '.csv'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_model_selection_table_' + str(i) + '.csv')
        print('Writing csv file with path: ', final_path)
        df.to_csv(final_path, sep='\t')
        self.model_selection_results = df

    def write_report(self, scores, root_dir, file_name):
        """
        Writes a txt report with evaluation results of the model
        :param scores: dictionary with evaluation results
        :param root_dir: name of the directory where the results will be stored
        :param file_name: name of the file to be stored
        :return: None
        """
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + file_name + '_report_' + str(
                i) + '.txt'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_report_' + str(i) + '.txt')
        print("Writing report file with path: " + final_path)
        out = open(final_path, 'w')
        out.write('Hyperparameters')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        for key in sorted(self.parameters):
            out.write(key + ": " + str(self.parameters[key]))
            out.write('\n')
        out.write('\n')
        out.write('Scores')
        out.write('\n')
        out.write("=" * 25)
        out.write('\n')
        for metric, score in scores.items():
            out.write(str(metric) + ': ' + str(score))
            out.write('\n')
        out.close()
        print("Report file successfully written.")

    def write_report_cv(self, mean_scores, sd_scores, raw_scores, root_dir, file_name):
        """
        Writes a txt report with cross-validation results of the model
        :param mean_scores: dictionary with mean of the cross-validation results
        :param sd_scores: dictionary with the standard deviations of the cross-valiation results
        :param raw_scores: dictionary with the raw scores of the cross-validation results
        :param root_dir: name of the directory where the results will be stored
        :param file_name: name of the file to be stored
        :return: None
        """
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + '/' + file_name + '_report_' + str(i) + '.txt'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_report_' + str(i) + '.txt')
        print("Writing report file with path: " + final_path)
        out = open(final_path, 'w')
        out.write('Hyperparameters')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        for key in sorted(self.parameters):
            out.write(key + ": " + str(self.parameters[key]))
            out.write('\n')
        out.write('\n')
        out.write('Scores')
        out.write('\n')
        out.write("=" * 25)
        out.write('\n')
        for metric, scores in mean_scores.items():
            out.write(str(metric) + ': ' + str(mean_scores[metric]) + ' +/- ' + str(sd_scores[metric]))
            out.write('\n')
        out.close()
        df = pd.DataFrame.from_dict(raw_scores)
        cv_df_path = os.path.join(root_dir, file_name + '_cv_results_' + str(i) + '.csv')
        print('Writing csv file with path: ', cv_df_path)
        df.to_csv(cv_df_path, sep='\t')
        print("Report files successfully written.")

    def model_fit_results(self, dropout, optimization, learning_rate, units_in_input_layer,
                  units_in_hidden_layers, nb_epoch, batch_size, early_stopping, patience, root_dir, file_name, cv):
        """
        Fits a model using a given set of hyperparameters and evaluates the model performance
        :param dropout: dropout rate to be used in the layers
        :param optimization: optimizer to be used in the DNN
        :param learning_rate: learning rate to be used by the optimizer
        :param units_in_input_layer: Number of units of the input layer
        :param units_in_hidden_layers: Number of units of the hidden layers
        :param nb_epoch: number of epochs to be used in the fitting
        :param batch_size: batch size to be used in the fitting
        :param early_stopping: boolean True if early stopping to be considered
        :param patience: number of epochs to tolerate before early stopping
        :param root_dir: name of the directory where the results will be stored
        :param file_name: name of the file to be stored
        :param cv: number of folds to be used in the cross-validation
        :return: None
        """
        self.parameters = {

            'dropout': dropout,
            # 'output_activation': output_activation,
            'optimization': optimization,
            'learning_rate': learning_rate,
            'units_in_input_layer': units_in_input_layer,
            'units_in_hidden_layers': units_in_hidden_layers,
            'nb_epoch': nb_epoch,
            'batch_size': batch_size,
            'early_stopping': early_stopping,
            'patience': patience

        }
        cv_scores, cv_history, time_fit = self.cv_fit(self.X, self.y, cv)
        self.plot_model_performance(cv_history, root_dir, file_name)
        mean_scores, sd_scores, raw_scores = self.format_scores_cv(cv_scores)
        self.write_report_cv(mean_scores, sd_scores, raw_scores, root_dir, file_name)
        time_fit = np.asarray(time_fit, dtype=float)
        del self.model

    def save_best_model(self):
        """
        Saves the model with best performance in json and HDF5 files
        :return: None
        """
        print("Saving the best model")
        i = 0
        while os.path.exists("best_DNN_models/" + "DNN_multi" + str(i) + '.json'):
            i += 1
        root = "best_DNN_models"
        if not os.path.exists(root):
            os.makedirs(root)
        file_name_model = os.path.join(root, "DNN_multi" + str(i) + '.json')
        file_name_weights = os.path.join(root, "DNN_multi" + str(i) + '.h5')
        model_json = self.model.to_json()
        with open(file_name_model, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(file_name_weights)
        print("Saved model to disk")

    def load_model(self,filename):
        """
        Loads a model using the json and a HDF5 file
        :param filename: name of the model to be loaded
        :return: None
        """
        file_name_model = "best_DNN_models\\" + filename + '.json'
        file_name_weights = "best_DNN_models\\" + filename + '.h5'
        print("Loading model from: " + file_name_model)
        json_file = open(file_name_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(file_name_weights)
        self.model = loaded_model
        print("Model loaded successfully!")

    def multi_model_selection(self, root_dir, experiment_designation, n_iter=10, cv=5):
        """
        Performs a random hyperparameters optimization with cross-validation, selects and evaluates the model with best hyperparameter configuration

        To be used when train and test matrices are used to create the instance

        :param root_dir: directory where the results will be saved
        :param experiment_designation: base name of the files to be stored
        :param n_iter: number of iterations to be executed in random search in the hyperparameter optimization
        :param cv: number of folds to be used in the cross-validation
        :return: None
        """
        file_name = experiment_designation
        self.model_selection(self.X_train.values, self.y_train.values, n_iter, cv)
        self.write_model_selection_results(root_dir, file_name)
        self.select_best_model()
        self.create_DNN_model()
        # X_train, X_valid, y_train, y_valid = train_test_split(self.X_train, self.y_train, test_size=0.25, stratify=self.y_train)
        # self.fit_model(X_train=X_train.values, X_test=X_valid.values, y_train=y_train.values, y_test=y_valid.values)
        self.fit_model_noval(X_train=self.X_train.values, y_train=self.y_train.values)
        scores = self.evaluate_model(X_test=self.X_test.values, y_test=self.y_test.values)
        self.write_report(scores,root_dir,experiment_designation)
        self.save_best_model()

    def multi_model_selection_cv(self, root_dir, experiment_designation, n_iter=100, cv=10):
        """
        Performs a random hyperparameters optimization with cross-validation, selects and evaluates the model with best hyperparameter configuration

        To be used when the whole dataset is used for fitting

        :param root_dir: directory where the results will be saved
        :param experiment_designation: base name of the files to be stored
        :param n_iter: number of iterations to be executed in random search in the hyperparameter optimization
        :param cv: number of folds to be used in the cross-validation
        :return: None
        """
        file_name = experiment_designation
        self.model_selection(self.X.values, self.y.values, n_iter, cv)
        self.write_model_selection_results(root_dir, file_name)
        self.select_best_model()
        self.create_DNN_model()
        cv_scores, cv_history, time_fit = self.cv_fit(self.X.values, self.y.values, cv)
        self.plot_model_performance(cv_history, root_dir, file_name)
        mean_scores, sd_scores, raw_scores = self.format_scores_cv(cv_scores)
        self.write_report_cv(mean_scores, sd_scores, raw_scores, root_dir, file_name)
        self.save_best_model()
