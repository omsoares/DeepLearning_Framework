# from NBHighThroughput import *
from external_functions import *
from numpy import *
from random import choice
import matplotlib
matplotlib.use('Agg')
import time
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, r2_score, precision_score, recall_score, \
    log_loss, mean_squared_error, mean_absolute_error
import os
from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())


class DNN_MT:
    """
    This is class is to be used for generating deep learning models for multi-task classification problems.

    The number of folds to be applied in cross-validation should be inputed in the "cv" argument

    The X and y data can be provided or X_train,X_test,y_train,y_test instead.

    An argument with a batch of hyperparameters can be provided

    the argument "labels" must be given as a list of column names
    the argument "type" must be given as a list ordered by the problem type of each label
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
            'output_activation': 'sigmoid',
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
        # model selection parameters
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
            assert kwargs['cv'] >= 2, 'cv value must be at least 2.'
            self.cv = kwargs['cv']
        if self.splitted == True:
            self.feature_number = self.X_train.shape[1]
        elif self.splitted == False:
            self.feature_number = self.X.shape[1]
        # self.parameters_batch = kwargs["parameters_batch"]
        self.types = kwargs["types"]
        self.loss_weights = None
        self.labels = kwargs["labels"]
        self.labels_number = len(self.labels)

    def insert_labels(self,labels):
        """
        Changes the list with the names of the columns with labels
        :param labels: list with the names of the columns with labels
        :return: None
        """
        self.labels = labels
        self.labels_number = len(labels)

    def insert_loss_weights(self,loss_weights):
        """
        Inserts or changes the weighs that each loss function has in the fitting process
        :param loss_weights: list with loss weights
        :return: None
        """
        self.loss_weights = loss_weights


    def create_DNN_model(self):
        """
        Creates a Keras DNN architecture and compiles it
        :return: None
        """
        print("Creating DNN model")
        fundamental_parameters = ['dropout', 'optimization', 'learning_rate',
                                  'units_in_input_layer',
                                  'units_in_hidden_layers', 'nb_epoch', 'batch_size','batch_normalization']
        for param in fundamental_parameters:
            if self.parameters[param] == None:
                print("Parameter not set: " + param)
                return
        self.print_parameter_values()
        # Input layer
        input = Input(shape=(self.feature_number,), name = "inputs")
        # constructing all hidden layers
        for units,i in zip(self.parameters['units_in_hidden_layers'],range(len(self.parameters['units_in_hidden_layers']))):
            if i == 0:
                x = Dense(units,activation="relu")(input)
            else:
                x = Dense(units,activation="relu")(x)
            if self.parameters['batch_normalization'] == True:
                x = BatchNormalization()(x)
            x = Dropout(self.parameters['dropout'])(x)
        output_dict = {}
        output_list = []

        for type in enumerate(self.types):
            if type[1] == "bin":
                units = 1
                output_activation = "sigmoid"
            elif type[1] == "multi":
                if self.splitted == True:
                    units = len(self.y_train.iloc[:,type[0]].unique())
                elif self.splitted == False:
                    units = len(self.y.iloc[:,type[0]].unique())
                output_activation = "softmax"
            elif type[1] == "reg":
                units = 1
                # output_activation = "linear"
                output_activation = None
            output_name = self.labels[type[0]]
            output_dict[output_name] = Dense(units, activation=output_activation, name=output_name)(x)
            output_list.append(output_name)

        loss_types = {"bin":"binary_crossentropy", "multi":"sparse_categorical_crossentropy","reg": "mean_squared_error"}
        metric_types = {"bin":"accuracy","multi": "accuracy","reg":r2_keras}
        metrics_dict = {}
        loss_dict = {}

        for output,type in zip(output_list,self.types):
            loss_dict[output] = loss_types[type]
            metrics_dict[output] = metric_types[type]


        if self.parameters['optimization'] == 'SGD':
            optim = SGD(lr=self.parameters['learning_rate'])
        elif self.parameters['optimization'] == 'RMSprop':
            optim = RMSprop(lr= self.parameters['learning_rate'])
        elif self.parameters['optimization'] == 'Adam':
            optim = Adam()
        elif self.parameters['optimization'] == 'Adadelta':
            optim = Adadelta()

        model = Model(outputs = list(output_dict.values()),input=[input])
        if self.loss_weights != None:
            model.compile(optimizer=optim, loss=list(loss_dict.values()), metrics=list(metrics_dict.values()),loss_weights=self.loss_weights)
        else:
            model.compile(optimizer=optim, loss = list(loss_dict.values()),metrics = list(metrics_dict.values()))
        if self.verbose == 1: str(model.summary())
        self.model = model
        print("DNN model sucessfully created")

    def print_parameter_values(self):
        """
        Prints the hyperparameters used in the DNN
        :return: None
        """
        print("Hyperparameters")
        for key in sorted(self.parameters):
            print(key + ": " + str(self.parameters[key]))

    def write_cv_results(self,cv_means,cv_std,cv_results,root_dir, file_name):
        """
        Writes a txt report and a csv report with cross-validation results of the model
        :param cv_means: dictionary with mean of the cross-validation results
        :param cv_std: dictionary with the standard deviations of the cross-valiation results
        :param cv_results: dictionary with the raw scores of the cross-validation results
        :param root_dir: name of the directory where the results will be stored
        :param file_name: name of the file to be stored
        :return: None
        """
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        for label in self.labels:
            while os.path.exists(root_dir + file_name + "_" + str(label) + '_report_' + str(i) + '.txt'):
                i += 1
            final_path = os.path.join(root_dir, file_name + "_" + str(label) + '_report_' + str(i) + '.txt')
            print("Writing report file with path: " + final_path)
            out = open(final_path, 'w')
            out.write('=' * 25)
            out.write('\n')
            out.write('Hyperparameters')
            out.write('\n')
            for key in sorted(self.parameters):
                out.write(key + ": " + str(self.parameters[key]))
                out.write('\n')
            out.write('\n')
            out.write('Scores')
            out.write('\n')
            out.write("=" * 25)
            out.write('\n')
            out.write("\n")
            print("Cross validation results:")
            for mean,std in zip(cv_means.keys(),cv_std.keys()):
                if mean == std:
                    if label in mean:
                        print(str(mean) + ": " + str(cv_means[mean]) + " +/- " + str(cv_std[std]))
                        out.write(str(mean) + ": " + str(cv_means[mean]) + " +/- " + str(cv_std[std]))
                        out.write('\n')
            out.close()
            df = pd.DataFrame.from_dict(cv_results)
            cv_df_path = os.path.join(root_dir, file_name + str(label) + '_cv_results_' + str(i) + '.csv')
            print('Writing csv file with path: ', cv_df_path)
            df_f = df.transpose()
            df_f.to_csv(cv_df_path, sep='\t', header= False)
        print("Report files successfully written.")

    def write_hold_out_results(self,scores,root_dir, file_name):
        """
        Writes a txt report hold-out results of the model
        :param scores: dictionary with results of the different metrics
        :param root_dir: name of the directory where the results will be stored
        :param file_name: name of the file to be stored
        :return: None
        """
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + file_name + '_report_' + str(i) + '.txt'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_report_' + str(i) + '.txt')
        print("Writing report file with path: " + final_path)
        out = open(final_path, 'w')
        out.write('=' * 25)
        out.write('\n')
        out.write('Hyperparameters')
        out.write('\n')
        for key in sorted(self.parameters):
            out.write(key + ": " + str(self.parameters[key]))
            out.write('\n')
        out.write('\n')
        out.write('Scores')
        out.write('\n')
        out.write("=" * 25)
        out.write('\n')
        out.write("\n")
        print("Hold out results:")
        for end in self.labels:
            out.write(str(end))
            out.write("\n")
            out.write("\n")
            for metric in scores.keys():
                if end in metric:
                    out.write(str(metric) + ": " + str(scores[metric]))
                    out.write('\n')
            out.write("\n")
        out.close()
        print("Report file successfully written.")


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

    def evaluate_model_selection(self, X_test, y_test):
        """
        Performs model evaluation for using during model selection
        :param X_test: Dataset with input test data
        :param y_test: Dataset with output test data (labels)
        :return: dictionary with evaluation results
        """
        # print("Evaluating model with hold out test set.")
        y_pred = self.model.predict(X_test)
        scores = dict()
        for i in range(self.labels_number):
            if self.types[i] == "bin":
                y_pred_v1 = [float(np.round(x)) for x in y_pred[i]]
                y_pred_v2 = np.ravel(y_pred_v1)
                scores[str(self.labels[i])] = accuracy_score(y_test[i], y_pred_v2)
            elif self.types[i]  == "multi":
                y_pred_v1 = [int(np.argmax(x)) for x in y_pred[i]]
                y_pred_v2 = np.ravel(y_pred_v1)
                scores[str(self.labels[i])] = accuracy_score(y_test[i], y_pred_v2)
            elif self.types[i] == "reg":
                scores[str(self.labels[i])] = r2_score(y_test[i], y_pred[i])
        for metric, score in scores.items():
            print(metric + ': ' + str(score))
        return scores

    def evaluate_model(self, X_test, y_test):
        """
        Performs model evaluation with multiple metrics
        :param X_test: Dataset with input test data
        :param y_test: Dataset with output test data (labels)
        :return: dictionary with evaluation results
        """
        # print("Evaluating model with hold out test set.")
        y_pred = self.model.predict(X_test)
        scores = dict()
        for i in range(self.labels_number):
            if self.types[i] == "bin":
                y_pred_v1 = [float(np.round(x)) for x in y_pred[i]]
                y_pred_v2 = np.ravel(y_pred_v1)
                scores[str(self.labels[i]) + "_accuracy"] = accuracy_score(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_roc_auc"] = roc_auc_score(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_f1_score"] = f1_score(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_mcc"] = matthews_corrcoef(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_precision"] = precision_score(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_recall"] = recall_score(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_log_loss"] = log_loss(y_test[i], y_pred_v2)
            elif self.types[i]  == "multi":
                y_pred_v1 = [int(np.argmax(x)) for x in y_pred[i]]
                y_pred_v2 = np.ravel(y_pred_v1)
                scores[str(self.labels[i]) + "_accuracy"] = accuracy_score(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_f1_score"] = f1_score(y_test[i], y_pred_v2,average="weighted")
                scores[str(self.labels[i]) + "_mcc"] = matthews_corrcoef(y_test[i], y_pred_v2)
                scores[str(self.labels[i]) + "_precision"] = precision_score(y_test[i], y_pred_v2,average="weighted")
                scores[str(self.labels[i]) + "_recall"] = recall_score(y_test[i], y_pred_v2,average="weighted")
            elif self.types[i] == "reg":
                scores[str(self.labels[i]) + "_r2"] = r2_score(y_test[i], y_pred[i])
                scores[str(self.labels[i]) + "_mse"] = mean_squared_error(y_test[i], y_pred[i])
                scores[str(self.labels[i]) + "_mae"] = mean_absolute_error(y_test[i], y_pred[i])
        for metric, score in scores.items():
            print(metric + ': ' + str(score))
        return scores

    def batch_parameter_shufller(self):
        """
        Creates a dictionary with a random choice of hyperaparameters
        :return: dictionary with random hyperparameters
        """
        chosen_param = {}
        for key in self.parameters_batch:
            chosen_param[key] = choice(self.parameters_batch[key])
        return chosen_param

    def set_new_parameters(self):
        """
        Sets a random set of hyperaparameters selected by batch_parameter_shufller
        :return: None
        """
        new_parameters = self.batch_parameter_shufller()
        dnn_parameters = {}
        for key in new_parameters:
            dnn_parameters[key] = new_parameters[key]
        self.parameters = dnn_parameters

    def y_splitter(self,y):
        """
        Separates the different types of labels and returns a list of lists
        :param y: group of different types of labels to be separated
        :return: a list of lists of different types of labels
        """
        y_list = []
        for i in range(y.shape[1]):
            y_list.append(y[:,i])
        return y_list


    def hold_out_fit(self,X_train,X_valid,y_train,y_valid):
        """
        Model fitting and evaluation using an hold out technique
        :param X_train: Dataset with input train data
        :param X_test: Dataset with input test data
        :param y_train: Dataset with output train data (labels)
        :param y_test: Dataset with output test data (labels)
        :return: dictionary with train and valid scores of hold out evaluation
        """
        res = {}
        self.create_DNN_model()
        y_train_splt = self.y_splitter(y_train.values)
        y_valid_splt = self.y_splitter(y_valid.values)
        self.fit_model(X_train, X_valid, y_train_splt, y_valid_splt)
        print("Train scores:")
        train_scores = self.evaluate_model_selection(X_train, y_train_splt)
        print("Validation scores:")
        valid_scores = self.evaluate_model_selection(X_valid, y_valid_splt)
        for label in self.labels:
            res[label + "_train_score: "] = train_scores[label]
            res[label + "_valid_score: "] = valid_scores[label]
        return res

    def fold_generator(self,n_folds,X):
        """
        Generate index to create folds for the cross validation
        :param n_folds: Number of folds to be created
        :param X: Dataset with input data
        :return: object to be used as folds in  the cross-validation
        """
        skf = KFold(n_splits=n_folds,shuffle=True)
        return skf.split(X)

    def cv_fit(self,X,y,n_folds = 3,shuffle = True,kf= None):
        """
        Performs a cross validation
        :param X: Dataset with input data
        :param y: Dataset with output data (labels)
        :param cv: number of folds to be used in the cv
        :param shuffle: boolean True if folds creation should be made randomly
        :param kf: object for the fold generation
        :return: three dictionaries with fitting evaluation scores means, standard deviations and raw results
        """
        # self.set_new_parameters()
        if kf == None:
            skf = KFold(n_splits=n_folds, shuffle=shuffle)
            kf = skf.split(X)
        self.create_DNN_model()
        init_weights = self.model.get_weights()
        cv_results = {}
        train_scores_kf = []
        valid_scores_kf = []
        cv_means = {}
        cv_std = {}
        i = 0
        for train, valid in kf:
            print("#" * 35)
            print("Running Fold " + str(i + 1) + str("/") + str(n_folds))
            X_train, X_valid = X.values[train], X.values[valid]
            y_train, y_valid = y.values[train], y.values[valid]
            y_train_splt = self.y_splitter(y_train)
            y_valid_splt = self.y_splitter(y_valid)
            self.model.set_weights(init_weights)
            self.fit_model(X_train, X_valid, y_train_splt, y_valid_splt)
            print("Train scores:")
            train_scores = self.evaluate_model_selection(X_train, y_train_splt)
            print("Validation scores:")
            valid_scores = self.evaluate_model_selection(X_valid, y_valid_splt)
            train_scores_kf.append(train_scores)
            valid_scores_kf.append(valid_scores)
            i += 1
        for label in self.labels:
            train_scores_list = []
            valid_scores_list = []
            for i in range(n_folds):
                train_scores_list.append(train_scores_kf[i][label])
                valid_scores_list.append(valid_scores_kf[i][label])
            cv_means[label + '_train_score_mean'] = np.mean(train_scores_list)
            cv_means[label + '_valid_score_mean'] = np.mean(valid_scores_list)
            cv_std[label + '_train_score_std'] = np.std(train_scores_list)
            cv_std[label + '_valid_score_std'] = np.std(valid_scores_list)
            cv_results[label] = valid_scores_list
        return cv_means,cv_std,cv_results

    def cv_fit_results(self,X,y,n_folds = 3,shuffle = True,kf= None):
        """
        Performs a cross validation
        :param X: Dataset with input data
        :param y: Dataset with output data (labels)
        :param cv: number of folds to be used in the cv
        :param shuffle: boolean True if folds creation should be made randomly
        :param kf: object for the fold generation
        :return: a dictionary with cross validation results
        """
        # self.set_new_parameters()
        if kf == None:
            skf = KFold(n_splits=n_folds, shuffle=shuffle)
            kf = skf.split(X)
        self.create_DNN_model()
        init_weights = self.model.get_weights()
        valid_scores_kf = []
        i = 0
        for train, valid in kf:
            print("#" * 35)
            print("Running Fold " + str(i + 1) + str("/") + str(n_folds))
            X_train, X_valid = X.values[train], X.values[valid]
            y_train, y_valid = y.values[train], y.values[valid]
            y_train_splt = self.y_splitter(y_train)
            y_valid_splt = self.y_splitter(y_valid)
            self.model.set_weights(init_weights)
            self.fit_model(X_train, X_valid, y_train_splt, y_valid_splt)
            print("Validation scores:")
            valid_scores = self.evaluate_model(X_valid, y_valid_splt)
            valid_scores_kf.append(valid_scores)
            i += 1

        cv_results = {}
        for label in self.labels:
            for fold in valid_scores_kf:
                for metric in fold.keys():
                    if label in metric:
                        if metric in cv_results.keys():
                            cv_results[metric].append(fold[metric])
                        else:
                            cv_results[metric] = [fold[metric]]
        return cv_results

    def format_cv_results(self,cv_results):
        """
        Calculates the metric means and standard deviations from a dictionary
        :param cv_results: dictionary with the metric results of a cross-validation
        :return: two dictionaries with metric means and standard deviations, respectively
        """
        cv_means = {}
        cv_sd = {}
        for key in cv_results.keys():
            cv_means[key] = np.mean(cv_results[key])
            cv_sd[key] = np.std(cv_results[key])
        return cv_means,cv_sd


    def model_selection(self,X,y,n_iter=2,cv=3):
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
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=True)
        for iteration in range(n_iter):
            print("Iteration no. " + str(iteration + 1))
            new_parameters = self.batch_parameter_shufller()
            temp_values = new_parameters.copy()
            for key in new_parameters:
                self.parameters[key] = new_parameters[key]
            if cv <= 2:
                hold_out_res = self.hold_out_fit(X_train,X_valid,y_train,y_valid)
                for key in hold_out_res:
                    if "val" in key:
                        temp_values[key] = hold_out_res[key]
            elif cv > 2:
                kf = self.fold_generator(cv,X)
                cv_means,cv_std,cv_results = self.cv_fit(X,y,n_folds = cv,shuffle= True,kf=kf)
                for key in cv_means:
                    if "val" in key:
                        temp_values[key] = cv_means[key]
                for key in cv_std:
                    if "val" in key:
                        temp_values[key] = cv_std[key]
            self.model_selection_history.append(temp_values)
            K.clear_session()
            del self.history
            del self.model
        self.parameters = old_parameters.copy()
        print("Best DNN model successfully selected")

    def find_best_model(self):
        """
        Selects the model with best performance
        :return: dictionary with the hyperparameters of the model with best performance, value of the metric used for evaluating model performance
        """
        if self.model_selection_history:
            best_model = None
            max_val_score= 0
            for dic in self.model_selection_history:
                valid_values = []
                for key in dic:
                    if "valid_score_mean" in key:
                        valid_values.append(dic[key])
                val_score = np.mean(valid_values)
                if val_score > max_val_score:
                    best_model = dic
                    max_val_score = val_score
        # If all models have score 0 assume the first one
        if best_model is None:
            best_model = self.model_selection_history[0]
        return best_model

    def select_best_model(self):
        """
        Selects the model with best performance and updates the hyperparameter configuration
        :return: None
        """
        best_model = self.find_best_model()
        for key in self.parameters:
            if key in best_model.keys():
                self.parameters[key] = best_model[key]
        valid_values = []
        for key in best_model:
            if "valid_score_mean" in key:
                valid_values.append(best_model[key])
        val_score = np.mean(valid_values)
        print("Best model:")
        self.print_parameter_values()
        print("Average_val_score:" + str(val_score))

    def multi_model_selection(self,root_dir, file_name,n_iter=2,cv=2):
        """
        Performs a random hyperparameters optimization with cross-validation, selects and evaluates the model with best hyperparameter configuration

        To be used when train and test matrices are used to create the instance

        :param root_dir: directory where the results will be saved
        :param experiment_designation: base name of the files to be stored
        :param n_iter: number of iterations to be executed in random search in the hyperparameter optimization
        :param cv: number of folds to be used in the cross-validation
        :return: None
        """
        self.model_selection(self.X_train,self.y_train,n_iter=n_iter,cv=cv)
        self.select_best_model()
        self.create_DNN_model()
        # X_train_f,X_valid,y_train_f,y_valid = train_test_split(self.X_train,self.y_train,test_size=0.3,shuffle=True)
        y_train_splt = self.y_splitter(self.y_train.values)
        # y_valid_splt = self.y_splitter(y_valid.values)
        y_test_splt = self.y_splitter(self.y_test.values)
        # self.fit_model(X_train_f,X_valid,y_train_splt,y_valid_splt)
        self.fit_model_noval(X_train=self.X_train,y_train=y_train_splt)
        print("#" * 20)
        print("Train test results: ")
        scores = self.evaluate_model(self.X_test,y_test_splt)
        print("#" * 20)
        self.write_hold_out_results(scores, root_dir, file_name)
        self.save_best_model()

    def multi_model_selection_cv(self,root_dir, file_name,n_iter=2,cv=2):
        """
        Performs a random hyperparameters optimization with cross-validation, selects and evaluates the model with best hyperparameter configuration

        To be used when the whole dataset is used for fitting

        :param root_dir: directory where the results will be saved
        :param experiment_designation: base name of the files to be stored
        :param n_iter: number of iterations to be executed in random search in the hyperparameter optimization
        :param cv: number of folds to be used in the cross-validation
        :return: None
        """
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.3,shuffle=True)
        self.model_selection(X_train,y_train,n_iter=n_iter,cv=cv)
        self.select_best_model()
        self.create_DNN_model()
        cv_results = self.cv_fit_results(self.X,self.y,self.cv)
        cv_means,cv_std = self.format_cv_results(cv_results)
        self.write_cv_results(cv_means,cv_std,cv_results,root_dir, file_name)
        self.save_best_model()


    def save_best_model(self):
        """
        Saves the model with best performance in json and HDF5 files
        :return: None
        """
        print("Saving the best model")
        i = 0
        while os.path.exists("best_DNN_models/" + "DNN_mt" + str(i) + '.json'):
            i += 1
        root = "best_DNN_models"
        if not os.path.exists(root):
            os.makedirs(root)
        file_name_model = os.path.join(root, "DNN_mt" + str(i) + '.json')
        file_name_weights = os.path.join(root, "DNN_mt" + str(i) + '.h5')
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