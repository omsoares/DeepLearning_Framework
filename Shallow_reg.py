from external_functions import *
from numpy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import matthews_corrcoef, make_scorer
mcc = make_scorer(matthews_corrcoef, greater_is_better=True)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
r2_metric = make_scorer(r2_score, greater_is_better= True)






class Shallow_reg:
    """
    This is class is to be used for generating traditional machine learning models for regression problems.

    The number of folds to be applied in cross-validation should be inputed in the "cv" argument

    The X and y data can be provided or X_train,X_test,y_train,y_test instead.

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
        self.list_models = ['en','rf','knn','svm']
        self.model = None
        self.model_name = None
        self.scoring = r2_metric
        if len(kwargs.keys())<=3:
            self.X = kwargs['X']
            self.y = kwargs['y']
            self.splitted = False
        elif len(kwargs.keys())>=4:
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



    def print_parameter_values(self):
        """
        Print the selected parameters after the GridSearchCV

        :return: None
        """
        print("Hyperparameters")
        for key in sorted(self.model.best_params_):
            print(key + ": " + str(self.model.best_params_[key]))

    def predict_values(self,X):
        """
        Return the predictions of the model for a given X input
        :param X: data used to generate the prediction
        :return: Return the predicted labels
        """
        y_pred = self.model.predict(X)
        return y_pred

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model performance by calculating different metrics using a test dataset

        :param X_test: Data to be used by the model to predict the labels
        :param y_test: Real labels
        :return: dictionary with the different metric scores.
        """
        y_pred = self.model.predict(X_test)
        scores = dict()
        scores['r2'] = r2_score(y_test, y_pred)
        scores['MAE'] = mean_absolute_error(y_test,y_pred)
        scores['MSE'] = mean_squared_error(y_test,y_pred)
        for metric, score in scores.items():
            print(metric + ': ' + str(score))
        return scores


    def calculate_scores_cv(self, X, y, cv):
        """
        Evaluates the model performance by calculating different metrics using cross validation

        :param X: Gene expression to be used to train and evaluate the model
        :param y: Clinical data to be used as label in model train and testing
        :param cv: number of cross-validation folds to be generated
        :return: Returns a dictionary with the different metrics for each fold
        """
        print("Evaluating model with cross validation.")
        #A testar
        self.model = self.model.best_estimator_
        scores_cv_list = []
        skf = KFold(n_splits=cv, shuffle=True)
        i=1
        for train, valid in skf.split(X, y):
            print("Fold: " + str(i))
            X_train, X_valid = X[train], X[valid]
            y_train, y_valid = y[train], y[valid]
            self.model.fit(X_train, y_train)
            scores_cv_list.append(self.evaluate_model(X_valid, y_valid))
            i+=1
        return scores_cv_list

    def format_scores_cv(self, scores_cv_list):
        """
        Formats the raw scores to calculate the mean and standard deviation

        :param scores_cv_list: dictionary with the metric values for each of the folds
        :return: Three dictionaries with mean scores, standard deviation scores and raw scores for each metric
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
        print("CV Results: ")
        for metric in mean_scores.keys():
            print(metric, ': ', str(mean_scores[metric]), ' +/- ', sd_scores[metric])
        return mean_scores, sd_scores, raw_scores

    def save_best_model(self,type):
        """
        Stores the model in a pkl format

        :param type: cv if cross-validation was applied, hold-out if it was evaluated with test dataset
        :return: None
        """
        print("Saving the best model for classificator: " + self.model_name)
        i = 0
        while os.path.exists("best_models/" + self.model_name + "_reg_" + str(i) + '.pkl'):
            i += 1
        root = "best_models"
        if not os.path.exists(root):
            os.makedirs(root)
        file_name = os.path.join(root, self.model_name + "_reg_" + str(i) + '.pkl')
        if type == "cv":
            joblib.dump(self.model, file_name, compress=1)
        elif type == "hold_out":
            joblib.dump(self.model.best_estimator_, file_name, compress=1)

    def load_model(self,model_name):
        """
        Loads a stored model

        :param model_name: file name of the model to be loaded
        :return: None
        """
        file_name = "best_models\\" + model_name + '.pkl'
        print("Loading model from: " + file_name)
        self.model_name = model_name
        self.model = joblib.load(file_name)
        print("Model loaded successfully!")


    def write_cv_results(self, root_dir, file_name):
        """
        Stores the results of the cross-validation in a csv file

        :param root_dir: directory where the file will be stored
        :param file_name: name of the file to be stored
        :return: None
        """
        # Create a file with parameters and results
        print("Saving file with results...")
        res = pd.DataFrame(self.model.cv_results_)
        resnew = res.join(pd.DataFrame(res["params"].to_dict()).T)
        resnew = resnew.filter(regex=('^(?!param)'))
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + file_name + str(i) + '.csv'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_' + str(i) + '.csv')
        print("Writing model selection file with path: " + final_path)
        resnew.to_csv(final_path, sep='\t')
        print("Model selection file successfully written.")


    def write_report(self, scores, root_dir, file_name):
        """
        Stores results of the evaluation using the test dataset

        :param scores: dictionary with the scores of the evaluation
        :param root_dir: directory where the file will be stored
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
        out.write('Best Model: ' + str(self.model_name))
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        out.write('Parameters:')
        out.write('\n')
        for key in sorted(self.model.best_params_):
            out.write(key + ": " + str(self.model.best_params_[key]))
            out.write('\n')
        out.write("=" * 25)
        out.write('\n')
        out.write('Scores')
        out.write('\n')
        for metric, score in scores.items():
            out.write(str(metric) + ': ' + str(score))
            out.write('\n')
        out.close()
        print("Report file successfully written.")

    def write_report_cv(self, mean_scores, sd_scores, raw_scores, root_dir, file_name):
        """
        Write the results of the cross validation evaluation

        :param mean_scores: dictionary with the mean scores
        :param sd_scores: dictionary with the standard deviations
        :param raw_scores: dictionary with raw scores
        :param root_dir: directory where the file will be stored
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
        out.write('Best Model: ' + str(self.model_name))
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        out.write('Parameters:')
        out.write('\n')
        for key in sorted(self.model.get_params()):
            out.write(key + ": " + str(self.model.get_params()[key]))
            out.write('\n')
        out.write("=" * 25)
        out.write('\n')
        out.write('Scores')
        out.write('\n')
        for metric, scores in mean_scores.items():
            out.write(str(metric) + ': ' + str(mean_scores[metric]) + ' +/- ' + str(sd_scores[metric]))
            out.write('\n')
        out.close()
        df = pd.DataFrame.from_dict(raw_scores)
        cv_df_path = os.path.join(root_dir, file_name + '_cv_results_' + str(i) + '.csv')
        print('Writing csv file with path: ', cv_df_path)
        df.to_csv(cv_df_path, sep='\t')
        print("Report file successfully written.")


    def model_selection_svm(self, X, y, cv):
        """
        SVM model training using GridSearchCV

        :param X: Input dataset to be used for model training
        :param y: Output dataset to be used for model training
        :param cv: number of folds to be used in the GridSearchCV
        :return: None
        """
        svm = SVR()
        param_range = [0.001,0.01,0.1,1,10,100,1000]
        param_grid = [{'C': param_range,
                       'kernel': ['linear']},
                      {'C': param_range,
                       'gamma': param_range,
                       'kernel': ['rbf']}]
        gs = GridSearchCV(estimator=svm, param_grid=param_grid, scoring=self.scoring, n_jobs=-1, cv=cv, verbose = 1)
        gs = gs.fit(X, y)
        self.model = gs
        self.model_name = 'SVM'

    def model_selection_rf(self, X, y, cv):
        """
        Random forest model training using GridSearchCV

        :param X: Input dataset to be used for model training
        :param y: Output dataset to be used for model training
        :param cv: number of folds to be used in the GridSearchCV
        :return: None
        """
        rf = RandomForestRegressor()
        n_estimators = [10, 50, 100, 200, 500]
        param_grid = [{'n_estimators': n_estimators}]
        gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=self.scoring, n_jobs=-1, cv=cv, verbose=1)
        gs.fit(X, y)
        print(gs.best_params_)
        self.model = gs
        self.model_name = 'RF'

    def model_selection_knn(self, X, y, cv):
        """
        K-nearest neighbor model training using GridSearchCV

        :param X: Input dataset to be used for model training
        :param y: Output dataset to be used for model training
        :param cv: number of folds to be used in the GridSearchCV
        :return: None
        """
        knn = KNeighborsRegressor()
        # creating odd list of K for KNN
        myList = list(range(1, 7))
        # subsetting just the odd ones
        neighbors = list(filter(lambda x: x % 2 != 0, myList))
        param_grid = [{'n_neighbors': neighbors,
                       'metric': ['euclidean', 'cityblock']}]
        gs = GridSearchCV(estimator=knn, param_grid=param_grid, scoring=self.scoring, n_jobs=-1, cv=cv, verbose=1)
        gs.fit(X, y)
        self.model = gs
        self.model_name = 'KNN'

    def model_selection_en(self, X, y, cv):
        """
        ElasticNet model training using GridSearchCV

        :param X: Input dataset to be used for model training
        :param y: Output dataset to be used for model training
        :param cv: number of folds to be used in the GridSearchCV
        :return: None
        """
        en = ElasticNet()
        param_grid = [{"fit_intercept":[True,False],"positive":[True,False],"selection":["cyclic","random"]}]
        gs = GridSearchCV(estimator=en, param_grid=param_grid, scoring=self.scoring, n_jobs=-1, cv=cv, verbose=1)
        gs.fit(X, y)
        self.model = gs
        self.model_name = 'ElasticNet'


    def multi_model_selection(self, root_dir, experiment_designation, cv=5):
        """
        Trains and selects the best model for each type of available model.

        The evaluation is performed using a test set

        :param root_dir: Directory where the models and reports will be stored
        :param experiment_designation: base name to be used in the generated models and results
        :param cv: number of folds to be used in the cross-validation process
        :return: None
        """
        for model in self.list_models:
            print(model)
            model_call = getattr(self, "model_selection_" + model)
            model_call(self.X_train.values, self.y_train.values, cv)
            file_name = model + '_' + experiment_designation
            self.print_parameter_values()
            scores = self.evaluate_model(self.X_test.values, self.y_test.values)
            self.write_report(scores, root_dir, file_name)
            self.write_cv_results(root_dir, file_name)
            self.save_best_model("hold_out")

    def multi_model_selection_cv(self,root_dir, experiment_designation,cv=5):
        """
        Trains and selects the best model for each type of available model.

        The evaluation is performed using cross-validation

        :param root_dir: Directory where the models and reports will be stored
        :param experiment_designation: base name to be used in the generated models and results
        :param cv: number of folds to be used in the cross-validation process
        :return: None
        """
        for model in self.list_models:
            print(model)
            model_call = getattr(self, "model_selection_" + model)
            model_call(self.X.values, self.y.values, cv)
            file_name = model + '_' + experiment_designation
            self.print_parameter_values()
            self.write_cv_results(root_dir, file_name)
            scores_cv_list = self.calculate_scores_cv(self.X.values, self.y.values, cv)
            mean, sd, raw = self.format_scores_cv(scores_cv_list)
            self.write_report_cv(mean, sd, raw, root_dir, file_name)
            self.save_best_model("cv")


