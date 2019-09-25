from Preprocessing import *
from external_functions import *
from numpy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer
mcc = make_scorer(matthews_corrcoef, greater_is_better=True)
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, \
    log_loss
import time
import os

accuracy = make_scorer(accuracy_score, greater_is_better=True)



class Shallow_multi:
    """
    This class is intended for performing model selection and testing of shallow models.

    For k-fold cross-validation with X and y matrices declare:

    For train/test split and cross-validation declare:


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
        self.list_models = ['knn', 'rf', 'lr','svm']
        self.model = None
        self.model_name = None
        self.scoring = accuracy
        #if validate_matrices(kwargs):
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
        print("Hyperparameters")
        for key in sorted(self.model.best_params_):
            print(key + ": " + str(self.model.best_params_[key]))

    def predict_values(self,X):
        y_pred = self.model.predict(X)
        return y_pred

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        scores = dict()
        scores['accuracy'] = accuracy_score(y_test, y_pred)
        scores['f1_score'] = f1_score(y_test, y_pred, average= 'weighted')
        scores['mcc'] = matthews_corrcoef(y_test, y_pred)
        scores['precision'] = precision_score(y_test, y_pred, average = 'weighted')
        scores['recall'] = recall_score(y_test, y_pred, average = 'weighted')
        for metric, score in scores.items():
            print(metric + ': ' + str(score))
        return scores


    def calculate_scores_cv(self, X, y, cv):
        print("Evaluating model with cross validation.")
        self.model = self.model.best_estimator_
        scores_cv_list = []
        skf = StratifiedKFold(n_splits=cv, shuffle=False)
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

    def save_best_model(self,type):
        print("Saving the best model for classificator: " + self.model_name)
        i = 0
        while os.path.exists("best_models/" + self.model_name + "_multi_" + str(i) + '.pkl'):
            i += 1
        root = "best_models"
        if not os.path.exists(root):
            os.makedirs(root)
        file_name = os.path.join(root, self.model_name + "_multi_" + str(i) + '.pkl')
        if type == "cv":
            joblib.dump(self.model, file_name, compress=1)
        elif type == "hold_out":
            joblib.dump(self.model.best_estimator_, file_name, compress=1)

    def load_model(self,model_name):
        file_name = "best_models\\" + model_name + '.pkl'
        print("Loading model from: " + file_name)
        self.model_name = model_name
        self.model = joblib.load(file_name)
        print("Model loaded successfully!")

    def write_cv_results(self, root_dir, file_name):
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
        # for key in sorted(self.model.best_params_):
        #     out.write(key + ": " + str(self.model.best_params_[key]))
        #     out.write('\n')
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

    def PCA(self):
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_valid)
        colors = ['r', 'b']
        markers = ['s', 'x']
        for l, c in zip(np.unique(self.y_train), colors, markers):
            plt.scatter(self.X_train[self.y_train == l, 0],
                        self.X_train[self.y_train == l, 1],
                        c=c, label=l, marker=m)
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.legend(loc='lower left')
            plt.show()

    def model_selection_svm(self, X, y, cv):
        svm = SVC()
        # param_range = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 50, 100, 150, 200, 500,
        #                750, 1000]
        param_range = [0.001,0.01,0.1,1,10,100,1000]
        param_grid = [{'C': param_range,
                       'kernel': ['linear']},
                      {'C': param_range,
                       'gamma': param_range,
                       'kernel': ['rbf']}]
        gs = GridSearchCV(estimator=svm, param_grid=param_grid, scoring=self.scoring, n_jobs=-1, cv=cv)
        gs = gs.fit(X, y)
        self.model = gs
        self.model_name = 'SVM'

    def model_selection_rf(self, X, y, cv):
        rf = RandomForestClassifier()
        n_estimators = [10, 50, 100, 200, 500]
        param_grid = [{'n_estimators': n_estimators}]
        gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=self.scoring, n_jobs=-1, cv=cv)
        gs.fit(X, y)
        self.model = gs
        self.model_name = 'RF'

    def model_selection_knn(self, X, y, cv):
        knn = KNeighborsClassifier()
        # creating odd list of K for KNN
        myList = list(range(1, 7))
        # subsetting just the odd ones
        neighbors = list(filter(lambda x: x % 2 != 0, myList))
        param_grid = [{'n_neighbors': neighbors,
                       'metric': ['euclidean', 'cityblock']}]
        gs = GridSearchCV(estimator=knn, param_grid=param_grid, scoring=self.scoring, n_jobs=-1, cv=cv)
        gs.fit(X, y)
        self.model = gs
        self.model_name = 'KNN'

    # def model_selection_lda(self, X, y, cv):
    #     LDA = LinearDiscriminantAnalysis()
    #     myList = list(range(1, 20))
    #     # subsetting just the odd ones
    #     n_components = list(filter(lambda x: x % 2 != 0, myList))
    #     param_grid = [{'n_components': n_components}]
    #     gs = GridSearchCV(estimator=LDA, param_grid=param_grid, scoring=self.scoring, n_jobs=10, cv=cv)
    #     gs.fit(X, y)
    #     self.model = gs
    #     self.model_name = 'LDA'

    def model_selection_lr(self, X, y, cv):
        LR = LogisticRegression()
        myList = list(range(1, 5))
        param_grid = {
            # 'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 50, 100, 150, 200, 500, 750,
                  # 1000],
            "C": [0.001,0.01,0.1,1,10,100,1000],
            'penalty': ['l1', 'l2']}
        gs = GridSearchCV(estimator=LR, param_grid=param_grid, scoring=self.scoring, n_jobs=10, cv=cv)
        gs.fit(X, y)
        self.model = gs
        self.model_name = 'LR'

    def multi_model_selection(self, root_dir, experiment_designation, cv=5):
        for model in self.list_models:
            print(model)
            model_call = getattr(self, "model_selection_" + model)
            # print(model_call)
            model_call(self.X_train.astype(np.float), self.y_train.astype(int), cv)
            # model_call(self.X_train, self.y_train, cv)
            file_name = model + '_' + experiment_designation
            self.print_parameter_values()
            # print(self.X_test)
            # print(self.y_test)
            # print(self.model_name)
            # print(self.model)
            scores = self.evaluate_model(self.X_test.astype(np.float), self.y_test.astype(int))
            # scores = self.evaluate_model(self.X_test, self.y_test)
            self.write_report(scores, root_dir, file_name)
            self.write_cv_results(root_dir, file_name)
            self.save_best_model("hold_out")

    def multi_model_selection_cv(self, root_dir, experiment_designation, cv=5):
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


