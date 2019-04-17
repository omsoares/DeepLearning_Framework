from Preprocessing import *
from Shallow_bin import *
from Shallow_multi import *
from Shallow_reg import *
from DNN_bin import *
from DNN_multi import *
from DNN_reg import *
from external_functions import *
import sys
from sklearn.datasets import load_breast_cancer, load_wine,load_diabetes
from DNN_MT_bin import *
from DNN_MT_all import *
from DNN_MT_simple import *


# rna_seq = HighThroughput("GSE49711_SEQC_NB_MAV_T_log2.20121127.txt","Clinical_data.txt")

# rna_seq.load_data()

# X,y = rna_seq.multi_task()

# dnn_mt_model_selection_cv(X,y,"DNN_mt","first_try",n_iter=1,cv=3)

'''
Test with TCGA-BRCA
'''
# rna_seq = Preprocessing("geq_data.csv","metadata.csv")

# rna_seq.load_data(1,None,"diagnoses.vital_status","file_id", sep = ",",transpose = False, equal = False)

# rna_seq.nom_to_num()

# X_train,X_test,y_train,y_test = rna_seq.split_dataset(split=1,normalize_method = None,filt_method = "KBest",features=5000, variance =0.01,test_size=0.3, stratify = True)

'''


Test with METABRIC Dataset

º
# 
# '''
# rna_seq = Preprocessing("data_mRNA_median_Zscores.txt","data_clinical_sample.txt", mt=True)
#
# # rna_seq.load_data("MB-", "Hugo_Symbol", "ER_STATUS", "PATIENT_ID")
#
#
# rna_seq.load_data(2, "Hugo_Symbol", ["ER_STATUS","TUMOR_SIZE"], "PATIENT_ID")
#
# rna_seq.nom_to_num(column="ER_STATUS")
#
# X,y = rna_seq.split_dataset(split=0,normalize_method = "standard",filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)
#
#
#
# parameters_batch = {
#             'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#             'optimization': ['SGD', 'Adam', 'RMSprop'],
#             'learning_rate': [0.015, 0.010, 0.005, 0.001],
#             'batch_size': [16, 32, 64, 128, 256],
#             'nb_epoch': [200],
#             'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
#                                        [2500, 100], [2500, 500]],
#             'units_in_input_layer': [5000],
#             'early_stopping': [True],
#             'patience': [80]
#         }
#
# endpoints = ["ER_STATUS","TUMOR_SIZE"]
# #
# types = ['bin','reg']
# #
# dnn_mt = DNN_MT_simple(X = X, y = y, cv= 3, parameters_batch= parameters_batch, types = types)
# #
# dnn_mt.multiple_y(endpoints)
#
# # dnn_mt.model_selection()
#
# cv_results = dnn_mt.cv_fit(dnn_mt.X,dnn_mt.y,n_folds=5)
#
# print(cv_results)

# X_train,X_test,y_train,y_test = rna_seq.split_dataset(split=1,normalize_method = "standard",filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)



'''

Test with Neuroblastoma dataset

'''
# rna_seq = Preprocessing("GSE49711_SEQC_NB_MAV_T_log2.20121127.txt","Clinical_data.txt",mt = False)

rna_seq = Preprocessing("GSE49711_SEQC_NB_MAV_T_log2.20121127.txt","Clinical_data.txt",mt = True)

#rna_seq.read_exprs_data("SEQC_", "#Gene")

# rna_seq.read_clinical_data("Sex_Imputed","NB ID")

# rna_seq.load_data(4, "#Gene", "INSS_Stage","NB ID",equal=False)

rna_seq.load_data(4, "#Gene", ["Sex_Imputed","INSS_Stage"],"NB ID",equal=False)

rna_seq.nom_to_num()

X,y = rna_seq.split_dataset(split=0,normalize_method = "standard",filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)

parameters_batch = {
            'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'optimization': ['SGD', 'Adam', 'RMSprop'],
            'learning_rate': [0.015, 0.010, 0.005, 0.001],
            'batch_size': [16, 32, 64, 128, 256],
            'nb_epoch': [120,200],
            'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
                                       [2500, 100], [2500, 500]],
            'units_in_input_layer': [5000],
            'early_stopping': [True],
            'patience': [80]
        }
#
endpoints = ["Sex_Imputed","INSS_Stage"]
#
types = ['bin','multi']
#
dnn_mt = DNN_MT_simple(X = X, y = y, cv= 3, parameters_batch= parameters_batch, types = types)
#
dnn_mt.multiple_y(endpoints)

# dnn_mt.model_selection()

cv_results = dnn_mt.cv_fit(dnn_mt.X,dnn_mt.y, n_folds=5)

print(cv_results)

# dnn_mt.set_new_parameters()

# dnn_mt.create_DNN_model()

# X_train,X_val,y_train,y_val = dnn_mt.val_data_generator(X,y)

# y_train_splt = dnn_mt.y_splitter(y_train)

# y_val_splt = dnn_mt.y_splitter(y_val)

# dnn_mt.fit_model(X_train,X_val,y_train_splt,y_val_splt)


'''
Test with breat cancer dataset NKI

# '''
# data = pd.read_csv("NKI_cleaned.csv", header = 0)
# X = data.iloc[:,16:]
#
# y_pre = data.loc[:,"timerecurrence"]

# y = nom_to_num(y_pre)
# X_train,X_test,y_train,y_test = train_test_split(X,y_pre,test_size = 0.3)

# shallow_bin_selection_split(X_train, X_test, y_train, y_test,"NKI", "shallow_bin")

# dnn_bin_selection_split(X_train,X_test,y_train,y_test,"NKI","dnn_bin",n_iter = 2, cv=5)

# shallow_multi_selection_split(X_train, X_test, y_train, y_test,"NKI", "shallow_multi")

# dnn_multi_selection_split(X_train,X_test,y_train,y_test,"NKI","dnn_multi",n_iter = 2, cv=3)

# shallow_reg_selection_split(X_train, X_test, y_train, y_test,"NKI", "shallow_reg")

# dnn_reg_selection_split(X_train,X_test,y_train,y_test,"NKI","dnn_reg",n_iter = 2, cv=3)

'''
Wisconsin dataset
'''


# X_pre,y_pre = load_breast_cancer(return_X_y=True)
#
# X = pd.DataFrame(X_pre)
# X_norm = normalize_data(X)
#
# y = pd.Series(y_pre)
#
# X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size = 0.3)
#
# shallow_bin_selection_split(X_train, X_test, y_train, y_test,"wisconsin", "shallow_bin")

# dnn_bin_selection_split(X_train,X_test,y_train,y_test,"wisconsin","dnn_bin",n_iter = 3, cv=5)

'''
Wine (multiclass classification)
'''
# X_pre,y_pre = load_wine(return_X_y=True)
#
# X = pd.DataFrame(X_pre)
# X_norm = normalize_data(X)

# y = nom_to_num(y_pre)

# X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size = 0.3)

# shallow_multi_selection_split(X_train, X_test, y_train, y_test,"wine", "shallow_multi")

# dnn_multi_selection_split(X_train,X_test,y_train,y_test,"wine","dnn_multi",n_iter = 3, cv=5)

'''
Diabetes dataset (regression)
'''
# X_pre,y_pre = load_diabetes(return_X_y=True)

# X = pd.DataFrame(X_pre)
# X_norm = normalize_data(X)

# y = pd.Series(y_pre)

# X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size = 0.3)

# shallow_reg_selection_split(X_train, X_test, y_train, y_test,"diabetes", "shallow_reg")

# dnn_reg_selection_split(X_train,X_test,y_train,y_test,"diabetes","dnn_reg",n_iter = 3, cv=5)

'''
Cancer reg (regression)
'''

# data = pd.read_csv("cancer_reg.csv")
#
# X_pre = data.drop(['binnedinc','geography','pctsomecol18_24','pctprivatecoveragealone'], axis = 1)
#
# X_pre = X_pre.dropna()
#
# y = X_pre.loc[:,'target_deathrate']
#
# X = X_pre.drop('target_deathrate',axis = 1)
#
# X_norm = normalize_data(X)
#
# X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size = 0.3)
#
# # shallow_reg_selection_split(X_train, X_test, y_train, y_test, "cancer", "shallow_reg")
#
# dnn_reg_selection_split(X_train,X_test,y_train,y_test,"cancer","dnn_reg",n_iter = 3, cv=5)

'''
Função que transforma dados nominais em dados numéricos
'''
# rna_seq.nom_to_num()


'''
Divisão dos dados e labels
'''

# X,y = rna_seq.split_dataset(split=0, features=5000, test_size=0.3)

# shallow_bin_selection_cv(X, y, "Shallow_results", "Second_try", cv=5)

# dnn_model_selection_cv(X, y, "DNN_results", "Second_try", n_iter=1, cv=3)

'''
Divisão em datasets de treino e teste
'''
# X,y= rna_seq.split_dataset(split=0, features=5000, test_size=0.3, stratify = True)

# shallow_bin = Shallow_bin(X=X, y=y, cv=5)

# shallow_bin.multi_model_selection_cv("Shallow_results","first_try_bin")

# shallow_multi = Shallow_multi(X=X, y=y, cv=5)

# shallow_multi.multi_model_selection_cv("Shallow_results_multi","first_try_multi")

# shallow_reg = Shallow_reg(X=X, y=y, cv=5)

# shallow_reg.multi_model_selection_cv("Shallow_results_reg","first_try_reg")

# parameters_batch = {
#             'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#             'output_activation': ['sigmoid'],
#             'optimization': ['SGD', 'Adam', 'RMSprop'],
#             'learning_rate': [0.015, 0.010, 0.005, 0.001],
#             'batch_size': [16, 32, 64, 128, 256],
#             'nb_epoch': [120,200],
#             'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
#                                        [2500, 100], [2500, 500]],
#             'units_in_input_layer': [5000],
#             'early_stopping': [True],
#             'patience': [80]
#         }
#

# dnn_bin = DNN_bin(X= X, y = y, cv = 3, parameters_batch = parameters_batch)

# dnn_bin.multi_model_selection_cv("DNN_results_bin","first_try_bin",1,3)

# dnn_multi = DNN_multi(X= X, y = y, cv = 3, parameters_batch = parameters_batch)

# dnn_multi.multi_model_selection_cv("DNN_results_multi","first_try_multi_cv",1,3)

# dnn_reg = DNN_reg(X= X, y = y, cv = 3, parameters_batch = parameters_batch)

# dnn_reg.multi_model_selection_cv("DNN_results_reg","first_try_reg_cv",1,3)


# X_train, X_test, y_train, y_test = rna_seq.split_dataset(split=1,normalize_method="standard",filt_method="KBest", features=2000,variance=1, test_size=0.3, stratify = True)

# shallow_bin = Shallow_bin(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_bin.multi_model_selection("Shallow_results_bin","TCGA-BRCA_bin")

# shallow_bin_2 = Shallow_bin(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_bin_2.load_model("KNN",1)


# shallow_multi = Shallow_multi(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_multi.multi_model_selection("Shallow_results_multi","TCGA-BRCA_multi")

# shallow_reg = Shallow_reg(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_reg.multi_model_selection("Shallow_results_reg","TCGA-BRCA_reg")


# shallow_bin_selection_split(X_train, X_test, y_train, y_test,"Shallow_results", "first_try_bin")

# shallow_multi_selection_split(X_train, X_test, y_train, y_test,"Shallow_results", "first_try_multi", cv=3)

# shallow_reg_selection_split(X_train, X_test, y_train, y_test,"Shallow_results", "first_try_reg", cv=3)


#
# import time
# args = sys.argv
# folder_prefix = time.strftime("%Y%m%d_%H%M%S")
# if len(sys.argv) > 1:
#     folder_prefix = args[1]

# dnn_bin_selection_split(X_train,X_test,y_train,y_test,"DNN_results_bin","first_try_bin",n_iter = 1, cv=3)

# dnn_bin = DNN_bin(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv = 3, parameters_batch = parameters_batch)

# dnn_bin.multi_model_selection("DNN_results_bin","TCGA-BRCA",1,3)


# dnn_multi_selection_split(X_train,X_test,y_train,y_test,"DNN_results_multi","first_try_multi",n_iter = 1, cv=3)

# dnn_multi = DNN_multi(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv = 3, parameters_batch = parameters_batch)

# dnn_multi.multi_model_selection("DNN_results_multi","TCGA-BRCA_multi",1,3)


# dnn_reg_selection_split(X_train,X_test,y_train,y_test,"DNN_reg_results","first_try_reg",n_iter = 1, cv = 3)

# dnn_reg = DNN_reg(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv = 3, parameters_batch = parameters_batch)

# dnn_reg.multi_model_selection("DNN_results_reg","TCGA-BRCA_reg",1,3)
