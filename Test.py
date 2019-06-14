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
# from DNN_MT_bin import *
# from DNN_MT_all import *
from DNN_MT import *



""" CCLE """

# rna_seq = Preprocessing("CCLE_RNAseq_rsem_genes_tpm_20180929.txt","CCLE_metabolomics_20190502.csv",mt = False)
#
# rna_seq.load_data(2,"gene_id","citrate","CCLE_ID", cli_sep=",")
#
# X,y = rna_seq.split_dataset(split=0, normalize_method=None, filt_method="variance",variance = 250, test_size=0.3,stratify= False)
#
# shallow = Shallow_reg(X=X, y=y, cv=5)
#
# shallow.multi_model_selection_cv("CCLE","citrate_first_try",cv=3)


'''
Test with TCGA-BRCA
'''
# rna_seq = Preprocessing("geq_data.csv","metadata.csv")

# rna_seq.load_data(1,None,"diagnoses.vital_status","file_id", sep = ",",transpose = False, equal = False)

# rna_seq.nom_to_num()

# X_train,X_test,y_train,y_test = rna_seq.split_dataset(split=1,normalize_method = None,filt_method = "KBest",features=5000, variance =0.01,test_size=0.3, stratify = False)

'''


Test with METABRIC Dataset

º
# 
# # '''
rna_seq = Preprocessing("data_mRNA_median_Zscores.txt","clinical_data", mt=False)
#
rna_seq.load_data(2, "Hugo_Symbol", "THREEGENE", "PATIENT_ID")
#
# rna_seq.load_data(2, "Hugo_Symbol", ["ER_STATUS","THREEGENE"], "PATIENT_ID")
#
rna_seq.nom_to_num()
#
# X,y = rna_seq.split_dataset(split=0,normalize_method = None,filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)
X_train,X_test,y_train,y_test = rna_seq.split_dataset(split=1,normalize_method = None,filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = True)
#
#
parameters_batch = {
            'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'optimization': ['Adadelta', 'Adam', 'RMSprop'],
            'learning_rate': [0.015, 0.010, 0.005, 0.001,0.0001],
            'batch_size': [16, 32, 64, 128, 256],
            'nb_epoch': [100,150,200],
            'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
                                       [2500, 100], [2500, 500]],
            'units_in_input_layer': [5000],
            'early_stopping': [True,False],
            'patience': [80]
        }


dnn = DNN_multi(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,parameters_batch=parameters_batch,cv=5)

dnn.multi_model_selection("THREEGENE","First_try",n_iter=50,cv=3)
# shallow = Shallow_reg(X = X, y = y, cv=5)
#
# shallow.multi_model_selection_cv("OS_MONTHS_metabric","shallow_first_try",cv=5)
# endpoints = ["ER_STATUS","THREEGENE"]

# types = ["bin","multi"]

# dnn = DNN_MT(X = X, y = y, cv=3, parameters_batch = parameters_batch, endpoints= endpoints, types= types)
#
# dnn.best_model_selection("DNN_MT","ER_THREE_first",n_iter = 1, cv = 3)



#
# endpoints = ["ER_STATUS","HER2_STATUS","GRADE"]
#
# types = ['bin','bin','multi']
#
# dnn_mt = DNN_MT(X = X, y = y, cv= 5, parameters_batch= parameters_batch, types = types, endpoints = endpoints)

# dnn_mt.insert_endpoints(endpoints)

# dnn_mt.insert_loss_weights([0.5,1.0])

# dnn_mt.multiple_y(endpoints)

# dnn_mt.best_model_selection(n_iter= 2,cv=3)


# dnn_mt.model_selection()

# cv_results = dnn_mt.cv_fit(dnn_mt.X,dnn_mt.y,n_folds=5)

# print(cv_results)

# X_train,X_test,y_train,y_test = rna_seq.split_dataset(split=1,normalize_method = "standard",filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)



'''

Test with Neuroblastoma dataset

'''
# rna_seq = Preprocessing("GSE49711_SEQC_NB_MAV_T_log2.20121127.txt","Clinical_data.txt",mt = False)

# rna_seq = Preprocessing("GSE49711_SEQC_NB_MAV_T_log2.20121127.txt","Clinical_data.txt",mt = True)
#
# # rna_seq.read_exprs_data("SEQC_", "#Gene")
#
# # rna_seq.read_clinical_data("Sex_Imputed","NB ID")
#
# rna_seq.load_data(4, "#Gene", "INSS_Stage","NB ID",equal=False)
#
# rna_seq.load_data(4, "#Gene", ["Sex_Imputed","INSS_Stage"],"NB ID",equal=False)
#
# rna_seq.nom_to_num()
#
# X,y = rna_seq.split_dataset(split=0,normalize_method = "standard",filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)
#
# parameters_batch = {
#             'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#             'optimization': ['Adadelta', 'Adam', 'RMSprop'],
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
# endpoints = ["Sex_Imputed","INSS_Stage"]
# #
# types = ['bin','multi']
# #
# dnn_mt = DNN_MT(X = X, y = y, cv= 3, parameters_batch= parameters_batch,endpoints=endpoints, types = types)
# dnn_mt = DNN_multi(X = X, y = y, cv= 3, parameters_batch= parameters_batch)
# dnn_mt.multi_model_selection_cv("Neurobl_multi","Neuroblastoma_multi_INSS",n_iter=1,cv=3)

# #
# dnn_mt.multiple_y(endpoints)
#
# dnn_mt.best_model_selection(n_iter= 2, cv=3)

# dnn_mt.model_selection()

# cv_results = dnn_mt.cv_fit(dnn_mt.X,dnn_mt.y, n_folds=5)

# dnn_mt.model_selection(n_iter = 3 , cv = 3)

# print(dnn_mt.model_selection_history)

# dnn_mt.set_new_parameters()

# dnn_mt.create_DNN_model()

# X_train,X_val,y_train,y_val = dnn_mt.val_data_generator(X,y)

# y_train_splt = dnn_mt.y_splitter(y_train)

# y_val_splt = dnn_mt.y_splitter(y_val)

# dnn_mt.fit_model(X_train,X_val,y_train_splt,y_val_splt)



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
# X,y= rna_seq.split_dataset(split=0, features=5000, test_size=0.3, stratify = False)

# shallow_bin = Shallow_bin(X=X, y=y, cv=5)

# shallow_bin.multi_model_selection_cv("Shallow_results","first_try_bin")

# shallow_multi = Shallow_multi(X=X, y=y, cv=5)

# shallow_multi.multi_model_selection_cv("Metabric_Intclust_Shallow","first_try_multi")

# shallow_reg = Shallow_reg(X=X, y=y, cv=5)

# shallow_reg.multi_model_selection_cv("Shallow_results_reg","first_try_reg")

# parameters_batch = {
#             'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#             # 'output_activation': ['sigmoid'],
#             'optimization': ['Adadelta', 'Adam', 'RMSprop'],
#             'learning_rate': [0.015, 0.010, 0.005, 0.001],
#             'batch_size': [16, 32, 64, 128, 256],
#             'nb_epoch': [120,200],
#             'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
#                                        [2500, 100], [2500, 500]],
#             'units_in_input_layer': [5000],
#             'early_stopping': [True],
#             'patience': [80]
#         }


# dnn_bin = DNN_bin(X= X, y = y, cv = 3, parameters_batch = parameters_batch)

# dnn_bin.multi_model_selection_cv("DNN_results_bin","first_try_bin",1,3)

# dnn_multi = DNN_multi(X= X, y = y, cv = 5, parameters_batch = parameters_batch)

# dnn_multi.multi_model_selection_cv("Metabric_DNN_Intclust","first_try_multi_cv",1,5)

# dnn_reg = DNN_reg(X= X, y = y, cv = 3, parameters_batch = parameters_batch)

# dnn_reg.multi_model_selection_cv("DNN_results_reg","first_try_reg_cv",1,3)


# X_train, X_test, y_train, y_test = rna_seq.split_dataset(split=1,normalize_method="standard",filt_method="mse", features=5000,variance=1, test_size=0.3, stratify = False)

# shallow_bin = Shallow_bin(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_bin.multi_model_selection("Shallow_results_bin","NB_bin")

# shallow_bin_2 = Shallow_bin(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_bin_2.load_model("KNN",1)


# shallow_multi = Shallow_multi(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_multi.multi_model_selection("Shallow_results_multi","NB_multi")

# shallow_reg = Shallow_reg(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv=5)

# shallow_reg.multi_model_selection("Shallow_results_reg","NB_reg")




# dnn_bin = DNN_bin(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv = 3, parameters_batch = parameters_batch)

# dnn_bin.multi_model_selection("DNN_results_bin","NB",1,3)


# dnn_multi = DNN_multi(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv = 3, parameters_batch = parameters_batch)

# dnn_multi.multi_model_selection("DNN_results_multi","NB_multi",1,3)


# dnn_reg = DNN_reg(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, cv = 3, parameters_batch = parameters_batch)

# dnn_reg.multi_model_selection("DNN_results_reg","NB_reg",1,3)
