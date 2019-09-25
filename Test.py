from Preprocessing import *
from Shallow_bin import *
from Shallow_multi import *
from Shallow_reg import *
from DNN_bin import *
from DNN_multi import *
from DNN_reg import *
from external_functions import *
import sys
from DNN_MT import *

# for argstr in sys.argv[1:]:
#     k, v = argstr.split('=')
#     print(k, v, argstr)
#     globals()[k] = v
#
# # label = label.split(',') if ',' in label else label
#
# print('Test label =', label)
# print('Model type =', model_type)
# print('Problem Type =', problem_type)
#
# parameters_batch = {
#     'dropout': [0.2, 0.3, 0.4, 0.5],
#     'optimization': ['Adadelta', 'Adam', 'RMSprop', 'SGD'],
#     'learning_rate': [0.015, 0.010, 0.005, 0.001, 0.0001],
#     'batch_size': [16, 32, 64, 128, 256],
#     'nb_epoch': [100, 150, 200],
#     'units_in_hidden_layers': [[2048, 1024, 512], [1024, 128], [2048, 1024, 512, 128], [2048, 128, 16],
#                                [2048, 128], [2048, 512]],
#     'units_in_input_layer': [5000],
#     'early_stopping': [True, False],
#     'patience': [80]
# }
# #
# if isinstance(label, list) and model_type == 'deep':
#
#     prepro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=True)
#     prepro.load_data(2, "Hugo_Symbol", label, "PATIENT_ID")
#     prepro.nom_to_num()
#     X, y = prepro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000, variance=0.01,
#                                  test_size=0.3, stratify=False)
#     labels = label
#     types = ["bin","multi"]
#     dnn = DNN_MT(X = X, y = y, cv=5, parameters_batch = parameters_batch, labels = labels, types= types)
#     dnn.best_model_selection("DNN_MT","DNN_MT_res",n_iter = 50, cv = 5)
#
# else:
#     if model_type == 'shallow':
#         if "bin" in problem_type:
#             prepro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=False)
#             prepro.load_data(2, "Hugo_Symbol", label, "PATIENT_ID")
#             prepro.nom_to_num()
#             X, y = prepro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000,
#                                          variance=0.01, test_size=0.3, stratify=True)
#             shallow = Shallow_bin(X=X, y=y, cv=5)
#             shallow.multi_model_selection_cv(label, "Shallow", cv=5)
#
#         elif "multi" in problem_type:
#             prepro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=False)
#             prepro.load_data(2, "Hugo_Symbol", label, "PATIENT_ID")
#             prepro.nom_to_num()
#             X, y = prepro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000,
#                                          variance=0.01, test_size=0.3, stratify=True)
#             shallow = Shallow_multi(X=X, y=y, cv=5)
#             shallow.multi_model_selection_cv(label, "Shallow", cv=5)
#
#         elif "reg" in problem_type:
#             prepro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=False)
#             prepro.load_data(2, "Hugo_Symbol", label, "PATIENT_ID")
#             X, y = prepro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000,
#                                          variance=0.01, test_size=0.3, stratify=False)
#             shallow = Shallow_reg(X=X, y=y, cv=5)
#             shallow.multi_model_selection_cv(label, "Shallow", cv=5)
#
#     elif model_type == 'deep':
#         if "bin" in problem_type:
#             prepro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=False)
#             prepro.load_data(2, "Hugo_Symbol", label, "PATIENT_ID")
#             prepro.nom_to_num()
#             X, y = prepro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000,
#                                         variance=0.01, test_size=0.3, stratify=True)
#             dnn = DNN_bin(X=X, y=y, parameters_batch=parameters_batch, cv=5)
#             dnn.multi_model_selection_cv(label, "DNN", n_iter=50, cv=5)
#
#         elif "multi" in problem_type:
#             prepro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=False)
#             prepro.load_data(2, "Hugo_Symbol", label, "PATIENT_ID")
#             prepro.nom_to_num()
#             X, y = prepro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000,
#                                         variance=0.01, test_size=0.3, stratify=True)
#             dnn = DNN_multi(X=X, y=y, parameters_batch=parameters_batch, cv=5)
#             dnn.multi_model_selection_cv(label, "DNN", n_iter=50, cv=5)
#
#         elif "reg" in problem_type:
#             prepro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=False)
#             prepro.load_data(2, "Hugo_Symbol", label, "PATIENT_ID")
#             X, y = prepro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000,
#                                         variance=0.01, test_size=0.3, stratify=False)
#             dnn = DNN_reg(X=X, y=y, parameters_batch=parameters_batch, cv=5)
#             dnn.multi_model_selection_cv(label, "DNN", n_iter=50, cv=5)
#     else:
#         print('The model type is invalid')

""" CCLE """

# rna_seq = Preprocessing("CCLE_RNAseq_rsem_genes_tpm_20180929.txt","CCLE_metabolomics_20190502.csv",mt = False)
#
# rna_seq.load_data(2,"gene_id","citrate","CCLE_ID", cli_sep=",")
#
# X,y = rna_seq.split_dataset(split=0, normalize_method=None, filt_method="variance",variance = 540, test_size=0.3,stratify= False)
#
# shallow = Shallow_reg(X=X, y=y, cv=5)
#
# shallow.multi_model_selection_cv("CCLE","citrate_first_try",cv=3)

# parameters_batch = {
#             'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#             'optimization': ['Adadelta', 'Adam', 'RMSprop'],
#             'learning_rate': [0.015, 0.010, 0.005, 0.001,0.0001],
#             'batch_size': [16, 32, 64, 128, 256],
#             'nb_epoch': [100,150,200],
#             'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
#                                        [2500, 100], [2500, 500]],
#             'units_in_input_layer': [5000],
#             'early_stopping': [True,False],
#             'patience': [80]
#         }
# dnn = DNN_reg(X = X, y = y, parameters_batch = parameters_batch, cv=5)
#
# dnn.multi_model_selection_cv('CCLE', 'First_try', n_iter = 1, cv=3)



'''


Test with METABRIC Dataset


# 
# # '''
# rna_seq = Preprocessing("data_mRNA_median_Zscores.txt","clinical_data", mt=False)
# #
# rna_seq.load_data(2, "Hugo_Symbol", "THREEGENE", "PATIENT_ID")
# #
# rna_seq.load_data(2, "Hugo_Symbol", ["ER_STATUS","THREEGENE"], "PATIENT_ID")
# #
# rna_seq.nom_to_num()
# #
# X,y = rna_seq.split_dataset(split=0,normalize_method = None,filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)
# # X_test = X.iloc[:750,:]
# # y_test = y.iloc[:750]
# # X_train,X_test,y_train,y_test = rna_seq.split_dataset(split=1,normalize_method = None,filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = False)
# #
# #
# parameters_batch = {
#             'dropout': [0.2, 0.3, 0.4, 0.5],
#             'optimization': ['Adadelta', 'Adam', 'RMSprop','SGD'],
#             'learning_rate': [0.015, 0.010, 0.005, 0.001,0.0001],
#             'batch_size': [16, 32, 64, 128, 256],
#             'nb_epoch': [100,150,200],
#             'units_in_hidden_layers': [[2048, 1024, 512], [1024, 128], [2048, 1024, 512, 128], [2048, 128, 16],
#                                        [2048, 128], [2048, 512]],
#             'units_in_input_layer': [5000],
#             'early_stopping': [True,False],
#             'patience': [80]
#         }

# dnn = DNN_reg(X=X, y=y,parameters_batch=parameters_batch,cv=5)
# dnn.multi_model_selection_cv("NPI","DNN",n_iter=1,cv=5)

# dnn = DNN_multi(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,parameters_batch=parameters_batch,cv=3)

# dnn.multi_model_selection("THREEGENE","First_try",n_iter=1,cv=3)
# shallow = Shallow_reg(X = X, y = y, cv=5)
# shallow.multi_model_selection_cv("NPI","Shallow",cv=5)
# labels = ["ER_STATUS","THREEGENE"]

# types = ["bin","multi"]

# dnn = DNN_MT(X = X, y = y, cv=5, parameters_batch = parameters_batch, labels = labels, types= types)

# dnn = DNN_MT(X_train = X_train,X_test = X_test,y_train = y_train, y_test = y_test, cv=3, parameters_batch = parameters_batch, labels = labels, types= types)
#
# dnn.multi_model_selection_cv("DNN_MT","ER_THREE_first",n_iter = 2, cv = 5)




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

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    micro = Preprocessing("data_mRNA_median_Zscores.txt", "clinical_data", mt=False)
    micro.load_data(2, "Hugo_Symbol","NPI", "PATIENT_ID")
    # micro.nom_to_num()
    # X, y = micro.split_dataset(split=0, normalize_method=None, filt_method="mse", features=5000, variance=0.01,
    #                              test_size=0.3, stratify=False)
    X_train,X_test,y_train,y_test= micro.split_dataset(split=1, normalize_method=None, filt_method='mse', features=5000, variance=0.01,
                                 test_size=0.25, stratify=False)


    parameters_batch = {
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'optimization': ['Adadelta', 'Adam', 'RMSprop', 'SGD'],
        'learning_rate': [0.015, 0.010, 0.005, 0.001, 0.0001],
        'batch_size': [16, 32, 64, 128, 256],
        'nb_epoch': [100, 150, 200],
        'units_in_hidden_layers': [[2048, 1024, 512], [1024, 128], [2048, 1024, 512, 128], [2048, 128, 16],
                                   [2048, 128], [2048, 512]],
        'units_in_input_layer': [5000],
        'early_stopping': [True, False],
        'patience': [80]
    }
    # labels = ["ER_STATUS","THREEGENE"]
    # types = ["bin","multi"]
    # dnn = DNN_reg(X=X, y=y, parameters_batch=parameters_batch, cv=5, labels = labels, types = types)
    # dnn.multi_model_selection_cv("DNN_MT", "V2", n_iter=1, cv=5)
    shallow = Shallow_reg(X_train = X_train, X_test = X_test, y_train = y_train,y_test = y_test,cv=5)
    shallow.multi_model_selection("NPI_V2","Shallow",cv=3)

if __name__=="__main__":
    main()