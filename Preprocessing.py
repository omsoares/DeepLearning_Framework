import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder


class Preprocessing:

    def __init__(self, expr_file, clinic_data_file,mt = False):
        self.expr_file = expr_file
        self.clinic_data_file = clinic_data_file
        self.exprs = None
        self.clinic_data = None
        self.n_features = None
        self.n_samples = None
        self.features = None
        self.X = None
        self.y = None
        self.mt = mt


    def read_exprs_data(self,column_index,gene_id,sep, transpose = True):
        """
        Reads high throughput expression data file.

        column_index corresponds to the index of the start of the columns corresponding to omics data

        gene_id corresponds to the string with name of columns with gene IDs

        """
        print("Reading High Throughput gene expression file...")
        exprs = pd.read_table(self.expr_file, header=0, sep=sep)
        # exprs_1 = exprs.filter(regex=(pat_tag)).transpose()
        exprs_1 = exprs.iloc[:,column_index:]
        if transpose:
            exprs_2 = exprs_1.transpose()
        elif transpose == False:
            exprs_2 = exprs_1
        if gene_id != None:
            exprs_2.columns = exprs[gene_id]
        self.exprs = exprs_2
        self.exprs = self.exprs.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        print("Expression data successfully load.")


    def read_clinical_data(self,cd_column,pat_id_column,sep):
        """
        Reads clinical data file.

        cd_column is the string of the column name corresponding to y values (values to be predicted)

        pat_id_column must be the string corresponding to the column with patient IDs

        """
        print("Reading clinical data file...")
        clinical_data = pd.read_csv(self.clinic_data_file, sep=sep, encoding="utf-8-sig")
        self.clinic_data = clinical_data
        # self.set_index(pat_id_column)
        self.clinic_data.index = self.clinic_data[pat_id_column]
        self.clinic_data = self.clinic_data.loc[:,cd_column]
        # self.clinic_data = self.clinic_data.reindex(self.exprs.index)
        # self.exprs = self.exprs[self.clinic_data.isna() == False]
        # self.clinic_data = self.clinic_data.dropna()
        print("Clinical data successfully load.")


    def set_feature_number(self):
        """
        Sets the number of features equal to the number of columns of expression matrix.

        """
        self.n_features = self.exprs.shape[1]

    def set_list_features(self):
        """
        Saves a list with the features of expression matrix.

        """
        self.features = list(self.exprs)

    def set_sample_number(self):
        """
        Sets the number of samples equal to the number of rows of expression matrix.

        """
        self.n_samples = self.exprs.shape[0]

    def set_index(self):
        """
            Sets the index for of the rows corresponding to patients ID

        """

        self.clinic_data = self.clinic_data.reindex(self.exprs.index)
        if self.mt == False:
            self.exprs = self.exprs[self.clinic_data.isna() == False]
        elif self.mt == True:
            # self.exprs = self.exprs[self.clinic_data.notna().any(axis=1) == True]
            self.exprs = self.exprs[self.clinic_data.isna().any(axis=1) == False]
        self.clinic_data = self.clinic_data.dropna()

    def nom_to_num(self,column = None):
        df = self.clinic_data.copy()
        lb_make = LabelEncoder()
        if self.mt == False:
            df_1 = pd.Series(lb_make.fit_transform(df), index= self.exprs.index)
        elif self.mt == True:
            if column == None:
                df_1 = df.apply(LabelEncoder().fit_transform)
                df_1.set_index(self.exprs.index)
            elif column != None:
                df[column] = lb_make.fit_transform(df[column])
                df_1 = df.set_index(self.exprs.index)
        self.clinic_data = df_1




    def load_data(self,column_index,gene_id, cd_column,pat_id_column,sep="\t",transpose = True, equal = True):
        """
        Runs previous functions to read all the data.

        If equal = True, then clinical data and expression data are reindex equally

        equal = False must be used when patient IDs are different between the clinical and expression data

        """
        self.read_exprs_data(column_index,gene_id,sep,transpose)
        self.read_clinical_data(cd_column,pat_id_column,sep)
        if equal:
            self.set_index()
        elif equal == False:
            self.exprs = self.exprs.set_index(self.clinic_data.index)
            self.set_index()
        self.set_feature_number()
        self.set_list_features()
        self.set_sample_number()

    def get_feature_number(self):
        return self.n_features

    def get_list_features(self):
        return self.features

    def get_sample_number(self):
        return self.n_samples


    def variance_filter(self,exprs, variance):
        """
        Filter the columns data by variance threshold.

        """
        print("Filtering by variance of ", variance)
        before = len(exprs.columns)
        variance_df = exprs.loc[:, (self.exprs.var() > int(variance))]
        after = len(variance_df.columns)
        print("Before: ", before)
        print("After: ", after)
        print("N feature filtered: ", before - after)
        return variance_df

    def mse_filter(self, exprs, num_mad_genes):
        """
        Determine most variably expressed genes and subset

        """
        print('Determining most variably expressed genes and subsetting')
        mad_genes = exprs.mad(axis=0).sort_values(ascending=False)
        top_mad_genes = mad_genes.iloc[0:num_mad_genes, ].index
        subset_df = exprs.loc[:, top_mad_genes]
        return subset_df

    def filter_genes(self, exprs, y, number_genes):
        """
        Filter top number_genes using sklearn SelectKBest with filter f_classif

        """
        print('Filtering top ' + str(number_genes) + ' genes.')
        filter = SelectKBest(score_func=f_classif, k=number_genes)
        rnaseq_filtered = filter.fit(exprs, y).transform(exprs)
        mask = filter.get_support()
        new_features = exprs.columns[mask]
        rnaseq_filtered_df = pd.DataFrame(rnaseq_filtered, columns=new_features, index=exprs.index)
        return rnaseq_filtered_df

    def normalize_zero_one(self, exprs):
        """
        Scale expression data using zero-one normalization

        """
        print('Zero one data normalization.')
        msc = MinMaxScaler()
        rnaseq_scaled_zeroone_df = msc.fit_transform(exprs)

        rnaseq_scaled_zeroone_df = pd.DataFrame(rnaseq_scaled_zeroone_df,
                                                columns=exprs.columns,
                                                index=exprs.index)
        return rnaseq_scaled_zeroone_df

    def normalize_data(self, exprs):
        """
        Scale expression data using StandardScaler normalization

        """
        print("Data normalization")
        rnaseq_scaled_df = StandardScaler().fit_transform(exprs)
        rnaseq_scaled_df = pd.DataFrame(rnaseq_scaled_df,
                                                columns=exprs.columns,
                                                index=exprs.index)
        return rnaseq_scaled_df

    def save_matrices_train_test(self, X_train, X_test, y_train, y_test, root, file_name):
        if not os.path.exists(root):
            os.makedirs(root)
        i = 0
        X_train_name = os.path.join(root, 'X_train' + file_name + '.csv')
        X_test_name = os.path.join(root, 'X_test' + file_name + '.csv')
        y_train_name = os.path.join(root, 'y_train' + file_name + '.csv')
        y_test_name = os.path.join(root, 'y_test' + file_name + '.csv')
        np.savetxt(X_train_name, X_train,fmt='%s')
        np.savetxt(X_test_name, X_test,fmt='%s')
        np.savetxt(y_train_name, y_train,fmt='%s')
        np.savetxt(y_test_name, y_test,fmt='%s')


    def split_dataset(self, split=1,normalize_method = "standard",filt_method = "mse",features=5000, variance =0.01,test_size=0.3, stratify = True):
        """

        :param  split = 0 - returns X,y not splited for k-fold cross-validation pipeline
                split = 1 - returns X_train, X_test, y_train, y_test and saves matrices


        :param features: number of final features (genes)

        :param test_size: test split factor

        :return: X, y or X_train, X_test, y_train, y_test matrices

        """
        print("Generating train and test datasets")
        X = self.exprs
        clinical_data = self.clinic_data
        y = clinical_data
        if normalize_method == "standard":
            X = self.normalize_data(X)
        elif normalize_method == "min_max":
            X = self.normalize_zero_one(X)
        if filt_method == "mse":
            X = self.mse_filter(X, features)
        elif filt_method == "variance":
            X = self.variance_filter(X, variance)
        elif filt_method == "KBest":
            X = self.filter_genes(X,y,features)
        if split == 0:
            print("Total: ", y.count())
            #print("M: ", y.value_counts()[0])
            #print("F: ", y.value_counts()[1])
            # y = y.values.astype(int)
            # X = X.values.astype(np.float)
            return X, y
        elif split == 1:
            # X = X.values.astype(np.float)
            # y = y.values.astype(int)
            if stratify == True:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            elif stratify == False:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            # Save *.csv files with stratified train/test subsets
            root = 'Train_test_matrices'
            file_name = "_matrix"
            self.save_matrices_train_test(X_train, X_test, y_train, y_test, root, file_name)
            return X_train, X_test, y_train, y_test

