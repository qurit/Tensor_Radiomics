
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    RepeatedStratifiedKFold, 
    StratifiedKFold, 
    PredefinedSplit,
    GridSearchCV
    )
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import defaultdict
import csv

def pool_reduction(pool_entries:pd.DataFrame, max_complex:float,
                   max_feats: float, complex_col='complexity', 
                   feat_col='features') -> dict:
    '''
    Function that chooses the best params from a pool of 
    params with scores.

    Prioritizes minimizing model complexity, and then minimizing
    the number of features.
    '''

    pool_min_complex = float(pool_entries[complex_col].min())

    if pool_min_complex > max_complex:
        best_params = {complex_col: max_complex,
                       feat_col: max_feats}

    else:
        pool_min_complex_entries = pool_entries.loc[pool_entries[complex_col]==pool_min_complex, :]
        if len(pool_min_complex_entries)==1:
            best_params = {complex_col: pool_min_complex,
                          feat_col: int(pool_min_complex_entries[feat_col])}
        # choose one with fewest features
        else:
            pool_min_feat = int(pool_min_complex_entries[feat_col].min())
            best_params = {complex_col: pool_min_complex,
                           feat_col: pool_min_feat}
    return best_params

def return_best(df:pd.DataFrame, complex_col='complexity',
                score_col='mean_test_score', stdev_col='stderr',
                feat_col='features') -> dict:
    '''
    Takes in dataframe of scores associated with params, returns 
    dictionary with best param values.

    Best parameter values chosen by taking pool of all the 
    parameter combos that give a score within one standard 
    error of the best score, and then choosing the least
    complex parameter combo.
    '''
    max_score = float(df[score_col].max())
    max_entries = df.loc[df[score_col]==max_score, :]

    if len(max_entries) == 1:
        max_entry_stdev = float(max_entries[stdev_col])
        max_entry_complex = float(max_entries[complex_col])
        max_entry_feats = int(max_entries[feat_col])

        # grab entries within one stdev
        lower_score = max_score - max_entry_stdev
        pool_entries = df.loc[df[score_col] >= lower_score, :]

        best_params = pool_reduction(pool_entries, max_score, max_entry_complex, 
                                     max_entry_feats, complex_col=complex_col,
                                     score_col=score_col, feat_col=feat_col)
    else:
        min_complex = max_entries[complex_col].min()
        min_complex_entries = max_entries.loc[max_entries[complex_col]==min_complex, :]
        if len(min_complex_entries)==1:
            ref_entry = min_complex_entries
            ref_entry_stdev = float(ref_entry[stdev_col])
            ref_entry_feats = int(ref_entry[feat_col])

            # grab entries within one stdev
            lower_score = max_score - ref_entry_stdev
            pool_entries = df.loc[df[score_col] >= lower_score, :]

            best_params = pool_reduction(pool_entries, max_score, min_complex, 
                                         ref_entry_feats, complex_col=complex_col,
                                         score_col=score_col, feat_col=feat_col)
        else:
            min_feat = int(min_complex_entries[feat_col].min())
            ref_entry = min_complex_entries.loc[min_complex_entries[feat_col]==min_feat, :]
            ref_entry_stdev = float(ref_entry[stdev_col])

                        # grab entries within one stdev
            lower_score = max_score - ref_entry_stdev
            pool_entries = df.loc[df[score_col] >= lower_score, :]

            best_params = pool_reduction(pool_entries, max_score, min_complex, 
                                         min_feat, complex_col=complex_col,
                                         score_col=score_col, feat_col=feat_col)

    return best_params

def make_fold(X_train, X_valid):
    """
    Helper for definng your own fold in scikit learn
    """
    train_fold = np.zeros(len(X_train)) - 1
    test_fold = np.zeros(len(X_valid))
    fold = np.concatenate([train_fold, test_fold], axis=0)

    return fold

def apply_pca(X_train:np.ndarray, X_valid:np.ndarray, col_names:list) -> tuple:
    """
    Applies PCA combination to feature set. Assumes there are 10 flavours.
    """
    pca_X_train = []
    pca_X_valid = []

    cycles = int(X_train.shape[1] / 10)
    indices = np.arange(10)
    pca = PCA(n_components=0.9)

    for i in range(cycles):
        category = col_names[indices[0]][0:-3]
        train_values = X_train[:, indices]
        valid_values = X_valid[:, indices]
        new_train_values = pca.fit_transform(train_values)
        new_valid_values = pca.transform(valid_values)

        # if you want to see explained variance n stuff
        #ratio = float(pca.explained_variance_ratio_)
        #print(f'{category}: {ratio:.3f}')
        #print(pca.components_)

        pca_X_train.append(new_train_values)
        pca_X_valid.append(new_valid_values)

        indices += 10

    X_train_array = np.concatenate(pca_X_train, axis=1)
    X_valid_array = np.concatenate(pca_X_valid, axis=1)

    return X_train_array, X_valid_array

def apply_lda(X_train:np.ndarray, y_train:np.ndarray, X_valid:np.ndarray, 
              col_names:list) -> tuple:
    """
    Applies LDA combination to feature set. Assumes there are 10 flavours.
    """
    lda_X_train = []
    lda_X_valid = []

    cycles = int(X_train.shape[1] / 10)
    indices = np.arange(10)
    lda = LinearDiscriminantAnalysis(n_components=1)

    for i in range(cycles):
        category = col_names[indices[0]][0:-3]
        train_values = X_train[:, indices]
        train_y_values = y_train.to_numpy()
        valid_values = X_valid[:, indices]
        new_train_values = lda.fit_transform(train_values, train_y_values)
        new_valid_values = lda.transform(valid_values)

        # if you want to see explained variance n stuff
        #ratio = float(pca.explained_variance_ratio_)
        #print(f'{category}: {ratio:.3f}')
        #print(pca.components_)

        lda_X_train.append(new_train_values)
        lda_X_valid.append(new_valid_values)

        indices += 10

    X_train_array = np.concatenate(lda_X_train, axis=1)
    X_valid_array = np.concatenate(lda_X_valid, axis=1)

    return X_train_array, X_valid_array

# choose between lda or pca
tr_type = 'lda'

# the models we will be testing
models = {
    'lr': LogisticRegression(class_weight='balanced', max_iter=800),
    'rf': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, n_estimators=150),
    'svc': SVC(class_weight='balanced', random_state=42, kernel='linear'),
    'knn': KNeighborsClassifier()
}

# tuning parameters for each model
model_param_grid = {
    'lr': {"logisticregression__C": [0.01, 0.1, 1.0, 10]},
    'rf': {"randomforestclassifier__max_depth": [2, 3, 5, 10]},
    'svc': {"svc__C": [0.01, 0.1, 1.0, 10]},
    'knn': {"kneighborsclassifier__n_neighbors": [2, 3, 4, 6]}
}

# feature selection methods being used, as well as their tuning parameters
# Choose between rfe or selectk

#fs_model_name = 'rfe'
#fs_model = RFE(estimator=LogisticRegression(C=1, solver='liblinear', penalty='l1', max_iter=800))
#fs_model_params = [10, 15, 20, 25]
#fs_model_param_name = 'rfe__n_features_to_select'

fs_model_name = 'selectk'
fs_model = SelectKBest(score_func=f_classif)
fs_model_params = [10, 15, 20, 25]
fs_model_param_name = 'selectkbest__k'

# results dataframe for later
results_df = pd.DataFrame()

# load our data
cases_csv = 'patient_endpoint_binary.csv'
cases = pd.read_csv(cases_csv)
cases = cases.loc[cases['Drop']!=1, :]
cases.drop(columns=['Drop', 'Progression free survival', 'Progression'], inplace=True)

# load features
haralick_feats_csv = 'features/haralick_features.csv'
shape_feats_csv = 'features/shape_features.csv'
firstorder_feats_csv = 'features/firstorder_features.csv'
haralick_feats = pd.read_csv(haralick_feats_csv)
shape_feats = pd.read_csv(shape_feats_csv)
firstorder_feats = pd.read_csv(firstorder_feats_csv)

haralick_data = pd.merge(cases, haralick_feats, how='left', on='PatientID')
# merge shape and firstorder features
shape_data = pd.merge(cases, shape_feats, how='left', on='PatientID')
shape_1st_data = pd.merge(shape_data, firstorder_feats, how='left', on='PatientID')

X, y = haralick_data.drop(columns=['Binary_progression', 'PatientID']), haralick_data['Binary_progression'].astype(int)
X_shape_1st = shape_1st_data.drop(columns=['Binary_progression', 'PatientID'])

# load features stuff for column names later
shape_1st_cols = X_shape_1st.columns.tolist()
haralick_cols = X.columns.tolist()[0:75] + X.columns.tolist()[-75:]# they are ordered 75 of a flavour, 75 of next flavour etc
all_cols = haralick_cols + shape_1st_cols
all_cols = pd.Series(all_cols)

# for feature analysis
feat_count = defaultdict(float)

# outer loop
train_test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

trial = 1
for train_index, test_index in train_test_skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_shape_1st_train, X_shape_1st_test = X_shape_1st.iloc[train_index], X_shape_1st.iloc[test_index]

    # inner loop
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    val_results_df = pd.DataFrame()

    ### VALIDATION ###  
    for sub_train_index, valid_index in skf.split(X_train, y_train):
        X_sub_train, X_valid = X_train.iloc[sub_train_index], X_train.iloc[valid_index]
        y_sub_train, y_valid = y_train.iloc[sub_train_index], y_train.iloc[valid_index]
        col_names = X_sub_train.columns.tolist()

        X_shape_1st_sub_train, X_shape_1st_valid = X_shape_1st_train.iloc[sub_train_index], X_shape_1st_train.iloc[valid_index]
        X_shape_1st_all = np.concatenate([X_shape_1st_sub_train.to_numpy(), X_shape_1st_valid.to_numpy()], axis=0)

        scale = StandardScaler()
        X_sub_train = scale.fit_transform(X_sub_train)
        X_valid = scale.transform(X_valid)

        if tr_type=='lda':
            X_sub_train_pca, X_valid_pca = apply_lda(X_sub_train, y_sub_train, X_valid, col_names)
        else:
            X_sub_train_pca, X_valid_pca = apply_pca(X_sub_train, X_valid, col_names)
        X_pca = np.concatenate([X_sub_train_pca, X_valid_pca], axis=0)
        X_all = np.concatenate([X_pca, X_shape_1st_all], axis=1)
        y_all = np.concatenate([y_sub_train, y_valid], axis=0)

        # Remove correlated features
        X_all_df = pd.DataFrame(X_all)
        corr_matrix = X_all_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_95 = [column for column in upper.columns if any(upper[column] > 0.95)]
        X_all_df.drop(X_all_df[to_drop_95], axis=1, inplace=True)
        X_all = X_all_df.to_numpy()

        # Now that we've cleaned up the data, we define our own fold
        test_fold = make_fold(X_sub_train_pca, X_valid_pca)
        ps = PredefinedSplit(test_fold)

        # Find best model parameters
        for model_name, model in models.items():
            # set up our pipelines
            pipe = make_pipeline(StandardScaler(),
                                 fs_model,
                                 model)
            param_key = list(model_param_grid[model_name].keys())[0]
            param_grid = model_param_grid[model_name]
            param_grid[fs_model_param_name] = fs_model_params

            # do some hyperparameter tuning
            grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=ps, scoring='balanced_accuracy')
            grid_search.fit(X_all, y_all)
            results = pd.DataFrame(grid_search.cv_results_)[
                [
                    "mean_test_score",
                    'param_' + param_key,
                    "param_" + fs_model_param_name,
                ]
            ]
            results.rename(columns={'param_' + param_key: param_key, 
                                    "param_" + fs_model_param_name: fs_model_param_name}, inplace=True)
            val_results_df = val_results_df.append(results)

    ### TESTING ###
    # clean up and format the train and test data
    col_names = X_train.columns.tolist()

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    if tr_type=='lda':
        X_train_pca, X_test_pca = apply_lda(X_train, y_train, X_test, col_names)
    else:
        X_train_pca, X_test_pca = apply_pca(X_train, X_test, col_names)

    X_all = np.concatenate([X_train_pca, X_shape_1st_train], axis=1)
    X_all_test = np.concatenate([X_test_pca, X_shape_1st_test], axis=1)

    # remove correlated features
    X_all_df = pd.DataFrame(X_all)
    corr_matrix = X_all_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_95 = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_all_df.drop(X_all_df[to_drop_95], axis=1, inplace=True)
    X_all = X_all_df.to_numpy()

    # which features were chosen?
    if tr_type=='lda':
        chosen_feats = all_cols.drop(to_drop_95).tolist()

    X_all_test_df = pd.DataFrame(X_all_test)
    X_all_test_df.drop(X_all_test_df[to_drop_95], axis=1, inplace=True)
    X_all_test = X_all_test_df.to_numpy()


    for model_name, model in models.items():
        pipe = make_pipeline(StandardScaler(),
                             fs_model,
                             model)
        param_key = list(model_param_grid[model_name].keys())[0]

        sub_results_df = val_results_df.loc[:, ['mean_test_score', param_key, fs_model_param_name]]
        sub_results_df.dropna(inplace=True)  # drop any rows that don't have value for that model
        sub_results_df_mean = sub_results_df.groupby([param_key, fs_model_param_name]).mean().reset_index()
        sub_results_df_stderr = sub_results_df.groupby([param_key, fs_model_param_name]).sem(ddof=0).reset_index()

        sub_results_df_mean['stderr'] = sub_results_df_stderr['mean_test_score'].copy()

        if model_name=='knn':  # because greater k is less complex
            sub_results_df_mean[param_key] *= -1

        best_params = return_best(sub_results_df_mean, complex_col=param_key, feat_col=fs_model_param_name)
        
        print(sub_results_df_mean)
        print(best_params)

        if model_name=='knn':
            # knn cannnot have a float number of nearest neighbours
            best_params[param_key] = -1 * int(best_params[param_key])

        pipe.set_params(**best_params)

        # fit and predict
        pipe.fit(X_all, y_train)
        y_preds = pipe.predict(X_all_test)

        # we want to exclude certain things from feature analysis
        if (model_name!='knn') & ((fs_model_name=='rfe') | (fs_model_name=='selectk')) & (tr_type!='pca'):  # knn doesnt have coef
            chosen_feat_indices = pipe[1].get_support(indices=True)
            final_chosen_feats = [chosen_feats[i] for i in chosen_feat_indices]

            if model_name=='rf':
                coefs = pd.DataFrame(data=np.abs(pipe[2].feature_importances_.T), 
                                     index=final_chosen_feats, columns=['Coefficient'])
            else:
                coefs = pd.DataFrame(data=np.abs(pipe[2].coef_.T), 
                                     index=final_chosen_feats, columns=['Coefficient'])

            coefs /= coefs['Coefficient'].sum()  # normalize importances
            coefs.sort_values(ascending=False, inplace=True, by='Coefficient')
            coefs = coefs.loc[coefs['Coefficient'] > 0, :]

            for index, row in coefs.iterrows():
                val = float(row.iloc[0])
                feat_count[index] += val

        print('\nTEST RESULTS: ')
        print(classification_report(y_test, y_preds, digits=3))
        balanced_acc = balanced_accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds)
        
        results_row = pd.DataFrame(
            {
            'model':[model_name], 'FS':[fs_model_name], 'model_param':[best_params[param_key]], 
            'FS_param':[best_params[fs_model_param_name]], 'balacc':[balanced_acc], 'f1':[f1],
            'trial':[trial]
            }
        )

        results_df = results_df.append(results_row)

    trial += 1

results_df.to_csv(f'results/tr_{tr_type}_bins_{fs_model_name}_results.csv', index=False)

# this part is only relevant if features were analzed
for model_name in models.keys():
    model_results = results_df.loc[results_df['model']==model_name]
    mean_score = model_results['balacc'].mean()
    stderr_score = model_results['balacc'].std() / np.sqrt(5)

    f1_mean_score = model_results['f1'].mean()
    f1_stderr_score = model_results['f1'].std() / np.sqrt(5)

    print(f'{model_name}: Balcc {mean_score:.4f} +- {stderr_score:.4f}')
    print(f'{model_name}: f1 {f1_mean_score:.4f} +- {f1_stderr_score:.4f}')

print(feat_count)
with open(f'results/tr_{tr_type}_{fs_model_name}_feats.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=feat_count.keys())
    writer.writeheader()
    writer.writerow(feat_count)

   




