import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    RepeatedStratifiedKFold, 
    StratifiedKFold, 
    GridSearchCV,
    )
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import mode
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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
    Takes in dataframe of scores, returns dictionary with
    best parameter values. 
    
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


# the models we will be testing
models = {
    'lr': LogisticRegression(class_weight='balanced', max_iter=800),
    'rf': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, n_estimators=100),
    'svc': SVC(class_weight='balanced', random_state=42, kernel='linear'),
    'knn': KNeighborsClassifier()
}

model_param_grid = {
    'lr': {"logisticregression__C": [0.01, 0.1, 1.0, 10]},
    'rf': {"randomforestclassifier__max_depth": [2, 3, 5, 10]},
    'svc': {"svc__C": [0.01, 0.1, 1.0, 10]},
    'knn': {"kneighborsclassifier__n_neighbors": [2, 3, 4, 6]}
}

# choose one of rfe, selectk, or lin_pca for dimensionality reduction

fs_model_name = 'rfe'
fs_model = RFE(estimator=LogisticRegression(C=1, solver='liblinear', penalty='l2', max_iter=800))
fs_model_params = [10, 15, 20, 25]
fs_model_param_name = 'rfe__n_features_to_select'

#fs_model_name = 'selectk'
#fs_model = SelectKBest(score_func=f_classif)
#fs_model_params = [10, 15, 20, 25]
#fs_model_param_name = 'selectkbest__k'

# fs_model_name = 'lin_pca'
# fs_model = KernelPCA(kernel='linear')
# fs_model_params = [10, 15, 20, 25, 35]
# fs_model_param_name = 'kernelpca__n_components'

# results dataframe for later
results_df = pd.DataFrame()

# read in patient data
cases_csv = 'patient_endpoint_binary.csv'
cases = pd.read_csv(cases_csv)
cases = cases.loc[cases['Drop']!=1, :]
cases.drop(columns=['Drop', 'Progression free survival', 'Progression'], inplace=True)

# read in features data and merge with patient data
features_csv = 'features/all_features.csv'
features = pd.read_csv(features_csv)
data = pd.merge(cases, features, how='left', on='PatientID')

# split into X and y
X, y = data.drop(columns=['Binary_progression', 'PatientID']), data['Binary_progression'].astype(int)

# split up into folds
train_test_skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# feature count for feature analysis
feat_count = defaultdict(float)

trial = 1
for train_index, test_index in train_test_skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    ### TRAINING ###
    # Remove correlated features.
    # Can do this now, it hardly affects validation results
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_95 = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_train.drop(X_train[to_drop_95], axis=1, inplace=True)

    # Do it on test set now, why not
    X_test.drop(X_test[to_drop_95], axis=1, inplace=True)

    for model_name, model in models.items():
        # set up our pipeline
        pipe = make_pipeline(StandardScaler(),
                                fs_model,
                                model)
        param_key = list(model_param_grid[model_name].keys())[0]
        param_grid = model_param_grid[model_name]
        param_grid[fs_model_param_name] = fs_model_params

        # do some hyperparameter tuning
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
        grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
        grid_search.fit(X_train, y_train)
        print('done')

        results = pd.DataFrame(grid_search.cv_results_)[
            [
                "mean_test_score",
                'std_test_score',
                'param_' + param_key,
                "param_" + fs_model_param_name,
            ]
        ]
        results['std_test_score'] /= np.sqrt(5)  # make it the standard error
        results.rename(columns={'param_' + param_key: param_key, 
                                "param_" + fs_model_param_name: fs_model_param_name,
                                'std_test_score':'stderr'}, inplace=True)
                                
        if model_name=='knn':  # because greater k is less complex
            results[param_key] *= -1

        best_params = return_best(results, complex_col=param_key, feat_col=fs_model_param_name)
        if model_name=='knn':
            # knn cannnot have a float number of nearest neighbours
            best_params[param_key] = -1 * int(best_params[param_key])

        pipe.set_params(**best_params)

        ### TESTING ###

        # fit and predict
        col_names = X_train.columns.tolist()
        pipe.fit(X_train, y_train)
        y_preds = pipe.predict(X_test)

        # we want to exclude certain things from feature analysis
        if (model_name!='knn') & ((fs_model_name=='rfe') | (fs_model_name=='selectk')):
            chosen_feat_indices = pipe[1].get_support(indices=True)
            chosen_feats = [col_names[i] for i in chosen_feat_indices]

            # rf is special and we get importances instead of direct coefficients
            if model_name=='rf':
                coefs = pd.DataFrame(data=np.abs(pipe[2].feature_importances_.T), 
                                     index=chosen_feats, columns=['Coefficient'])
            else:
                coefs = pd.DataFrame(data=np.abs(pipe[2].coef_.T), 
                                     index=chosen_feats, columns=['Coefficient'])

            coefs /= coefs['Coefficient'].sum()  # normalize importances
            coefs.sort_values(ascending=False, inplace=True, by='Coefficient')

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

results_df.to_csv(f'results/tr_{fs_model_name}_ridge_results.csv', index=False)
for model_name in models.keys():
    model_results = results_df.loc[results_df['model']==model_name]
    mean_score = model_results['balacc'].mean()
    stderr_score = model_results['balacc'].std() / np.sqrt(5)

    f1_mean_score = model_results['f1'].mean()
    f1_stderr_score = model_results['f1'].std() / np.sqrt(5)

    print(f'{model_name}: Balcc {mean_score:.4f} +- {stderr_score:.4f}')
    print(f'{model_name}: f1 {f1_mean_score:.4f} +- {f1_stderr_score:.4f}')

# this part is only relevant if features were analyzed
with open(f'results/tr_{fs_model_name}_feats.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=feat_count.keys())
    writer.writeheader()
    writer.writerow(feat_count)

   




