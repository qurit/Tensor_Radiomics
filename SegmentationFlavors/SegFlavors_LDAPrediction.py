import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import SimpleITK as sitk
from scipy import stats
import radiomics
from radiomics import featureextractor
import logging
import matplotlib.pyplot as plt
import numpy as np
import os, time
import pandas as pd
import torchio as tio
import torch
from PIL import Image
from scipy.ndimage import zoom, rotate, convolve1d, map_coordinates, generate_binary_structure, binary_dilation, \
    binary_erosion, shift, binary_closing
from sklearn.model_selection import ParameterGrid
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries, find_boundaries

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.svm import SVC
from sklearn.utils import class_weight
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)


def freedman_diaconis(data, returnas="width"):
    """
  Parameters
  ----------
  data: np.ndarray
      One-dimensional array.

  returnas: {"width", "bins"}
      If "width", return the estimated width for each histogram bin.
      If "bins", return the number of bins suggested by rule.
  """
    data = np.asarray(data, dtype=np.float_)
    IQR = stats.iqr(data, nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    if returnas == "width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return (result)


def contour_randomization(image, mask):
    mask[mask > 0.0] = 1.
    # loop over the number of segments

    # apply SLIC and extract (approximately) the supplied number
    # of segments
    # N_p = nb of pixels in I_s.
    # A_px = Area of each pixel.
    N_p = np.sum(mask)
    A_px = 0.1 ** 2  # i think because of resampling, all of the pixel areas are due to be 1mm^2
    N_sx = np.ceil((N_p * A_px) / 0.01)
    segments = slic(image, n_segments=N_sx, sigma=1, compactness=0.01)

    new_mask = np.zeros_like(mask, dtype='uint8')
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        # print("[x] inspecting segment %d" % i)
        mask_sp = np.zeros(image.shape[:2], dtype="uint8")
        mask_sp[segments == segVal] = 1

        result = np.multiply(mask, mask_sp)
        if np.any(result):
            m_k = np.sum(mask_sp)
            m_eta_k = np.sum(result)
            eta_k = m_eta_k / m_k
            if 0.9 <= eta_k:
                new_mask += mask_sp
            if 0.2 > eta_k:
                pass
            if 0.2 <= eta_k < 0.9:
                x_k = np.random.uniform(0., 1.)
                if x_k <= eta_k:
                    new_mask += mask_sp

    st_elem = generate_binary_structure(2, 3)
    new_mask = binary_closing(new_mask, st_elem)
    return image, new_mask


def translation_randomization(image, eta):
    x_dir = np.random.uniform(0., 1.)
    y_dir = np.random.uniform(0., 1.)
    if x_dir <= 0.5:
        x_term = 1
    else:
        x_term = -1

    if y_dir <= 0.5:
        y_term = 1
    else:
        y_term = -1
    image_out = shift(image, np.array([y_term * eta, x_term * eta]), order=3)
    # mask_out = shift(mask, np.array([eta, eta]), order=1)
    # mask_out[mask_out > 0.33] = 1.
    return image_out  # , mask_out


def volume_adaptation(mask, tau):
    mask[mask > 0.0] = 1.

    vol = np.sum(mask)
    st_elem = generate_binary_structure(2, 3)

    vol_a = np.floor(vol * (1 + tau))

    mask_p = mask

    #  Two loops,
    mask_n = mask_p
    vol_n = np.sum(mask_n)
    threshold = 5
    while True:
        if tau > 0.:
            mask_n = binary_dilation(mask_p, st_elem).astype(float)
            vol_n = np.sum(mask_n)

            if vol_n <= 0.:
                break

            if vol_n > vol_a:
                break
            mask_p = mask_n

        if tau < 0.:
            mask_n = binary_erosion(mask_p, st_elem).astype(float)
            vol_n = np.sum(mask_n)

            if vol_n <= 0.:
                break

            if vol_n < vol_a:
                break

            mask_p = mask_n

        if tau == 0.:
            vol_n = np.sum(mask_n)
            break

    if vol_n != vol_a:
        vol_p = np.sum(mask_p)
        px_delta_N = np.abs(vol_a - vol_p)
        rim_Rr = np.logical_xor(mask_n, mask_p)
        rim_flat = rim_Rr.flatten()

        top = np.abs(vol_a - vol_p)
        patience = 5
        counter = 0
        while px_delta_N:
            sel_px = np.random.choice(rim_flat, size=mask_p.shape)
            sel_px = np.reshape(sel_px, rim_Rr.shape)

            px_in_rim = rim_Rr.astype(float) * sel_px.astype(float)

            vol_p = np.sum(mask_p)
            px_delta_N = np.abs(vol_a - vol_p)
            if top - px_delta_N <= 0:  # means we're going in the wrong direction...
                counter += 1
                if counter >= patience:
                    # print('patience reach... busting out of here')
                    break
            # print(px_delta_N)
            if px_delta_N > 0 and tau > 0.:
                mask_p = mask_p + px_in_rim
                mask_p[mask_p > 0.0] = 1.
            elif px_delta_N > 0 and tau < 0.:
                mask_p = mask_p - px_in_rim
                mask_p[mask_p > 0.0] = 1.
            elif tau == 0.:
                break
            if px_delta_N <= threshold:
                break
            top = px_delta_N

    return mask_p


def get_all_files(path_to):
    flist = []
    for (_, _, filenames) in os.walk(path_to):
        for f in filenames:
                flist.append(f)

    return flist


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def segmentation_and_image_perturbation(path_to_images, path_to_masks, images_and_labels_path, perturb_protcol_name="No_perturb"):
    """
    first step of Tensor Radiomics (TR) generation. Creates all new images and masks depending on the protcol.

    This method assumes all masks have been resampled to 1mm^2 px. This method has not be verified on voxels.

    I am aware that I am adapting the area of the segmentation and the "V" in the protcol implies a Volume. I am only
    maintaining naming conventions from the Zwanenberg et al. 2019 paper such that this could be adapted to volumetric
    data.

    Due to data ownership, we are unable to provide the original dataset for this study.

    :param path_to_images: str r"..\.\path\to\images\"
    :param path_to_masks: str r"..\.\path\to\masks\"
    :param labels_path: str, path to labels of images. Must be csv. must be binary, label column must be labeled as "Labels" and must contain
    matching column of image names that are contained in path_to_images.
    :param peturb_protcol:  str, name of the protocol you want to use. You can select from the following list. Default
    is "No_perturb" and will not perturbate the image+masks at all.
    :return:
    """

    seg_path = r"..\.\data\seg_perturbs\\"
    ensure_dir(seg_path)

    df_labels = pd.read_csv(images_and_labels_path)

    np.random.seed(42)
    # -------------- params -------------
    default_bin_width = 25
    mask_val = 1

    # available perturbations
    R = {'R': np.linspace(-13, 13, 27)}  # rotation. "R" modificiations Not implemented as of March 1st, 2022.
    N = {'N': [30]}
    T = {'T': [0.0, 0.333, 0.667]}
    V = {'V': np.linspace(-0.25, 0.25, 10)}
    C_only = {'C': [30]}
    RT = {'R': np.linspace(-6, 6, 4), 'T': [0.25, 0.75]}
    RNT = {'R': np.linspace(-6, 6, 4), 'N': 1, 'T': [0.25, 0.75]}
    RV = {'R': np.linspace(-6, 6, 4)}
    NTVC = {'N': 1, 'T': [0.25, 0.75], 'V': np.linspace(-0.2, 0.2, 5), 'C': [1, 2, 3, 4]}
    TVC = {'T': [0.25, 0.75], 'V': np.linspace(-0.2, 0.2, 5), 'C': [1]}

    No_perturb = {'No_perturb': [0.]}

    # Feel free to expand this section to include the perturbations of your own choice. I have merely
    # replicated the ones from the TR paper as of March 1st, 2022

    if perturb_protcol_name == 'No_perturb':
        seg_perturb_protocol = No_perturb
        seg_perturb_protocol_str = [perturb_protcol_name]
    elif perturb_protcol_name == "V":
        seg_perturb_protocol = V
        seg_perturb_protocol_str = [char for char in perturb_protcol_name]
    elif perturb_protcol_name == "TVC":
        seg_perturb_protocol = TVC
        seg_perturb_protocol_str = [char for char in perturb_protcol_name]
    else:
        print('YOU DID NOT SELECT A VIABLE SEG. PROTOCOL METHOD')
        return

    # seg_perturb_protocol_str = [char for char in perturb_protcol_name] # ['T', 'V', 'C']
    protocol_grid = ParameterGrid(seg_perturb_protocol)

    # this is assuming all masks and images have the same naming scheme and ordered in the dir the same way
    flist_img = get_all_files(path_to_images)
    flist_mask = get_all_files(path_to_masks)

    print(flist_img)
    results = pd.DataFrame()

    # outcome_values = df_labels[['Image', 'Label']]

    i = 0
    for f_img, f_mask in zip(flist_img, flist_mask):
        img_path = path_to_images + f_img
        img = Image.open(img_path)

        mask_path = path_to_masks + f_mask
        mask_orig = np.array(Image.open(mask_path))

        featureVector = pd.Series(dtype='float64')

        outcome_val_idx = df_labels[df_labels['Image']==f_img].index.values.astype(int)[0]
        outcome_val = df_labels['Label'].iloc[outcome_val_idx]  # there's gotta be a cleaner way to do this.

        ii = 0
        for protocol in protocol_grid:
            image_space = img
            mask_space = mask_orig
            mask_keep = mask_space
            protocol_sorted = {k: protocol[k] for k in seg_perturb_protocol_str}

            protocol_suffix = [f'{key}_{protocol_sorted[key]}' for key in protocol_sorted]
            print(f'Working on image# {f_img} for protocol {protocol_sorted}')
            for key in protocol_sorted:
                mask_space[mask_space > 0.] = 1.
                mask_keep = mask_space
                if key == 'C':
                    reps = protocol_sorted[key]
                    for contour_count in range(reps):
                        try:
                            # randomize contour. You can repeat this N times. I only do it once.
                            image_space, mask_space = contour_randomization(image_space, mask_space)
                        except:
                            pass

                        # print('contouring done.')
                        pass
                    pass
                elif key == 'T':
                    eta = protocol_sorted[key]
                    image_space = translation_randomization(image_space, eta)
                    # print('translation done')

                elif key == 'V':
                    # Adapt the "volume".
                    tau = protocol_sorted[key]
                    mask_space = volume_adaptation(mask_space, tau)
                    # print('area adaptation done.')
                elif key == 'N':
                    reps = protocol_sorted[key]  # this is more complicated...
                    pass
                elif key == 'R':
                    pass

                if np.sum(mask_space) == 0.:  # stop gap measure...
                    print('THIS STEP WAS A PROBLEM: ', key, protocol, i)
                    mask_space = mask_keep

            # print('DONE PROTOCOLS')
            # plt.figure()
            # plt.imshow(mask_space)
            # plt.show()
            # time.sleep(1)
            img_suffix = str(ii)
            mask_space = mask_space.astype(float)
            mask_space[mask_space > 0.] = mask_val
            # save new images here.
            if type(image_space) != type(mask_space):
                image_space = np.array(image_space)
            image_to_save = Image.fromarray(image_space)
            mask_to_save = Image.fromarray(mask_space.astype('uint8'))
            # if image_to_save.mode != 'RGB':
            #     image_to_save = image_to_save.convert('RGB')
            #     mask_to_save = mask_to_save.convert('RGB')  # this is providing empty masks????

            image_to_save_str = seg_path + f'IMAGE_{i}_{"".join(seg_perturb_protocol_str)}_{str(ii)}' + '.png'
            mask_to_save_str = seg_path + f'_MASK_{i}_{"".join(seg_perturb_protocol_str)}_{str(ii)}' + '.png'
            image_to_save.save(image_to_save_str)
            mask_to_save.save(
                mask_to_save_str)  # okay, the masks LOOK empty, but they're not. The range is just 0-1.
            time.sleep(0.1)  # needed my system to calm down during saving or else get a FileNotFound error

            # set up settings

            settings = {}
            settings['binWidth'] = default_bin_width
            settings['label'] = mask_val
            settings['resampledPixelSpacing'] = [1, 1]  # [3,3,3]
            settings['interpolator'] = sitk.sitkBSpline
            settings['force2D'] = True
            settings['shape2D'] = True


            binwidth = default_bin_width
            outputFilepath = f'.././results/features/' + f'_Features_{i}_' \
                                                         f'{"".join(seg_perturb_protocol_str)}_{str(ii)}.csv'

            ensure_dir(f'.././results/features/')

            imageFilepath = image_to_save_str
            maskFilepath = mask_to_save_str

            image_arr = np.array(Image.open(imageFilepath))
            mask_arr = np.array(Image.open(maskFilepath))

            # I don't know why your mask or image is empty, but something is wrong with it...
            if np.sum(mask_arr) == 0. or np.sum(image_arr) == 0.:
                featureVector = pd.Series(dtype='float64')  # blank slate
                featureVector['Image'] = imageFilepath
                featureVector['Mask'] = maskFilepath
                featureVector['Label'] = outcome_val

                featureVector.name = f"Image_{i}"
                results = results.join(featureVector, how='outer')  # this should toss an error
                continue

            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName('shape2D')
            extractor.enableFeatureClassByName('firstorder')
            extractor.enableFeatureClassByName('glcm')
            extractor.enableFeatureClassByName('glrlm')
            extractor.enableFeatureClassByName('glszm')
            extractor.enableFeatureClassByName('gldm')
            extractor.enableFeatureClassByName('ngtdm')

            # This is a pandas Series
            if ii == 0:
                featureVector['Image'] = os.path.basename(imageFilepath).split('.')[0]
                featureVector['Mask'] = os.path.basename(maskFilepath).split('.')[0]
                featureVector['Label'] = outcome_val
            # featureVector['binWidth'] = bin
            # feats = extractor.execute(imageFilepath, maskFilepath, label=255)
            image_sitk = sitk.ReadImage(image_to_save_str, sitk.sitkInt8)  # sitk is so stupid
            mask_sitk = sitk.ReadImage(mask_to_save_str, sitk.sitkInt8)
            feats = extractor.execute(image_sitk, mask_sitk, label=mask_val)
            feat_names = []
            new_names = []
            for key_name in feats.keys():
                feat_names.append(key_name)
                new_name = f'{"".join(seg_perturb_protocol_str)}_{str(ii)}_' + key_name
                new_names.append(new_name)

            for (key_name, new_name) in zip(feat_names, new_names):
                feats[new_name] = feats.pop(key_name)

            result = pd.Series(feats)
            featureVector = pd.concat([featureVector, result])  # featureVector.append(result)
            ii += 1


        featureVector.name = f'Image_{i}_{"".join(seg_perturb_protocol_str)}'
        # print(featureVector)
        df_fV = featureVector.to_frame()
        # results = results.merge(featureVector, on='Image', how='outer')
        results = pd.concat([results, df_fV], axis=1)
        # print('RESULTS: ', results)
        i += 1
        # print(results)
        # print('-----------------------------------------------------------------')
        # print(results)

    outputFilepath = f'.././results/features/SegmentationProtocol_' \
                     f'{"".join(seg_perturb_protocol_str)}_{len(protocol)}_all_feats.csv'
    results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
    return outputFilepath


def ml_prediction(features_path, mdl_name='LDA', nb_splits=5, min_feats=5, max_feats=10, scoring='roc_auc',
                  oversample=False):
    df_feats = pd.read_csv(features_path)
    df_feats = df_feats[df_feats.columns.drop(list(df_feats.filter(regex='diagnostic')))]
    pn_names = df_feats['Image']

    col_drop_list_shapefeats = []
    for col_name in df_feats.columns:
        if 'original_shape2D' not in col_name:
            continue

        if 'bin1' not in col_name:
            col_drop_list_shapefeats.append(col_name)

    df_feats = df_feats.drop(labels=col_drop_list_shapefeats, axis=1)

    cols_to_drop = ['Image', 'Mask']  # hmmmm drop binWidth or not?
    seed = 3000
    class_weight_str = 'balanced'

    # ==========================================================================================
    np.random.seed(seed)
    img_names = df_feats['Image']

    i = 0
    group_list = []

    prev_name = 'whatever'
    for name in img_names:
        if name != prev_name:
            i += 1
            prev_name = name
        group_list.append(i)

    group_arr = np.array(group_list)
    df_feats = df_feats.drop(cols_to_drop, axis=1)

    y = df_feats['Label'].values.astype(int)
    df_feats_in = df_feats.drop(['Label'], axis=1)

    # ======================== Other thing ==========
    kfold = KFold(n_splits=nb_splits)  # original work had to do GroupKFold splitting.
    scores = []
    preds_out = []
    y_out = []
    pred_probs_out = []
    p_scores = []
    pr_scores = []
    # =============== removing correlated features ============================
    before_drop = df_feats_in.shape[-1]

    corr_matrix = df_feats_in.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # Drop features
    df_feats_in.drop(df_feats_in[to_drop], axis=1, inplace=True)

    df_feats_in.to_csv('.././results/features/df_feats_in.csv', index=False)  # for inspection
    feats_names = df_feats_in.columns
    x = df_feats_in.values

    after_drop = df_feats_in.shape[-1]
    print('We dropped this many feats: %d and have %d remaining to train on' % (before_drop - after_drop, after_drop))

    cval = list(kfold.split(x, y, group_arr))
    mdl = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None, n_components=None,
                                     store_covariance=False, tol=0.0001)

    sfs1 = SFS(estimator=mdl,
               k_features=(min_feats, max_feats),
               forward=True,
               floating=True,
               scoring=scoring,
               cv=cval,
               n_jobs=1)

    print('Fitting SFS...')
    sfs1.fit(x, y)
    print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
    print(feats_names[np.array(sfs1.k_feature_idx_)])

    nSel = feats_names[np.array(sfs1.k_feature_idx_)]
    y_preds_all = []
    truth_all = []
    names_all = []
    k_count = 0
    for train_index, test_index in kfold.split(x, y, group_arr):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        scaler.fit(x_train)

        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        df_xtrain = pd.DataFrame(x_train, columns=feats_names)
        df_xtest = pd.DataFrame(x_test, columns=feats_names)

        x_train, x_test = np.array(df_xtrain.values), np.array(df_xtest.values)
        mdl = None

        if mdl_name == 'RF':
            mdl = RandomForestClassifier(bootstrap=True, class_weight=class_weight_str, criterion='gini',
                                         max_depth=None, max_features='auto', max_leaf_nodes=None,
                                         min_impurity_decrease=0, min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2,
                                         min_weight_fraction_leaf=0, n_estimators=1000, n_jobs=-1,
                                         oob_score=False, random_state=seed, verbose=1, warm_start=False)
        elif mdl_name == 'KNN':
            mdl = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto',
                                       leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

        elif mdl_name == 'LDA':
            x_train = sfs1.transform(x_train)
            x_test = sfs1.transform(x_test)
            mdl = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None, n_components=None,
                                             store_covariance=False, tol=0.0001)

        elif mdl_name == 'Bagging':
            x_train = sfs1.transform(x_train)
            x_test = sfs1.transform(x_test)
            mdl = BaggingClassifier(
                base_estimator=LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                                          store_covariance=False, tol=0.0001),

                n_estimators=100, max_samples=1.0,
                max_features=1.0, bootstrap=True, bootstrap_features=False,
                oob_score=False, warm_start=False, n_jobs=None, random_state=seed, verbose=0)

        elif mdl_name == 'SVC':
            x_train = sfs1.transform(x_train)
            x_test = sfs1.transform(x_test)
            mdl = SVC(C=1.0, cache_size=200, class_weight=class_weight_str, coef0=0.0, decision_function_shape='ovr',
                      degree=2,
                      gamma='auto', kernel='rbf',
                      max_iter=-1, probability=True, random_state=seed, shrinking=True, tol=0.001, verbose=False)

        elif mdl_name == 'GradBoost':
            mdl = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, max_features=1, max_depth=2,
                                             random_state=seed)

        elif mdl_name == 'XGBoost':
            mdl = XGBClassifier()

        elif mdl_name == 'CART':
            mdl = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                         min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                                         max_leaf_nodes=None, min_impurity_decrease=0.0,
                                         min_impurity_split=None, class_weight=class_weight_str, presort='deprecated',
                                         ccp_alpha=0.0)

        elif mdl_name == 'QDA':
            mdl = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)

        elif mdl_name == 'LR':
            x_train = sfs1.transform(x_train)
            x_test = sfs1.transform(x_test)
            mdl = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                     intercept_scaling=1,
                                     class_weight=class_weight_str, random_state=None,
                                     solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False,
                                     n_jobs=None, l1_ratio=None)

        class_weights = class_weight.compute_class_weight(class_weight_str,  # unused
                                                          np.unique(y_train),
                                                          y_train)

        if oversample:
            imb_ratio = np.sum(y_train) / len(y_train)
            over = SMOTE(sampling_strategy=imb_ratio)
            steps = [('o', over)]
            pipeline = Pipeline(steps=steps)  # create pipeline for SMOTE oversampling

            x_train, y_train = pipeline.fit_resample(x_train, y_train)

        mdl.fit(x_train, y_train)

        preds_train = mdl.predict(x_train)
        preds_test = mdl.predict(x_test)
        preds_prob_test = mdl.predict_proba(x_test)
        preds_test_out = []
        for kk in range(len(preds_test)):
            pred_val = preds_test[kk]
            pred_prob = preds_prob_test[kk][pred_val]
            if pred_val == 0.:
                pred_prob = 1 - pred_prob
            preds_test_out.append(pred_prob)

        print(f'~~~~~~~ Training scores split {k_count} ~~~~~~~~~')
        # pred_train.append(mdl.predict(X_train))
        # predictions_probaTrain.append(mdl.predict_proba(X_train))
        cfMat = confusion_matrix(y_train, preds_train)
        print(accuracy_score(y_train, preds_train))
        print(cfMat)
        print(classification_report(y_train, preds_train))

        print(f'~~~~~~~ Test scores split {k_count} ~~~~~~~~~')
        cfMat = confusion_matrix(y_test, preds_test)
        print('Bal Acc: ', balanced_accuracy_score(y_test, preds_test))
        print('F1: ', f1_score(y_test, preds_test))
        print('ROC AUC:', roc_auc_score(y_test, preds_test))
        print('ROC AUC (prob):', roc_auc_score(y_test, preds_test_out))
        prec, reca, _ = precision_recall_curve(y_test, preds_test_out)
        pr_auc_k = auc(reca, prec)
        print('PR AUC: ', pr_auc_k)
        print(cfMat)
        print(classification_report(y_test, preds_test))

        if scoring == 'roc_auc':
            score = roc_auc_score(y_test, preds_test)
            p_score = roc_auc_score(y_test, preds_test_out)
            prec, recall, _ = precision_recall_curve(y_test, preds_test_out)
            pr_auc = auc(recall, prec)
        elif scoring == 'f1':
            score = f1_score(y_test, preds_test)
        elif scoring == 'bal_acc':
            score = balanced_accuracy_score(y_test, preds_test)
        else:
            scoring = 'accuracy'
            score = accuracy_score(y_test, preds_test)
        scores.append(score)
        p_scores.append(p_score)
        pr_scores.append(pr_auc)
        print('Score: ', score)

        preds_out.extend(list(preds_test))
        # pred_probs_out.extend(list(preds_prob_test))
        y_out.extend(list(y_test))
        print('------------------------------------------')

        names_here = pn_names[test_index].values
        [y_preds_all.append(prediction) for prediction in preds_test]  # binary outcome
        [truth_all.append(label) for label in y_test]
        [names_all.append(n) for n in names_here]
        [pred_probs_out.append(p) for p in preds_test_out]  # probability outcome
        k_count += 1

    print(f'\n {mdl_name} Final average {scoring} score: ', np.average(scores), '\n')
    print(f'\n {mdl_name} Final average probability {scoring} score: ', np.average(p_scores), '\n')
    print(f'\n {mdl_name} Final average PR {scoring} score: ', np.average(pr_scores), '\n')
    print(confusion_matrix(y_out, preds_out))
    print('===========================================')


    roc_std = np.std(scores)
    roc_p_std = np.std(p_scores)
    pr_p_std = np.std(pr_scores)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('\nENTIRE TEST SET RESULTS PER IMAGE: ')
    print(classification_report(truth_all, y_preds_all, digits=3))
    print(confusion_matrix(truth_all, y_preds_all))
    print(f'ROC AUC score: {roc_auc_score(truth_all, pred_probs_out)} +/- {roc_p_std}\n')
    prec_all, recall_all, _ = precision_recall_curve(truth_all, pred_probs_out)
    pr_auc_all = auc(recall_all, prec_all)
    print(f'PR AUC probabilistic score: {pr_auc_all} +/- {pr_p_std}')
    # print('\nENTIRE TEST SET RESULTS PER CASE (max): ')
    # print(classification_report(truth_lesion_all, pred_lesion_max, digits=3))
    # print('\nENTIRE TEST SET RESULTS PER CASE (min): ')
    # print(classification_report(truth_lesion_all, pred_lesion_min, digits=3))
    print(f'\nUsed {len(nSel)} features: ')
    [print(feat) for feat in list(nSel)]
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


path_to_images = r'..\.\path\to\images\\'
path_to_masks = r'..\.\path\to\masks\\'
path_to_labels = r"..\.\path\to\labels\sample.csv"
perturb_protocol = 'V'  # 'No perturb', 'TVC', 'V'

feats_out_path = segmentation_and_image_perturbation(path_to_images, path_to_masks, path_to_labels,
                                                     perturb_protcol_name=perturb_protocol)
ml_prediction(feats_out_path, nb_splits=2)
