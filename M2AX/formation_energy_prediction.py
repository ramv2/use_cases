from enum import Enum
import os
from os.path import isfile
import time
import joblib
import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from attributes.generators.composition.ChargeDependentAttributeGenerator import \
    ChargeDependentAttributeGenerator
from attributes.generators.composition.ElementFractionAttributeGenerator import \
    ElementFractionAttributeGenerator
from attributes.generators.composition.ElementalPropertyAttributeGenerator \
    import ElementalPropertyAttributeGenerator
from attributes.generators.composition.IonicCompoundProximityAttributeGenerator import \
    IonicCompoundProximityAttributeGenerator
from attributes.generators.composition.IonicityAttributeGenerator import \
    IonicityAttributeGenerator
from attributes.generators.composition.StoichiometricAttributeGenerator \
    import StoichiometricAttributeGenerator
from attributes.generators.composition.ValenceShellAttributeGenerator import \
    ValenceShellAttributeGenerator
from attributes.generators.composition.YangOmegaAttributeGenerator import \
    YangOmegaAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry
import itertools
from data.materials.util.LookUpData import LookUpData

class M2AXCompounds:
    def __init__(self, n):
        self.n = n

    def load_composition_entry_input(self, file_path):
        entries = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                entry = CompositionEntry(composition=line.strip())
                entries.append(entry)

        return entries

    def load_y(self, file_path):
        y = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                y.append(float(line.strip()))

        return y

    def cross_validate_n_fold(self, n, model, X, y):
        rmses = []
        scores = []
        mae = []
        kf = KFold(n_splits=n, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            error = y_test - y_pred
            rmses.append(np.sqrt(np.mean(error ** 2)))
            scores.append(model.score(X_test, y_test))
            mae.append(np.mean(abs(error)))
        return rmses, scores, mae

    def create_param_grid(self, model, n_features=None, simple=False):
        if model == MLModels.SVR:
            # Create parameter grid for SVR.
            C = [0.1, 1, 10, 100, 300, 500, 1000] if not simple else [1]
            gamma = [1e-4, 1e-3, 1e-2, 1e-1] if not simple else [0.1]
            if not simple and n_features is not None and n_features != 0:
                gamma += [1.0 / n_features]
            epsilon = [0.01, 0.05, 0.1, 0.2] if not simple else [0.1]
            kernel = ['rbf', 'poly', 'linear'] if not simple else ['rbf']
            param_grid = [{'kernel': k, 'C': c, 'gamma': g, 'epsilon': e} for k
                          in kernel for c in C for g in gamma for e in epsilon]
            return param_grid
        elif model == MLModels.NN:
            # Create parameter grid for neural network.
            l = [128, 256, 512, 1024] if not simple else [100, 50]
            layers = [l_i for i in range(1, len(l)) for l_i in
                      itertools.permutations(l, i)]
            act = ['logistic', 'tanh', 'relu'] if not simple else ['tanh']
            solver = ['lbfgs', 'sgd', 'adam'] if not simple else ['lbfgs']
            alpha = [1e-5, 1e-4, 1e-3] if not simple else [1e-4]
            l_r = ['invscaling', 'adaptive'] if not simple else ['adaptive']
            param_grid = [{'hidden_layer_sizes': h, 'activation': a,
                           'learning_rate': lr, 'alpha': al,
                           'early_stopping': True, 'solver': s} for h in
                          layers for a in act for lr in l_r for al in alpha
                          for s in solver]
            return param_grid
        elif model == MLModels.RF or model == MLModels.GBR:
            # Create parameter grid for random forest regressor or gradient
            # boosting regressor.
            n_estimators = [100, 300, 500, 800, 1000] if not simple else [100]
            max_depths = [None, 3, 5, 8, 10, 20] if not simple else [None]
            param_grid = [{'n_estimators': n, 'max_depth': max_depth} for n in
                    n_estimators for max_depth in max_depths]
            return param_grid
        else:
            max_depths = [None, 3, 5, 8, 10, 20] if not simple else [None]
            param_grid = [{'max_depth': max_depth} for max_depth in max_depths]
            return param_grid

    def sanitize(self, df, drop_na=False, verbose=False):
        """
        Function to check if a data frame contains NaN's or Infinity's and
        handle them appropriately. We have a couple of options:

        1) We can get rid of these troublesome columns:
        This is fairly straightforward. Simply delete these columns in
        the input data frame.

        2) We can replace NaN's with some meaningful numerical values so
        that they don't affect the training and testing process. It is
        tricky to find the correct numerical value with which to
        replace. For instance, NaN values of Bulk Modulus and Shear
        Modulus may be replaced with zero without loosing much
        information. However, NaN values of charge can't be replaced
        with zero as this will lead to erroneous results. On a similar
        note, it doesn't make physical sense to define a boiling point
        for a solid material. Therefore, we add one extra column with
        zeros in place of NaN's and ones everywhere else. We need to
        add this additional column for every column that has even one
        NaN value.

        :param df: Data frame to check.
        :param drop_na: Whether to drop the troublesome columns or to add one
        extra column as described above.
        :param verbose: Whether to print a lot of information for the purpose
        of debugging.
        :return df: Data frame with NaN's or Infinity's replaced with
        appropriate numerical values.
        """

        # First identify troublesome columns.
        nan_cols = df.columns[df.isnull().any()].tolist()
        if not nan_cols:
            # Great! Nothing to do.
            if verbose:
                print "None of the columns have NaN's or Infinity's."
            return df
        else:
            if verbose:
                print "{} columns have NaN's or Infinity's.".format(len(nan_cols))
                print "These are:"
                print "Column:\t Number of NaN's"
            for col in nan_cols:
                nan_count = df.loc[:, col].isnull().sum()
                print col, "\t", nan_count
                if nan_count == df.shape[0]:
                    if verbose:
                        print "Deleting column: "+col
                    del df[col]

            if drop_na:
                new_df = df.dropna(axis=1)
            else:
                nan_cols = df.columns[df.isnull().any()].tolist()
                for col in nan_cols:
                    df["extra_"+col] = df.loc[:, col].notnull().astype(int)
                new_df = df.fillna(df.mean())

        if verbose:
            # Check that there are no more NaN's in the data frame.
            print new_df.isnull().any()

            # Print the shape of the new data frame.
            print new_df.shape

        return new_df

class MLModels(Enum):
    SVR = SVR
    NN = MLPRegressor
    RF = RandomForestRegressor
    GBR = GradientBoostingRegressor
    DT = DecisionTreeRegressor

if __name__ == "__main__":
    pd.set_option('display.width', 240)
    verbose = False
    simple_test = False
    test_size = 0.2
    n_folds = 10 if not simple_test else 3
    m2ax = M2AXCompounds(1)
    y = np.array(m2ax.load_y("new_y.txt"))
    file_name = "new_features_352.csv"

    if not isfile(file_name):
        entries = m2ax.load_composition_entry_input("new_input_python.txt")
        sg = StoichiometricAttributeGenerator()
        # For generating Ward et al. features, use_default_properties=True.
        # eg = ElementalPropertyAttributeGenerator(use_default_properties=True)
        eg = ElementalPropertyAttributeGenerator(use_default_properties=False)
        eg.add_elemental_properties(LookUpData.all_properties)
        vg = ValenceShellAttributeGenerator()
        ig = IonicityAttributeGenerator()
        cg = ChargeDependentAttributeGenerator()
        fg = ElementFractionAttributeGenerator()
        # pg = IonicCompoundProximityAttributeGenerator()
        yg = YangOmegaAttributeGenerator()
        X_reduced = pd.DataFrame()

        # Generate features.
        f_sg = sg.generate_features(entries)
        f_eg = eg.generate_features(entries)
        f_vg = vg.generate_features(entries)
        f_ig = ig.generate_features(entries)
        f_cg = cg.generate_features(entries)
        f_fg = fg.generate_features(entries)
        # f_pg = pg.generate_features(entries)
        f_yg = yg.generate_features(entries)

        # Concatenate them all.
        X = pd.concat([f_sg, f_eg, f_vg, f_ig, f_cg, f_fg, f_yg], axis=1)
        if verbose:
            print "Shape before sanitizing: ", X.shape
        X = m2ax.sanitize(X, drop_na=False, verbose=verbose)
        if verbose:
            print "Shape after sanitizing: ", X.shape
            tol = 1e-10
            X_reduced = X.loc[:, X.var() > tol]
            deleted_features = list(X.loc[:, X.var() < tol])
            n_features = X_reduced.shape[1]
            X_reduced.to_csv("new_features_{}.csv".format(n_features),
                             index=False)
            if verbose:
                print len(deleted_features), deleted_features
                print "Shape after removing low variance columns: ", \
                    X_reduced.shape
    else:
        X_reduced = pd.read_csv(file_name)

    if verbose:
        print X_reduced.isnull().any().any()
    # Mean normalization and feature scaling.
    X_nm = (X_reduced - X_reduced.mean()) / X_reduced.std()
    if verbose:
        print X_nm.isnull().any().any()
    # positive = len(y[y > 0])
    # print positive, len(y) - positive

    X_train, X_test, y_train, y_test = train_test_split(X_nm, y,
                                                        test_size=test_size)
    time_stamp = time.strftime("%b-%d_%H:%M")

    log_file_name = "log/"+time_stamp+".log"
    buf_size = 0
    overall_best_params = None
    overall_best_scores = []
    overall_best_rmses = []
    overall_best_mae = []
    overall_best_ratio = -float("inf")
    overall_best_model = None
    with open(log_file_name, 'w', buf_size) as f:
        for model in MLModels:
            best_params = None
            best_scores = []
            best_rmses = []
            best_mae = []
            best_ratio = -float("inf")
            best_model = None
            param_grid = m2ax.create_param_grid(model, n_features=X_nm.shape[
                1], simple=simple_test)
            for params in param_grid:
                mdl = model.value(**params)
                rmses, scores, mae = m2ax.cross_validate_n_fold(n_folds, mdl,
                                                                X_nm, y)
                mean_scores = np.mean(scores)
                mean_rmses = np.mean(rmses)
                mean_mae = np.mean(mae)
                ratio = mean_scores / mean_rmses
                f.write("Model : " + mdl.__class__.__name__ + "\n")
                f.write("Parameters : " + str(params) + "\n")
                f.write("Scores : " + str(scores) + " mean : " + str(
                    mean_scores) + "\n")
                f.write("Rmses : " + str(rmses) + " mean : " + str(
                    mean_rmses) + "\n")
                f.write("MAE : " + str(mae) + " mean : " + str(mean_mae) +
                        "\n")
                f.write("Ratio : " + str(ratio) + "\n")
                f.write("\n")
                f.flush()
                if ratio > best_ratio:
                    best_model = mdl
                    best_ratio = ratio
                    best_params = params
                    best_scores = scores
                    best_rmses = rmses
                    best_mae = mae

            f.write("\n")
            f.write("\n")
            f.write("Best model : " + best_model.__class__.__name__ + "\n")
            f.write("Best model parameters : " + str(best_params) + "\n")
            f.write("Best scores : " + str(best_scores) + " mean : " + str(
                np.mean(best_scores)) + "\n")
            f.write("Best rmses : " + str(best_rmses) + " mean : " + str(
                np.mean(best_rmses)) + "\n")
            f.write("Best mae : " + str(best_mae) + " mean : " + str(
                np.mean(best_mae)) + "\n")
            f.write("Best ratio : " + str(best_ratio) + "\n")
            f.write("\n")
            f.write("\n")
            f.flush()
            if best_ratio > overall_best_ratio:
                overall_best_model = best_model
                overall_best_ratio = best_ratio
                overall_best_params = best_params
                overall_best_scores = best_scores
                overall_best_rmses = best_rmses
                overall_best_mae = best_mae

        f.write("Overall best model : " +
                overall_best_model.__class__.__name__ + "\n")
        f.write("Overall best model parameters : " + str(overall_best_params)
                + "\n")
        f.write("Overall best scores : " + str(overall_best_scores) + " mean "
                ": " + str(np.mean(overall_best_scores)) + "\n")
        f.write("Overall best rmses : " + str(overall_best_rmses) + " mean : "
                + str(np.mean(overall_best_rmses)) + "\n")
        f.write("Overall best mae : " + str(overall_best_mae) + " mean : "
                + str(np.mean(overall_best_mae)) + "\n")
        f.write("Overall best ratio : " + str(overall_best_ratio) + "\n")
        f.write("\n")
        f.write("\n")
        f.flush()
    model_dir = "models/{}".format(time_stamp)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    overall_best_model_str = model_dir + "/{}".format(
        overall_best_model.__class__.__name__)
    joblib.dump(overall_best_model, overall_best_model_str)