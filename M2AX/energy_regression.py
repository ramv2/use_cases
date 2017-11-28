import os
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

from scipy.stats import pearsonr, linregress

from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.initializers import TruncatedNormal, VarianceScaling
from keras.optimizers import RMSprop, SGD

global m
read_from_excel = False
if read_from_excel:
	data_A = pd.read_excel('../data/max data -dibbs.xlsx', sheetname = 'Descriptors for A', skiprows = [0])
	all_columns = list(data_A.columns.values)
	site_columns = all_columns[: 7]
	feature_columns = all_columns[7 : 17]
	E = data_A['Energy'].values
	elems = data_A[site_columns].values
	featA = data_A[feature_columns].values
	data_M = pd.read_excel('../data/max data -dibbs.xlsx', sheetname = 'Descriptors for M', skiprows = [0])
	all_columns = list(data_M.columns.values)
	feature_columns = all_columns[7 : 17]
	featM = data_M[feature_columns].values
	data_X = pd.read_excel('../data/max data -dibbs.xlsx', sheetname = 'Descriptors for X', skiprows = [0])
	all_columns = list(data_X.columns.values)
	feature_columns = all_columns[7 : 17]
	featX = data_X[feature_columns].values
	print E.shape, featA.shape, featM.shape, featX.shape
	XC = [4, 0.64, 2.55, 11.26,	0.77]
	XN = [5, 0.54, 3.04, 14.54,	0.71]
	for i, elemr in enumerate(elems):
		if elemr[-1] == 'N':
			featX[i, 5 :] = np.asarray(XN)
		else:
			featX[i, 5 :] = np.asarray(XC)
	# featmodes = ['concat12', 'concat21', 'avgdiff']
	# featA_ = np.hstack([(featA[:, : 5] + featA[:, 5 :]) / 2., np.abs(featA[:, : 5] - featA[:, 5 :])])
	# featM_ = np.hstack([(featM[:, : 5] + featM[:, 5 :]) / 2., np.abs(featM[:, : 5] - featM[:, 5 :])])
	# featX_ = np.hstack([(featX[:, : 5] + featX[:, 5 :]) / 2., np.abs(featX[:, : 5] - featX[:, 5 :])])
	# y = E
	# X = np.hstack([featA_, featM_, featX_])
# if read_from_excel:
# 	data = pd.read_excel('../data/max phase data.xlsx')
# 	E = data['Energy'].values
# 	with open('../data/MAX_energy.csv', 'w') as f:
# 		for e in E:
# 			f.write(str(e) + '\n')

def range_normalize_01(X, min_ = None, max_ = None):
	min_ = np.amin(X, axis = 0).reshape(1, -1) if min_ is None else min_
	max_ = np.amax(X, axis = 0).reshape(1, -1) if max_ is None else max_
	range_ = min_ - max_
	out = (X - min_) / range_
	return out.squeeze(), min_, max_

def range_normalize_11(X, min_ = None, max_ = None):
	min_ = np.amin(X, axis = 0).reshape(1, -1) if min_ is None else min_
	max_ = np.amax(X, axis = 0).reshape(1, -1) if max_ is None else max_
	range_ = min_ - max_
	out = (2. * (X - min_) / range_) - 1
	return out.squeeze(), min_, max_

def preprocess(train_X, test_X):
	train_X, train_min, train_max = range_normalize_01(train_X)
	X_mean = np.mean(train_X, axis = 0)
	X_std = np.std(train_X, axis = 0)
	train_X -= X_mean
	train_X /= X_std
	test_X, _, _ = range_normalize_01(test_X, train_min, train_max)
	test_X -= X_mean
	test_X /= X_std
	return train_X, test_X

def check_feature_variance(X):
	n_features = X.shape[1]
	min_ = np.amin(X, axis = 0).reshape(1, -1)
	max_ = np.amax(X, axis = 0).reshape(1, -1)
	# find features that dont vary
	static_idxs = np.where(min_ == max_)[1] # replace this by a variance threshold??
	if len(static_idxs) == 0:
		return X
	idxs = np.asarray([i for i in np.arange(X.shape[1]) if i not in static_idxs])
	# for s_idx in static_idxs:
	# 	print np.mean(X[:, s_idx]), np.min(X[:, s_idx]), np.max(X[:, s_idx])
	X = X[:, idxs]
	print 'removed {} of {} features with very low variance'.format(len(static_idxs), n_features)
	print 'feature indexes are', static_idxs
	return X, static_idxs

def clean_data(data_, labels_):
	data = np.hstack([data_, labels_.reshape(-1, 1)])
	n_samples = data.shape[0]
	data = data[~np.isnan(data).any(axis = 1)]
	data = data[~np.isinf(data).any(axis = 1)]
	if data.shape[0] != n_samples:
		print 'removed {} of {} rows with NaN or inf'.format(n_samples - data.shape[0], n_samples)
	return data[:, : -1], data[:, -1]

def read_from_csv(featfile, targetfile):
	with open(featfile, 'r') as f:
		lines = [line.strip() for line in f.readlines()[1 :]]
		features = [map(float, line.split(',')[: -1]) for line in lines]
	with open(targetfile, 'r') as f:
		lines = [line.strip() for line in f.readlines()]
		targets = map(float, lines)
	if len(features) != len(targets):
		print 'data lengths mismatch in features and targets'
		return None, None
	features, targets = np.asarray(features), np.asarray(targets)
	clean_data(features, targets)
	return features, targets

def read_from_csv_features(featfile):
	with open(featfile, 'r') as f:
		feature_names = f.readline().strip().split(',')
		lines = [line.strip() for line in f.readlines()[1 :]]
		features = [map(float, line.split(',')[: -1]) for line in lines]
	features = np.asarray(features)
	clean_data(features, np.ones((features.shape[0], )))
	return features, feature_names

def get_features(featmode):
	y = E
	if featmode == 'concat21':
		featA_ = np.hstack([featA[:, 5 :], featA[:, : 5]])
		featM_ = np.hstack([featM[:, 5 :], featM[:, : 5]])
		featX_ = np.hstack([featX[:, 5 :], featX[:, : 5]])
	elif featmode == 'avg':
		featA_ = (featA[:, : 5] + featA[:, 5 :]) / 2.
		featM_ = (featM[:, : 5] + featM[:, 5 :]) / 2.
		featX_ = (featX[:, : 5] + featX[:, 5 :]) / 2.
	elif featmode == 'diff':
		featA_ = np.abs(featA[:, : 5] - featA[:, 5 :])
		featM_ = np.abs(featM[:, : 5] - featM[:, 5 :])
		featX_ = np.abs(featX[:, : 5] - featX[:, 5 :])
	elif featmode == 'avgdiff':
		featA_ = np.hstack([(featA[:, : 5] + featA[:, 5 :]) / 2., np.abs(featA[:, : 5] - featA[:, 5 :])])
		featM_ = np.hstack([(featM[:, : 5] + featM[:, 5 :]) / 2., np.abs(featM[:, : 5] - featM[:, 5 :])])
		featX_ = np.hstack([(featX[:, : 5] + featX[:, 5 :]) / 2., np.abs(featX[:, : 5] - featX[:, 5 :])])
	else:
		featA_ = featA[...]
		featM_ = featM[...]
		featX_ = featX[...]
	X = np.hstack([featA_, featM_, featX_])
	clean_data(X, y)
	return X, y

def get_enet_grid():
	l1_ratios = [0.01, 0.25, 0.5, 0.75, 1]
	grid = [{'l1_ratio': l1_ratio} for l1_ratio in l1_ratios]
	return grid

def get_svr_grid():
	# SVR(kernel = 'rbf', C = C, gamma = gamma)
	C = [100, 300, 500, 800, 1000]
	gamma = [0.5, 0.1, 0.05, 0.01, 1. / 145]
	kernels = ['rbf'] #, 'linear']
	grid = [{'kernel': kernel, 'C': c, 'gamma': g} for kernel in kernels for c in C for g in gamma]
	return grid

def get_gbr_grid():
	# GradientBoostingRegressor(n_estimators = n_estimators, max_depth = max_depth)
	n_estimators = [100, 300, 500, 800, 1000]
	max_depths = [None, 3, 5, 8, 10, 20]
	grid = [{'n_estimators': n, 'max_depth': max_depth} for n in n_estimators for max_depth in max_depths]
	return grid

def get_nn_models(layer_strs, X, y):
	mdls = []
	init = TruncatedNormal(mean = 0.0, stddev = 0.01)
	# init = VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal')
	opt = RMSprop(lr = 0.1, rho = 0.9, epsilon = 1e-08, decay = 0.95)
	# opt = SGD(lr = 0.01, momentum = 0.9, decay = 0.95, nesterov = False)
	for layer_str in layer_strs:
		model = Sequential()
		model.add(Dense(layer_str[0], input_dim = X.shape[-1], kernel_initializer = init, activation = 'tanh'))
		for l in layer_str[1 :]:
			model.add(Dense(l, kernel_initializer = init, activation = 'tanh'))
		model.add(Dense(1, kernel_initializer = init, activation = 'linear'))
		model.compile(loss = 'mean_squared_error', optimizer = opt)
		mdls.append(model)
	return mdls

def train_test_nfolds(mdl_, X, y, n_folds = 5):
	rmses = []
	kf = KFold(len(y), n_folds = n_folds, shuffle = True)
	for train_idx, test_idx in kf:
		train_X = X[train_idx]
		train_y = y[train_idx]
		test_X = X[test_idx]
		test_y = y[test_idx]
		train_X, test_X = preprocess(train_X, test_X)
		if isinstance(mdl_, list):
			clf, hyperparams = mdl_
			mdl = clf(**hyperparams)
		if isinstance(mdl_, Sequential):
			mdl = mdl_
			mdl.fit(x = train_X, y = train_y, epochs = 100, batch_size = train_X.shape[0] / 100, verbose = 0)
		mdl.fit(train_X, train_y)
		pred_y = mdl.predict(test_X).squeeze()
		rmses.append(np.sqrt(np.mean((test_y - pred_y) ** 2)))
	return rmses, np.mean(rmses)

def train_test_grid(clf, grid, X, y):
	n_folds = 5
	min_rmse = 100000
	best_mdlf = '../models/best_model_MAXn1_Ef_{}_{}.mdl'.format(clf.__name__, m)
	for hyperparams in grid:
		print 'performing', n_folds, 'fold regression for', clf.__name__, hyperparams,
		mdl = [clf, hyperparams]
		rmses, rmse = train_test_nfolds(mdl, X, y, n_folds = n_folds)
		print rmse
		if rmse <= min_rmse:
			min_rmse = rmse
			best_mdl = mdl
	print 'best model --->', best_mdlf, min_rmse
	clf, hyperparams = best_mdl
	print clf.__name__, hyperparams
	best_mdl = clf(**hyperparams)
	joblib.dump(best_mdl, best_mdlf)
	return best_mdl

def train_test_nn(X, y):
	# layer_strs = [(100, ), (100, 50), (100, 50 ,25),
	# 			  (100, 80), (100, 80, 50), (100, 80, 30),
	# 			  (120, ), (120, 100), (120, 100, 80), (120, 100, 80, 50)]
	# layer_strs.extend([layer_str[::-1] for layer_str in layer_strs])
	layer_strs = [(512, ), (512, 1024),
				  (1024, ), (1024, 2048), (1024, 4096), (1024, 2048, 4096)]
	mdls = get_nn_models(layer_strs, X, y)
	n_folds = 5
	min_rmse = 100000
	best_mdlf = '../models/best_model_MAXn1_Ef_nnet_{}.mdl'.format(m)
	best_mdlf_str = '../models/best_model_MAXn1_Ef_nnet_{}.json'.format(m)
	for mdl in mdls:
		rmses, rmse = train_test_nfolds(mdl, X, y, n_folds = n_folds)
		print rmses, rmse
		if rmse <= min_rmse:
			min_rmse = rmse
			best_mdl = mdl
	print 'best model --->', best_mdlf, min_rmse
	print best_mdl.summary()
	with open(best_mdlf_str, 'w') as f:
		f.write(mdl.to_json())
	mdl.save_weights(best_mdlf)
	return best_mdl

def error_analysis(mdl, X, y):
	n_samples = y.shape[0]
	val_fraction = 0.2
	train_fraction = 1 - val_fraction
	train_length = int(n_samples * train_fraction)
	shuffle = np.random.permutation(n_samples)
	train_y = y[shuffle[: train_length]]
	train_X = X[shuffle[: train_length], :]
	test_y = y[shuffle[train_length: ]]
	test_X = X[shuffle[train_length: ], :]
	train_X, test_X = preprocess(train_X, test_X)
	mdl.fit(train_X, train_y)
	pred_y = mdl.predict(test_X).squeeze()
	rmse = np.sqrt(np.mean((test_y - pred_y) ** 2))
	_, _, r, _, _ = linregress(test_y, pred_y)
	print rmse
	print max(y), min(y)
	print max(pred_y), min(pred_y)
	error_y = np.abs(pred_y - test_y) # / test_y
	print np.sum(error_y <= 0.05), len(test_y)
	print 'r squared', r ** 2
	# plt.plot(test_y, error_y, 'x')
	# plt.show()
	# plt.hist(error_y, bins = 100)
	# plt.show()
	fig2, ax2 = plt.subplots()
	plt.plot(test_y, pred_y, 'bx')
	plt.plot(test_y, test_y, 'r')
	plt.title('Prediction vs DFT Formation Energy {}'.format(mdl.__class__.__name__))
	plt.xlabel('DFT Ef')
	plt.ylabel('Predicted Ef')
	# plt.show(block = False)
	fig2.savefig('../results/{}_{}.png'.format(mdl.__class__.__name__, X.shape[-1]))

def error_analysis2(mdl1, mdl2, X1, X2, y1, y2):
	n_samples = y.shape[0]
	val_fraction = 0.2
	train_fraction = 1 - val_fraction
	train_length = int(n_samples * train_fraction)
	shuffle = np.random.permutation(n_samples)
	train_y1 = y1[shuffle[: train_length]]
	train_y2 = y2[shuffle[: train_length]]
	train_X1 = X1[shuffle[: train_length], :]
	train_X2 = X2[shuffle[: train_length], :]
	test_y1 = y1[shuffle[train_length: ]]
	test_y2 = y2[shuffle[train_length: ]]
	test_X1 = X1[shuffle[train_length: ], :]
	test_X2 = X2[shuffle[train_length: ], :]
	train_X1, test_X1 = preprocess(train_X1, test_X1)
	train_X2, test_X2 = preprocess(train_X2, test_X2)
	mdl1.fit(train_X1, train_y1)
	mdl2.fit(train_X2, train_y2)
	pred_y1 = mdl1.predict(test_X1).squeeze()
	pred_y2 = mdl2.predict(test_X2).squeeze()
	rmse1 = np.sqrt(np.mean((test_y1 - pred_y1) ** 2))
	rmse2 = np.sqrt(np.mean((test_y2 - pred_y2) ** 2))
	_, _, r1, _, _ = linregress(test_y1, pred_y1)
	_, _, r2, _, _ = linregress(test_y2, pred_y2)
	print rmse1
	print rmse2
	print max(y1), min(y1)
	print max(y2), min(y2)
	print max(pred_y1), min(pred_y1)
	print max(pred_y2), min(pred_y2)
	error_y1 = np.abs(pred_y1 - test_y1) # / test_y1
	error_y2 = np.abs(pred_y2 - test_y2) # / test_y2
	print np.sum(error_y1 <= 0.05), len(test_y1)
	print np.sum(error_y2 <= 0.05), len(test_y2)
	print 'r squared 1', r1 ** 2
	print 'r squared 2', r2 ** 2
	# plt.plot(test_y, error_y, 'x')
	# plt.show()
	# plt.hist(error_y, bins = 100)
	# plt.show()
	fig2, ax2 = plt.subplots()
	p, = plt.plot(test_y1, test_y1, 'r', markersize = 0.5)
	p2, = plt.plot(test_y1, pred_y2, 'go', markersize = 4)
	p1, = plt.plot(test_y1, pred_y1, 'bx', markersize = 6)
	plt.legend([p2, p1], [r'Ashton et al.$^a$', r'Ward et al.$^b$'], loc = 'lower right')
	# plt.plot(test_y2, test_y2, 'm')
	# plt.title('Comparison of different descriptors for predicting Formation Energy (Ef)')
	# y axis label: Predicted $\Delta E_f$ (meV/atom)
	# x axis label: DFT $\Delta E_f$ (meV/atom)
	plt.xlabel(r'DFT $\Delta E_f$ (meV/atom)')
	plt.ylabel(r'Predicted $\Delta E_f$ (meV/atom)')
	fig2.savefig('../results/descriptorcomparisonforM2AX.png')
	# plt.show(block = False)

def plot_feature_var_corr(X, y, feature_names):
	feature_names = np.asarray(feature_names)
	X1, _, _ = range_normalize_01(X)
	corrs = []
	ps = []
	print X.shape, X1.shape
	for i in xrange(X1.shape[1]):
		corr, p = pearsonr(X1[:, i], y)
		corrs.append(corr)
		ps.append(p)
	corrs = np.abs(np.asarray(corrs))
	ps = np.asarray(ps)
	var = np.std(X1, axis = 0)
	var /= max(var)
	sorted_idx = np.argsort(corrs)
	pos = np.arange(sorted_idx.shape[0])
	print feature_names[sorted_idx[-20 :]]
	print corrs[sorted_idx[-20 :]]
	print ps[sorted_idx[-20 :]]
	xcorr = np.empty((X.shape[1], X.shape[1]))
	for i in xrange(X1.shape[1]):
		for j in xrange(X1.shape[1]):
			xcorr[i, j], _ = pearsonr(X1[:, i], X1[:, j])
	# xcorr /= np.max(xcorr)
	# print xcorr
	plt.figure()
	heatmap = plt.pcolor(xcorr, cmap = plt.cm.seismic)
	cbar = plt.colorbar(heatmap)
	plt.show(block = False)
	fig1, ax1 = plt.subplots()
	plt.bar(pos, var[sorted_idx], align='center')
	plt.bar(pos, corrs[sorted_idx], alpha = 0.7)
	plt.xticks(pos, sorted_idx)
	plt.ylabel('Relative Variance')
	plt.title('Variance of Features')
	plt.setp(ax1.get_xticklabels(), rotation = 90) #, horizontalalignment = 'right')
	ax1.tick_params(axis = 'x', labelsize = 6.)
	ax1.set_xlim(0, len(pos))
	plt.show(block = False)

def plot_feature_importance(mdl, X, y, feature_names):
	mdl.fit(X, y)
	feature_importance = mdl.feature_importances_
	# feature_importance = 1. * (feature_importance / np.max(feature_importance))
	feature_names = np.asarray(feature_names)
	sorted_idx = np.argsort(feature_importance)
	print feature_names[sorted_idx[-20 :]]
	joblib.dump(feature_names[sorted_idx[-20 :]], '../models/important_features_{}_{}.list'.format(mdl.__class__.__name__, m))
	importance_map = {name : importance for name, importance in zip(feature_names, feature_importance)}
	joblib.dump(importance_map, '../models/important_features_{}_{}.dict'.format(mdl.__class__.__name__, m))
	sorted_idx = sorted_idx[-20 :]
	pos = np.arange(sorted_idx.shape[0])
	fig, ax = plt.subplots()
	plt.bar(pos, feature_importance[sorted_idx] / np.max(feature_importance[sorted_idx]), align='center')
	plt.xticks(pos, feature_names[sorted_idx])
	plt.ylabel('Relative Importance')
	plt.title('Importance of Top 20 Features')
	plt.setp(ax.get_xticklabels(), rotation = 45) #, horizontalalignment = 'right')
	ax.tick_params(axis = 'x', labelsize = 8.)
	ax.set_xlim(0, len(pos))
	plt.show(block = False)

def write_prediction(mdl, featfile, outfile):
	X, _ = read_from_csv_features(featfile = featfile)
	X, _ = check_feature_variance(X)
	pred = mdl.predict(X)
	with open(outfile, 'w') as f:
		f.write('\n'.join(map(str, pred)))

def consolidated_rank_1(ranks):
	count = float(len(ranks))
	rank_count = {}
	for rank in ranks:
		rank_count[rank] = 1 if rank not in rank_count else rank_count[rank] + 1
	rank_ = 0.
	for rank in rank_count:
		rank_ += rank * (rank_count[rank] / count)
	return rank_

def consolidated_rank_2(ranks):
	return np.sum(21 - ranks) / 400.

def plot_consolidated_feature_importances(importances):
	consolidated_feature_importance = {}
	for feature_importance in importances:
		for name, importance in feature_importance.iteritems():
			if name in consolidated_feature_importance:
				consolidated_feature_importance[name] += importance
			else:
				consolidated_feature_importance[name] = importance
	max_importance = np.max([cfi[1] for cfi in consolidated_feature_importance.iteritems()])
	sum_importance = np.sum([cfi[1] for cfi in consolidated_feature_importance.iteritems()])
	# consolidated_feature_importance = {name : float(importance) / max_importance for name, importance in consolidated_feature_importance.iteritems()}
	sorted_importances = sorted(consolidated_feature_importance.items(), key = lambda x : x[1], reverse = True)
	fig, ax = plt.subplots()
	pos = np.arange(20) + 0.5
	plt.bar(pos, [si[1] / max_importance for si in sorted_importances[: 20]], align='center')
	plt.xticks(pos, [si[0] for si in sorted_importances[: 20]])
	plt.ylabel('Consolidated Relative Importance')
	plt.title('Consolidated Importance of Top 20 Features')
	plt.setp(ax.get_xticklabels(), rotation = 45) #, horizontalalignment = 'right')
	ax.tick_params(axis = 'x', labelsize = 8.)
	ax.set_xlim(0, len(pos))
	plt.show(block = False)
	cumul_importance = [sorted_importances[0][1]]
	for i in xrange(1, len(sorted_importances)):
		cumul_importance.append(cumul_importance[i - 1] + sorted_importances[i][1])
	cumul_importance = np.asarray(cumul_importance) # / cumul_importance[-1]
	for i in [20, 40, 60] + range(75, 100, 5):
		print i, np.sum(cumul_importance / cumul_importance[-1] <= i / 100.)
	fig1, ax1 = plt.subplots()
	pos = np.arange(len(consolidated_feature_importance)) + 0.5
	# plt.bar(pos, cumul_importance / cumul_importance[-1], align='center', alpha = 0.7, fill = False)
	# print len(cumul_importance), len(pos)
	plt.plot(pos, cumul_importance * 100. / cumul_importance[-1], alpha = 0.7)
	plt.title('Cumulative Relative Importance')
	ax1.set_xlabel('No. of Features')
	ax1.set_ylabel('Percentage of Total MDI')
	ax1.set_yticks([75, 85, 90, 95], minor = True)
	ax1.set_xticks([5, 13, 24, 37, 43, 50, 58, 72, 129], minor = False)
	ax1.set_xlim(0, len(pos))
	ax1.yaxis.grid(True, which = 'both')
	ax1.xaxis.grid(True, which = 'both')
	ax1.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
	ax1.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
	ax1.grid(True)
	plt.show(block = False)
	return sorted_importances

def visualize_formation_energy_2D(X, y):
	embedding_f = '../models/TSNE_2_{}.embedding'.format(X.shape[-1])
	if not os.path.isfile(embedding_f):
		print 'learning',
		X_embedding = TSNE(n_components = 2).fit_transform(X)
		joblib.dump(X_embedding, embedding_f)
	else:
		print 'loading',
		X_embedding = joblib.load(embedding_f)
	print embedding_f, X_embedding.shape
	fig3, ax3 = plt.subplots()
	c = np.ones_like(y)
	c[y <= 0.] = -1
	# c = y
	plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c = c, marker = '^', s = 0.5)#, cmap = plt.cm.seismic, s = 0.5)
	plt.colorbar(ticks = [-1, 1]) #np.linspace(-1, 1, 20))
	fig3.savefig('../results/Visualization2d_{}.png'.format(X.shape[-1]))

def visualize_formation_energy_3D(X, y):
	embedding_f = '../models/TSNE_3_{}.embedding'.format(X.shape[-1])
	if not os.path.isfile(embedding_f):
		print 'learning',
		X_embedding = TSNE(n_components = 3).fit_transform(X)
		joblib.dump(X_embedding, embedding_f)
	else:
		print 'loading',
		X_embedding = joblib.load(embedding_f)
	print embedding_f, X_embedding.shape
	fig4 = plt.figure(1)
	ax4 = fig4.add_subplot(111, projection = '3d')
	ax4.set_title('Visualization3D_{}'.format(X.shape[-1]))
	c = np.ones_like(y)
	c[y <= 0.] = -1
	# c = y
	p = ax4.scatter(X_embedding[:, 0], X_embedding[:, 1], X_embedding[:, 2], c = c, marker = '^', s = 0.5)#, cmap = plt.cm.seismic, s = 0.5)
	fig4.colorbar(p, ticks = [-1, 1]) #np.linspace(-1, 1, 20))
	if not os.path.isdir('../results/TSNE_3_{}'.format(X.shape[-1])):
		os.makedirs('../results/TSNE_3_{}'.format(X.shape[-1]))
	for angle in range(0, 360, 10):
		ax4.view_init(30, angle)
		plt.draw()
		plt.pause(.001)
		fig4.savefig('../results/TSNE_3_{}/{}.png'.format(X.shape[-1], angle))
	raw_input('press a key to continue')
	plt.close(1)


if __name__ == '__main__':
	clfs = [GradientBoostingRegressor] #, RandomForestRegressor, SVR, ElasticNet]
	grids = [get_gbr_grid()] #, get_gbr_grid(), get_svr_grid(), get_enet_grid()]
	select = [129]
	# clfs = [Ridge]
	# grids = [[{'fit_intercept' : True}]]
	# select = [37, 43, 50, 58, 72, 129]
	for clf, grid in zip(clfs, grids):
		for m_ in xrange(20, 21):
			for s in select:
				m = m_
				X, y = read_from_csv(featfile = '../data/featuresn1.csv', targetfile = '../data/energyn1.csv')
				X, low_var_idxs = check_feature_variance(X)
				_, feature_names = read_from_csv_features('../data/featuresn1.csv')
				low_var_names = [feature_names[i] for i in low_var_idxs]
				print '\n'.join(low_var_names)
				feature_names = [feature_names[i] for i in range(len(feature_names)) if i not in low_var_idxs]
				sorted_importances = joblib.load('../models/relevant_features_{}.dict'.format(GradientBoostingRegressor.__name__))
				sorted_important_feature_names = [si[0] for si in sorted_importances[: s]]
				cols = [i for i in xrange(X.shape[-1]) if feature_names[i] in sorted_important_feature_names]
				X = X[:, cols]
				print X.shape
				# print 'stable', len(y[y <= 0.]), 'unstable', len(y[y > 0.])
				visualize_formation_energy_2D(X, y)
				# visualize_formation_energy_3D(X, y)
				continue
				# X1, y1 = get_features('avgdiff')
				# print len(y), len(y1), np.sum((y - y1) ** 2)
				mdl = train_test_grid(clf, grid, X, y)
				# mdl1 = joblib.load('../models/best_model_MAXn1_Ef_{}_{}.mdl'.format(clf.__name__, m))
				mdl = joblib.load('../models/best_model_MAXn1_Ef_{}_{}.mdl'.format(clf.__name__, m))
				# mdl = train_test_nn(X, y)
				# with open('../models/best_model_MAXn1_Ef_nn_{}.json'.format(m), 'r') as f:
				# 	mdl = model_from_json(f.read())
				# mdl.load_weights('../models/best_model_MAXn1_Ef_nn_{}.nnet'.format(m))
				# mdl.fit(X, y)
				# write_prediction(mdl = mdl, featfile = '../data/featuresn2.csv', outfile = '../data/predicted_energies_n2.csv')
				# write_prediction(mdl = mdl, featfile = '../data/featuresn3.csv', outfile = '../data/predicted_energies_n3.csv')
				# plot_feature_var_corr(X, y, feature_names)
				error_analysis(mdl, X = X, y = y)
				# plot_feature_importance(mdl, X, y, feature_names)
				# raw_input()
				# break
				# error_analysis2(mdl, mdl1, X, X1, y, y1)
				# raw_input()
		feature_count = {}
		rankings = {}
		importances = []
		for m__ in range(0):
			model = joblib.load('../models/best_model_MAXn1_Ef_{}_{}.mdl'.format(clf.__name__, m__))
			# print m__, model.n_estimators, model.max_depth
			feature_list = joblib.load('../models/important_features_{}.list'.format(m__))
			importances.append(joblib.load('../models/important_features_{}_{}.dict'.format(clf.__name__, m__)))
			for rank, feature in enumerate(feature_list[::-1]):
				if feature in feature_count:
					feature_count[feature] += 1
				else:
					feature_count[feature] = 1
				if feature in rankings:
					rankings[feature].append(rank + 1)
				else:
					rankings[feature] = [rank + 1]
		rankings = {feature : np.asarray(rankings[feature]) for feature in rankings}
		# import operator
		# # sorted_feat = sorted(feature_count.items(), key = lambda x : (-x[1], consolidated_rank_1(rankings[x[0]])))
		# sorted_feat = sorted(feature_count.items(), key = lambda x : (consolidated_rank_2(rankings[x[0]])), reverse = True)[: 20]
		# for feat, votes in sorted_feat:
		# 	print feat, votes, rankings[feat], consolidated_rank_2(rankings[feat])
		# 	# print np.round(np.mean(rankings[feat])), np.std(rankings[feat]), np.min(rankings[feat]), np.max(rankings[feat])
		# fig, ax = plt.subplots()
		# pos = np.arange(len(sorted_feat)) + 0.5
		# plt.bar(pos, [consolidated_rank_2(rankings[feat]) for feat, votes in sorted_feat], align='center')
		# plt.xticks(pos, [feat for feat, votes in sorted_feat])
		# plt.ylabel('Relative Importance')
		# plt.title('Most Important Features for Prediction of Formation Energy')
		# plt.setp(ax.get_xticklabels(), rotation = 45) #, horizontalalignment = 'right')
		# ax.tick_params(axis = 'x', labelsize = 8.)
		# ax.set_xlim(0, len(pos))
		# plt.show(block = False)
		if len(importances) > 0:
			sorted_importances = plot_consolidated_feature_importances(importances)
			property_importances = {}
			for si in sorted_importances:
				name = si[0].split('_', 1)[-1]
				if name not in property_importances:
					property_importances[name] = si[1]
				else:
					property_importances[name] += si[1]
			property_importances = sorted(property_importances.iteritems(), key = lambda x: x[1], reverse = True)
			for pi in property_importances:
				print pi[0], pi[1] / np.max([pi[1] for pi in property_importances])
			# joblib.dump(sorted_importances, '../models/relevant_features_{}.dict'.format(clf.__name__))
		raw_input()
	pass

