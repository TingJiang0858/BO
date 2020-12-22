# Random Forest
import sys
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot

def init_data():
	T1, T2 = make_classification(n_samples=100, n_features=20,  n_redundant=5, n_informative=15, random_state=3)

	return T1, T2

def get_models_fea():
	models = dict()

	for i in range(1,8):
		models[str(i)] = RandomForestClassifier(max_features=i)

	return models

def get_models_samples():
	models = dict()
	for i in range(0.1, 1.1, 0.1):
		key = '%.1f' % i
		if i == 1.0:         # !!!!!
			i = None
		models[key] = RandomForestClassifier(max_samples=i)

	return models

def get_models_tree():
	models = dict()
	tree_NO = [10, 30, 50, 80, 100, 150]
	for i in tree_NO:
		models[str(i)] = RandomForestClassifier(n_estimators=i)

	return models


def eva_kfold(model, T1, T2):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, T1, T2, scoring='accuracy', cv=cv, n_jobs=-1)

	return scores

def model_test_fea(T1, T2):
	#print(T1,T2)
	print("For different number of features:")
	modelfea = get_models_fea()
	score = []
	names = []
	for name, model in modelfea.items():
		scores = eva_kfold(model, T1, T2)
		score.append(scores)
		names.append(name)
		#print("fea:", score)
		print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

	pyplot.boxplot(score, labels=names)
	pyplot.show()

def model_test_sample(T1, T2):
	print("For different number of samples:")
	modelsam = get_models_samples()
	score_sam = []
	names = []
	for name, model in modelsam.items():
		scores = eva_kfold(model, T1, T2)
		score_sam.append(scores)
		# print("s:", score_sam)
		names.append(name)
		print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

	pyplot.boxplot(score_sam, labels=names)
	pyplot.show()

def model_test_tree(T1, T2):
	modeltree = get_models_tree()
	score_tree = []
	names = []
	print("For different number of trees:")
	for name, model in modeltree.items():
		scores = eva_kfold(model, T1, T2)

		score_tree.append(scores)
		names.append(name)
		
		print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	pyplot.boxplot(score_tree, labels=names)
	pyplot.show()

def main():
	T1, T2 = init_data()
	# test the accuracy for number of features, samples and trees
	model_test_fea(T1, T2)
	model_test_sample(T1, T2)
	model_test_tree(T1, T2)

if __name__ == "__main__":
    main()