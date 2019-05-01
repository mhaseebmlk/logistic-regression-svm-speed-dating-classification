import statistics
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time

ORG_DATA_FILE_NAME = 'dating-full.csv'
ORG_DATA_PATH = '../dataset/' + ORG_DATA_FILE_NAME
TRAINING_DATE_FILE_NAME = 'trainingSet.csv'
TRAINING_DATE_FILE_NAME_NBC = 'trainingSet_NBC.csv'
TRAINING_DATE_PATH = '../1_preprocessing/' + TRAINING_DATE_FILE_NAME

RANDOM_STATE = 18
FRAC = 1

__TIME_DEBUG = False

# Required for NBC
NUM_BINS = 5
continuous_valued_columns = ['age', 'age_o', 'importance_same_race', 'importance_same_religion', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'like']
col_ranges = {'age': (18, 58), 'age_o': (18, 58), 'importance_same_race': (0, 10), 'importance_same_religion': (0, 10), 'pref_o_attractive': (0, 1), 'pref_o_sincere': (0, 1), 'pref_o_intelligence': (0, 1), 'pref_o_funny': (0, 1), 'pref_o_ambitious': (0, 1), 'pref_o_shared_interests': (0, 1), 'attractive_important': (0, 1), 'sincere_important': (0, 1), 'intelligence_important': (0, 1), 'funny_important': (0, 1), 'ambition_important': (0, 1), 'shared_interests_important': (0, 1), 'attractive': (0, 10), 'sincere': (0, 10), 'intelligence': (0, 10), 'funny': (0, 10), 'ambition': (0, 10), 'attractive_partner': (0, 10), 'sincere_partner': (0, 10), 'intelligence_parter': (0, 10), 'funny_partner': (0, 10), 'ambition_partner': (0, 10), 'shared_interests_partner': (0, 10), 'sports': (0, 10), 'tvsports': (0, 10), 'exercise': (0, 10), 'dining': (0, 10), 'museums': (0, 10), 'art': (0, 10), 'hiking': (0, 10), 'gaming': (0, 10), 'clubbing': (0, 10), 'reading': (0, 10), 'tv': (0, 10), 'theater': (0, 10), 'movies': (0, 10), 'concerts': (0, 10), 'music': (0, 10), 'shopping': (0, 10), 'yoga': (0, 10), 'interests_correlate': (-1, 1), 'expected_happy_with_sd_people': (0, 10), 'like': (0, 10)}	
DATA_COLUMNS = ['gender', 'age', 'age_o', 'race', 'race_o', 'samerace', 'importance_same_race', 'importance_same_religion', 'field', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'like'] 
label_class_name = 'decision'

def perform_label_encoding(column):
    column = column.astype('category')
    codes_for_column = {}
    for i, category in enumerate(column.cat.categories):
        codes_for_column[category] = i
    return column.cat.codes

def remove_quotes(x):
    if "'" in x:
        return x.replace("'", "")
    else:
        return x

def to_lower(x):
    if x.islower():
        return x
    else:
        return x.lower()

def update_anomalous_max_values(row, col_min, col_max , true_range):
	true_max = true_range[1]
	row = true_max if row > true_max else row
	return row

def bin_row(row, intervals):
	bin_num = intervals.index(row)
	return bin_num

def discretize(df, num_bins):
	bin_interval_mappings = dict()
	for col in continuous_valued_columns:
		col_min, col_max = df[col].min(), df[col].max()
		true_range = col_ranges[col]
		df[col] = df[col].apply(update_anomalous_max_values, args=(col_min,col_max, true_range ))

		true_min, true_max = true_range[0], true_range[1]
		bin_min_val = -0.01 if true_min==0 else (true_min-(0.1/100 * true_min))

		bins = list(np.linspace(true_min, true_max, num_bins+1))
		bins = list(zip(bins, bins[1:]))
		bins[0] = (bin_min_val,bins[0][1])
		bins = pd.IntervalIndex.from_tuples(bins)

		bin_interval_mappings[col] = {bin: interval for bin, interval in enumerate(bins)}

		bins_list = bins.tolist()
		df[col] = pd.cut(df[col], bins).apply(bin_row, intervals= bins_list)

		counts = [len(df[df[col] == i])  for i in range(len(bins_list))]

	return df

def preprocess_nbc():
	#Read the dataset
	data = pd.read_csv(ORG_DATA_PATH, nrows=6500)
	decision_col = data['decision']

	#Remove quotes
	data['race'] = data['race'].apply(remove_quotes)
	data['race_o'] = data['race_o'].apply(remove_quotes)
	data['field'] = data['field'].apply(remove_quotes)

	#Convert to lowercase
	data['field'] = data['field'].apply(to_lower)

	#Label encode
	data[['race','race_o','gender','field']] = data[['race','race_o','gender','field']].apply(perform_label_encoding)

	#Normalize preference scores of the participant
	columns1  = ['attractive_important', 'sincere_important', 'intelligence_important','funny_important', 'ambition_important', 'shared_interests_important']
	columns2  = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
	             'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
	data[columns1] = data[columns1].div(data[columns1].sum(axis=1), axis=0)
	data[columns2] = data[columns2].div(data[columns2].sum(axis=1), axis=0)

	#Move the target class to the end
	data = data.drop(['decision'], axis = 1)
	data['decision'] = decision_col

	# Binning
	data = discretize(data, NUM_BINS)

	random_state_ = 25
	frac_ = 0.2

	#Save the csv file
	df_test = data.sample(frac=frac_, random_state=random_state_)
	df_train = data.drop(df_test.index)
	df_train.to_csv(TRAINING_DATE_FILE_NAME_NBC, index = False)

def nbc_predict(row, class_labels, nbc_params):
	probs = []
	for i in class_labels:
		prior_prob = nbc_params['prior_probs'][i]
		prob = prior_prob
		for col in DATA_COLUMNS:
			key = '{}={}|{}={}'.format(col,row[col],label_class_name,i)
			conditional_prob = 0
			try:
				conditional_prob = nbc_params['conditional_probs'][key]
			except:
				pass
			prob = prob * conditional_prob
		probs.append((i,prob))

	assigned_label = -1
	max_prob = -1
	for p in probs:
		if p[1] > max_prob:
			max_prob = p[1]
			assigned_label = p[0]

	return 'Correct' if assigned_label==row[label_class_name] else 'Incorrect'

def nbc(df_training, df_testing):
	# calculate the prior probs of the class labels
	parameters=dict()
	prior_probs = dict()
	class_labels = df_training[label_class_name].unique()
	for v in class_labels:
		label_filter = df_training[label_class_name].isin([v])
		filtered_df = df_training[label_filter]
		prob = len(filtered_df)/len(df_training)
		prior_probs[v] = prob

	parameters['prior_probs'] = prior_probs

	# get the naive bayes probabilities for each class label
	conditional_probs = dict()
	for i in class_labels:
		for col in DATA_COLUMNS:
			col_vals = df_training[col].unique()
			for j in col_vals:
				key = '{}={}|{}={}'.format(col,j,label_class_name,i)
				col_val_filter = df_training[col].isin([j])
				label_filter = df_training[label_class_name].isin([i])
				filtered_df = df_training[col_val_filter & label_filter]
				label_counts = len(df_training[label_filter])
				prob = (len(filtered_df))/(label_counts)
				conditional_probs[key] = prob

	parameters['conditional_probs'] = conditional_probs

	# model has been learned. now get accuracies
	test_accuracy = 0.0

	# testing accuracy
	df_testing['predictions'] = df_testing.apply(nbc_predict, axis=1, args=(class_labels,parameters))
	prediction_counts = df_testing['predictions'].value_counts().values.tolist()
	num_correct_predictions = prediction_counts[0]
	test_accuracy = round(num_correct_predictions/(sum(prediction_counts)),2)

	return test_accuracy

def l2_norm(v): return np.linalg.norm(v)

def sigmoid(x): return 1/(1+np.exp(-x))

def lr(training_set, test_set):
	LAMBDA = 0.01
	STEP_SZ = 0.01
	MAX_NUM_EPOCHS = 500
	DIFF_THRESHOLD = 10**-6

	# add the bias feature in the example and the weight vector
	training_set.insert(loc=0, column='bias', value=1)
	test_set.insert(loc=0, column='bias', value=1)

	#################### CONVERTING TO NUMPY ARRAY - START ####################
	training_set_examples = training_set.drop('decision', axis=1)
	training_set_labels = training_set['decision']

	test_set_examples = test_set.drop('decision', axis=1)
	test_set_labels = test_set['decision']

	col_names_idxs = {val:idx for idx,val in enumerate(training_set_examples.columns)}
	NUM_FEATURES = len(col_names_idxs)

	training_set_examples = training_set_examples.values
	training_set_labels = training_set_labels.values

	test_set_examples = test_set_examples.values
	test_set_labels = test_set_labels.values
	############################## END ########################################

	weights = np.zeros(NUM_FEATURES)

	for e in range(1,MAX_NUM_EPOCHS+1):	
		predictions = sigmoid(np.dot(training_set_examples, weights))
		delta_w = np.sum((training_set_examples.T * (-training_set_labels + predictions)).T, axis=0) + LAMBDA*weights

		old_weights = np.copy(weights)
		weights = weights - STEP_SZ*delta_w

		diff = weights - old_weights
		if l2_norm(diff) < DIFF_THRESHOLD:
			break

	return get_accuracy(test_set_examples, test_set_labels, weights,'1')

def get_accuracy(examples, labels, weights, model_idx):
	num_examples = examples.shape[0]
	num_predicted_correct = 0
	for i, x_i in enumerate(examples):
		y_i = labels[i]
		act = np.dot(weights, x_i)
		predicted_label = 0
		if model_idx == '1':
			predicted_label = 1 if act >= 0.5 else 0
		else:
			predicted_label = 1 if act > 0 else -1
		if y_i == predicted_label:
			num_predicted_correct += 1
	return round(num_predicted_correct/num_examples,2)

def svm(training_set, test_set):
	LAMBDA = 0.01
	STEP_SZ = 0.5
	MAX_NUM_EPOCHS = 500
	DIFF_THRESHOLD = 10**-6

	# add the bias feature in the example and the weight vector
	training_set.insert(loc=0, column='bias', value=1)
	test_set.insert(loc=0, column='bias', value=1)

	#################### CONVERTING TO NUMPY ARRAY - START ####################
	training_set_examples = training_set.drop('decision', axis=1)
	training_set_labels = training_set['decision']

	test_set_examples = test_set.drop('decision', axis=1)
	test_set_labels = test_set['decision']

	col_names_idxs = {val:idx for idx,val in enumerate(training_set_examples.columns)}
	NUM_FEATURES = len(col_names_idxs)

	training_set_examples = training_set_examples.values
	training_set_labels = training_set_labels.values
	training_set_labels[training_set_labels == 0] = -1

	test_set_examples = test_set_examples.values
	test_set_labels = test_set_labels.values
	test_set_labels[test_set_labels == 0] = -1
	############################## END ########################################

	N = training_set_examples.shape[0]
	weights = np.zeros(NUM_FEATURES)

	for e in range(MAX_NUM_EPOCHS):

		predictions = np.dot(training_set_examples, weights)
		
		delta_j = (training_set_examples.T * training_set_labels).T

		for i, x_i in enumerate(training_set_examples):
			prediction = 1 if predictions[i] > 0 else -1
			if training_set_labels[i] * prediction >= 1:
				delta_j[i] = np.zeros(NUM_FEATURES)

		delta_j[:,0] =  - delta_j[:,0]
		delta_j[:,1:] = LAMBDA*weights[1:] - delta_j[:,1:]

		delta_j = np.sum(delta_j, axis=0)
		delta_j = delta_j/N

		old_weights = np.copy(weights)
		weights = weights - STEP_SZ*delta_j

		diff = weights - old_weights
		if l2_norm(diff) < DIFF_THRESHOLD:
			break

	return get_accuracy(test_set_examples, test_set_labels, weights,'2')

def main():
	t0 = time.time()

	preprocess_nbc()

	df_training_org = pd.read_csv(TRAINING_DATE_PATH)
	df_training_org_nbc = pd.read_csv(TRAINING_DATE_FILE_NAME_NBC)

	# Shuffle
	df_training = df_training_org.sample(frac=FRAC, random_state=RANDOM_STATE)
	df_training_nbc = df_training_org_nbc.sample(frac=FRAC, random_state=RANDOM_STATE)

	# Partitioning
	K = 10
	org_index = df_training.index.tolist()
	org_index_nbc = df_training_nbc.index.tolist()
	S, i, fold_size = list(), 0, 520
	S_nbc = list()
	for _ in range(K):
		S.append(df_training.loc[org_index[i:i+fold_size]])
		S_nbc.append(df_training_nbc.loc[org_index_nbc[i:i+fold_size]])
		i=i+fold_size

	# K-Fold Cross Validation
	T_FRACs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
	random_state_ = 32
	model_statistics = {'NBC': list(), 'LR': list(), 'SVM': list()}
	for t_frac in T_FRACs:
		model_accuracies = {'NBC': list(), 'LR': list(), 'SVM': list()}
		for idx in range(K):
			test_set = S[idx]
			test_set_nbc = S_nbc[idx]

			other_folds = [S[i] for i in range(K) if i != idx]
			other_folds_nbc = [S_nbc[i] for i in range(K) if i != idx]

			S_c = pd.concat(other_folds)
			S_c_nbc = pd.concat(other_folds_nbc)
			
			train_set = S_c.sample(frac=t_frac, random_state=random_state_)
			train_set_nbc = S_c_nbc.sample(frac=t_frac, random_state=random_state_)

			train_set_sz = len(train_set)
			train_set_nbc_sz = len(train_set_nbc)

			for model in model_accuracies:
				if model == 'NBC':
					model_accuracies[model].append(nbc(train_set_nbc, test_set_nbc))
				elif model == 'LR':
					model_accuracies[model].append(lr(train_set.copy(), test_set.copy()))
				elif model == 'SVM':
					model_accuracies[model].append(svm(train_set.copy(), test_set.copy()))
				else:
					raise Exception('Unknown model:', model)

		for model in model_accuracies:
			accuracies = model_accuracies[model]
			avg_accuracy = sum(accuracies) / len(accuracies) 
			sd = statistics.stdev(accuracies)
			standard_error = sd / math.sqrt(K)
			model_statistics[model].append((train_set_sz, avg_accuracy, standard_error))

	# Graph
	training_set_sizes = [t[0] for t in model_statistics['NBC']]
	nbc_avg_accuracies = [t[1] for t in model_statistics['NBC']]
	nbc_standard_errors = [t[2] for t in model_statistics['NBC']]
	lr_avg_accuracies = [t[1] for t in model_statistics['LR']]
	lr_standard_errors = [t[2] for t in model_statistics['LR']]
	svm_avg_accuracies = [t[1] for t in model_statistics['SVM']]
	svm_standard_errors = [t[2] for t in model_statistics['SVM']]
	file_name = 'learning_curves.png'
	fig, ax = plt.subplots()
	ax.errorbar(training_set_sizes, nbc_avg_accuracies, yerr=nbc_standard_errors, label='NBC')
	ax.errorbar(training_set_sizes, lr_avg_accuracies, yerr=lr_standard_errors, label='LR')
	ax.errorbar(training_set_sizes, svm_avg_accuracies, yerr=svm_standard_errors, label='SVM')
	ax.legend()
	title='Model Accuracy vs. Size of Training Data'
	plt.xlabel('Size of Training Data')
	plt.ylabel('Model Accuracy')
	plt.title(title)
	plt.savefig(file_name)

	if __TIME_DEBUG:
		t1 = time.time()
		print ('Time taken: ',t1-t0)

if __name__ == '__main__':
	main()
