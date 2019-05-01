import sys

import pandas as pd
import numpy as np

import time

if len(sys.argv) != 4:
	print('Usage: \n\tpython lr_svm.py trainingSet.csv testSet.csv [1, 2]')
	exit(1)

TRAINING_DATE_FILE_NAME = sys.argv[1]
TRAINING_DATE_PATH = '../1_preprocessing/' + TRAINING_DATE_FILE_NAME
TEST_DATE_FILE_NAME = sys.argv[2]
TEST_DATE_PATH = '../1_preprocessing/' + TEST_DATE_FILE_NAME

MODEL_IDX = sys.argv[3]

if MODEL_IDX not in {'1','2'}:
	print('Model Index must be either 1 or 2.')
	exit(1)

__DEBUG = False

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
		if __DEBUG:
			print ('Epoch: {}'.format(e))

		predictions = sigmoid(np.dot(training_set_examples, weights))
		delta_w = np.sum((training_set_examples.T * (-training_set_labels + predictions)).T, axis=0) + LAMBDA*weights

		old_weights = np.copy(weights)
		weights = weights - STEP_SZ*delta_w

		diff = weights - old_weights
		if l2_norm(diff) < DIFF_THRESHOLD:
			break

		if __DEBUG:
			print ('Training Accuracy LR: {}'.format(get_accuracy(training_set_examples, training_set_labels, weights,'1')))
			print ('Testing Accuracy LR: {}'.format(get_accuracy(test_set_examples, test_set_labels, weights,'1')))

	print ('Training Accuracy LR: {}'.format(get_accuracy(training_set_examples, training_set_labels, weights,'1')))
	print ('Testing Accuracy LR: {}'.format(get_accuracy(test_set_examples, test_set_labels, weights,'1')))

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

	t0 = time.time()
	for e in range(MAX_NUM_EPOCHS):	
		if __DEBUG:
			print ('Epoch: {}'.format(e))		

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

		if __DEBUG:
			print ('Training Accuracy SVM: {}'.format(get_accuracy(training_set_examples, training_set_labels, weights,'2')))
			print ('Testing Accuracy SVM: {}'.format(get_accuracy(test_set_examples, test_set_labels, weights,'2')))

	print ('Training Accuracy SVM: {}'.format(get_accuracy(training_set_examples, training_set_labels, weights,'2')))
	print ('Testing Accuracy SVM: {}'.format(get_accuracy(test_set_examples, test_set_labels, weights,'2')))

	if __DEBUG:
		t1 = time.time()	
		print ('Time taken: ',t1-t0)

def main():
	df_training = pd.read_csv(TRAINING_DATE_PATH)
	df_test = pd.read_csv(TEST_DATE_PATH)

	if MODEL_IDX == '1':
		lr(df_training, df_test)
	else:
		svm(df_training, df_test)

if __name__ == '__main__':
	main()
