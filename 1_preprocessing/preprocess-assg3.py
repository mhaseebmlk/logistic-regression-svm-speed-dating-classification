import sys

import pandas as pd
import numpy as np

DATA_FILE_NAME = 'dating-full.csv'
DATA_PATH = '../dataset/' + DATA_FILE_NAME
OUTPUT_FILE_NAME_TRAINING = 'trainingSet.csv'
OUTPUT_FILE_NAME_TEST = 'testSet.csv'

RANDOM_STATE = 25
FRAC = 0.2

df = pd.read_csv(DATA_PATH, nrows=6500)

preference_scores_of_participant_columns = [
	'attractive_important', 
	'sincere_important', 
	'intelligence_important',
	'funny_important', 
	'ambition_important', 
	'shared_interests_important'
]
for col in preference_scores_of_participant_columns:
	df[col] = df[col].astype('float64')

preference_scores_of_partner_columns = [
	'pref_o_attractive', 
	'pref_o_sincere',
	'pref_o_intelligence', 
	'pref_o_funny',
	'pref_o_ambitious', 
	'pref_o_shared_interests'
]

count_quotes_removed = count_standardized = 0

cols_to_split_quotes = ['race', 'race_o','field']
cols_to_standardize = ['field']
cols_to_encode = ['gender', 'race', 'race_o', 'field']

def strip_quotes(row):
	global count_quotes_removed
	stripped_row = row
	if row[0] == '\'' or  row[-1] == '\'':
		row = row.strip('\'')
		count_quotes_removed += 1
	return row

def standardize(row):
	global count_standardized
	if not row.islower():
		row = row.lower()
		count_standardized += 1
	return row

def encode(row, val):
	return 1 if row == val else 0

for col in cols_to_split_quotes:
	df[col] = df[col].apply(strip_quotes)

for col in cols_to_standardize:
	df[col] = df[col].apply(standardize)

# normalizing
for i, row in df.iterrows():
	total_pref_scores_participant = total_pref_scores_partner = 0.0
	for col in preference_scores_of_participant_columns:
		total_pref_scores_participant += row[col]	
	for col in preference_scores_of_participant_columns:
		df.at[i, col] = row[col]/total_pref_scores_participant

	for col in preference_scores_of_partner_columns:
		total_pref_scores_partner += df.at[i, col]
	for col in preference_scores_of_partner_columns:
		df.at[i, col] = row[col]/total_pref_scores_partner

"""print ('Quotes removed from {} cells.'.format(count_quotes_removed))
print ('Standardized {} cells to lower case.'.format(count_standardized))"""

# One-Hot encoding
one_hot_encodings = dict()
for col in cols_to_encode:
	one_hot_encodings[col] = dict()
	sorted_vals = sorted(df[col].unique())
	for idx,val in enumerate(sorted_vals):
		bit_vector = [0] * (len(sorted_vals)-1)
		if idx < (len(sorted_vals)-1):
			bit_vector[idx] = 1
		one_hot_encodings[col][val] = bit_vector

	# encodings for this column done. can now use it 
	encodings = one_hot_encodings[col]
	for val in list(encodings.keys())[:-1]:
		new_col_name = col+'_'+val
		df[new_col_name] = df[col].apply(encode, args=(val,))

	df = df.drop(columns=[col])

df_decision = df['decision'] # temporariily store all the decisions
df = df.drop(columns=['decision'])
df['decision'] = df_decision

to_print = {'gender':'female', 
			'race':'Black/African American', 
			'race_o':'Other',
			'field': 'economics'}
for col,val in to_print.items():
	print ('Mapped vector for {} in column {}: {}.'.format(val,col,one_hot_encodings[col][val]))

# print the means of the columns
"""for col in preference_scores_of_participant_columns:
	mean = round(df[col].mean(),2)
	print('Mean of {}: {}.'.format(col,mean))
for col in preference_scores_of_partner_columns:
	mean = round(df[col].mean(),2)
	print('Mean of {}: {}.'.format(col,mean))"""

# make test and training data sets
df_test = df.sample(frac=FRAC, random_state=RANDOM_STATE)
df_train = df.drop(df_test.index)

df_test.to_csv(OUTPUT_FILE_NAME_TEST, index=False)
df_train.to_csv(OUTPUT_FILE_NAME_TRAINING, index=False)

# df.to_csv('dating-full-onehot-encoded.csv',index=False)