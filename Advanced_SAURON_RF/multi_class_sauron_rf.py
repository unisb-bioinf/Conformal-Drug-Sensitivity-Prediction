# Author:Kerstin Lenhof
# Date: 7.11.2022
# Multiclass SAURON-RF
import operator

import numpy
import math
import time
import copy


# from sklearn.datasets import make_regression
from collections import Counter
from collections.abc import Mapping
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import resample
from scipy.stats import pearsonr
from scipy.stats import spearmanr


class SAURONRF:
	'multiclass SAURON-RF implementation, compatible with Conformal Prediction Framework by Lisa-Marie Rolli'

	
	def __init__(self, X_train, y_train,  sample_names_train, min_number_of_samples_per_leaf, number_of_trees_in_forest, number_of_features_per_split, class_assignment_samples_train,  name_of_analysis, mse_included, classification_included, feature_imp_output_file, feature_names, threshold, upsampling, time_file, sample_weights_included, leaf_assignment_file_train, sample_info_file, all_available_labels, debug_file):
		self.X_train = X_train
		self.y_train = y_train
		self.sample_names_train = sample_names_train
		self.min_number_of_samples_per_leaf = min_number_of_samples_per_leaf
		self.number_of_trees_in_forest = number_of_trees_in_forest
		self.number_of_features_per_split = number_of_features_per_split
		self.class_assignment_samples_train = class_assignment_samples_train
		self.name_of_analysis = name_of_analysis
		self.mse_included = mse_included
		self.classification_included = classification_included
		self.feature_imp_output_file = feature_imp_output_file
		self.feature_names = feature_names
		self.thresholds = threshold
		self.upsampling = upsampling
		self.time_file = time_file
		self.sample_weights_included = sample_weights_included
		self.leaf_assignment_file_train = leaf_assignment_file_train
		self.sample_info_file = sample_info_file
		self.debug_file = debug_file
		self.all_available_labels = all_available_labels

		print("Your input parameters are: ")
		#print("Analysis mode: " + self.analysis_mode)
		print("Number of trees: " + str(self.number_of_trees_in_forest))
		print("Min number of samples per leaf: " + str(self.min_number_of_samples_per_leaf))
		print("Number of features per split: " + str(self.number_of_features_per_split))

		print("The dimensions of your training matrix are: ")
		print("Number of rows (samples): " + str(self.X_train.shape[0]))
		print("Number of columns (features): " + str(self.X_train.shape[1]))

		self.original_Xtrain = copy.deepcopy(self.X_train)
		self.original_ytrain = copy.deepcopy(self.y_train)
		self.original_sample_names_train = copy.deepcopy(self.sample_names_train)
		self.original_class_assignment_samples_train = copy.deepcopy(self.class_assignment_samples_train)
		self.class_count = Counter(self.class_assignment_samples_train)
		self.majority_class_training = int(self.class_count.most_common(1)[0][0])

		self.model = RandomForestRegressor(n_estimators=self.number_of_trees_in_forest, random_state=numpy.random.RandomState(2),
									  min_samples_leaf=self.min_number_of_samples_per_leaf,
									  max_features=self.number_of_features_per_split, bootstrap=True,
									  oob_score=True, n_jobs=10)

	def calculate_prediction_single_sample_simple_average(self, sample_prediction):
		'''
		@param sample_prediction: a dictionary of trees to the predictions for a particular sample
		'''

		used_trees = 0.0
		sum_trees = 0.0
		for current_tree in range(0, self.number_of_trees_in_forest):

			current_tree_prediction = sample_prediction[current_tree]

			if not math.isnan(current_tree_prediction):
				used_trees = used_trees + 1
				sum_trees = sum_trees + current_tree_prediction

		if used_trees == 0.0:
			print("Something went wrong, no trees were used for a prediction")

		return sum_trees / used_trees

	def upsample_train_data(self, upsampling, X_train, y_train, class_assignment_samples_train, sample_names_train):
		'''
		@param X_train: the model matrix for the training samples
		@param y_train: the response values for the training samples
		@param class_assignment_samples_train: the class assignments of the training samples
		@param sample_names_train: the sample names of the training samples
		'''
		if upsampling == "simple":

			return self.upsample_train_data_simple(X_train, y_train, class_assignment_samples_train, sample_names_train)
		# may become supported
		# elif upsampling == "proportional":

		#        return upsample_train_data_proportional(X_train, y_train, class_assignment_samples_train, sample_names_train, threshold)
		else:

			print(
				"The given upsampling mode was not supported. Given mode is: " + upsampling + " Supported upsampling modes: simple")
			print("Using mode simple instead.")
			return self.upsample_train_data_simple(X_train, y_train, class_assignment_samples_train, sample_names_train)

	def calculate_prediction_single_sample_simple_weighted_average(self, sample_prediction,
																   leaf_assignment_current_sample, majority_class):
		'''
		@param sample_prediction: a dictionary of trees to the predictions for a particular sample
		@param leaf_assignment_current_sample: the leaf assignments of one sample to all of the trees
		@param majority_class: the majority class of the current sample

		'''

		weighted_average = 0.0
		sum_weights = 0.0
		for current_tree in range(0, self.number_of_trees_in_forest):

			current_tree_prediction = sample_prediction[current_tree]

			if not math.isnan(current_tree_prediction):

				leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][current_tree]

				if majority_class in self.tree_to_leaf_to_percentage_class_dict[current_tree][
					leaf_assignment_current_sample_current_tree]:  # can be that a node is purely of the other class, then it is not included

					current_sample_weight = \
					self.tree_to_leaf_to_percentage_class_dict[current_tree][leaf_assignment_current_sample_current_tree][
						majority_class]

					sum_weights = sum_weights + current_sample_weight

					weighted_average = weighted_average + current_sample_weight * current_tree_prediction

		if sum_weights == 0.0:
			print("Something went wrong, no trees were used for a prediction")

		return weighted_average / sum_weights

	def my_own_apply(self, X_train):
		'''
		@param X_train: training model matrix
		'''
		# Have to draw bootstrap samples on my own using the same random instance as RandomForestRegressor uses

		tree_to_leaf_to_samples_dict = {}
		for tree_idx in range(0, self.number_of_trees_in_forest):

			random_instance = numpy.random.RandomState(self.model.estimators_[tree_idx].random_state)
			samples_current_tree = random_instance.randint(0, X_train.shape[0], X_train.shape[0])
			bootstrap_samples_current_tree = numpy.array([X_train[sample_idx] for sample_idx in samples_current_tree])

			current_tree_leaf_assignment = self.model.estimators_[tree_idx].apply(bootstrap_samples_current_tree)

			for sample in range(0, len(samples_current_tree)):

				leaf_of_current_sample = current_tree_leaf_assignment[sample]

				true_sample_id = samples_current_tree[sample]

				if not tree_idx in tree_to_leaf_to_samples_dict:

					tree_to_leaf_to_samples_dict[tree_idx] = {leaf_of_current_sample: [true_sample_id]}

				else:
					if leaf_of_current_sample not in tree_to_leaf_to_samples_dict[tree_idx]:

						tree_to_leaf_to_samples_dict[tree_idx][leaf_of_current_sample] = [true_sample_id]


					else:
						tree_to_leaf_to_samples_dict[tree_idx][leaf_of_current_sample].append(true_sample_id)

		return tree_to_leaf_to_samples_dict

	def determine_majority_class(self, leaf_assignment_current_sample):
		'''
		@param leaf_assignment_current_sample: the leaf assignments of one sample to all of the trees
		@return best_class: the majority class of the samples in the leaf

		'''
		class_votes = {}
		for tree in self.tree_to_leaf_to_percentage_class_dict.keys():

			leaf = leaf_assignment_current_sample[0][tree]  # there is only a single sample and several trees

			current_best_percentage = 0.0
			best_prediction_class = None

			for prediction_class in self.tree_to_leaf_to_percentage_class_dict[tree][leaf]:

				percentage = self.tree_to_leaf_to_percentage_class_dict[tree][leaf][prediction_class]

				if percentage > current_best_percentage:
					best_prediction_class = prediction_class
					current_best_percentage = percentage

			if best_prediction_class not in class_votes:

				class_votes[best_prediction_class] = 1.0

			else:
				class_votes[best_prediction_class] = class_votes[best_prediction_class] + 1.0

		best_class = None
		best_votes = 0

		for prediction_class in class_votes.keys():
			if class_votes[prediction_class] > best_votes:
				best_votes = class_votes[prediction_class]
				best_class = prediction_class

		total_sum =  sum(class_votes.values()) #should be number of trees
		return [best_class, best_votes /total_sum, {key: value/total_sum for key, value in class_votes.items()} ]


	def fit(self):

		#decide which model should be fit: with weights, with upsampling, or plain
		self.train_sample_weights = []
		if self.sample_weights_included in ["simple", "linear"]:

			if not math.isnan(self.thresholds[0]):
				self.train_sample_weights = numpy.array(self.calculate_weights(self.thresholds, self.y_train, self.sample_weights_included))
				#print(len(self.y_train))
				#print(len(self.train_sample_weights))
				#print(len(self.sample_names_train))
				print_train_samples_to_file(self.sample_names_train, self.train_sample_weights, self.sample_info_file)
				start_time = time.perf_counter()
				self.model.fit(self.X_train, self.y_train, sample_weight=self.train_sample_weights)
				end_time = time.perf_counter()
				self.elapsed_time = end_time - start_time
			else:

				print("No Threshold was given to calculate sample weights.")
				print("Please provide a threshold. Calculations aborted.")

		elif not self.upsampling == "":#in this case the data set will be changed (X_train, y_train etc.) and original_X_train etc. contains the original data
			new_training_data = self.upsample_train_data(self.upsampling, self.X_train, self.y_train, self.class_assignment_samples_train,
													self.sample_names_train)
			up_Xtrain = new_training_data[0]
			up_ytrain = new_training_data[1]
			up_sample_names_train = new_training_data[2]
			new_class_assignment_samples_train = new_training_data[3]

			print_train_samples_to_file_upsample(up_sample_names_train, self.sample_info_file)

			self.X_train = numpy.array(up_Xtrain)
			self.y_train = numpy.array(up_ytrain)
			self.sample_names_train = up_sample_names_train
			self.class_assignment_samples_train = new_class_assignment_samples_train
			self.train_sample_weights = [1] * len(self.sample_names_train)
			start_time = time.perf_counter()
			self.model.fit(self.X_train, self.y_train)
			end_time = time.perf_counter()
			self.elapsed_time = end_time - start_time

		else:
			self.train_sample_weights = [1] * len(self.sample_names_train)
			start_time = time.perf_counter()
			self.model.fit(self.X_train, self.y_train)
			end_time = time.perf_counter()
			self.elapsed_time = end_time - start_time

		number_of_samples_train = self.X_train.shape[0]
		#number_of_samples_test = self.X_test.shape[0]

		feature_imp_fit_model = self.model.feature_importances_

		print_feature_importance_to_file(feature_imp_fit_model, self.feature_imp_output_file, self.feature_names)

		# get table with leaf-assignment for each train sample in each tree

		# Have to write this method on my own because apply does not know which samples were not in a bootstrap sample of a tree
		# This is the code that would do it if if the bootstrap samples were handled correctly
		# leaf_assignment_train_data = model.apply(X_train)
		# convert table to dict
		# tree_to_leaf_to_samples_dict = convert_leaf_assignment_to_dict(leaf_assignment_train_data)

		self.tree_to_leaf_to_samples_dict = self.my_own_apply(self.X_train)

		# Can be removed when it has been tested for several samples
		test_equality_by_prediction(self.tree_to_leaf_to_samples_dict, self.model, self.X_train, self.y_train, self.train_sample_weights)

		# global global_tree_to_leaf_to_samples_dict
		# global_tree_to_leaf_to_samples_dict = tree_to_leaf_to_samples_dict

		# global if_test_set_used
		# if_test_set_used = False
		write_leaf_assignment_to_file(self.leaf_assignment_file_train, self.tree_to_leaf_to_samples_dict, self.sample_names_train)

		#test data will later be used separately
		#leaf_assignment_all_test_data = model.apply(X_test)
		#tree_to_leaf_to_samples_dict_all_test_data = convert_leaf_assignment_to_dict(leaf_assignment_all_test_data)
		#write_leaf_assignment_to_file(leaf_assignment_file_test, tree_to_leaf_to_samples_dict_all_test_data,
		#							  sample_names_test)

		# Determine purity (and majority class) of the leafs per tree
		# Determine percentage of samples in specific class per tree leaf of one tree

		self.tree_to_leaf_to_percentage_class_dict = {}

		if self.sample_weights_included in ["simple"]:
			self.tree_to_leaf_to_percentage_class_dict = self.determine_percentage_of_classes_sample_weights(self.tree_to_leaf_to_samples_dict, self.class_assignment_samples_train, self.train_sample_weights)
		else:
			self.tree_to_leaf_to_percentage_class_dict = self.determine_percentage_of_classes(self.tree_to_leaf_to_samples_dict,
																					self.class_assignment_samples_train)

		# To understand the distribution of values in nodes, we use some variance information
		# This information can also be used to pick specific trees for predicting

		self.tree_to_leaf_to_sample_variance_dict = self.determine_variance_of_samples(self.tree_to_leaf_to_samples_dict, self.y_train)

	def is_majority_in_current_tree(self, majority_class_sample, tree_distribution):
		'''
		@param majority_class_sample: the majority class of the current sample
		@param tree_distribution:  a dictionary with the distribution of classes in the current tree

		'''

		best_class_of_tree = None
		best_percentage = 0.0
		for prediction_class in tree_distribution.keys():

			current_percentage = tree_distribution[prediction_class]

			if current_percentage > best_percentage:
				best_percentage = current_percentage
				best_class_of_tree = prediction_class

		if majority_class_sample == best_class_of_tree:

			return True

		else:
			return False

	def calculate_predictions_binary_sensitive_average_q(self, X_test, number_of_samples):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param number_of_samples: the number of samples for which to perform the calculations

		'''

		predictions_samples = []
		classification_samples = []
		certainty_samples = []
		leaf_purity_samples = []
		leaf_variance_samples = []

		for sample in range(0, number_of_samples):

			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

			majority_class_and_class_probabilities = self.determine_majority_class(leaf_assignment_current_sample)
			class_probabilities = majority_class_and_class_probabilities[2]
			majority_class_for_sample = majority_class_and_class_probabilities[0]
			percentage_majority_class = majority_class_and_class_probabilities[1]
			classification_samples.append(majority_class_for_sample)
			certainty_samples.append(class_probabilities)
			predictions_trees = {}
			percentages_all_trees = {}
			variances_all_trees = {}

			if int(majority_class_for_sample) == self.majority_class_training:

				for tree in range(0, self.number_of_trees_in_forest):
					leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][
						tree]  # is an ndarray
					tree_x = self.model.estimators_[tree]
					pred_single_tree = self.my_own_predict_q(tree, leaf_assignment_current_sample_current_tree)

					if tree not in predictions_trees:

						predictions_trees[tree] = pred_single_tree

					else:
						print("Found tree " + str(tree) + " twice for prediction.")

					percentage_current_tree = 0

					if majority_class_for_sample in self.tree_to_leaf_to_percentage_class_dict[tree][
						leaf_assignment_current_sample_current_tree]:
						percentage_current_tree = \
						self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
							majority_class_for_sample]

					if tree not in percentages_all_trees:
						percentages_all_trees[tree] = percentage_current_tree

					else:
						print("Found tree " + str(tree) + "twice")

					variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
						leaf_assignment_current_sample_current_tree]

					if tree not in variances_all_trees:
						variances_all_trees[tree] = variance_current_tree

					else:
						print("Found tree " + str(tree) + "twice")

			else:
				for tree in range(0, self.number_of_trees_in_forest):

					leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][
						tree]  # is an ndarray

					is_majority_current_tree = self.is_majority_in_current_tree(majority_class_for_sample,
																		   self.tree_to_leaf_to_percentage_class_dict[tree][
																			   leaf_assignment_current_sample_current_tree])

					if is_majority_current_tree:
						tree_x = self.model.estimators_[tree]
						pred_single_tree = self.my_own_predict_q(tree, leaf_assignment_current_sample_current_tree)

						if tree not in predictions_trees:

							predictions_trees[tree] = pred_single_tree

						else:
							print("Found tree " + str(tree) + " twice for prediction.")

						percentage_current_tree = \
						self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
							majority_class_for_sample]

						if tree not in percentages_all_trees:
							percentages_all_trees[tree] = percentage_current_tree

						else:
							print("Found tree " + str(tree) + "twice")

						variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
							leaf_assignment_current_sample_current_tree]

						if tree not in variances_all_trees:
							variances_all_trees[tree] = variance_current_tree

						else:
							print("Found tree " + str(tree) + "twice")
					else:

						pred_single_tree = float("NaN")

						if tree not in predictions_trees:
							predictions_trees[
								tree] = pred_single_tree  # was not included as a tree because majority was not the right class

						else:
							print("Found tree " + str(tree) + " twice for prediction.")

						if tree not in percentages_all_trees:
							percentages_all_trees[tree] = float("NaN")

						else:
							print("Found tree " + str(tree) + "twice")

						if tree not in variances_all_trees:
							variances_all_trees[tree] = float("NaN")

						else:
							print("Found tree " + str(tree) + "twice")

			predictions_samples.append(predictions_trees)
			leaf_purity_samples.append(percentages_all_trees)
			leaf_variance_samples.append(variances_all_trees)

		return (
		[predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])

	def calculate_predictions_binary_sensitive_average(self, X_test, number_of_samples):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param number_of_samples: the number of samples for which to perform the calculations

		'''

		predictions_samples = []
		classification_samples = []
		certainty_samples = []
		leaf_purity_samples = []
		leaf_variance_samples = []

		for sample in range(0, number_of_samples):

			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

			majority_class_and_class_probabilities = self.determine_majority_class(leaf_assignment_current_sample)
			class_probabilities = majority_class_and_class_probabilities[2]
			majority_class_for_sample = majority_class_and_class_probabilities[0]
			percentage_majority_class = majority_class_and_class_probabilities[1]
			classification_samples.append(majority_class_for_sample)
			certainty_samples.append(class_probabilities)
			predictions_trees = {}
			percentages_all_trees = {}
			variances_all_trees = {}

			if int(majority_class_for_sample) == self.majority_class_training:

				for tree in range(0, self.number_of_trees_in_forest):
					leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][
						tree]  # is an ndarray
					tree_x = self.model.estimators_[tree]
					pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

					if tree not in predictions_trees:

						predictions_trees[tree] = pred_single_tree[0]  # is an array

					else:
						print("Found tree " + str(tree) + " twice for prediction.")

					percentage_current_tree = 0

					if majority_class_for_sample in self.tree_to_leaf_to_percentage_class_dict[tree][
						leaf_assignment_current_sample_current_tree]:
						percentage_current_tree = \
						self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
							majority_class_for_sample]

					if tree not in percentages_all_trees:
						percentages_all_trees[tree] = percentage_current_tree

					else:
						print("Found tree " + str(tree) + "twice")

					variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
						leaf_assignment_current_sample_current_tree]

					if tree not in variances_all_trees:
						variances_all_trees[tree] = variance_current_tree

					else:
						print("Found tree " + str(tree) + "twice")

			else:
				for tree in range(0, self.number_of_trees_in_forest):

					leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][
						tree]  # is an ndarray

					is_majority_current_tree = self.is_majority_in_current_tree(majority_class_for_sample,
																		   self.tree_to_leaf_to_percentage_class_dict[tree][
																			   leaf_assignment_current_sample_current_tree])

					if is_majority_current_tree:
						tree_x = self.model.estimators_[tree]
						pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

						if tree not in predictions_trees:

							predictions_trees[tree] = pred_single_tree[0]  # is an array

						else:
							print("Found tree " + str(tree) + " twice for prediction.")

						percentage_current_tree = \
						self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
							majority_class_for_sample]

						if tree not in percentages_all_trees:
							percentages_all_trees[tree] = percentage_current_tree

						else:
							print("Found tree " + str(tree) + "twice")

						variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
							leaf_assignment_current_sample_current_tree]

						if tree not in variances_all_trees:
							variances_all_trees[tree] = variance_current_tree

						else:
							print("Found tree " + str(tree) + "twice")
					else:

						pred_single_tree = float("NaN")

						if tree not in predictions_trees:
							predictions_trees[
								tree] = pred_single_tree  # was not included as a tree because majority was not the right class

						else:
							print("Found tree " + str(tree) + " twice for prediction.")

						if tree not in percentages_all_trees:
							percentages_all_trees[tree] = float("NaN")

						else:
							print("Found tree " + str(tree) + "twice")

						if tree not in variances_all_trees:
							variances_all_trees[tree] = float("NaN")

						else:
							print("Found tree " + str(tree) + "twice")

			predictions_samples.append(predictions_trees)
			leaf_purity_samples.append(percentages_all_trees)
			leaf_variance_samples.append(variances_all_trees)

		return (
		[predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])

	def calculate_prediction_single_sample_simple_average_q(self, sample_prediction):
		'''
		@param sample_prediction: a dictionary of trees to their predictions
		'''
		sample_to_weights_dict = {}
		number_of_used_trees = 0
		for sample in range(0, len(self.train_sample_weights)):
			number_of_used_trees = 0
			for current_tree in range(0, self.number_of_trees_in_forest):

				current_tree_samples_to_weights = sample_prediction[current_tree]

				if isinstance(current_tree_samples_to_weights, Mapping):
					number_of_used_trees = number_of_used_trees +1
					if sample in current_tree_samples_to_weights:
						current_weight = current_tree_samples_to_weights[sample]
						#print(current_weight)

						if not sample in sample_to_weights_dict:
							sample_to_weights_dict[sample] = [current_weight]
						else:
							sample_to_weights_dict[sample].append(current_weight)
		#print(number_of_used_trees)
		sample_to_final_weight_dict = {}
		#sumi = 0.0
		for sample in sample_to_weights_dict.keys():
			#print(sample_to_weights_dict[sample])
			final_weight = float(sum(sample_to_weights_dict[sample]))/float(number_of_used_trees)
			#sumi = sumi + final_weight
			#print(final_weight)
			sample_to_final_weight_dict[sample] = final_weight

		#print(sumi)
		assert math.isclose(sum(sample_to_final_weight_dict.values()), 1.0, abs_tol=10**-2) #should be probabilities

		return sample_to_final_weight_dict

	def quantile_prediction(self, quantile, train_samples_to_weight):

		#print(train_samples_to_weight)
		y_train_values_copy = enumerate(list(copy.deepcopy(self.y_train)))
		y_train_values_sorted = sorted(y_train_values_copy, key=operator.itemgetter(1))

		sum_weight = 0.0
		for index in range(0, len(self.y_train)):
			sample = y_train_values_sorted[index][0]
			current_y_value = y_train_values_sorted[index][1]

			if sample in train_samples_to_weight:
				current_weight = train_samples_to_weight[sample]
				sum_weight = sum_weight + current_weight

			if sum_weight >= quantile: # infimum y that fulfills the condition that the sum of the weights are larger than the wanted quantile
				return current_y_value

		print("Warning: sum of weights did not equal one")
		return y_train_values_sorted[len(self.y_train)-1][1]



	def calculate_final_predictions_binary_simple_average_q(self, predictions_samples, quantile):
		'''
		@param predictions_samples: a list of dictionaries of dictionary from samples to the trees and trees to samples to their weights
		'''

		final_predictions_samples = []
		for sample in range(0, len(predictions_samples)):
			train_samples_to_weights = self.calculate_prediction_single_sample_simple_average_q(predictions_samples[sample])
			inf_y_higher_than_quantile = self.quantile_prediction(quantile, train_samples_to_weights)
			final_predictions_samples.append(inf_y_higher_than_quantile)

		return final_predictions_samples

	def calculate_final_predictions_binary_simple_average(self, predictions_samples):
		'''
		@param predictions_samples: a list of samples with dictionary from samples to the trees and their prediction for each sample
		'''

		final_predictions_samples = []
		for sample in range(0, len(predictions_samples)):
			final_prediction_sample = self.calculate_prediction_single_sample_simple_average(predictions_samples[sample])
			final_predictions_samples.append(final_prediction_sample)

		return final_predictions_samples
	def calculate_predictions_binary_simple_average(self, X_test, number_of_samples):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param number_of_samples: the number of samples for which to perform the calculations
		'''

		predictions_samples = []
		classification_samples = []
		certainty_samples = []
		leaf_purity_samples = []
		leaf_variance_samples = []

		for sample in range(0, number_of_samples):

			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

			majority_class_and_class_probabilities = self.determine_majority_class(leaf_assignment_current_sample)
			class_probabilities = majority_class_and_class_probabilities[2]
			majority_class_for_sample = majority_class_and_class_probabilities[0]
			percentage_majority_class = majority_class_and_class_probabilities[1]
			classification_samples.append(majority_class_for_sample)
			certainty_samples.append(class_probabilities)
			predictions_trees = {}
			percentages_all_trees = {}
			variances_all_trees = {}
			for tree in range(0, self.number_of_trees_in_forest):

				leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

				is_majority_current_tree = self.is_majority_in_current_tree(majority_class_for_sample,
																	   self.tree_to_leaf_to_percentage_class_dict[tree][
																		   leaf_assignment_current_sample_current_tree])

				if is_majority_current_tree:
					tree_x = self.model.estimators_[tree]
					pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

					if tree not in predictions_trees:

						predictions_trees[tree] = pred_single_tree[0]  # is an array

					else:
						print("Found tree " + str(tree) + " twice for prediction.")

					percentage_current_tree = \
					self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
						majority_class_for_sample]

					if tree not in percentages_all_trees:
						percentages_all_trees[tree] = percentage_current_tree

					else:
						print("Found tree " + str(tree) + "twice")

					variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
						leaf_assignment_current_sample_current_tree]

					if tree not in variances_all_trees:
						variances_all_trees[tree] = variance_current_tree

					else:
						print("Found tree " + str(tree) + "twice")
				else:

					pred_single_tree = float("NaN")

					if tree not in predictions_trees:
						predictions_trees[
							tree] = pred_single_tree  # was not included as a tree because majority was not the right class

					else:
						print("Found tree " + str(tree) + " twice for prediction.")

					if tree not in percentages_all_trees:
						percentages_all_trees[tree] = float("NaN")

					else:
						print("Found tree " + str(tree) + "twice")

					if tree not in variances_all_trees:
						variances_all_trees[tree] = float("NaN")

					else:
						print("Found tree " + str(tree) + "twice")

			predictions_samples.append(predictions_trees)
			leaf_purity_samples.append(percentages_all_trees)
			leaf_variance_samples.append(variances_all_trees)

		return (
		[predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])


	def my_own_predict_q(self, tree_number, leaf_assignment_current_sample):

		samples = self.tree_to_leaf_to_samples_dict[tree_number][leaf_assignment_current_sample]
		#print(samples)
		samples_to_weights = {}

		sum_weights = 0.0
		for sample in samples:

			sum_weights = self.train_sample_weights[sample] + sum_weights

			if sample not in samples_to_weights:
				samples_to_weights[sample] =float(self.train_sample_weights[sample])
			else:
				samples_to_weights[sample] = samples_to_weights[sample] + float(self.train_sample_weights[sample])


		for sample in  list(set(samples)):

			samples_to_weights[sample] = samples_to_weights[sample]/float(sum_weights)
			
		return samples_to_weights

	def calculate_predictions_binary_simple_average_q(self, X_test, number_of_samples):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param number_of_samples: the number of samples for which to perform the calculations
		'''

		predictions_samples = []
		classification_samples = []
		certainty_samples = []
		leaf_purity_samples = []
		leaf_variance_samples = []

		for sample in range(0, number_of_samples):

			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

			majority_class_and_class_probabilities = self.determine_majority_class(leaf_assignment_current_sample)
			class_probabilities = majority_class_and_class_probabilities[2]
			majority_class_for_sample = majority_class_and_class_probabilities[0]
			percentage_majority_class = majority_class_and_class_probabilities[1]
			classification_samples.append(majority_class_for_sample)
			certainty_samples.append(class_probabilities)
			predictions_trees = {}
			percentages_all_trees = {}
			variances_all_trees = {}
			for tree in range(0, self.number_of_trees_in_forest):

				leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

				is_majority_current_tree = self.is_majority_in_current_tree(majority_class_for_sample,
																	   self.tree_to_leaf_to_percentage_class_dict[tree][
																		   leaf_assignment_current_sample_current_tree])

				if is_majority_current_tree:
					pred_single_tree = self.my_own_predict_q(tree, leaf_assignment_current_sample_current_tree)

					if tree not in predictions_trees:

						predictions_trees[tree] = pred_single_tree

					else:
						print("Found tree " + str(tree) + " twice for prediction.")

					percentage_current_tree = \
					self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
						majority_class_for_sample]

					if tree not in percentages_all_trees:
						percentages_all_trees[tree] = percentage_current_tree

					else:
						print("Found tree " + str(tree) + "twice")

					variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
						leaf_assignment_current_sample_current_tree]

					if tree not in variances_all_trees:
						variances_all_trees[tree] = variance_current_tree

					else:
						print("Found tree " + str(tree) + "twice")
				else:

					pred_single_tree = float("NaN")

					if tree not in predictions_trees:
						predictions_trees[
							tree] = pred_single_tree  # was not included as a tree because majority was not the right class

					else:
						print("Found tree " + str(tree) + " twice for prediction.")

					if tree not in percentages_all_trees:
						percentages_all_trees[tree] = float("NaN")

					else:
						print("Found tree " + str(tree) + "twice")

					if tree not in variances_all_trees:
						variances_all_trees[tree] = float("NaN")

					else:
						print("Found tree " + str(tree) + "twice")

			predictions_samples.append(predictions_trees)
			leaf_purity_samples.append(percentages_all_trees)
			leaf_variance_samples.append(variances_all_trees)

		return (
		[predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])

	def calculate_final_predictions_binary_simple_weighted_average(self, X_test, predictions_samples,
																   classification_predictions):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param predictions_samples: a dictionary from samples to the trees and their prediction for each sample
		@param classification_predictions: the classification for all of the samples accodring to SAURON-RF
		@return final_predictions_samples: the weighted SAURON-RF predictions

		'''
		final_predictions_samples = []

		for sample in range(0, len(predictions_samples)):
			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees
			majority_class = classification_predictions[sample]
			final_prediction_sample = self.calculate_prediction_single_sample_simple_weighted_average(
				predictions_samples[sample],
				leaf_assignment_current_sample, majority_class)
			final_predictions_samples.append(final_prediction_sample)

		return final_predictions_samples


	def calculate_prediction_single_sample_simple_weighted_average_q(self, sample_prediction, leaf_assignment_current_sample, majority_class):
		'''
		@param sample_prediction: a dictionary of trees to a dictionary of samples to their weights
		@param leaf_assignment_current_sample: the leaf assignments of one sample to all of the trees
		@param majority_class: the majority class of the current sample
		'''

		sample_to_weights_dict = {}
		for sample in range(0, len(self.train_sample_weights)):
			sum_weights = 0.0
			weighted_average = 0.0
			for current_tree in range(0, self.number_of_trees_in_forest):
				leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][current_tree]
				current_tree_samples_to_weights = sample_prediction[current_tree]

				if isinstance(current_tree_samples_to_weights, Mapping):

					if majority_class in self.tree_to_leaf_to_percentage_class_dict[current_tree][leaf_assignment_current_sample_current_tree]:  # can be that a node is purely of the other class, then it is not included

						current_sample_weight = self.tree_to_leaf_to_percentage_class_dict[current_tree][
								leaf_assignment_current_sample_current_tree][
								majority_class]

						#print(current_sample_weight)
						sum_weights = sum_weights + current_sample_weight

						if sample in current_tree_samples_to_weights:

							current_weight = current_tree_samples_to_weights[sample]
							#sum_weights = sum_weights + current_sample_weight

							weighted_average = weighted_average + current_sample_weight * current_weight
			if sum_weights != 0.0:

				if not sample in sample_to_weights_dict:
					sample_to_weights_dict[sample] = float(weighted_average)/float(sum_weights)
				else:
					print("Found sample twice")

		#print(sample_to_weights_dict)
		#print(len(sample_to_weights_dict))
		#print(sum(sample_to_weights_dict.values()))
		assert math.isclose(sum(sample_to_weights_dict.values()) , 1.0, abs_tol=10**-2) #should be probabilities

		return sample_to_weights_dict

	def calculate_final_predictions_binary_simple_weighted_average_q(self, X_test, predictions_samples,
																   classification_predictions, quantile):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param predictions_samples: a dictionary from samples to the trees and their prediction for each sample
		@param classification_predictions: the classification for all of the samples accodring to SAURON-RF
		@param quantile: the quantile for quantile prediction
		@return final_predictions_samples: the weighted SAURON-RF predictions

		'''
		final_predictions_samples = []

		for sample in range(0, len(predictions_samples)):
			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees
			majority_class = classification_predictions[sample]
			train_samples_to_weights = self.calculate_prediction_single_sample_simple_weighted_average_q(
				predictions_samples[sample], leaf_assignment_current_sample, majority_class)
			inf_y_higher_than_quantile = self.quantile_prediction(quantile, train_samples_to_weights)
			final_predictions_samples.append(inf_y_higher_than_quantile)

		return final_predictions_samples

	def calculate_predictions_binary_simple_weighted_average(self, X_test, number_of_samples):
		'''
		@param X_test: a list of samples for which to perform the calculations
		@param number_of_samples: the number of samples for which to perform the calculations

		'''

		predictions_samples = []
		classification_samples = []
		certainty_samples = []
		leaf_purity_samples = []
		leaf_variance_samples = []

		for sample in range(0, number_of_samples):

			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

			majority_class_and_class_probabilities = self.determine_majority_class(leaf_assignment_current_sample)
			class_probabilities = majority_class_and_class_probabilities[2]
			majority_class_for_sample = majority_class_and_class_probabilities[0]
			percentage_majority_class = majority_class_and_class_probabilities[1]
			classification_samples.append(majority_class_for_sample)
			certainty_samples.append(class_probabilities)
			predictions_trees = {}
			percentages_all_trees = {}
			variances_all_trees = {}
			for tree in range(0, self.number_of_trees_in_forest):

				tree_x = self.model.estimators_[tree]
				pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

				if tree not in predictions_trees:

					predictions_trees[tree] = pred_single_tree[0]  # is an array

				else:
					print("Found tree " + str(tree) + " twice for prediction.")

				leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray
				percentage_current_tree = 0
				if majority_class_for_sample in self.tree_to_leaf_to_percentage_class_dict[tree][
					leaf_assignment_current_sample_current_tree]:
					percentage_current_tree = \
					self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
						majority_class_for_sample]

				if tree not in percentages_all_trees:
					percentages_all_trees[tree] = percentage_current_tree

				else:
					print("Found tree " + str(tree) + "twice")

				variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
					leaf_assignment_current_sample_current_tree]

				if tree not in variances_all_trees:
					variances_all_trees[tree] = variance_current_tree
				else:
					print("Found tree " + str(tree) + "twice")

			leaf_purity_samples.append(percentages_all_trees)

			predictions_samples.append(predictions_trees)
			leaf_variance_samples.append(variances_all_trees)

		return (
		[predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])


	def calculate_predictions_binary_simple_weighted_average_q(self, X_test, number_of_samples):
		'''
		@param X_test: a list of samples for which to perform the calculations
		@param number_of_samples: the number of samples for which to perform the calculations

		'''

		predictions_samples = []
		classification_samples = []
		certainty_samples = []
		leaf_purity_samples = []
		leaf_variance_samples = []

		for sample in range(0, number_of_samples):

			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

			majority_class_and_class_probabilities = self.determine_majority_class(leaf_assignment_current_sample)
			class_probabilities = majority_class_and_class_probabilities[2]
			majority_class_for_sample = majority_class_and_class_probabilities[0]
			percentage_majority_class = majority_class_and_class_probabilities[1]

			classification_samples.append(majority_class_for_sample)
			certainty_samples.append(class_probabilities)
			predictions_trees = {}
			percentages_all_trees = {}
			variances_all_trees = {}
			for tree in range(0, self.number_of_trees_in_forest):
				leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree] #is an nd array

				pred_single_tree = self.my_own_predict_q(tree, leaf_assignment_current_sample_current_tree)

				if tree not in predictions_trees:

					predictions_trees[tree] = pred_single_tree

				else:
					print("Found tree " + str(tree) + " twice for prediction.")

				percentage_current_tree = 0
				if majority_class_for_sample in self.tree_to_leaf_to_percentage_class_dict[tree][
					leaf_assignment_current_sample_current_tree]:
					percentage_current_tree = \
					self.tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
						majority_class_for_sample]

				if tree not in percentages_all_trees:
					percentages_all_trees[tree] = percentage_current_tree

				else:
					print("Found tree " + str(tree) + "twice")

				variance_current_tree = self.tree_to_leaf_to_sample_variance_dict[tree][
					leaf_assignment_current_sample_current_tree]

				if tree not in variances_all_trees:
					variances_all_trees[tree] = variance_current_tree
				else:
					print("Found tree " + str(tree) + "twice")

			leaf_purity_samples.append(percentages_all_trees)

			predictions_samples.append(predictions_trees)
			leaf_variance_samples.append(variances_all_trees)

		return (
		[predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])

	def calculate_final_predictions_binary_sensitive_weighted_average(self, X_test, predictions_samples,
																	  classification_predictions,
																	  majority_class_training):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param predictions_samples: a dictionary from samples to the trees and their prediction for each sample
		@param classification_predictions: the classification for all of the samples accodring to SAURON-RF
		@param majority_class_training: the majority class of the training samples
		@return final_predictions_samples: the weighted SAURON-RF predictions

		'''

		final_predictions_samples = []

		for sample in range(0, len(predictions_samples)):
			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees
			majority_class = classification_predictions[sample]

			if int(majority_class) == majority_class_training:  # just usual random forest average without weighting
				final_prediction_sample = self.calculate_prediction_single_sample_simple_average(predictions_samples[sample])
				final_predictions_samples.append(final_prediction_sample)

			else:
				final_prediction_sample = self.calculate_prediction_single_sample_simple_weighted_average(
					predictions_samples[sample],
					leaf_assignment_current_sample, majority_class)
				final_predictions_samples.append(final_prediction_sample)

		return final_predictions_samples

	def calculate_final_predictions_binary_sensitive_weighted_average_q(self, X_test, predictions_samples,
																	  classification_predictions,
																	  majority_class_training, quantile):

		'''
		@param X_test: a list of samples for which to perform the calculations
		@param predictions_samples: a dictionary from samples to the trees and their prediction for each sample
		@param classification_predictions: the classification for all of the samples accodring to SAURON-RF
		@param majority_class_training: the majority class of the training samples
		@return final_predictions_samples: the weighted SAURON-RF predictions

		'''

		final_predictions_samples = []

		for sample in range(0, len(predictions_samples)):
			current_sample = X_test[sample, :]  # current sample
			leaf_assignment_current_sample = self.model.apply(
				current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees
			majority_class = classification_predictions[sample]

			if int(majority_class) == majority_class_training:  # just usual random forest average without weighting
				train_samples_to_weights = self.calculate_prediction_single_sample_simple_average_q(
					predictions_samples[sample])
				inf_y_higher_than_quantile = self.quantile_prediction(quantile, train_samples_to_weights)
				final_predictions_samples.append(inf_y_higher_than_quantile)

			else:
				train_samples_to_weights = self.calculate_prediction_single_sample_simple_weighted_average_q(
					predictions_samples[sample], leaf_assignment_current_sample, majority_class)
				inf_y_higher_than_quantile = self.quantile_prediction(quantile, train_samples_to_weights)
				final_predictions_samples.append(inf_y_higher_than_quantile)

		return final_predictions_samples


	def predict_binary_no_weights(self, X_test, quantile, X_train_cal):

		if math.isnan(quantile):
			if X_train_cal:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_average(self.original_Xtrain,
																										   self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_average(self.predictions_samples_train)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_average(X_test,
																									  X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_average(self.predictions_samples_test)

		else:
			if X_train_cal:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_average_q(
					self.original_Xtrain,
					self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_average_q(
					self.predictions_samples_train, quantile)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_average_q(X_test,
																												X_test.shape[
																													0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_average_q(
				self.predictions_samples_test, quantile)

	def predict_binary_no_weights_sensitive(self, X_test, quantile, X_train_calc):
		# Predict with own routine

		if math.isnan(quantile):
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_sensitive_average(self.original_Xtrain,
																											  self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_average(self.predictions_samples_train)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_sensitive_average(X_test,
																										 X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_average(self.predictions_samples_test)

		else:
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_sensitive_average_q(
					self.original_Xtrain,
					self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_average_q(
					self.predictions_samples_train, quantile)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_sensitive_average_q(
				X_test,
				X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_average_q(
				self.predictions_samples_test, quantile)

	def predict_binary_weights(self, X_test, quantile, X_train_calc):

		if math.isnan(quantile):
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_average(self.original_Xtrain,
																										   self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_weighted_average(self.original_Xtrain,
																											 self.predictions_samples_train,
																											 self.classification_prediction_samples_train)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_average(X_test,
																									  X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_weighted_average(X_test,
																										self.predictions_samples_test,
																										self.classification_prediction_samples_test)

		else:
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_average_q(
					self.original_Xtrain,
					self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_weighted_average_q(
					self.original_Xtrain,
					self.predictions_samples_train,
					self.classification_prediction_samples_train, quantile)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_average_q(X_test,
																												X_test.shape[
																													0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_weighted_average_q(
				X_test,
				self.predictions_samples_test,
				self.classification_prediction_samples_test, quantile)

	def predict_majority_weights(self, X_test, quantile, X_train_calc):

		if math.isnan(quantile):
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_weighted_average(
					self.original_Xtrain,
					self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_weighted_average(self.original_Xtrain,
																											 self.predictions_samples_train,
																											 self.classification_prediction_samples_train)

				self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_weighted_average(X_test,
																												   X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_weighted_average(X_test,
																										self.predictions_samples_test,
																										self.classification_prediction_samples_test)

		else:
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_weighted_average_q(
					self.original_Xtrain,
					self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_simple_weighted_average_q(
					self.original_Xtrain,
					self.predictions_samples_train,
					self.classification_prediction_samples_train, quantile)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_weighted_average_q(
					X_test,
					X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_simple_weighted_average_q(X_test,
																												  self.predictions_samples_test,
																												  self.classification_prediction_samples_test, quantile)

	def predict_majority_weights_sensitive(self, X_test, quantile, X_train_calc):

		if math.isnan(quantile):
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_weighted_average(
					self.original_Xtrain,
					self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_sensitive_weighted_average(self.original_Xtrain,
																												self.predictions_samples_train,
																												self.classification_prediction_samples_train,
																												self.majority_class_training)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_weighted_average(X_test,
																											   X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_sensitive_weighted_average(self.X_test,
																										   self.predictions_samples_test,
																										   self.classification_prediction_samples_test,
																										   self.majority_class_training)

		else:
			if X_train_calc:
				self.predictions_samples_train_and_classification = self.calculate_predictions_binary_simple_weighted_average_q(
					self.original_Xtrain,
					self.original_Xtrain.shape[0])
				self.predictions_samples_train = self.predictions_samples_train_and_classification[0]
				self.classification_prediction_samples_train = self.predictions_samples_train_and_classification[1]
				self.certainty_samples_train = self.predictions_samples_train_and_classification[2]
				self.leaf_purity_samples_train = self.predictions_samples_train_and_classification[3]
				self.leaf_variance_samples_train = self.predictions_samples_train_and_classification[4]
				self.final_predictions_samples_train = self.calculate_final_predictions_binary_sensitive_weighted_average_q(
					self.original_Xtrain,
					self.predictions_samples_train,
					self.classification_prediction_samples_train,
					self.majority_class_training, quantile)

			self.predictions_samples_test_and_classification = self.calculate_predictions_binary_simple_weighted_average_q(
				X_test,
				X_test.shape[0])
			self.predictions_samples_test = self.predictions_samples_test_and_classification[0]
			self.classification_prediction_samples_test = self.predictions_samples_test_and_classification[1]
			self.certainty_samples_test = self.predictions_samples_test_and_classification[2]
			self.leaf_purity_samples_test = self.predictions_samples_test_and_classification[3]
			self.leaf_variance_samples_test = self.predictions_samples_test_and_classification[4]
			self.final_predictions_samples_test = self.calculate_final_predictions_binary_sensitive_weighted_average_q(
				self.X_test,
				self.predictions_samples_test,
				self.classification_prediction_samples_test,
				self.majority_class_training, quantile)


	def set_analysis_mode(self, analysis_mode, output_sample_prediction_file_train, train_error_file, feature_imp_output_file, output_leaf_purity_file_train, output_variance_file_train, output_sample_prediction_file_test, test_error_file,  output_leaf_purity_file_test, output_variance_file_test, leaf_assignment_file_test):#user has to ensure that new calculations are performed if the analysis mode is changed
		self.analysis_mode = analysis_mode
		self.output_sample_prediction_file_train = output_sample_prediction_file_train
		self.train_error_file = train_error_file
		self.feature_imp_output_file = feature_imp_output_file
		self.output_leaf_purity_file_train = output_leaf_purity_file_train
		self.output_variance_file_train = output_variance_file_train
		self.output_sample_prediction_file_test = output_sample_prediction_file_test
		self.test_error_file = test_error_file
		self.output_leaf_purity_file_test = output_leaf_purity_file_test
		self.output_variance_file_test = output_variance_file_test
		self.leaf_assignment_file_test = leaf_assignment_file_test

		print("Please ensure that you perform new predictions each time you change your analysis mode")
		return


	def predict(self, X_test, y_test, class_assignments_samples_test, sample_names_test, quantile = float('nan'), X_train_calc =True ):

		'''
				@param X_test: matrix with test samples
				@param sample_names_test: the names of the samples in the order of X_test
				@param quantile: if quantile regression is desired, this should be set to the desired quantile
				@param X_train_calc: should predictions be done for the train set too?

		'''


		self.X_test = X_test
		self.y_test = y_test
		self.class_assignment_samples_test = class_assignments_samples_test
		self.sample_names_test = sample_names_test

		if self.analysis_mode == "binary_no_weights":
			print("Analysis mode binary_no_weights started")

			self.predict_binary_no_weights(X_test, quantile, X_train_calc)

		elif self.analysis_mode == "binary_no_weights_sensitive":
			print("Analysis mode binary_no_weights_sensitive started")

			self.predict_binary_no_weights_sensitive(X_test, quantile, X_train_calc)

		elif self.analysis_mode == "binary_weights":
			print("Analysis mode binary_weights started")

			self.predict_binary_weights(X_test, quantile, X_train_calc)

		elif self.analysis_mode == "majority_weights":
			# Predict with own routine

			print("Analysis mode majority_weights started")
			self.predict_majority_weights(X_test, quantile, X_train_calc)

		elif self.analysis_mode == "majority_weights_sensitive":
			# Predict with own routine
			self.predict_majority_weights_sensitive(X_test, quantile, X_train_calc)

		else:
			print("The given analysis mode was not allowed. It was " + self.analysis_mode + ". Default mode majority_weights is being used now.")

			self.predict_majority_weights(X_test, quantile, X_train_calc)

		# Print solutions to file
		parameters = ["#Trees:" + str(self.number_of_trees_in_forest), "#MinSamplesLeaf:" + str(self.min_number_of_samples_per_leaf), "#FeaturesPerSplit:" + str(self.number_of_features_per_split)]

		write_time_to_file(self.time_file, self.elapsed_time, self.name_of_analysis, ",".join(parameters))

		if X_train_calc:
			print_prediction_to_file(self.predictions_samples_train, self.final_predictions_samples_train, self.number_of_trees_in_forest, self.original_Xtrain.shape[0], self.output_sample_prediction_file_train, self.original_sample_names_train, self.certainty_samples_train, self.classification_prediction_samples_train, self.class_count)
			print_error_to_file(self.train_error_file, self.name_of_analysis, ",".join(parameters), self.mse_included, self.classification_included, self.final_predictions_samples_train, self.original_ytrain, self.classification_prediction_samples_train, self.original_class_assignment_samples_train, self.all_available_labels)
			print_leaf_purity_per_sample_to_file(self.leaf_purity_samples_train,  self.number_of_trees_in_forest, self.original_Xtrain.shape[0], self.output_leaf_purity_file_train, self.original_sample_names_train)
			print_leaf_purity_per_sample_to_file(self.leaf_variance_samples_train, self.number_of_trees_in_forest, self.original_Xtrain.shape[0], self.output_variance_file_train, self.original_sample_names_train, "variance")


		print_prediction_to_file(self.predictions_samples_test, self.final_predictions_samples_test, self.number_of_trees_in_forest,
									X_test.shape[0], self.output_sample_prediction_file_test, sample_names_test, self.certainty_samples_test, self.classification_prediction_samples_test, self.class_count)
		print_error_to_file(self.test_error_file, self.name_of_analysis, ",".join(parameters), self.mse_included,
							self.classification_included, self.final_predictions_samples_test, self.y_test,
							self.classification_prediction_samples_test, self.class_assignment_samples_test, self.all_available_labels)
		print_leaf_purity_per_sample_to_file(self.leaf_purity_samples_test,  self.number_of_trees_in_forest, X_test.shape[0], self.output_leaf_purity_file_test, sample_names_test)
		print_leaf_purity_per_sample_to_file(self.leaf_variance_samples_test, self.number_of_trees_in_forest, X_test.shape[0], self.output_variance_file_test, sample_names_test, "variance")

		return [self.final_predictions_samples_test, self.certainty_samples_test]

	def calculate_weights(self, threshold, response_values, weighting_scheme):
		'''
		@param threshold: the thresholds that are used to discretize the response variables (increasingly sorted)
		@param response_values: the continuous response values for each sample
		@param weighting_scheme: weighting scheme used for calculating the weights
		@return weights: the calculated weights for each sample (in the order of the response values)

		'''

		if weighting_scheme == "simple":
			return self.calculate_simple_weights(threshold, response_values)
		elif weighting_scheme == "linear":
			return self.calculate_linear_weights(threshold, response_values)

		# elif weighting_scheme == "quadratic":
		#	return calculate_quadratic_weights(threshold, response_values)

		else:
			print("The given weighting scheme is not supported. Supported schemes are: simple.")
			print("Using weighting sheme simple instead")
			return self.calculate_simple_weights(threshold, response_values)



	def calculate_linear_weights(self, sorted_thresholds, response_values):
		'''
		@param sorted_thresholds: the thresholds that are used to discretize the response variables (increasingly sorted)
		@param response_values: the continuous response values for each sample
		@return weights: the calculated weights for each sample (in the order of the response values)

		'''

		assert (all(sorted_thresholds[i] <= sorted_thresholds[i+1] for i in range(len(sorted_thresholds)-1))),"Thresholds not sorted"



		sum_weights = [0] * (len(sorted_thresholds)+1)

		for value in response_values:

			found_threshold = False
			for i in range(len(sorted_thresholds)+1):

				if not found_threshold:


					if i == 0 and value < sorted_thresholds[i]:

						found_threshold = True
						sum_weights[i] = sum_weights[i] +  abs(value - sorted_thresholds[i])

					elif i == len(sorted_thresholds):
						found_threshold = True
						sum_weights[i] = sum_weights[i] + abs(value - sorted_thresholds[i-1])

					elif value < sorted_thresholds[i]:

						found_threshold = True
						sum_weights[i] = sum_weights[i] + abs(value - sorted_thresholds[i]) + abs(value -sorted_thresholds[i-1])


		weights = []
		norm_factor = len(sorted_thresholds) +1

		for value in response_values:
			found_threshold = False
			for i in range(len(sorted_thresholds)+1):

				if not found_threshold:

					if i == 0 and value < sorted_thresholds[i]:

						found_threshold = True
						new_weight =  abs(value - sorted_thresholds[i]) / (sum_weights[i]*norm_factor)
						weights.append(new_weight)

					elif i == len(sorted_thresholds):
						found_threshold = True
						new_weight = abs(value - sorted_thresholds[i - 1]) /(sum_weights[i]*norm_factor)
						weights.append(new_weight)

					elif value < sorted_thresholds[i]:

						found_threshold = True
						sum_weights[i] = abs(value - sorted_thresholds[i]) + abs(value - sorted_thresholds[i - 1]) /(sum_weights[i]*norm_factor)

		return weights

	def calculate_simple_weights(self, sorted_thresholds, response_values):
		'''
		@param sorted_thresholds: the thresholds that are used to discretize the response variables (increasingly sorted)
		@param response_values: the continuous response values for each sample
		@return weights: the calculated weights for each sample (in the order of the response values)

		'''

		assert (all(sorted_thresholds[i] <= sorted_thresholds[i+1] for i in range(len(sorted_thresholds)-1))),"Thresholds not sorted"



		sum_weights = [0] * (len(sorted_thresholds)+1)

		for value in response_values:

			found_threshold = False
			for i in range(len(sorted_thresholds)+1):

				if not found_threshold:

					if i == len(sorted_thresholds):
						found_threshold = True
						sum_weights[i] = sum_weights[i] +1

					elif value < sorted_thresholds[i]:

						found_threshold = True
						sum_weights[i] = sum_weights[i] +1


		max_sum = max(sum_weights)
		weights = []


		for value in response_values:
			found_threshold = False
			for i in range(len(sorted_thresholds)+1):

				if not found_threshold:
					if i == len(sorted_thresholds):
						found_threshold = True
						new_weight = float(max_sum)/float(sum_weights[i])
						weights.append(new_weight)
					elif value < sorted_thresholds[i]:

						found_threshold = True
						new_weight = float(max_sum)/float(sum_weights[i])
						weights.append(new_weight)


		return weights

	def add_missing_samples(self, upsampled_samples, all_samples):
		'''
		@param upsampled_samples: the upsampled data set
		@param all_samples: all input (training) samples
		'''
		for sample in all_samples:

			if sample not in upsampled_samples:
				upsampled_samples.append(sample)

	def upsample_train_data_simple(self, X_train, y_train, class_assignment_samples_train, sample_names_train):
		'''
		@param X_train: the model matrix for the training samples
		@param y_train: the response values for the training samples
		@param class_assignment_samples_train: the class assignments of the training samples
		@param sample_names_train: the sample names of the training samples
		'''
		class_counts = {}

		for sample_idx in range(0, len(class_assignment_samples_train)):

			class_sample = class_assignment_samples_train[sample_idx]

			if class_sample not in class_counts:

				class_counts[class_sample] = [1, [sample_idx]]

			else:
				class_counts[class_sample][0] = class_counts[class_sample][0] + 1
				class_counts[class_sample][1].append(sample_idx)

		max_class = ""
		max_class_count = 0

		for class_as in class_counts.keys():

			current_class_count = class_counts[class_as][0]

			if current_class_count > max_class_count:
				max_class_count = current_class_count
				max_class = class_as

		upsampled_dict = {}

		for class_as in class_counts.keys():

			current_class_count = class_counts[class_as][0]

			if current_class_count != max_class_count:
				current_samples = class_counts[class_as][1]

				# print(max_class_count)
				upsampled_samples = resample(current_samples, replace=True, n_samples=max_class_count, random_state=223)

				self.add_missing_samples(upsampled_samples,
									current_samples)  # no sample of the minority class should be left out

				if class_as not in upsampled_dict:
					upsampled_dict[class_as] = upsampled_samples

				else:
					print("Sampled class " + str(class_as) + " twice.")


			else:
				current_samples = class_counts[class_as][1]

				upsampled_samples = current_samples

				if class_as not in upsampled_dict:
					upsampled_dict[class_as] = upsampled_samples

				else:
					print("Sampled class " + str(class_as) + " twice.")

		# print(upsampled_dict)
		new_Xtrain = []
		new_ytrain = []
		new_sample_names = []
		new_class_assignment_samples_train = []

		for class_as in upsampled_dict.keys():

			for sample_idx in upsampled_dict[class_as]:
				xtrain_sample = X_train[sample_idx]
				new_Xtrain.append(xtrain_sample)

				y_train_sample = y_train[sample_idx]
				new_ytrain.append(y_train_sample)

				new_sample_name = sample_names_train[sample_idx]
				new_sample_names.append(new_sample_name)

				new_class_assignment = class_assignment_samples_train[sample_idx]
				new_class_assignment_samples_train.append(new_class_assignment)

		return [new_Xtrain, new_ytrain, new_sample_names, new_class_assignment_samples_train]


	def determine_percentage_of_classes_sample_weights(self, tree_to_leaf_to_samples_dict, class_assignment_samples,
													   weights_samples):

		'''
		@param tree_to_leaf_to_samples_dict: a dictionary of dictionaries containing mapping of random forests trees to leaves to samples in the leaves
		@param class_assignment_samples: the class assignment of the samples according to the trained regression random forest
		@param weights_samples: weights of the samples
		@return tree_to_leaf_to_percentage_class_dict: the calculated percentages of the classes in the leaves

		'''
		tree_to_leaf_to_percentage_class_dict = {}

		for tree_number in tree_to_leaf_to_samples_dict.keys():

			for leaf in tree_to_leaf_to_samples_dict[tree_number].keys():

				current_samples_in_leaf = tree_to_leaf_to_samples_dict[tree_number][leaf]

				class_count = {}

				total_weight_count = 0
				for sample in current_samples_in_leaf:

					current_class = class_assignment_samples[sample]

					current_weight = weights_samples[sample]

					total_weight_count = total_weight_count + current_weight
					if current_class not in class_count:

						class_count[current_class] = current_weight

					else:
						class_count[current_class] = class_count[current_class] + current_weight

				class_percentage = {}

				for current_class in class_count:
					class_percentage[current_class] = class_count[current_class] / total_weight_count

				if tree_number not in tree_to_leaf_to_percentage_class_dict:

					tree_to_leaf_to_percentage_class_dict[tree_number] = {leaf: class_percentage}

				else:

					if leaf not in tree_to_leaf_to_percentage_class_dict[tree_number]:
						tree_to_leaf_to_percentage_class_dict[tree_number][leaf] = class_percentage

		return tree_to_leaf_to_percentage_class_dict

	def determine_percentage_of_classes(self, tree_to_leaf_to_samples_dict, class_assignment_samples):
		'''
		@param tree_to_leaf_to_samples_dict: a dictionary of dictionaries containing mapping of random forests trees to leaves to samples in the leaves
		@param class_assignment_samples: the class assignment of the samples according to the trained regression random forest
		@return tree_to_leaf_to_percentage_class_dict: the calculated percentages of the classes in the leaves

		'''
		tree_to_leaf_to_percentage_class_dict = {}

		for tree_number in tree_to_leaf_to_samples_dict.keys():

			for leaf in tree_to_leaf_to_samples_dict[tree_number].keys():

				current_samples_in_leaf = tree_to_leaf_to_samples_dict[tree_number][leaf]

				class_count = {}

				for sample in current_samples_in_leaf:

					current_class = class_assignment_samples[sample]

					if current_class not in class_count:

						class_count[current_class] = 1

					else:
						class_count[current_class] = class_count[current_class] + 1

				class_percentage = {}

				for current_class in class_count:
					class_percentage[current_class] = class_count[current_class] / len(current_samples_in_leaf)

				if tree_number not in tree_to_leaf_to_percentage_class_dict:

					tree_to_leaf_to_percentage_class_dict[tree_number] = {leaf: class_percentage}

				else:

					if leaf not in tree_to_leaf_to_percentage_class_dict[tree_number]:
						tree_to_leaf_to_percentage_class_dict[tree_number][leaf] = class_percentage

		return tree_to_leaf_to_percentage_class_dict

	def determine_variance_of_samples(self, tree_to_leaf_to_samples_dict, y_train):
		'''
		@param tree_to_leaf_to_samples_dict: a dictionary of dictionaries containing mapping of random forests trees to leaves to samples in the leaves
		@param y_train: the drug responses
		@return tree_to_leaf_to_sample_variance_dict: the calculated variances of the samples in the leaves

		'''
		tree_to_leaf_to_sample_variance_dict = {}

		for tree_number in tree_to_leaf_to_samples_dict.keys():

			for leaf in tree_to_leaf_to_samples_dict[tree_number].keys():

				current_samples_in_leaf = tree_to_leaf_to_samples_dict[tree_number][leaf]

				values_samples_in_leaf = []

				for sample in current_samples_in_leaf:
					current_value = y_train[sample]

					values_samples_in_leaf.append(current_value)

				leaf_variance = numpy.var(values_samples_in_leaf, ddof=1)

				if tree_number not in tree_to_leaf_to_sample_variance_dict:

					tree_to_leaf_to_sample_variance_dict[tree_number] = {leaf: leaf_variance}

				else:

					if leaf not in tree_to_leaf_to_sample_variance_dict[tree_number]:
						tree_to_leaf_to_sample_variance_dict[tree_number][leaf] = leaf_variance

		return tree_to_leaf_to_sample_variance_dict



def get_non_zero_feature_importances(feature_importances):
	'''
	@param feature_importances: features importances from random forest
	@return non_zero_indices: all features with non-zero feature importance
	'''
	non_zero_indices = []
	for idx in range(0, feature_importances.shape[0]):

		if feature_importances[idx] != 0.0:
			non_zero_indices.append(idx)
	return non_zero_indices




def print_prediction_to_file(predictions_samples, final_predictions_samples, number_of_trees, number_of_samples,
                             output_sample_prediction_file, sample_names, certainty_samples, classification_prediction, class_count):
	with open(output_sample_prediction_file, "w", encoding="utf-8") as output_file:

		#print(certainty_samples)
		first_line = ["Prediction_Tree_" + str(tree) for tree in range(0, number_of_trees)]

		first_line.append("Total_Prediction")

		ordered_classes = list(class_count.keys())
		first_line.extend(["Probability_Class_" +str(key) for key in ordered_classes])
		first_line.append("Class_Prediction")

		all_samples_all_predictions = []
		for current_sample in range(0, number_of_samples):

			one_sample_all_predictions = []
			sample_name = sample_names[current_sample]
			one_sample_all_predictions.append(sample_name)

			for current_tree in range(0, number_of_trees):
				one_sample_single_tree_prediction = predictions_samples[current_sample][current_tree]

				one_sample_all_predictions.append(one_sample_single_tree_prediction)

			one_sample_all_predictions.append(final_predictions_samples[current_sample])
			for cur_class in ordered_classes:

				if cur_class in certainty_samples[current_sample]:
					one_sample_all_predictions.append(certainty_samples[current_sample][cur_class])
				else:
					#print(certainty_samples[current_sample])
					one_sample_all_predictions.append(0)
			one_sample_all_predictions.append(classification_prediction[current_sample])

			all_samples_all_predictions.append(one_sample_all_predictions)

		output_file.write("\t".join(first_line) + "\n")

		for line in all_samples_all_predictions:
			output_file.write("\t".join([str(x) for x in line]) + "\n")


def print_leaf_purity_per_sample_to_file(leaf_purity_samples,  number_of_trees, number_of_samples,
                             output_leaf_purity_file, sample_names, purity_or_var = "purity"):
	with open(output_leaf_purity_file, "w", encoding="utf-8") as output_file:

		first_line = []
		if purity_or_var == "purity":
			first_line = ["Leaf_Purity_Tree_" + str(tree) for tree in range(0, number_of_trees)]
		elif purity_or_var == "variance":
			first_line = ["Leaf_Variance_Tree_" + str(tree) for tree in range(0, number_of_trees)]
		elif purity_or_var == "distance":
			first_line = ["Leaf_Distance_Tree_" + str(tree) for tree in range(0, number_of_trees)]

		else:
			first_line = ["Leaf_Purity_Tree_" + str(tree) for tree in range(0, number_of_trees)]


		all_samples_all_predictions = []
		for current_sample in range(0, number_of_samples):

			one_sample_all_predictions = []
			sample_name = sample_names[current_sample]
			one_sample_all_predictions.append(sample_name)

			for current_tree in range(0, number_of_trees):
				one_sample_single_tree_prediction = leaf_purity_samples[current_sample][current_tree]

				one_sample_all_predictions.append(one_sample_single_tree_prediction)

			all_samples_all_predictions.append(one_sample_all_predictions)

		output_file.write("\t".join(first_line) + "\n")

		for line in all_samples_all_predictions:
			output_file.write("\t".join([str(x) for x in line]) + "\n")



def print_error_to_file(filename, name_of_analysis, parameters, mse_included, classification_included,
						predictions_samples, true_values, classification_prediction_samples,
						true_classification_samples, all_available_labels):
	output_line = [name_of_analysis, parameters]
	first_line = ["Name_of_Analysis", "Parameters"]

	#print("Prediction: ")
	#print(classification_prediction_samples)

	#print("Known values: ")
	#print(true_classification_samples)
	with open(filename, "w", encoding="utf-8") as output_file:
		if mse_included:
			mse = mean_squared_error(y_pred=predictions_samples, y_true=true_values)
			pcc = pearsonr(predictions_samples, true_values)

			output_line.append(mse)
			output_line.append(pcc)
			first_line.append("MSE")
			first_line.append("PCC")

		if classification_included:

			accuracy = accuracy_score(y_true=true_classification_samples, y_pred=classification_prediction_samples,
										normalize=True)
			mcc = matthews_corrcoef(y_true=true_classification_samples, y_pred=classification_prediction_samples)
			#print(all_available_labels)
			#print(true_classification_samples)
			if len(all_available_labels) == 2:
				#print(sorted(list(set(all_available_labels))))
				confusion_mtx = confusion_matrix(y_true=true_classification_samples,
												 y_pred=classification_prediction_samples, labels = sorted(list(set(all_available_labels))))
				#if confusion_mtx.shape == (1, 1):
				#	print(confusion_mtx)
				tn = confusion_mtx[0][0]
				fp = confusion_mtx[0][1]
				fn = confusion_mtx[1][0]
				tp = confusion_mtx[1][1]


				output_line.append(tn)
				output_line.append(tp)
				output_line.append(fn)
				output_line.append(fp)
				output_line.append(accuracy)
				output_line.append(mcc)

				first_line.append("TN")
				first_line.append("TP")
				first_line.append("FN")
				first_line.append("FP")
				first_line.append("Accuracy")
				first_line.append("MCC")
			else:
				possible_classes = sorted(list(set(all_available_labels)))
				confusion_mtx = confusion_matrix(y_true=true_classification_samples,
												 y_pred=classification_prediction_samples, labels=possible_classes)

				c_1 = -1
				for t_class in possible_classes:
					c_1 = c_1 +1

					c_2 = -1
					for p_class in possible_classes:
						c_2 = c_2 +1
						first_line.append("Tr_" + str(t_class) + "_Pr_" + str(p_class))
						output_line.append(confusion_mtx[c_1][c_2])

				first_line.append("Accuracy")
				first_line.append("MCC")

				output_line.append(accuracy)
				output_line.append(mcc)

				
		output_file.write("\t".join(first_line) + "\n")
		output_file.write("\t".join([str(x) for x in output_line]) + "\n")

def print_feature_importance_to_file(feature_imp_fit_model, feature_imp_output_file, feature_names):

	with open(feature_imp_output_file, "w", encoding = "utf-8") as feature_imp_output:
		#print(feature_imp_fit_model.shape[0])
		#print(len(feature_names))
		for feature_idx in range(0, feature_imp_fit_model.shape[0]):
			current_feature_name = feature_names[feature_idx]

			feature_importance = feature_imp_fit_model[feature_idx]

			feature_imp_output.write(current_feature_name + "\t" + str(feature_importance) + "\n")


def write_time_to_file(time_file, elapsed_time, name_of_analysis, parameters):

	with open(time_file, "w", encoding = "utf-8") as time_output:
		first_line = ["Name_of_Analysis", "Parameters", "Time"]

		time_output.write("\t".join(first_line) + "\n")
		time_output.write("\t".join([name_of_analysis, parameters, str(elapsed_time)]) + "\n")


def write_leaf_assignment_to_file(leaf_assignment_file_train, tree_to_leaf_to_samples_dict, sample_names_train):

	with open(leaf_assignment_file_train, "w", encoding = "utf-8") as leaf_assignment_output:

		unique_names = numpy.unique(numpy.array(sample_names_train))
		leaf_assignment_output.write("Tree" + "\t" + "\t".join([str(x) for x in unique_names]) + "\n")

		for tree in tree_to_leaf_to_samples_dict.keys():

			current_row = [str(tree)]
			for sample_name in unique_names:

				found_sample = False
				current_row_string = str(float("NaN"))

				for leaf in tree_to_leaf_to_samples_dict[tree].keys():

					real_current_sample_names = [str(sample_names_train[idx_name]) for idx_name in tree_to_leaf_to_samples_dict[tree][leaf]]

					if sample_name in real_current_sample_names:

						if found_sample == True:
							print("Found a sample twice in the same tree")

						else:
							found_sample = True
							current_row_string = str(leaf)

				current_row.append(current_row_string)


			leaf_assignment_output.write("\t".join(current_row) + "\n")



def test_equality_by_prediction(tree_to_leaf_to_samples_dict, model, X_train, y_train, weights):

	#actual_predictions = model.predict(X_train)

	actual_leaf_assignments = model.apply(X_train)


	for sample in range(0, X_train.shape[0]):

		leaf_assignment_current_sample = actual_leaf_assignments[sample]

		pred_values = []
		for tree in range(0, len(tree_to_leaf_to_samples_dict.keys())):

			leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[tree]
			leaf_samples = tree_to_leaf_to_samples_dict[tree][leaf_assignment_current_sample_current_tree]


			if len(weights) == 0:
				pred_value = numpy.mean([y_train[sample_idx] for sample_idx in leaf_samples])

			else:
				current_weights = [weights[sample_idx] for sample_idx in leaf_samples]
				current_leaf_samples = [y_train[sample_idx] for sample_idx in leaf_samples]
				sum_weights = sum(current_weights)
				pred_value = (1/sum_weights) * (sum(numpy.multiply(current_leaf_samples, current_weights)))

			actual_pred = model.estimators_[tree].predict(X_train[sample, :].reshape(1, -1))[0]

			if not abs(pred_value-actual_pred) <= 0.0001:

				print("Actual prediction value: " + str(actual_pred))
				print("Our prediction value: " + str(pred_value))


def print_final_prediction_to_file(predictions_samples_train, original_sample_names_train, output_train_file):


	with open(output_train_file, "w", encoding = "utf-8") as output_train:

		output_train.write("Total_Prediction" + "\n")

		for sample_idx in range(0, len(original_sample_names_train)):

			current_sample_name = original_sample_names_train[sample_idx]
			current_prediction = predictions_samples_train[sample_idx]

			output_train.write(current_sample_name + "\t" + str(current_prediction) + "\n")


def print_train_samples_to_file(train_sample_names, train_sample_weights, sample_info_file):

	with open(sample_info_file, "w", encoding = "utf-8") as sample_info_output:
		#print(len(train_sample_names))
		#print(len(train_sample_weights))
		sample_info_output.write("Sample_name" + "\t" + "Weight" + "\n")
		for train_sample in range(0, len(train_sample_names)):

			sample_info_output.write(str(train_sample_names[train_sample]) + "\t" + str(train_sample_weights[train_sample]) + "\n")


def print_train_samples_to_file_upsample(train_sample_names, sample_info_file):


	counts = Counter(train_sample_names)
	unique_elements = numpy.unique(numpy.array(train_sample_names))


	with open(sample_info_file, "w", encoding = "utf-8") as sample_info_output:

		sample_info_output.write("Sample_name" + "\t" + "Count" + "\n")

		for elem in unique_elements:
			sample_info_output.write(str(elem) + "\t" + str(counts[elem]) + "\n")
		

