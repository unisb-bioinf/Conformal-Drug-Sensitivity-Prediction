import sys
import json
from multi_class_sauron_rf import SAURONRF
import numpy as np
import math
from conformal_prediction import *


def read_gene_expression_matrix(gene_expression_matrix_file: str) -> list:
    '''
        reads the gene expression matrix

        @param gene_expression_matrix_file: path to a file with the gene expression matrix
        @return feature_names: list containing the names of the genes in the matrix
        @return cell_line_to_values_dict: dict with the cell line names as keys and the respective gene expression vectors as value
    '''
    feature_names = []
    cell_line_to_values_dict = {}

    with open(gene_expression_matrix_file, "r", encoding="utf-8") as gene_expression_matrix:

        first_line = gene_expression_matrix.readline()
        feature_names = first_line.strip('\n').split('\t')

        for line in gene_expression_matrix:
            sline = line.strip('\n').split('\t')

            sample_name = sline[0].strip()
            values = [float(x) for x in sline[1:]]

            if sample_name not in cell_line_to_values_dict:

                cell_line_to_values_dict[sample_name] = values
            else:
                print("Found sample " + sample_name + " twice")

    return [feature_names, cell_line_to_values_dict]


def filter_gene_expression_features(feature_names: list, wanted_names: list, sample_gene_expression_dict: dict) -> list:
    '''
        filters the gene expression matrix st only selected genes are considered as features

        @param feature_names: a list with the features/genes that are in the gene expression dict
        @param wanted_genes: a list with the wanted/selected genes to be considered for this analysis
        @param sample_gene_expression_dict: a dictionary with feature names as keys and expression values as values
        @return wanted_features: the wanted genes
        @return updated_dict: the updated sample gene expression dict only containing the considered genes
    '''
    delete_ids = []
    wanted_features = []
    for feature_idx in range(0, len(feature_names)):

        feature = feature_names[feature_idx]

        if feature not in wanted_names:

            delete_ids.append(feature_idx)
        else:

            wanted_features.append(feature)

    s_delete_ids = sorted(delete_ids, reverse=True)

    updated_dict = {}
    for sample in sample_gene_expression_dict.keys():

        values = sample_gene_expression_dict[sample]
        for deleted_idx in s_delete_ids:
            del values[deleted_idx]

        updated_dict[sample] = values
    return [wanted_features, updated_dict]


def read_wanted_gene_list(wanted_gene_list_file: str, number_of_wanted_genes: float) -> list:
    '''
        reads which genes are wanted

        @param wanted_gene_list_file: path to a file with all genes that should be considered 
            in each line of the file we expect the gene name followed by a tab and its score
        @param number_of_wanted_genes: int, number of genes to consider
        @return: a list with the names of the wanted genes
    '''
    wanted_genes = []
    with open(wanted_gene_list_file, "r", encoding="utf-8") as wanted_gene_list:

        for line in wanted_gene_list:

            if len(wanted_genes) >= number_of_wanted_genes:
                break
            else:
                wanted_gene_and_score = line.strip().split('\t')  # Assuming the list is sorted
                # the score does not have to exist as long the list is sorted by importance of the features
                wanted_gene = wanted_gene_and_score[0]
                if wanted_gene not in wanted_genes:

                    wanted_genes.append(wanted_gene)

                else:
                    print(
                        "A gene was twice in the list of the selected genes. Gene name was:  " + wanted_gene)
    return wanted_genes


def read_classification_file(classification_filename: str) -> dict:
    '''
        @param classification_filename: path to a file containing the classification responses, 
            we expect each line to contain the  tab separated cell line name and respective class 
        @return: a dict with the cell line names as keys and the class as value
    '''
    sample_to_classification_dict = {}

    with open(classification_filename, "r", encoding="utf-8") as classification_file:

        for line in classification_file:

            sline = line.strip('\n').split('\t')

            if len(sline) != 2:
                print("Classification file: Line " +
                      line + " was not long enough")
                continue

            else:

                cell_line = sline[0].strip()
                value = str(sline[1].strip())

                if cell_line not in sample_to_classification_dict:

                    sample_to_classification_dict[cell_line] = value
                else:
                    print("Found sample " + cell_line +
                          " twice in classification file.")

    return sample_to_classification_dict


def read_drug_file(drug_filename: str) -> list:
    '''

        @param drug_filename: path to a file containing the continuous drug response,
            we expect each line to contain the  tab separated cell line name and sensitivity measure (e.g., cmax, IC50)
        @return sample_list: list of all sample/cell line names
        @return value_list: list of all response values
    '''
    sample_list = []
    value_list = []
    with open(drug_filename, "r", encoding="utf-8") as drug_file:

        for line in drug_file:

            sline = line.strip('\n').split('\t')

            if len(sline) != 2:
                print("Drug Response File: Line " +
                      line + " was not long enough")
                continue

            else:

                cell_line = sline[0].strip()
                drug_response = float(sline[1].strip())

                if not cell_line in sample_list:

                    sample_list.append(cell_line)
                    value_list.append(drug_response)

                else:
                    print("Found sample " + cell_line +
                          " twice in drug response file")

    return [sample_list, value_list]


def split_classification_file(training_samples: list, test_samples: list, calibration_samples: list, classification_matrix: dict) -> list:
    '''
        @param training_samples: list of all sample names that are used for training
        @param test_samples: list of all sample names that are used for testing
        @param calibration_samples: list of all sample names that are used for calibration
        @param classification_matrix: a dict with the cell line names as keys and the class as value
        @return training_classification_samples: list containing the classes for the samples that are used for training
        @return test_classification_samples: list containing the classes for the samples that are used for testing
        @return calibration_classification_samples: list containing the classes for the samples that are used for calibration
    '''
    training_classification_samples = []
    test_classification_samples = []
    calibration_classification_samples = []

    for training_sample in training_samples:

        if training_sample not in classification_matrix:

            print(
                "Invalid sample id. No classification value for sample " + training_sample)
        else:
            training_classification_samples.append(
                classification_matrix[training_sample])

    for test_sample in test_samples:

        if test_sample not in classification_matrix:
            print("Invalid sample id. No classification value for sample " + test_sample)

        else:

            test_classification_samples.append(
                classification_matrix[test_sample])

    for calibration_sample in calibration_samples:

        if calibration_sample not in classification_matrix:
            print(
                "Invalid sample id. No classification value for sample " + calibration_sample)

        else:

            calibration_classification_samples.append(
                classification_matrix[calibration_sample])

    return [training_classification_samples, test_classification_samples, calibration_classification_samples]


def split_gene_expression_matrix(gene_expression_matrix_dict: dict, training_samples: list, calibration_samples: list, testing_samples: list):
    '''
        @param gene_expression_matrix_dict: a dict with the cell line names as keys and the gene expression as value
        @param training_samples: list of all sample names that are used for training
        @param test_samples: list of all sample names that are used for testing
        @param calibration_samples: list of all sample names that are used for calibration
        @return training_classification_samples: list containing the gene expression values for the samples that are used for training
        @return test_classification_samples: list containing the gene expression values for the samples that are used for testing
        @return calibration_classification_samples: list containing the gene expression values for the samples that are used for calibration
    '''
    gene_expression_training_matrix = []
    gene_expression_test_matrix = []
    gene_expression_calibration_matrix = []
    for sample in training_samples:

        if sample in gene_expression_matrix_dict:
            gene_expression_training_matrix.append(
                gene_expression_matrix_dict[sample])

        else:
            print("Sample " + sample + " was not found in gene expression matrix")

    for sample in testing_samples:

        if sample in gene_expression_matrix_dict:
            gene_expression_test_matrix.append(
                gene_expression_matrix_dict[sample])

        else:
            print("Sample " + sample + "was not found in gene expression matrix")

    for sample in calibration_samples:

        if sample in gene_expression_matrix_dict:
            gene_expression_calibration_matrix.append(
                gene_expression_matrix_dict[sample])

        else:
            print("Sample " + sample + "was not found in gene expression matrix")

    return [gene_expression_training_matrix, gene_expression_test_matrix, gene_expression_calibration_matrix]


def perform_conformal_prediction(json_dict: json.JSONDecoder):
    '''
        performs the conformal prediction for the settings given in the json config file

        @param json_dict: the config file
    '''
    training_samples_file = json_dict["tr_matrix_file"]
    test_samples_file = json_dict["te_matrix_file"]
    calibration_samples_file = json_dict["cal_matrix_file"]
    classification_matrix_file = json_dict["cl_matrix_file"]
    gene_expression_matrix_file = json_dict["ge_matrix_file"]
    wanted_gene_list_file = json_dict["wanted_genes"]
    nr_of_wanted_genes = float(json_dict["nr_of_w_genes"])

    training_samples_and_drug_response_values = read_drug_file(
        training_samples_file)
    test_samples_and_drug_response_values = read_drug_file(test_samples_file)
    calibration_samples_and_drug_response_values = read_drug_file(
        calibration_samples_file)

    sample_to_classification_dict = read_classification_file(
        classification_matrix_file)
    wanted_genes = read_wanted_gene_list(
        wanted_gene_list_file, nr_of_wanted_genes)

    samples_names_train = training_samples_and_drug_response_values[0]
    samples_names_test = test_samples_and_drug_response_values[0]
    samples_names_calibration = calibration_samples_and_drug_response_values[0]

    classifications_training_testing_calibration = split_classification_file(
        samples_names_train, samples_names_test, samples_names_calibration, sample_to_classification_dict)

    gene_expression_feature_names_and_matrix_dict = read_gene_expression_matrix(
        gene_expression_matrix_file)
    gene_expression_feature_names = gene_expression_feature_names_and_matrix_dict[0]
    gene_expression_matrix_dict = gene_expression_feature_names_and_matrix_dict[1]
    wanted_feature_names_and_filtered_gene_expression_matrix_dict = filter_gene_expression_features(
        gene_expression_feature_names, wanted_genes, gene_expression_matrix_dict)
    feature_names = wanted_feature_names_and_filtered_gene_expression_matrix_dict[0]
    filtered_gene_expression_matrix_dict = wanted_feature_names_and_filtered_gene_expression_matrix_dict[
        1]
    gene_expression_training_testing_calibration = split_gene_expression_matrix(
        gene_expression_matrix_dict=filtered_gene_expression_matrix_dict, training_samples=samples_names_train, testing_samples=samples_names_test, calibration_samples=samples_names_calibration)

    X_train = np.array(gene_expression_training_testing_calibration[0])
    X_test = np.array(gene_expression_training_testing_calibration[1])
    X_cal = np.array(gene_expression_training_testing_calibration[2])
    y_train = np.array(training_samples_and_drug_response_values[1])
    y_test = np.array(test_samples_and_drug_response_values[1])
    y_cal = np.array(calibration_samples_and_drug_response_values[1])

    upsampling = json_dict["upsample"].strip()

    class_assignment_samples_train = classifications_training_testing_calibration[0]
    class_assignment_samples_test = classifications_training_testing_calibration[1]
    class_assignment_samples_calibration = classifications_training_testing_calibration[2]

    classes = np.unique(class_assignment_samples_train)
    classes = np.sort(classes)
    num_classes = len(classes)

    min_number_of_samples_per_leaf = int(json_dict["samples_per_leaf"])
    number_of_features_per_split = json_dict["number_of_features_per_split"]

    # regression
    if not "regression_classification" in json_dict:

        if number_of_features_per_split == "breiman_default":
            # number of features Breiman default
            number_of_features_per_split = math.floor(
                float(X_train.shape[1]) / 3.0)

            if number_of_features_per_split == 0:
                number_of_features_per_split = float(X_train.shape[1])
        else:

            number_of_features_per_split = int(
                number_of_features_per_split)  # assuming it is an integer

    # regression
    elif json_dict["regression_classification"] == "regression":
        if number_of_features_per_split == "breiman_default":
            # number of features Breiman default
            number_of_features_per_split = math.floor(
                float(X_train.shape[1]) / 3.0)

            if number_of_features_per_split == 0:

                number_of_features_per_split = float(X_train.shape[1])
        else:

            number_of_features_per_split = int(
                number_of_features_per_split)  # assuming it is an integer

    number_of_trees_in_forest = int(json_dict["number_of_trees"])

    mse_included = json_dict["mse_included"]

    if mse_included in ["True", "true", "TRUE"]:

        mse_included = True

    else:
        mse_included = False

    classification_included = json_dict["classification_errors_included"]

    if classification_included in ["True", "true", "TRUE"]:

        classification_included = True

    else:
        classification_included = False

    name_of_analysis = json_dict["analysis_name"]

    output_directory = json_dict["output_dir"]

    threshold_g = json_dict["threshold"]
    threshold = 0.0
    if threshold_g != "":

        threshold = [float(t) for t in threshold_g.strip().split(',')]
    else:
        threshold = [float("NaN")]

    sample_weights_included = json_dict["sample_weights"]

    # percentage_top_candidates_g = json_dict["percentage_top_candidates"]

    # percentage_top_candidates = 0.0
    # if percentage_top_candidates_g == "":
    #   percentage_top_candidates = float("NaN")
    # else:
    #    percentage_top_candidates = float(percentage_top_candidates_g)

    analysis_mode = [str(x)
                     for x in json_dict["analysis_mode"].strip().split(',')]

    leaf_assignment_file_train = output_directory + \
        name_of_analysis + "_Training_Set_LeafAssignment.txt"
    sample_info_file = output_directory + name_of_analysis + \
        "_Additional_Sample_Information.txt"
    feature_imp_output_file = output_directory + \
        name_of_analysis + "_Feature_Importance.txt"
    time_file = output_directory + name_of_analysis + "_ElapsedTimeFitting.txt"
    debug_file = output_directory + name_of_analysis + "_DebugFile.txt"

    score = json_dict['score_classification']
    class_function = None
    if score == 'summation':
        class_function = get_pred_score_summation
    elif score == 'true_class':
        class_function = get_pred_score_true_class
    elif score == 'mondrian':
        class_function = get_pred_score_mondrian
    else:
        print('Invalid classification score. We use summation instead.')
        class_function = get_pred_score_summation

    error_rate = float(json_dict['error_rate'])
    minimal_certainty = 1-error_rate

    # Fitting is equal for all analysis modes, only predictions differ
    # Therefore, the RF object can be set and built before actual predictions need to be performed
    mult_sauron = SAURONRF(X_train=X_train, y_train=y_train, sample_names_train=samples_names_train,
                           min_number_of_samples_per_leaf=min_number_of_samples_per_leaf,
                           number_of_trees_in_forest=number_of_trees_in_forest,
                           number_of_features_per_split=number_of_features_per_split,
                           class_assignment_samples_train=class_assignment_samples_train,
                           name_of_analysis=name_of_analysis, mse_included=mse_included,
                           classification_included=classification_included,
                           feature_imp_output_file=feature_imp_output_file,
                           feature_names=feature_names, threshold=threshold,
                           upsampling=upsampling, time_file=time_file,
                           sample_weights_included=sample_weights_included,
                           leaf_assignment_file_train=leaf_assignment_file_train,
                           sample_info_file=sample_info_file, debug_file=debug_file)

    mult_sauron.fit()

    for am in analysis_mode:
        output_sample_prediction_file_train = output_directory + \
            name_of_analysis + '_' + am + "_Training_Set_Predictions.txt"
        output_sample_prediction_file_test = output_directory + \
            name_of_analysis + '_' + am + "_Test_Set_Predictions.txt"
        train_error_file = output_directory + \
            name_of_analysis + '_' + am + "_" + "Train_Error.txt"
        test_error_file = output_directory + \
            name_of_analysis + '_' + am + "_" + "Test_Error.txt"
        output_leaf_purity_file_train = output_directory + \
            name_of_analysis + '_' + am + "_Training_Set_LeafPurity.txt"
        output_leaf_purity_file_test = output_directory + \
            name_of_analysis + '_' + am + "_Test_Set_LeafPurity.txt"
        output_variance_file_train = output_directory + \
            name_of_analysis + '_' + am + "_Training_Set_Variance.txt"
        output_variance_file_test = output_directory + \
            name_of_analysis + '_' + am + "_Test_Set_Variance.txt"
        leaf_assignment_file_test = output_directory + \
            name_of_analysis + '_' + am + "_Test_Set_LeafAssignment.txt"

        # regression

        if not "regression_classification" in json_dict or json_dict["regression_classification"] == "regression":

            mult_sauron.set_analysis_mode(analysis_mode=am, output_sample_prediction_file_train=output_sample_prediction_file_train, train_error_file=train_error_file,
                                          feature_imp_output_file=feature_imp_output_file, output_leaf_purity_file_train=output_leaf_purity_file_train,
                                          output_variance_file_train=output_variance_file_train, output_sample_prediction_file_test=output_sample_prediction_file_test,
                                          test_error_file=test_error_file,  output_leaf_purity_file_test=output_leaf_purity_file_test,
                                          output_variance_file_test=output_variance_file_test, leaf_assignment_file_test=leaf_assignment_file_test)
        y_pred_test_class1, y_pred_test_reg1 = predict(mult_sauron, X_cal=X_cal, y_cal=y_cal, class_assignment_samples_calibration=class_assignment_samples_calibration, samples_names_calibration=samples_names_calibration, class_function=class_function, num_classes=num_classes,
                                                       classes=classes, X_test=X_test, y_test=y_test, class_assignment_samples_test=class_assignment_samples_test, samples_names_test=samples_names_test, minimal_certainty=minimal_certainty, score=score)
        if json_dict['swap_test_claibration'] in ['True', 'true']:
            y_pred_test_class2, y_pred_test_reg2 = predict(mult_sauron, X_cal=X_test, y_cal=y_test, class_assignment_samples_calibration=class_assignment_samples_test, samples_names_calibration=samples_names_test, class_function=class_function, num_classes=num_classes,
                                                           classes=classes, X_test=X_cal, y_test=y_cal, class_assignment_samples_test=class_assignment_samples_calibration, samples_names_test=samples_names_calibration, minimal_certainty=minimal_certainty, score=score)
    output_regression_test = output_directory + \
        name_of_analysis + f'_{minimal_certainty}' + "_regression1_test.txt"
    output_classification_test = output_directory + \
        name_of_analysis + f'_{minimal_certainty}' + \
        "_classification1_test.txt"
    y_pred_test_class1.to_csv(
        output_classification_test, sep='\t', index=False)
    y_pred_test_reg1.to_csv(output_regression_test, sep='\t', index=False)
    if json_dict['swap_test_claibration'] in ['True', 'true']:
        output_regression_test = output_directory + \
            name_of_analysis + f'_{minimal_certainty}' + \
            "_regression2_test.txt"
        output_classification_test = output_directory + \
            name_of_analysis + f'_{minimal_certainty}' + \
            "_classification2_test.txt"
        y_pred_test_class2.to_csv(
            output_classification_test, sep='\t', index=False)
        y_pred_test_reg2.to_csv(output_regression_test, sep='\t', index=False)


def predict(fitted_model: SAURONRF, X_cal: np.array, y_cal: np.array, class_assignment_samples_calibration: np.array, samples_names_calibration: np.array, class_function, num_classes: int, classes: np.array, X_test: np.array, y_test: np.array, class_assignment_samples_test: np.array, samples_names_test: np.array, minimal_certainty: float, score: str):

    q_class, q_reg = conformal_prediction(
        estimator=fitted_model, X_cal=X_cal, y_cal=y_cal, y_cal_discretized=class_assignment_samples_calibration, sample_names=samples_names_calibration, score_function=class_function, minimal_certainty=minimal_certainty, SAURON=True, num_classes=num_classes, class_names=classes)
    p_upper, y_probas_class = fitted_model.predict(X_test=X_test, y_test=y_test, class_assignments_samples_test=class_assignment_samples_test, sample_names_test=samples_names_test,
                                                   quantile=1-((1-minimal_certainty)/2), X_train_calc=False)
    p_lower, y_probas_class = fitted_model.predict(X_test=X_test, y_test=y_test, class_assignments_samples_test=class_assignment_samples_test, sample_names_test=samples_names_test,
                                                   quantile=((1-minimal_certainty)/2), X_train_calc=False)
    # convert from dict to np.array such that the results for Sauron and normal classifier are the same
    plain_regression_rf, dummy = fitted_model.predict(X_test=X_test, y_test=y_test, class_assignments_samples_test=class_assignment_samples_test,
                                                      sample_names_test=samples_names_test, quantile=float('NaN'), X_train_calc=False)
    y_probas = []
    for sample in y_probas_class:
        present_classes = sample.keys()
        sample_pred = [0 for i in range(0, num_classes)]
        for class_name in classes:
            if class_name in present_classes:
                sample_pred[np.where(
                    classes == class_name)[0][0]] = sample[class_name]
        y_probas.append(sample_pred)
    y_probas = np.array(y_probas)
    y_pred_test_reg = eval_regression(
        predictions=list(zip(p_lower, p_upper)), true_y=y_test, sample_names=samples_names_test, minimal_certainty=minimal_certainty, q=q_reg)
    y_pred_test_reg['plain_rf'] = plain_regression_rf
    if score == 'summation':
        y_pred_test_class = eval_classification_summation(
            y_pred_proba=y_probas, y_test=class_assignment_samples_test, sample_names=samples_names_test, minimal_certainty=minimal_certainty, q=q_class, num_classes=num_classes, class_names=classes)
    elif score == 'true_class':
        y_pred_test_class = eval_classification_true_class(
            y_pred_proba=y_probas, y_test=class_assignment_samples_test, sample_names=samples_names_test, minimal_certainty=minimal_certainty, q=q_class, num_classes=num_classes, class_names=classes)
    elif score == 'mondrian':
        y_pred_test_class = eval_classification_mondrian(
            y_pred_proba=y_probas, y_test=class_assignment_samples_test, sample_names=samples_names_test, minimal_certainty=minimal_certainty, q=q_class, num_classes=num_classes, class_names=classes)
    return y_pred_test_class, y_pred_test_reg


################ Main function ##################################


def main(config_filename):

    # print(config_filename)
    json_file = open(config_filename)

    data = json.load(json_file)
    print(data)
    perform_conformal_prediction(data)


# Start of program
if __name__ == "__main__":
    # print(sys.argv)
    if len(sys.argv) < 2:
        sys.exit("This program needs the following arguments:\
                    \n- Config json file with information about \
                    \n a) number of trees (number_of_trees)\
                    \n b) number of samples per leaf (samples_per_leaf)\
                    \n c) number of features per split (number_of_features_per_split)\
                    \n d) analysis mode for HARF (analysis_mode, accepted: binary_no_weights, binary_weights, majority_weights, binary_no_weights_sensitive)\
                    \n e) output directory for results (output_dir)\
                    \n f) Should MSE and PCC be included (mse_included, accepted: true, false)\
                    \n g) Should classification errors be included (classification_errors_included, accepted: true, false)\
                    \n h) Training matrix cell line file name (tr_matrix_file)\
                    \n i) Test matrix cell line file name (te_matrix_file)\
                    \n j) Gene Expression matrix (ge_matrix_file)\
                    \n k) Classification info file (cl_matrix_file)\
                    \n l) File with gene names that should be used for analysis (wanted_genes)\
                    \n m) Number of wanted genes from the sorted list (nr_of_w_genes)\
                    \n n) Name for the analysis (analysis_name, e.g. Fold0, CompleteTraining, etc.)\
                    \n o) Threshold(s) for calculating weights (threshold). Can be used with sample weights.\
                    \n p) Should training data be upsampled for minority class (upsample, accepted: simple, linear). Cannot be combined with sample_weights\
                    \n q) Should sample weights be used for fitting trees? (sample_weights, accepted: simple, no)\
                    \n r) Information whether classification or regression trees should be fit (regression_classification, accepted: regression, classification not yet implemented)\
                    \n s) quantile (a number between 0 and 1, can also be an empty string, then usual RF is fitted)\
                    \n t) score classification (summation, mondrian, or true_class)")
    print(sys.argv[1])
    main(sys.argv[1])
