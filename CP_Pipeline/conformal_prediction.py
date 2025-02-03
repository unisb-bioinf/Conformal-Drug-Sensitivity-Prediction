from typing import Union
import numpy as np
import math
import bisect
from operator import itemgetter
import pandas as pd
from copy import deepcopy


def conformal_prediction(estimator, X_cal: np.array, y_cal: np.array, y_cal_discretized: np.array, sample_names: np.array, score_function=None, minimal_certainty: float = 0.9, SAURON: bool = False, num_classes: int = 2, class_names: np.ndarray = np.array(['0', '1'])) -> Union[float, tuple, list]:
    '''
        @param estimator: any fitted estimator for which we want to perform conformal prediction: 
                        if classifier: should support a function "predict_proba(X_cal)" that outputs a list of lists
                            where each entry represents a list of prediction probabilities for each class and the classes are 
                            positional encoded (c.f., https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba for instance)
                        if regression: should be able to perform a quantile regression, called using a function "predict(X_cal, qntl=[quantile])" that takes
                            a feature matrix and a list of quantiles and output a list of predictions for these quanitles (e.g., https://github.com/jnelson18/pyquantrf/issues)
        @param X_cal: a numpy array, the features of the calibration set which we need to compute q
        @param y_cal: a numpy array, the drug responses
                    (discretized or contiuous, depending on if we are doing classification or regression)
                    for the cell lines in X
        @param y_cal_discretized: a numpy array, the discretized drug response (only used if SAURON is set to True) then
                    y_cal is considered to be the continuous response
        @param sample_names: a numpy array containing the sample names (e.g., the cell line ids), only needed if SAURON is set to True
        @param score_function: function, default None, the scoring function which is needed to compute the score list
                    if None, 0 is returned which will lead to the fact that every prediction will just be accepted
                    if SAURON is set to True we expect this function to be the classification score
                        while the regression result is calculated using the one regression score that is implemented in this package
        @param t: confidence level for conformal prediction (1-alpha according to the notation in our paper), 0.9 if not given
        @param SAURON: Tells us whether we want to use the Sauron RF
        @param num_classes: default 2, number of classes
        @param class_names: default '0' and '1', the class assignment names
        @return q: the adjusted quantile
                if Sauron was set to true, it is a tuple of two quantiles while the first position is the quantile for the classification
                    and the second the quantile for regression
                if we used mondrian, we return a list of tuples where each position encodes for another class analogously to the encoding of
                    the prediciton probabilities. Each tuple contains at position one the class specific quantile and at position two the respective list
                    the lists are used later on two sort the predicted classes
    '''
    if score_function:
        if SAURON:
            # we expect estimator to be an instance of Sauron RF that is fitted
            # and that the analysis mode is already set
            # also since, we need the names of the cell lines, we expect the datasets to be numpy arrays
            p_upper, y_proba = estimator.predict(X_test=X_cal, y_test=y_cal, class_assignments_samples_test=y_cal_discretized,
                                                 sample_names_test=sample_names, quantile=1-((1-minimal_certainty)/2), X_train_calc=False)
            p_lower, y_proba = estimator.predict(X_test=X_cal, y_test=y_cal, class_assignments_samples_test=y_cal_discretized,
                                                 sample_names_test=sample_names, quantile=((1-minimal_certainty)/2), X_train_calc=False)
            y_proba_converted = []
            # SAURON does not predict the class probabilities as positional encoded lists but as dict
            # therefore, we have to convert them first
            for sample in y_proba:
                present_classes = sample.keys()
                sample_pred = [0 for i in range(0, num_classes)]
                for class_name in class_names:
                    if class_name in present_classes:
                        sample_pred[np.where(
                            class_names == class_name)[0][0]] = sample[class_name]
                y_proba_converted.append(sample_pred)
            y_proba_converted = np.array(y_proba_converted)
            # now, we can predict the scores for the class using the function
            scores_class = score_function(
                y_proba=y_proba_converted, y=y_cal_discretized, class_names=class_names)

            # the regression results can directly be scored using the quantile regression score
            scores_reg = get_pred_score_regression(
                y=y_cal, p_lower=p_lower, p_upper=p_upper)

            n = len(y_cal)
            # we have a uncertainty score -> we want to calculate the roughly 1-alpha qunatile
            # if alpha denotes our uncertainty
            if score_function == get_pred_score_mondrian:
                # this here is a bit special since we have different lists for the classes
                q_class = []
                for class_list in scores_class:
                    if len(class_list) == 0:
                        q = 1 #class was not existing in calibration set, to fulfil cp guarantee, we add class nevertheless
                    else:
                        q = np.quantile(class_list, math.ceil((n+1)*(minimal_certainty))/n)
                    # we now have a list of tuples containing the list and the resulting quantile
                    q_class.append((class_list, q))
            else:
                q_class = np.quantile(
                    scores_class, math.ceil((n+1)*minimal_certainty)/n)
            # our regression score is always a score with only one list
            q_reg = np.quantile(scores_reg, math.ceil(
                (n+1)*minimal_certainty)/n)
            # we return a tuple containing the results for classification and regression
            return q_class, q_reg
        else:
            # we do not perform simultaneous regression and classification but only one of them
            # the call for all our scoring functions looks the same
            scores = score_function(estimator, X_cal, y_cal,
                                    minimal_certainty, class_names=class_names)
            n = len(y_cal)
            if score_function == get_pred_score_mondrian:
                # this score is a bit special since we have different lists for the classes
                q = []
                scores = score_function(estimator, X_cal, y_cal,
                                        minimal_certainty, class_names=class_names)
                for class_list in scores:
                    q_class = np.quantile(class_list, math.ceil(
                        (n+1)*(minimal_certainty)/n))
                    q.append((class_list, q_class))
            else:
                # for certainty we need the 1-alpha (i.e., minimal certainty) quantile
                q = np.quantile(scores, math.ceil((n+1)*minimal_certainty)/n)

        # we return the quantile of the score list or a list of quantiles if our score was the mondrian score
        return q
    else:
        # there was no score function given so we return 0
        return 0


######################## scoring functions ########################

def get_pred_score_summation(estimator=None, X: np.array = None, y: np.array = None, minimal_certainty: float = 0, y_proba: np.array = None, class_names: np.array = None) -> list:
    '''
        @param estimator: the classifier for which we want to perform conformal prediction
            should support a function "predict_proba(X_cal)" that outputs a list of lists
            where each entry represents a list of prediction probabilities for each class and the classes are 
            positional encoded (c.f., https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba for instance)
        @param X: a numpy array, the features of the calibration set
        @param y: a numpy array, the discretized drug responses for the cell lines in X
        @param minimal_certainty: not used, just needed that all scoring functions have the same signature
        @param y_proba: instead of passing estimator and X, we also can directly pass the prediction probabilities
            Note that y is still needed to calculate the score
        @param class_names: a numpy array containing the class assignment names
        @return scores: an numpy array, containing all scores
    '''
    if not (estimator is None or X is None):
        # we predict each class and store the confidence level for each class and prediction [0, 1]
        y_proba = estimator.predict_proba(X)
    else:
        if y_proba is None:
            raise ValueError(
                'Either an estimator and X must be given or y_proba.')
    scores = []
    # now we need to compute the score (Romano et al(video min 24)):
    # this simple score is computed by sorting the confidence level desc and adding all of
    # them until we reach the true class
    for i in range(0, len(y_proba)):
        # len(y_proba) = len(y)
        probabilities = y_proba[i]
        # sort the probabilites decreasingly
        probabilities = -np.sort(-probabilities)
        score = 0
        # y[i] contains the true class, since range stops at the second arg -1,
        # we have to add 1 to acutally end with the true class
        normalized_class_index = np.where(class_names == y[i])[0][0]
        for j in range(0, normalized_class_index+1):
            score += probabilities[j]
            scores.append(score)
    return scores


def get_pred_score_true_class(estimator=None, X: np.array = None, y: np.array = None, minimal_certainty: float = 0, y_proba: np.array = None, class_names: np.array = None) -> list:
    '''
        @param estimator: the classifier for which we want to perform conformal prediction
            should support a function "predict_proba(X_cal)" that outputs a list of lists
            where each entry represents a list of prediction probabilities for each class and the classes are 
            positional encoded (c.f., https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba for instance)
        @param X: a numpy array, the features of the calibration set
        @param y: a numpy array, the discretized drug responses for the cell lines in X
        @param minimal_certainty: not used, just needed that all scoring functions have the same signature
        @param y_proba: instead of passing estimator and X, we also can directly pass the prediction probabilities
                        Note that y is still needed to calculate the score
        @param class_names: a numpy array containing the class assignment names
        @return scores: an numpy array, containing all scores
    '''
    if not (estimator is None or X is None):
        # we predict each class and store the confidence level for each class and prediction [0, 1]
        y_proba = estimator.predict_proba(X)
    else:
        if y_proba is None:
            raise ValueError(
                'Either an estimator and X must be given or y_proba.')
    scores = []
    # now we need to compute the score:
    # this simple score is computed by scoring each prediction with its true class
    # prediction probability
    for i in range(0, len(y_proba)):
        normalized_class_index = np.where(class_names == y[i])[0][0]
        score = 1-y_proba[i][normalized_class_index]
        scores.append(score)
    return scores


def get_pred_score_mondrian(estimator=None, X: np.array = None, y: np.array = None, minimal_certainty: float = 0, y_proba: np.array = None, class_names: np.array = None) -> list:
    '''
        @param estimator: the classifier for which we want to perform conformal prediction
            should support a function "predict_proba(X_cal)" that outputs a list of lists
            where each entry represents a list of prediction probabilities for each class and the classes are 
            positional encoded (c.f., https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba for instance)
        @param X: a numpy array, the features of the calibration set
        @param y: a numpy array, the discretized drug responses for the cell lines in X
        @param minimal_certainty: not used, just needed that all scoring functions have the same signature
        @param y_proba: instead of passing estimator and X, we also can directly pass the prediction probabilities
                        Note that y is still needed to calculate the score
        @param class_names: a numpy array containing the class assignment names
        @return scores: an numpy array, containing all scores
    '''
    if not (estimator is None or X is None):
        # we predict each class and store the confidence level for each class and prediction [0, 1]
        y_proba = estimator.predict_proba(X)
    else:
        if y_proba is None:
            raise ValueError(
                'Either an estimator and X must be given or y_proba.')
    scores = [[] for i in range(len(class_names))]
    # now we need to compute the score:
    # this simple score is computed by scoring each prediction with its true class
    # prediction probability
    for i in range(0, len(y_proba)):
        normalized_class_index = np.where(class_names == y[i])[0][0]
        score = 1-y_proba[i][normalized_class_index]
        bisect.insort(scores[normalized_class_index], score)
    return scores


def get_pred_score_regression(estimator=None, X: np.array = None, y: np.array = None, minimal_certainty: float = 0, p_upper: np.array = None, p_lower: np.array = None, class_names: np.array = None) -> list:
    '''
        @param estimator: the regressor for which we want to perform conformal prediction
            should be able to perform a quantile regression, called using a function "predict(X_cal, qntl=[quantile])" that takes
            a feature matrix and a list of quantiles and output a list of predictions for these quanitles
        @param X: a numpy array, the features of the calibration set
        @param y: a numpy array, the discretized drug responses for the cell lines in X
        @param minimal_certainty: the minimal certainty that should be guaranteed, needed to claculate the quantile predicitons
        @param p_upper: instead of passing estimator and X, we also can directly pass the quantile predictions, 
            this one is a numpy array containing the predicitions for the upper quantile
        @param p_lower: this one is a numpy array containing the predicitions for the lower quantile
        @param class_names: not used, just needed that all scoring functions have the same signature
        @return scores: an numpy array, containing all scores
    '''
    # t is our certainty score not uncertainty -> alpha in formula in paper must be replaced
    # by 1-minimal_certainty
    # example minimal_certainty=0.9 => (1-minimal_certainty)/2 = 0.1/2 = 0.05 = lower bound
    # && 1-((1-minimal_certainty)/2) = 1-0.05 = 0.95 = upper bound
    if not (estimator is None or y is None):
        p_upper = estimator.predict(X, qntl=[1-((1-minimal_certainty)/2)])
        p_lower = estimator.predict(X, qntl=[(1-minimal_certainty)/2])
    else:
        # neither the estimator and the features nor both quantile predictions are passed
        if p_upper is None or p_lower is None:
            raise ValueError(
                'Either the estimator and X or the quantile prediction results must be present.')
    scores = []
    for upper, lower, true in zip(p_upper, p_lower, y):
        scores.append(max([lower-true, true-upper]))
    return scores


######################## evaluation functions ########################

def eval_regression(predictions: np.array, true_y: np.array, sample_names: np.array, minimal_certainty: float, q: float) -> pd.DataFrame:
    '''
    creates a pandas DataFrame containing all the interesting information about the (conformal) prediction using the 
    quantile regression based score

    @param predictions: numpy array containing the predictions of the quantile regression
    @param true_y: the true continuous drug response
    @param sample_names: the cell line names
    @param minimal_certainty: the minimal certainty, we need to guarantee
    @param q: the quantile from the score list
    @return: a pandas data frame containing the results
    '''
    hitted = []
    pred_lower = list(map(itemgetter(0), predictions))
    pred_upper = list(map(itemgetter(1), predictions))
    for lower, upper, true in zip(pred_lower, pred_upper, true_y):
        if true >= lower-q and true <= upper+q:
            hitted.append(True)
        else:
            hitted.append(False)
    lower = [pred - q for pred in pred_lower]
    upper = [pred + q for pred in pred_upper]
    data = [sample_names, lower, upper, pred_lower,
            pred_upper, true_y, hitted]
    pred = pd.DataFrame(data=data, columns=sample_names,
                        index=['CL_name', f'conformal_prediction_{int(minimal_certainty*100)}_quantile_lower',
                               f'conformal_prediction_{int(minimal_certainty*100)}_quantile_upper', 'quantile_lower', 'quantile_upper',
                               'actual', 'actual_in_interval']).transpose()
    return pred


def eval_classification_mondrian(y_pred_proba: np.array, y_test: np.array, sample_names: np.array, minimal_certainty: float, q: list, num_classes: int, class_names: np.array) -> pd.DataFrame:
    '''
        creates a pandas DataFrame containing all the interesting information about the (conformal) prediction using the 
        mondrian score 

        @param y_pred_proba: an 2 dimensional array containing the list of prediction probabilities
                for each prediction. The classes should be positional encoded.
        @param y_test: array with the true outcomes
        @param sample_names: the names of the cell line
        @param minimal_certainty: confidence level
        @param q: list containing tuples for each class containing the score list at position one and the quantile at position two
        @param num_classes: the number of classes
        @param class_names: the names of the classes

        @return: a pandas dataframe, containing the prediction, conformal prediction, true value,
                if it was TP, TN, FP or FN for the prediction and the conformal prediction respectively

    '''
    y_pred_con = []
    # if our prediction probability is lower than the score q which is the quantile, computed by the
    # conformal prediction method, we are to unsure to predict 1 class and say that it is either 0 or 1
    # otherwise we are sure and predict the class with the high score
    y_pred = []
    for pred in y_pred_proba:
        indices = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        y_pred.append(class_names[indices[0]])
        pred_set = []
        sorting_list = []
        for i in range(len(pred)):
            class_pred = pred[i]
            # the quantile is the second element of the tuple
            quantile = q[i][1]
            # the score list is the first element of the tuple
            scores = deepcopy(q[i][0])
            # we sort in ascending order
            scores = sorted(scores, reverse=True)
            if 1-class_pred <= quantile:
                pred_set.append(class_names[i])
                bisect.insort(scores, 1-class_pred)
                # if the value is stored in this list several times we take the highest index
                # therefore, we reverse the list and find the first occurence of our class probability
                scores.reverse()
                # then, we convert the index back to the original un-reversed list index
                inserted_index = len(scores) - scores.index(1-class_pred) - 1
                # we divide the index of the newly inserted element by the size of the list
                # this will give us values in [0,1]
                sorting_criterion = inserted_index/(len(scores))
                sorting_list.append(sorting_criterion)
        # we first want to sort our set according to the probability with which the class would have been added
        # based on the score lists
        pred_set = [x for _, x in sorted(
            zip(sorting_list, pred_set), reverse=True)]
        y_pred_con.append(deepcopy(pred_set))

    return eval_classification(y_pred, y_pred_con, y_test, y_pred_proba, sample_names, minimal_certainty, num_classes, class_names)


def eval_classification_summation(y_pred_proba: np.array, y_test: np.array, sample_names: np.array, minimal_certainty: float, q: float, num_classes: int, class_names: np.array) -> pd.DataFrame:
    '''
        function that computes the conformal prediction set for classification if the summation score was used

        @param y_pred_proba: an 2 dimensional array containing the list of prediction probabilities
                for each prediction. The 0th position is always the probability for class 0 and the
                1st for class 1.
        @param y_test: array with the true outcomes
        @param sample_names: the names of the cell line
        @param minimal_certainty: confidence level
        @param q: score which was computed by the conformal prediction function
        @param num_classes: the number of classes
        @param class_names: the names of the classes

        @return: a pandas dataframe, containing the prediction, conformal prediction, true value,
                if it was TP, TN, FP or FN for the prediction and the conformal prediction respectively

    '''
    y_pred_con = []
    # if our prediction probability is lower than the score q which is the quantile, computed by the
    # conformal prediction method, we are to unsure to predict 1 class and say that it is either 0 or 1
    # otherwise we are sure and predict the class with the high score
    y_pred = []
    for pred in y_pred_proba:
        indices = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        y_pred.append(class_names[indices[0]])
        pred_set = []
        sum = 0
        for i in range(0, len(indices)):
            pred_set.append(class_names[indices[i]])
            sum += pred[indices[i]]
            if sum >= q:
                break

        y_pred_con.append(deepcopy(pred_set))

    return eval_classification(y_pred, y_pred_con, y_test, y_pred_proba, sample_names, minimal_certainty,  num_classes, class_names)


def eval_classification_true_class(y_pred_proba: np.array, y_test: np.array, sample_names: np.array, minimal_certainty: float, q: float, num_classes: int, class_names: np.array) -> pd.DataFrame:
    '''
        function that computes the conformal prediction set for classification if the true class' score
        was used
        given the threshold q and the confidence level t
        based od the prediction probabilities and the true values

        @param y_pred_proba: an 2 dimensional array containing the list of prediction probabilities
                for each prediction. The 0th position is always the probability for class 0 and the
                1st for class 1.
        @param y_test: array with the true outcomes
        @param sample_names: names of the cell lines
        @param minimal_certainty: confidence level
        @param q: score which was computed by the conformal prediction function
        @param num_classes: the number of classes
        @param class_names: the names of the classes

        @return: a pandas dataframe, containing the prediction, conformal prediction, true value,
                if it was TP, TN, FP or FN for the prediction and the conformal prediction respectively

    '''
    y_pred_con = []
    # we predict a set containing all classes which have a score exceeding q
    y_pred = []
    for pred in y_pred_proba:
        indices = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        y_pred.append(class_names[indices[0]])
        pred_set = []
        for i in range(0, len(indices)):
            if pred[indices[i]] >= 1-q:
                pred_set.append(class_names[indices[i]])
        y_pred_con.append(deepcopy(pred_set))
    return eval_classification(y_pred, y_pred_con, y_test, y_pred_proba, sample_names, minimal_certainty, num_classes, class_names)


def eval_classification(y_pred: np.array, y_pred_con: np.array, y_test: np.array, y_pred_proba, sample_names: np.array, minimal_certainty: float,  num_classes: int, class_names: np.array) -> pd.DataFrame:
    '''
        function that computes for a prediction and a conformal prediction if
        it was a TP, TN, FN, FP or NaN given the true values

        @param y_pred: an array containing the predicitons
        @param y_pred_con: an array containing the conformal prediction sets
        @param y_test: an array containing the true values
        @param y_pred_proba: an 2 dimensional array containing the prediction probabilites
                (just needed to write it)
        @param sample_names: names of the cell lines
        @param t: confidence level (needed to write the dataframe properly)
        @param q: threshold computed by conformal prediction (important to decide
            if there was actually a conformal prediction performed, which is not the
            case for cv training error)
        @param num_classes: the number of classes
        @param class_names: the names of the classes

        @return a pandas dataframe, containing the prediction, conformal prediction, true value,
                if it was TP, TN, FP or FN for the prediction and the conformal prediction respectively
    '''
    outcome = []
    outcome_con = []
    for y_pr, y_con, y in zip(y_pred, y_pred_con, y_test):
        if (y_pr == y):
            outcome.append('T' + y)
        else:
            outcome.append('F' + y_pr)
        if len(y_con) == num_classes:
            outcome_con.append('NaN')
        elif len(y_con) < num_classes:
            if y in y_con:
                outcome_con.append('T' + y)
            elif len(y_con) > 0:
                outcome_con.append('F' + y_pr)
            else:
                outcome_con.append('empty')

    pred = None
    data = [sample_names, y_pred, y_pred_con, y_test, outcome, outcome_con]
    index = ['CL_name', 'predicted', f'conformal_prediction_{int(minimal_certainty*100)}_quantile',
             'actual', 'outcome',
             f'outcome_after_conformal_prediction_{int(minimal_certainty*100)}_quantile']
    for i in range(num_classes):
        proba_i = list(map(itemgetter(i), y_pred_proba))
        data.append(proba_i)
        index.append(f'score{class_names[i]}')
    pred = pd.DataFrame(data=data, columns=sample_names,
                        index=index).transpose()
    pred[f'conformal_prediction_{int(minimal_certainty*100)}_quantile'] = pred[f'conformal_prediction_{int(minimal_certainty*100)}_quantile'].apply(lambda x: [str(element) for element in x])
    return pred
