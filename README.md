# Conformal Prediction
This is a framework for conformal prediction (CP), which we developed to perform reliable drug sensitivity prediction and prioritization. It can be used to obtain reliable classification and regression results, also for simultaneous regression and classification approaches.  If you use our conformal prediction framework, SAURON-RF, or the code in this repository, please cite our paper.


## Usage

The CP framework together with SAURON-RF can be executed as a python3 script in the console. It requires a single json-config file (see example_Json_config.txt in Example_Data folder) as input. If you want to use the conformal prediction framework with another estimator this is also possible. See conformal_prediction.py to read the respective doc strings that explain the required arguments and main.py for an example usage of CP with SAURON-RF as estimator.

used python3 libraries: pandas numpy typing math bisect operator copy sklearn time collections scipy

Example call: python3 cp_main.py example_Json_config.json
