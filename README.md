# Conformal Prediction
This is a conformal prediction (CP) framework, which we developed to perform reliable drug sensitivity prediction and prioritization. A detailed description and application of the framework can be found in [our article 'Reliable anti-cancer drug sensitivity prediction and prioritization'](https://doi.org/10.1038/s41598-024-62956-6). The framework can be used to obtain reliable classification and regression results, also for simultaneous regression and classification approaches like our own [already published approach SAURON-RF](https://www.nature.com/articles/s41598-022-17609-x) that we extended to be eligible for CP. If you use our conformal prediction framework, (extended) SAURON-RF, or the code in this repository, please cite the corresponding publications. There, we also provide detailed mathematical descriptions of the framework including the non-conformity scores we implemented for CP.

For issues and questions, please contact Kerstin Lenhof (research[at]klenhof.de) or Lea Eckhart (lea.eckhart[at]uni-saarland.de). 

## Usage

The CP framework together with SAURON-RF can be executed as a python3 script in the console. It requires a single json-config file (see example_Json_config.txt in Example_Data folder) as input. If you want to use the conformal prediction framework with another estimator, this is also possible. To this end, please read the doc strings in the conformal_prediction.py python file, where we explain the required arguments. In main.py, we depict an example usage of CP with SAURON-RF as estimator.

used python3 libraries: pandas numpy typing math bisect operator copy sklearn time collections scipy

After downloading at least the directories `CP_Pipeline`,`Example_Data`, and the file `Advanced_SAURON_RF/multi_class_sauron_rf.py` you can execute the conformal prediction pipeline combined with SAURON RF as follows: 
```
cd Example_Data/
python3 ../CP_Pipeline/cp-main.py example_Json_config.json
```
Note that the directory tree should be kept and the path to the output folder should be edited in the file `Example_Data/example_JSON_config.json`. In this folder 10 output files will be generated (12 if the field swap_test_calibration in the config file is set tu 'True'). The conformal prediction results for classification will be found in ```<output_dir>/<analysis_name>_<1-error_rate>_classification_1_test.txt ``` and the regression results are stored in ```<output_dir>/<analysis_name>_<1-error_rate>_regression_1_test.txt```. If if the field swap_test_calibration in the config file is set to 'True' there will be one additional file per task, respectively, where the '1' in the file name is replaced by a '2'.
