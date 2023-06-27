# Conformal Prediction
This is a conformal prediction (CP) framework, which we developed to perform reliable drug sensitivity prediction and prioritization. It can be used to obtain reliable classification and regression results, also for simultaneous regression and classification approaches like our own [already published approach SAURON-RF](https://www.nature.com/articles/s41598-022-17609-x) that we extended to be eligible for CP. If you use our conformal prediction framework, (extended) SAURON-RF, or the code in this repository, please cite the corresponding publications. There, we also provide detailed mathematical descriptions of the framework including the non-conformity scores we implemented for CP.

For issues and questions, please contact Kerstin Lenhof (klenhof[at]bioinf.uni-sb.de) or Lea Eckhart (lea.eckhart[at]bioinf.uni-sb.de). 

## Usage

The CP framework together with SAURON-RF can be executed as a python3 script in the console. It requires a single json-config file (see example_Json_config.txt in Example_Data folder) as input. If you want to use the conformal prediction framework with another estimator, this is also possible. To this end, please read the doc strings in the conformal_prediction.py python file, where we explain the required arguments. In main.py, we depict an example usage of CP with SAURON-RF as estimator.

used python3 libraries: pandas numpy typing math bisect operator copy sklearn time collections scipy

Example call: python3 cp_main.py example_Json_config.json
