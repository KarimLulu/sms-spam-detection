## What's SMS Spam Detection

It is the final project for the course ["Data Science. NLP"](https://github.com/vseloved/prj-nlp). The main goal of the project was to build a detection system for a Ukrainian user. Presentation from the project defense can be accessed [here](https://docs.google.com/presentation/d/e/2PACX-1vQDGvXABqpZcqhVr1FFcxMPaG0WkaSkYYxoCj5zcXVlUmAK_KZxdyFKajixlsEsp6XJ80GY4r-ATFsw/pub?start=false&loop=false&delayms=3000&slide=id.g588e0d797198c72c_10) and downloaded from [here](https://drive.google.com/file/d/1-rD9f____FSUDrRRnh66Md5jiaUzr2vW/view?usp=sharing).

## Table of Contens

<!--ts-->
* [What's SMS Spam Detection](#whats-sms-spam-detection)
* [Table of Contens](#table-of-contens)
* [Data Collection](#data-collection)
* [Machine Learning Pipeline](#machine-learning-pipeline)
* [Model Description](#model-description)
	* [Estimator](#estimator)
	* [Features](#features)
	* [Performance](#performance)
* [Getting Started](#getting-started)
* [License](#license)
<!--te-->

## Data Collection

The data was collected in multiple ways and labeled manually since there is no open dataset with labeled SMS messages from Ukrainian users. Here are the results:

* 269 responses for [Google Form](https://docs.google.com/forms/d/18Uh1cJIkXQg6_UBJ63u0HGQFy2uKvWTOCZtUYGFZ4U0/) with ~62% spam rate
* ~5800 self-annotated SMS messages from multiple devices with ~18% spam rate

So, here is basic statistics about the final data set:

* ~6100 SMS messages
* ~20% spam rate (imbalancement)
* Two classes (spam and ham)
* Multiple languages - Ukrainian, Russian, translit (Cyrillic letters are encoded with Latin ones), English

## Machine Learning Pipeline

ML Pipeline and Transformers were built on top of scikit's Pipeline/TransformerMixin classes. Transformers and Pipeline can be accessed [here](src/transformers.py) and [here](src/pipeline.py) respectively. Pipeline consists of several steps:

* Data loading and preparation
	* Text cleaning
	* Tokenization
* Feature building (will be discussed in the next section)
* Stratified CV splitting
* Grid search with 5-fold CV

After determining the best model and parameters, it is fitted to a whole data set and dumped altogether with detailed metadata (performance on folds) [here](data/models).


## Model Description

### Estimator

Logistic regression with L2 penalty was selected as the best performing model.

### Features

Here is the final set of features (please see [here](src/pipeline.py)):

* [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) character 4-grams with top 4000 features
* Pattern based features:
	* dot included?
	* uppercased/lowercased word? (+ratios)
	* bunch of RegExes (phone number, custom spam words, currency symbols, dates, etc.)
	* emoji included?
* Length based features:
	* No. of words
	* No. of chars
	* No. of N-grams

### Performance

* ROC-AUC: 99.5%
* Accuracy: 98.4%
* Precision: 96.4%
* F1: 96.03%
* Recall: 95.6%
* Averaged across folds confusion matrix:

    | | Ham | Spam |
    |---| --- | --- |
    | Pred Ham | 968.8 | 10.6
    | Pred Spam | 8.6 | 232.6


## Getting Started


1. Clone from the git repository:
	
		$ git clone https://github.com/KarimLulu/sms-spam-detection.git
	
2. Change directory to `sms-spam-detection` and start up the application:

		$ cd sms-spam-detection
		$ docker-compose up

3. Go to `http://localhost:8000` and submit text messages to test the system

4. Enjoy!

## License

SMS Spam Detector is released under the [MIT License](https://opensource.org/licenses/MIT).
