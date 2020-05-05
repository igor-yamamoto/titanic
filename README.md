# Titanic chalenge #

This is my attempt at kaggle's ["Titanic: Machine Learning from Disaster"](https://www.kaggle.com/c/titanic) challenge. It contains a notebook explaining all the steps I followed in order to train the model, located in the **notebooks** directory, and it can be ran via terminal by using the codes in the  **code** directory.


## Downloading ##

* Clone this repo to your computer
* Access the directory by using `cd titanic`
* To run the codes, the csv files need to be extracted in the **datasets** directory. They can either be [manually](https://www.kaggle.com/c/titanic/data) or automatically downloaded by triggering the `down` variable to `True` in the `download_data` function, defined on **data_analysis.py**. 
   * Manually downloading the data fetches a .zip file. Paste it in the **datasets** directory and run `unzip titanic.zip`
   * The later option requires [Kaggle API](https://github.com/Kaggle/kaggle-api) to be installed and properlly configured. Read Kaggle API's documentation in order to follow this step
* Move back to root directory by `cd ..`

## Installation ##

* To run the codes, several packages must be installed. The list of packages can be assessed by `cat requirements.txt`. To install them, use `pip install -r requirements.txt`
   * It is advisable to use a virtual environment. To create one over this project, run `virtualenv ../titanic`

## Usage ## 
* The codes are separated into *analysis*, *preparation* and *model testing* methods, and it is advisable to run them in this order (althought the **analysis.py** script don't create any file that is to be used in the other scripts, the output from **data_process.py** is piped to **model_test.py**). 


:dragon_face:
