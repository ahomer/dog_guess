# Data Scientist Nanodegree - Capstone Project

### Prerequisites

You need to install python,such as python 3 from https://www.python.org/.
Then you should install these packages listed below using pip:

```
pip install matplotlib
pip install pandas 
pip install keras 
pip install scikit-learn
pip install Flask
pip install tqdm
```

### Contents
app/run.py                      -- Website for find dog's breed <br>
models/weights.best.VGG19.hdf5  -- Pre-traind VGG19 modle file for dog's breed classifying <br>
data/test_images                -- Test images,you can upload these images to check the model


### Instructions
1. Install the python packages.

2. Run the following command in the app's directory to run your web app.

    ```
    cd app
    python run.py
    ```

3. Go to http://127.0.0.0:3001/

4. Upload a dog's image and click 'Find Dog's Breed' button,then you will get the dog's breed

### Model train detail
Please open the dog_app_notebook.pdf to get the dog's breed model training detail.


# Acknowledgements
Thanks to Udacity(https://udacity.com ).I have followed the of machine learning pipeline courses(ud025).This is the capstone project of the Data Scientist Nanodegree.


# License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/ahomer/dog_guess/blob/master/LICENSE) for additional details.
