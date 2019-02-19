This project is originated from " LSTM to predict gender of a name"

https://github.com/divamgupta/lstm-gender-predictor

I add a little modification and show a confusion matrix in the end of the notebook to visualize the results


#### training set:

`female.txt` contains female names

`male.txt` contains male names

### How to use :
To train the model:

	`python gender_v3.py train`
    
To predict the names with your file:

	`python gender_v3.py predict <MY_NAME.txt>`

### requirements
* Keras
* numpy

