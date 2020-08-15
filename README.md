# Yulin Xue
# Tag-Based Model

Tag-based model is an improved version of matrix factorization by adding user tags and item tags as extra information to recommender system.

## Installing Pandas

The standard Python distribution does not come with the Pandas module. To use this 3rd party module, you must install it.
The nice thing about Python is that it comes bundled with a tool called pip that can be used for the installation of Pandas. The do the installation, you need to run the following command:

```bash
pip install pandas
```
If you have installed Anaconda on your system, just run the following command to install Pandas:
```bash
conda install pandas
```

# Download from GitHub
```bash
git clone https://github.com/HelloElijah/TagBased-Recommender-System.git
```

## Usage

```python
python TagBased.py
```

## Output

```bash
Iteration: 1 ; error = 1.1516
Absolute Error is 0.957
Iteration: 2 ; error = 1.1291
Absolute Error is 0.927
Iteration: 3 ; error = 1.1145
Absolute Error is 0.907
Iteration: 4 ; error = 1.1013
Absolute Error is 0.889
Iteration: 5 ; error = 1.0989
Absolute Error is 0.879
Iteration: 6 ; error = 1.1073
Absolute Error is 0.878
Iteration: 7 ; error = 1.1212
Absolute Error is 0.885
```

## Description
Can get a graph of Cross Validation Result, by using:
Can change the parameter on mse_xxx.py file of different models
```python
python graph_result.py
```
Can change the parameter on mse_xxx.py file of different models
Then copy the result from mse_xxx.py file to graph_result.py

## Function
```bash
print_matrix: Print the Matrix, for debugging use
split_data: Split the date to training and testing set (9:1)
load_data: Load User rating data to a matrix
load_recipe_tag: Load recipe tag to a matrix(recipe_tag)
load_user_tag: Load user tag to a matrix(user_tag)

Class MF:
train: Generate Initial Random Value for User, Recipe, Tag Matrix and train the Recommender System
mse: Calculate the Mean Square Error and Absolute Error
sgd: Perform stochastic graident descent in order to get minimal error
get_rating: Helper Function, can get rating of User i and Recipe j
full_matrix: result of rating
```
