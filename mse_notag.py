import numpy as np
import pandas
from sklearn.model_selection import train_test_split

class MF():
    
    def __init__(self, train_data, test_data, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix (train)
        - T (ndarray)   : user-item rating matrix (test)
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.train_data = train_data
        self.test_data = test_data
        self.num_users, self.num_items = train_data.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        # mean = 0, sd = 1./self.K
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.train_data[np.where(self.train_data != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.train_data[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.train_data[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        list = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            # if (i+1) % 10 == 0:
            #     print("Iteration: %d ; error = %.4f" % (i+1, mse))
            # print("Iteration: %d ; error = %.4f" % (i+1, mse))
            list.append(mse)

        return training_process, list

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.test_data.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.test_data[x, y] - predicted[x, y], 2)
        return np.sqrt(error/ len(xs))

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])


    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        self.Q[j, :].T is transpose
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)



def print_matrix(matrix):
    for i in range(1, 4):
        print("User: {}: ".format(i+1))
        for j in range(len(matrix[0])):
            print("{:5.2f}".format(matrix[i][j]), end=', ')
            if (j+1) % 15 == 0:
                print()
        print('\n\n')
    print()
    return

def split_data(data):
    Y = data['rate']
    Y = Y[1:]
    X = np.column_stack((data['id'], data['user_id'], data['recipe_id']))
    X = X[1:]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=0)
    train = np.column_stack((X_train, Y_train))
    test = np.column_stack((X_test, Y_test))

    return train, test

def load_data(data):
    # (1214, 238) full data value
    matrix = np.zeros((1214, 238))

    for i in range(len(data)):
        user_id = int(data[i][1])
        recipe_id = int(data[i][2])
        rate = float(data[i][3])

        # if user_id > 1214:
        #     break

        matrix[user_id-1][recipe_id-1] = rate
    return matrix


names = ['id', 'user_id', 'recipe_id', 'rate']
# data = pandas.read_csv("userrate_small_modify.csv", names=names)
data = pandas.read_csv("userrate_modify.csv", names=names)
train_rows, test_rows = split_data(data)

train = load_data(train_rows)
test = load_data(test_rows)

result_list = [0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    list = []
    mf = MF(train, test, K=6, alpha=0.025, beta=0.005, iterations=10)
    training_process, list = mf.train()
    for j in range(10):
        result_list[j] += list[j]


for i in range(10):
    result_list[i] = result_list[i] / 10
print('list: ', result_list)

