import numpy as np
import pandas
from sklearn.model_selection import train_test_split

class MF():
    
    def __init__(self, train_data, test_data, user_tag, recipe_tag, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - train_data (ndarray)   : user-item rating matrix (train)
        - test_data  (ndarray)   : user-item rating matrix (test)
        - user_tag   (ndarray)   : tag of each user matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        self.NUM_RECIPE_TAG = 1075 
        self.NUM_USER_TAG = 3560
        
        self.train_data = train_data
        self.test_data = test_data
        self.user_tag = user_tag
        self.recipe_tag = recipe_tag
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
        self.Utag = np.random.normal(scale=1./self.K, size=(self.NUM_USER_TAG, self.K))
        self.Rtag = np.random.normal(scale=1./self.K, size=(self.NUM_RECIPE_TAG, self.K))

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
            print("Iteration: %d ; error = %.4f" % (i+1, mse))
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
            ab_error += abs(self.test_data[x, y] - predicted[x, y])
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

            # Fetch all the tags from to User i, to calculate sum
            # If User i taged any recipe
            tag_flag = False
            num_user_tags = 1
            sum_user_tag = np.zeros(self.K)
            if i in self.user_tag[0]:
                tag_flag = True
                tag_row = self.user_tag[0].index(i)
                # print(tag_row, i)
                num_user_tags = len(self.user_tag[tag_row+1])
                for tag_count in range(num_user_tags):
                    tag_id = self.user_tag[tag_row+1][tag_count]
                    sum_user_tag += self.Utag[tag_id-1, :]
            
            # Fetch all the tags attached to Recipe j, to calculate sum
            num_recipe_tags = len(self.recipe_tag[j])
            sum_recipe_tag = np.zeros(self.K)
            for tag_count in range(num_recipe_tags):
                tag_id = self.recipe_tag[j][tag_count]
                sum_recipe_tag += self.Rtag[tag_id-1, :]

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            Q_i = self.Q[j, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * ( Q_i + sum_recipe_tag/num_recipe_tags ) - self.beta * P_i)
            self.Q[j, :] += self.alpha * (e * ( P_i + sum_user_tag/num_user_tags )-  self.beta * Q_i)


            # Fetch all the tags attached to User i, to update
            if tag_flag == True:
                for tag_count in range(num_user_tags):
                    tag_id = self.user_tag[tag_row+1][tag_count]
                    self.Utag[tag_id-1, :] += self.alpha * ( e / num_user_tags * ( Q_i + sum_recipe_tag/num_recipe_tags ) - self.beta * self.Utag[tag_id-1, :]  )


            # Fetch all the tags attached to Recipe j, to update
            # print("Tag of Recipe id", j+1, end=": ")
            for tag_count in range(num_recipe_tags):
                tag_id = self.recipe_tag[j][tag_count]     
                # Need to -1 to get actual row number of tag
                self.Rtag[tag_id-1, :] += self.alpha * ( e / num_recipe_tags * ( P_i + sum_user_tag/num_user_tags ) - self.beta * self.Rtag[tag_id-1, :]  )

            #     print(tag_id, end=', ')
            # print()



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
        # (1, 4) means User 2 to User 4
        print("User: {}: ".format(i+1))
        for j in range(len(matrix[i])):
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

def load_recipe_tag(data):
    """
    matrix format:
    [
        [(tag of recipe id 1) ....],
        [(tag of recipe id 2) 313, 463, 168, 359, 857...],
        ....
    ]
    """
    matrix = [[] for y in range(238)] 
    for i in range(1, len(data)):
        recipe_id = int(data['recipename_id'][i])
        tag_id = int(data['tagclass_id'][i])
        matrix[recipe_id-1].append(tag_id)

    return matrix

def load_user_tag(data):
    """
    matrix format:
    [
        [(user id contains tag) 945, 980, 981, 982, 986, 989, 991.....],
        [(tag of user id 945) 213, 993, 1868, 650, 205....],
        [(tag of user id 980) 1155, 205, 128, 478, 1259...],
        ....
    ]
    """
    matrix = [[] for y in range(132)] 
    for i in range(1, len(data)):
        user_id = int(data['user_id'][i])
        tag_id = int(data['tag_id'][i])
        if user_id not in matrix[0]:
            matrix[0].append(user_id)
            matrix[len(matrix[0])].append(tag_id)
        else:
            index = matrix[0].index(user_id)
            matrix[index+1].append(tag_id)

    return matrix


rate_names = ['id', 'user_id', 'recipe_id', 'rate']
# rate_data = pandas.read_csv("userrate_small_modify.csv", names=rate_names)
rate_data = pandas.read_csv("userrate_modify.csv", names=rate_names)
train_rows, test_rows = split_data(rate_data)

train = load_data(train_rows)
test = load_data(test_rows)

recipetag_names = ['tagclass_id', 'Tag', 'recipename_id', 'id', 'Weight']
recipe_tag_data = pandas.read_csv("tags.csv", names=recipetag_names)
recipe_tag_matrix = load_recipe_tag(recipe_tag_data)

user_names = ['id', 'user_id', 'recipe_id', 'tag_id', 'Weight']
user_tag_data = pandas.read_csv("usertag.csv", names=user_names)
user_tag_matrix = load_user_tag(user_tag_data)


# print_matrix(test)

result_list = [0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    list = []
    mf = MF(train, test, user_tag_matrix, recipe_tag_matrix, K=6, alpha=0.025, beta=0.005, iterations=10)
    training_process, list = mf.train()
    for j in range(10):
        result_list[j] += list[j]


for i in range(10):
    result_list[i] = result_list[i] / 10
print('list: ', result_list)
