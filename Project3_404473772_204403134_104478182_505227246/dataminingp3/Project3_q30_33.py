from surprise import Dataset
from surprise import Reader
from sklearn.metrics import mean_squared_error
from surprise.model_selection import KFold
from math import sqrt
import numpy as np

#reimporting the files from the dataset
def retrieve_data():
    f_path = './ml-latest-small/ratings.csv'
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5), skip_lines=1)
    data = Dataset.load_from_file(f_path, reader=reader)
    return data

###########################################################################
#returns a dictionary of ratings with movieID key
def ret_ratings(data):
    ratings = {}
    for r in data.raw_ratings:
        if r[1] not in ratings:
            ratings[r[1]] = []
        ratings[r[1]].append(r[2])
    return ratings

#returns variance of each key in dict
def ret_var(ratings):
    movie_var = {}
    for k in ratings:
        movie_var[k] = np.var(ratings[k])
    return movie_var

###########################################################################

#returns a dictionary of ratings with user key
def ret_user_dict(data):
    user_movies = {}
    for r in data.raw_ratings:
        if r[0] not in user_movies:
            user_movies[r[0]] = []
        user_movies[r[0]].append(r[2])
    return user_movies

#outputs the mean of each user key in array
def ret_user_means(user_movies):
    user_mean = {}
    for k in user_movies:
        user_mean[k] = np.mean(user_movies[k])
    return user_mean

###########################################################################

# trims the TESTSET depending on choice
def trim(data, testset, choice):
    ratings = ret_ratings(data)
    movie_var = ret_var(ratings)

    if choice == 0:
        return testset

    elif choice == 1:
        popular_movies = [x for x in ratings if len(ratings[x]) > 2]
        test_trim = [x for x in testset if x[1] in popular_movies]

    elif choice == 2:
        unpopular_movies = [x for x in ratings if len(ratings[x]) <= 2]
        test_trim = [x for x in testset if x[1] in unpopular_movies]

    elif choice == 3:
        high_var_movies = [x for x in ratings if len(ratings[x])>= 5 and movie_var[x] >= 2]
        test_trim = [x for x in testset if x[1] in high_var_movies]

    return test_trim



def ret_avg_rmse_10(data, choice = 0):
    # setting up the cross validation scheme and init
    RMSE = 0.0
    RMSE_per = 0.0

    user_mean = ret_user_means(ret_user_dict(data))
    kf = KFold(n_splits=10)

    for trainset, testset in kf.split(data):

        test_manip = trim(data, testset, choice)
        RMSE_per = sqrt(mean_squared_error([y[2] for y in test_manip], [user_mean.get(x[0]) for x in test_manip]))
        RMSE = RMSE + RMSE_per

    return RMSE / 10.0


if __name__ == '__main__':
    # re-configuring the dataset
    data = retrieve_data()
    print ""
    print "****** Naive Collaboratory Filtering ******"
    print ""
    print "Method of Trim.\t\t\tRoot Mean Squared Error"  
    print "No trimming\t\t\t" + str(ret_avg_rmse_10(data, 0)) #30
    print "Popular Movie Trimming\t\t" + str(ret_avg_rmse_10(data, 1)) #31
    print "Unpopular Movie Trimming\t" + str(ret_avg_rmse_10(data, 2)) #32
    print "High Var Movie Trimming\t\t" + str(ret_avg_rmse_10(data, 3)) #33
    print ""




