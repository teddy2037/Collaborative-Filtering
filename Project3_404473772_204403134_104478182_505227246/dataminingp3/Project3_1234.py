import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def swap(R):
	R = R[:,~np.all(R == 0, axis=0)]
	R = R[:,np.sum((R>0.0).astype(int), axis = 0).argsort()[::-1]]
	R = R[np.sum((R>0.0).astype(int), axis = 1).argsort()[::-1],:]
	# print R
	return R

def return_ratings_mat():
    # Returns the matrix R that we use for everything in the project
    # Rij is the rating by user i in movie j
    r = pd.read_csv("ml-latest-small/ratings.csv")
    r_prime = r.values

    max_user_id = max(r_prime[:,0])
    max_movie_id = max(r_prime[:,1])

    actual_no_of_movies = np.unique(r_prime[:,1]).size
    max_index = len(r.index)

    # Need to verify lowest rating != 0
    # print min(r_prime[:,2])
    R = np.zeros((int(max_user_id),int(max_movie_id)), dtype=np.float)

    for i in range(0, max_index):
        R[int(r_prime[i,0]-1),int(r_prime[i,1]-1)] = float(r_prime[i,2])

    R_real = swap(R)
    return R_real

def sparsity(R):
    return float(np.count_nonzero(R)) / (R.shape[0] * R.shape[1])

def plot_freq_ratings(R):
    # Plots
    plt.hist(np.ravel(R), bins = [0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25])
    # Emphasizes the disparity of the ratings
    plt.xlabel("Rating Bins")
    plt.ylabel("Frequency of Bin")
    plt.show()

def plot_movie_rate_freq_desc(R):
    Count_R_sub = np.sum((R > 0.0).astype(int), axis = 0)
    plt.plot(Count_R_sub)
    plt.xlabel("Index of Movie")
    plt.ylabel("Number of Ratings")
    plt.show()

def plot_user_vote_freq_desc(R):
    Count_R_sub = np.sum((R > 0.0).astype(int), axis = 1)
    plt.plot(Count_R_sub)
    plt.xlabel("Index of User")
    plt.ylabel("Number of Ratings")
    plt.show()

if __name__ == '__main__':
    R = return_ratings_mat()
    # print R.shape
    print sparsity(R) # Q1
    plot_freq_ratings(R) # Q2
    plot_movie_rate_freq_desc(R) # Q3
    plot_user_vote_freq_desc(R) # Q4
