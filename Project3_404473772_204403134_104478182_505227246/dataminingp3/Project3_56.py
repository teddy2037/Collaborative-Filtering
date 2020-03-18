'''
# Q5: The resulting graph is exponentially decaying, implying the following:
1. The ratings for movies is not uniform. Some movies are more frequently rated than others;
as a result, sparsely rated movies will have a poorer success rate in the recommendation.

2. Further if our recommendation model predicted on quality of rating and number of ratings,
movies with few but healthy evaluations might be a fringe recommendation in our model further moving them
down the curve, despite being "good" movies. (This is in exchange to the "tried-and-tested" movies.)

Thus there are concerns to implement an "exploration factor". A good recommendation system would have to create:
(a graph that is as "less steep" as possible)
1. Sufficient number of movies that are rated in high frequency.
2. A rating that correlates to this number.
'''

from Project3_1234 import return_ratings_mat
import numpy as np
import matplotlib.pyplot as plt

def plot_var_wrt_index(R):
	# print max(R.var(0))
	bind = np.linspace(0.0,5.5,12)

	# Don't use var! Too many 0s which are not ratings
	number_of_entries = np.sum((R>0.0).astype(int), axis = 0)
	mean = np.sum(R, axis = 0)/number_of_entries
	ssq = np.sum(R**2, axis = 0)/number_of_entries
	fin = ssq - mean**2

	plt.hist(fin, bins = bind)
	plt.xlabel('Variance Bins')
	plt.ylabel('Movies in Bin')
	plt.show()

if __name__ == '__main__':
	plot_var_wrt_index(return_ratings_mat()) # Q6a
	''' 
	The variance of most movies is low. This indicates that viewers in general have a
	good understanding of what others think of the movie and rate similarly. 
	(This is why collaborative filtering is a suitable strategy.) # Q6b
	'''
