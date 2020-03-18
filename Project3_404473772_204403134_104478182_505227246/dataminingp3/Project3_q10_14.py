import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Project3_1234 import *
import csv
from surprise import KNNWithMeans
from surprise.model_selection import *
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import metrics

f_path = './ml-latest-small/ratings.csv'
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5), skip_lines=1)
data = Dataset.load_from_file(f_path, reader=reader)


def plotgraphs(x_axis, y_axis, nameofxaxis = 'x-axis', nameofyaxis = 'y-axis', title = 'Title'):

    plt.plot(x_axis,y_axis)
    plt.xlabel(nameofxaxis, size = 15)
    plt.ylabel(nameofyaxis, size = 15)
    plt.title(title,size = 15)
    plt.show()

def plot_roc(fpr, tpr,t):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label='area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', size=15)
    ax.set_ylabel('True Positive Rate', size=15)
    ax.legend(loc="lower right")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    plt.title('Threshold  = %0.2f' % t, size=15)
    plt.show()


if __name__ == '__main__':
    sim_options = {'name' : 'pearson' , 'user_based' : True} ############### user based #################

    x_axis = range(2,102,2)
    dim = len(x_axis)
    rmse_test_store = np.zeros(dim)
    mae_test_store = np.zeros(dim)

    for i in x_axis:
        algo = KNNWithMeans(k=i, sim_options=sim_options,verbose=True)
        result = cross_validate(algo, data, measures=['rmse', 'mae'], cv=10, verbose=True)

        rmse_score = np.mean(result['test_rmse'])
        mae_score = np.mean(result['test_mae'])

        ##################### Index to store values in rmse and mae ###################
        ind = int(i/2-1)

        rmse_test_store[ind] = rmse_score
        mae_test_store[ind] = mae_score


    pd.DataFrame(rmse_test_store).to_csv("rmse_test_store_10.csv")
    pd.DataFrame(mae_test_store).to_csv("mae_test_store_10.csv")

    plotgraphs(x_axis,rmse_test_store,'K','Mean rmse scores','Plot')
    plotgraphs(x_axis,mae_test_store,'K','Mean Mae scores','Plot')


    ################################################   storing and ploting #################################
    #
    # rmsescores = []
    #
    # with open('rmse_test_store_10.csv', newline='') as csvDataFile:
    #     csvReader = csv.reader(csvDataFile)
    #     for row in csvReader:
    #         rmsescores.append(row[1])
    #
    # rmsescores.pop(0)
    # score = np.asarray(rmsescores)

    #


    #########################################################################################

    ratings = {}
    for r in data.raw_ratings:

        if r[1] not in ratings:
            ratings[r[1]] = []
        ratings[r[1]].append(r[2])

    ###############################################################################################


    popular_movies = [x for x in ratings if len(ratings[x]) > 2]
    unpopular_movies = [x for x in ratings if len(ratings[x]) <= 2]

    ###################################################################################
    kf = KFold(n_splits=10)
    rmse_popular_store = []
    for i in x_axis:

        algo = KNNWithMeans(k=i, sim_options=sim_options,verbose=True)
        accu = []
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            test_trim = [x for x in testset if x[1] in popular_movies]
            predictions = algo.test(test_trim)
            accu.append(accuracy.rmse(predictions, verbose=True))
        s = np.mean(accu)
        rmse_popular_store.append(s)


    pd.DataFrame(rmse_popular_store).to_csv("rmse_popular_store_12.csv")
    plotgraphs(x_axis,rmse_popular_store,'K','Mean RMSE scores','Plot of popular movies')

    ##########################################################################################

    kf = KFold(n_splits=10)
    rmse_unpopular_store = []
    for i in x_axis:

        algo = KNNWithMeans(k=i, sim_options=sim_options,verbose=True)
        accu = []
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            test_trim = [x for x in testset if x[1] in unpopular_movies]
            predictions = algo.test(test_trim)
            accu.append(accuracy.rmse(predictions, verbose=True))
        s = np.mean(accu)
        rmse_unpopular_store.append(s)



    pd.DataFrame(rmse_unpopular_store).to_csv("rmse_unpopular_store_13.csv")
    plotgraphs(x_axis,rmse_unpopular_store,'K','Mean RMSE scores','Plot of unpopular movies')

    ############ rates  "key" id, values are ratings #######################################
    movie_var = {}
    for k in ratings:
        # print(k)
        movie_var[k] = np.var(ratings[k])

    ####################################################################################
    highvar_movies = [ x for x in ratings if len(ratings[x])>= 5 and movie_var[x] >= 2 ]
    ##################################################################################

    kf = KFold(n_splits=10)
    rmse_highvar_store = []
    for i in x_axis:

        algo = KNNWithMeans(k=i, sim_options=sim_options,verbose=True)
        accu = []
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            test_trim = [x for x in testset if x[1] in highvar_movies]
            predictions = algo.test(test_trim)
            accu.append(accuracy.rmse(predictions, verbose=True))
        s = np.mean(accu)
        rmse_highvar_store.append(s)


    pd.DataFrame(rmse_highvar_store).to_csv("rmse_highvar_store_13.csv")
    plotgraphs(x_axis,rmse_highvar_store,'K','Mean RMSE scores','Plot of high variance movies')



    ######################### ROc plots ###################################



    Threshold = [2.5, 3, 3.5, 4]
    best_k = 20
    trainset, testset = train_test_split(data, test_size=0.1, train_size=None, random_state=None, shuffle=True)
    algo = KNNWithMeans(k=best_k, sim_options=sim_options,verbose=True)
    algo.fit(trainset)
    predictions = algo.test(testset)

    print(predictions)


    trui = [ getattr(r,'r_ui') for r in predictions ]
    est = [ getattr(r,'est') for r in predictions ]

    for t in Threshold:
        trui_bin = [1 if r > t else 0 for r in trui]
        # est_bin = [1 if r > t else 0 for r in est]
        fpr, tpr, _ = metrics.roc_curve(trui_bin,est)
        plot_roc(fpr,tpr,t)