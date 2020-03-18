# local imports
from Project3_q30_33 import retrieve_data
from Project_3_q17_22_q34_functions import plotgraphs
from Project3_q30_33 import ret_user_dict
from collections import defaultdict
from operator import itemgetter


# global imports
from surprise.model_selection import KFold
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise import KNNWithMeans


# global VARs
KNN_no_of_LF = 20
NMF_no_of_LF = 20
MF_no_of_LF = 20
threshold = 3.0
sim_options = {'name' : 'pearson' , 'user_based' : True}


def ret_mod_user_dict(data):
    user_movies = {}
    for r in data.raw_ratings:
        if r[0] not in user_movies:
            user_movies[r[0]] = []
        if r[2] >= 3.0:
            user_movies[r[0]].append(r[1])
    return user_movies


def filter_test_set(testset, G_max, user_movie_dict, t):
    testfin = []
    user_movie_count = []

    for key in user_movie_dict:
        if len(user_movie_dict[key]) < t:
            user_movie_count.append(key)

    for test_iter in testset:
        if len(G_max[test_iter[0]]) > 0 and (test_iter[0] not in user_movie_count):
            testfin.append(test_iter)

    return testfin


def metrics(predictions, t, threshold=threshold):

    test_user_dict = defaultdict(list)
    for userid, _, true_rating, est_rating, possibility in predictions:
        test_user_dict[userid].append((true_rating, est_rating, possibility['was_impossible']))

    pr_avg_per_fold = 0.0
    re_avg_per_fold = 0.0
    number_of_users = 0.0

    prec = 0.0
    reca = 0.0

    for userid, user_ratings in test_user_dict.items():
        user_ratings = sorted(user_ratings,key=itemgetter(1))
        user_ratings.reverse()

        if len(user_ratings) < t: # Constraint 2
            continue

        # Size of G per user
        magG = sum((true_rating >= threshold) for true_rating,_,_ in user_ratings)

        if magG == 0: # Constraint 3
            continue

        # constraint 1
        magS = sum(bolo == False for true_rating,est_rating, bolo in user_ratings[:t])

        if magS == 0:
            continue

        magG_I_S = sum(((true_rating >= threshold) and (est_rating >= threshold) and bolo == False)
                              for (true_rating, est_rating, bolo) in user_ratings[:t])

        prec = prec + magG_I_S / float(magS)
        reca = reca + magG_I_S / float(magG)

        number_of_users = number_of_users + 1.0

    pr_avg_per_fold = prec / number_of_users
    re_avg_per_fold = reca / number_of_users

    return pr_avg_per_fold, re_avg_per_fold


def cross_val_(data, G_max, t, algo):

    pr = 0.0
    re = 0.0
    user_movie_dict = ret_user_dict(data)
    kf = KFold(n_splits=10)
    for trainset, testset in kf.split(data):
        print "Fold for" + str(t)
        algo.fit(trainset)
        # print testset
        predictions = algo.test(testset)
        Prec, Reca = metrics(predictions, t)
        pr = pr + Prec
        re = re + Reca

    return pr / 10.0, re / 10.0


if __name__ == '__main__':
    data = retrieve_data()
    G_max = ret_mod_user_dict(data)

    algo_NMF = NMF(NMF_no_of_LF, verbose=False)
    algo_SVD = SVD(n_factors=MF_no_of_LF)
    algo_KNN = KNNWithMeans(k=KNN_no_of_LF, sim_options=sim_options,verbose=False)

    # Q36
    Pr1 = []
    Re1 = []
    t = list(range(1,26))
    for l in t:
        Precision, Recall = cross_val_(data, G_max, l, algo_KNN)
        Pr1.append(Precision)
        Re1.append(Recall)

    plotgraphs(t, Pr1, "Number of Suggestions", "Precision", "Precision Curve for KNN")
    plotgraphs(t, Re1, "Number of Suggestions", "Recall", "Recall Curve for KNN")
    plotgraphs(Re1, Pr1, "Recall", "Precision", "Precision-Recall Curve for KNN")

    # Q37
    Pr2 = []
    Re2 = []
    for l in t:
        Precision, Recall = cross_val_(data, G_max, l, algo_NMF)
        Pr2.append(Precision)
        Re2.append(Recall)

    plotgraphs(t, Pr2, "Number of Suggestions", "Precision", "Precision Curve for NNMF")
    plotgraphs(t, Re2, "Number of Suggestions", "Recall", "Recall Curve for NNMF")
    plotgraphs(Re2, Pr2, "Recall", "Precision", "Precision-Recall Curve for NNMF")

    # Q38
    Pr3 = []
    Re3 = []
    for l in t:
        Precision, Recall = cross_val_(data, G_max, l, algo_SVD)
        Pr3.append(Precision)
        Re3.append(Recall)

    plotgraphs(t, Pr3, "Number of Suggestions", "Precision", "Precision Curve for MF")
    plotgraphs(t, Re3, "Number of Suggestions", "Recall", "Recall Curve for MF")
    plotgraphs(Re3, Pr3, "Recall", "Precision", "Precision-Recall Curve for MF")

plt.plot(Re1,Pr1, label = 'KNN')
plt.plot(Re2,Pr2,label = 'NNMF')
plt.plot(Re3,Pr3, label = 'MF')
plt.xlabel('Recall', size=15)
plt.ylabel('Precision', size=15)
plt.title('Comparison of Precision-Recall Trade-Off', size=15)
plt.legend(fontsize=10)
plt.draw()
plt.savefig('PR_all.png', bbox_inches='tight')
plt.show()










