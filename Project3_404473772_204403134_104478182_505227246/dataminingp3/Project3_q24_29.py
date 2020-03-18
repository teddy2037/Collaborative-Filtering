import numpy as np
from Project3_q30_33 import trim, retrieve_data
from surprise.model_selection import KFold, cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.accuracy import rmse
import matplotlib.pyplot as plt


def Question24(data):
    ks = range(2,51,2)
    RMSE = []
    MAE = []
    for k in ks:
        model = SVD(n_factors=k)
        pred = cross_validate(model, data, cv=10)
        RMSE.append(np.mean(pred['test_rmse']))
        MAE.append(np.mean(pred['test_mae']))

    # Plot
    plt.plot(ks, RMSE)
    plt.xlabel('k')
    plt.ylabel('Average RMSE')
    plt.savefig('Q24_RMSE.png')
    plt.figure()
    plt.plot(ks, MAE)
    plt.xlabel('k')
    plt.ylabel('Average MAE')
    plt.savefig('Q24_MAE.png')

    
    index = np.argmin(RMSE)
    print("Best k: %i" % ks[index] )
    print("Lowest RMSE: %f" % RMSE[index] )
    print("Lowest MAE: %f" % np.min(MAE) )


def trimmed_test_MF(data, choice = 0):
    ks = range(2,51,2)
    avg_RMSEs = []
    for k in ks:
        kf = KFold(n_splits=10)
        rmse_total = 0
        for trainset, testset in kf.split(data):
            trimmed_testset = trim(data, testset, choice)
            model = SVD(n_factors=k).fit(trainset)
            pred = model.test(trimmed_testset)
            rmse_total += rmse(pred, verbose=False)
        rmse_total = rmse_total / 10.0
        avg_RMSEs.append(rmse_total)
        
    # Plot
    plt.plot(ks, avg_RMSEs)
    plt.xlabel('k')
    plt.ylabel('Average RMSE')
    plt.savefig('RMSE_' + str(choice) + '.png')

    index = np.argmin(avg_RMSEs)
    print("Best k: %i" % ks[index] )
    print("Lowest RMSE: %f" % avg_RMSEs[index] )



if __name__ == '__main__':
    data = retrieve_data()
    Question24(data)
    # Question 26
    print("Trimmed Test Set: Popular movies")
    trimmed_test_MF(data, 1)
    # Question 27
    print("\nTrimmed Test Set: Unpopular movies")
    trimmed_test_MF(data, 2)
    # Question 28
    print("\nTrimmed Test Set: High-variance movies")
    trimmed_test_MF(data, 3)
    #Question 29
    Threshold = [2.5, 3, 3.5, 4]  # thresholds
    # MF
    best_k = 20
    fpr_svd, tpr_svd, t_svd = get_roc_params(algo=SVD(best_k, verbose=False), data=data, threshold=Threshold)

    for i, thresh in enumerate(Threshold):
        fig, ax = plt.subplots()
        roc_auc_svd = auc(fpr_svd[i], tpr_svd[i])

        ll3 = 'MF area under curve = %0.4f' % roc_auc_svd

        l3, = ax.plot(fpr_svd[i], tpr_svd[i], lw=2)
        ax.grid(color='0.7', linestyle='--', linewidth=1)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', size=15)
        ax.set_ylabel('True Positive Rate', size=15)
        ax.legend((l3,), (ll3,), loc="lower right")

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        plt.title('Threshold  = %0.2f' % thresh, size=15)
        plt.savefig('q29'+str(thresh)+'.png', bbox_inches='tight')
        plt.show()


    





