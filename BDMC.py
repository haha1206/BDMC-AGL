import numpy as np
from data_loader import load_mat
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def generate_random_matrix(rows, cols):
    data = np.zeros((rows, cols))
    for i in range(rows):
        data[i, np.random.randint(cols)] = 1
    return data
def BDMC(X,num_cluster,k,n_max,n_min,INTER=10):
    n_view = X.size
    num = X[0].shape[0]

    #normalization
    for i in range(n_view):
        for j in range(num):
            X[i][j,:] = (X[i][j,:]-np.mean(X[i][j,:]))/np.std(X[i][j,:])

    lambda_n = 1e-2
    sum = np.zeros(num)
    disX_In = np.zeros((num,num,n_view))
    for i in range(n_view):
        disX_In[:,:,i] =  EuDist2(X[i],X[i])
        sum = sum+disX_In[:,:,i]

    disX = sum*(1/n_view)
    index = np.argsort(disX,axis=1)
    #initialize S
    eps = 2.2204e-16
    S = np.zeros((num,num))
    rr = np.zeros((num,1))
    for i in range(num):
        id = index[i, 1:(k + 2)]
        di = disX[i, id]
        rr[i] = 0.5 * (k * di[k] - np.sum(di[:k]))
        S[i,id] = (di[k]-di)/(k*di[k]-np.sum(di[:k])+eps)
    #initialize F
    F = generate_random_matrix(num,num_cluster)

    for inter in range(INTER):
        sum = np.zeros((num,num))
        for i in range(n_view):
            if i==0:
                disX_Up = disX_In
            disX_Up[:,:,i] = (0.5/np.power(np.sum(np.dot(disX_Up[:,:,i],S)),0.5))*disX_Up[:,:,i]
            sum = sum+disX_Up[:,:,i]
        disX = sum
        #update S
        disF = EuDist2(F,F)
        S = np.zeros_like(S)
        for i in range(num):
            indxa0 = index[i,1:(k+2)]
            dfi = disF[i,indxa0]
            dxi = disX[i,indxa0]
            ddi = dxi + lambda_n * dfi
            ad = ddi / (2*rr[i])
            S[i,indxa0] = EProjSimplex_new(-ad)
        #update F
        S = (S + S.T) / 2
        F,G = ALM(S,F,n_max=n_max, n_min=n_min)

    return F

def EProjSimplex_new(v,k=1):
    ft = 1
    n = v.size
    v_0 = v - np.mean(v) + k/n
    v_min = np.min(v_0)
    if v_min<0:
        f = 1
        lambda_m = 0
        while abs(f)> 1e-10:
            v_1 = v_0 - lambda_m
            posidx = v_1>0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v_1[posidx])-k
            lambda_m = lambda_m-f/g
            ft = ft+1
            if ft>100:
                v_1[v_1<0]=0
                x = v_1
                break
            v_1[v_1<0]=0
            x = v_1
    else:
        x = v_0
    return x
def ALM(A, F, n_max,n_min,mu=0.1,rou=1.005):
    N_inter = 20
    threshold = 1e-10
    val = 0
    cnt = 0
    n,n_k = F.shape
    G = F
    lambda_n = np.ones(F.shape)
    for a in range(N_inter):
        #update G
        H1 = A@F
        H2 = F + (1/mu)*lambda_n
        H = H1/mu + H2
        for j in range(n_k):
            G[:, j] = G[:, j] - 0.2 * (2 * (G[:, j] - H[:, j]))
            if np.sum(G[:, j]) >= n_max:
                G[:, j] = EProjSimplex_new(G[:, j], k=n_max)
            if np.sum(G[:, j]) <= n_min:
                G[:, j] = EProjSimplex_new(G[:, j], k=n_min)
            else:
                G[:, j][G[:, j] < 0] = 0

        M1 = A@G
        M2 = lambda_n/mu - G
        M = M1/mu - M2
        F1 = np.zeros((n,n_k))
        for i in range(n):
            F1[i,np.argmax(M[i,:])]=1
        F = F1
        #update lambda_n
        lambda_n = lambda_n + mu*(F-G)
        #update mu
        mu = rou*mu
        val_old = val
        val = -np.trace(F.T@A@F)
        if abs(val - val_old) < threshold:
            if cnt >=5:
                break
            else:
                cnt +=1
        else:
            cnt = 0

    return F,G
def acc(y_true, y_pred):
    # Calculate clustering accuracy
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size
def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)
def F_score(labels_true, labels_pred):
    n = len(labels_true)
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            if labels_true[i] == labels_true[j] and labels_pred[i] == labels_pred[j]:
                tp += 1
            elif labels_true[i] != labels_true[j] and labels_pred[i] == labels_pred[j]:
                fp += 1
            elif labels_true[i] == labels_true[j] and labels_pred[i] != labels_pred[j]:
                fn += 1
            else:
                tn += 1

    Pre = tp / (tp + fp)
    Rec = tp / (tp + fn)

    F_score = 2*Pre*Rec / (Pre+Rec)

    return F_score

def NE(y_pred):
    n = len(y_pred)
    ar, num = np.unique(y_pred, return_counts=True)
    c = len(ar)
    ne = 0
    for i in range(c):
        ne = ne + (num[i]/n)*np.log(num[i]/n)
    return (-1/np.log(c))*ne
if __name__=="__main__":

    X, GT = load_mat('MVCdata/MSRCV1.mat')
    N, _ = X[0].shape
    c = len(np.unique(GT))
    k=10
    Inter = 10
    Max_list = [np.floor(1.2* N / c), np.floor(1.4* N / c), np.floor(1.6* N / c), np.floor(1.8* N / c)] # For most datasets
    # Max_list = [np.floor(4 * N / c), np.floor(6 * N / c), np.floor(8 * N / c), np.floor(10 * N / c)] # For Caltech101
    # Max_list = [np.floor(3 * N / c), np.floor(4 * N / c), np.floor(5 * N / c), np.floor(6 * N / c)]  # For NUS
    Min_list = [np.floor(0.7 * N / c), np.floor(0.5 * N / c), np.floor(0.3 * N / c), np.floor(0.1 * N / c)]

    for i in range(len(Max_list)):
        for j in range(len(Min_list)):
            ACC_res = []
            NMI_res = []
            Purtiy_res = []
            ARI_res = []
            F1_res = []
            for inter in range(10):
                GT = GT.reshape(np.max(GT.shape), )
                F = BDMC(X, c, k, Max_list[i],Min_list[j],Inter)
                y_pred = np.argmax(F, axis=1)
                GT = GT.reshape(np.max(GT.shape), )
                ACC = acc(GT, y_pred)
                NMI = metrics.normalized_mutual_info_score(GT, y_pred)
                Purity = purity_score(GT, y_pred)
                ARI = metrics.adjusted_rand_score(GT, y_pred)
                F1_score = F_score(GT, y_pred)
                ne = NE(y_pred)
                print('ACC: {}, NMI: {}, Purity: {},ARI:{},F1:{},MAX:{},MIN:{}'.format(ACC, NMI, Purity, ARI, F1_score,
                                                                                       Max_list[i], Min_list[j]))
                ACC_res.append(ACC)
                NMI_res.append(NMI)
                Purtiy_res.append(Purity)
                ARI_res.append(ARI)
                F1_res.append(F1_score)

            print('ACC_mean: {}, ACC_std: {}'.format(np.array(ACC_res).mean(), np.array(ACC_res).std()),Max_list[i], Min_list[j])
            print('NMI_mean: {}, NMI_std: {}'.format(np.array(NMI_res).mean(), np.array(NMI_res).std()),Max_list[i], Min_list[j])
            print('Purtiy_mean: {}, Purtiy_std: {}'.format(np.array(Purtiy_res).mean(), np.array(Purtiy_res).std()),Max_list[i], Min_list[j])
            print('ARI_mean: {}, ARI_std: {}'.format(np.array(ARI_res).mean(), np.array(ARI_res).std()),Max_list[i], Min_list[j])
            print('F1_mean: {}, F1_std: {}'.format(np.array(F1_res).mean(), np.array(F1_res).std()),Max_list[i], Min_list[j])





