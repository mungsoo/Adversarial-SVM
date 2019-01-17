# from lz import *
import logging, numpy as np, matplotlib.pyplot as plt
logging.root.setLevel(logging.ERROR)
import svm, data


def test_on_dataset(dataset = 'sonar', kernel = 'rbf', n_samples = 208, n_features = 60):

    l = 1;
    fig = plt.figure(figsize=(8,3))
    for C in [1, 10, 100, 1000]:
        ax = fig.add_axes([0.25*l-0.20, 0.3,0.18,0.4])
        
        acc_1 = []
        acc_2 = []
        acc_3 = []
        for L in [0, n_samples//10, n_samples//5, int(n_samples//3.3333), int(n_samples//2.5)]:
            X_, y_ = data.get_train_data(dataset, n_samples, n_features)
            X_c, y_c, = data.split_train_test(X_, y_)
            # trainer = svm.SVMTrainer(kernel, C)
            # predictor = trainer.train(X, y, remove_zero=True)
            # print(predictor.error(X_val, y_val))
            # print(acc_1, acc_2, acc_3)
            acc_1.append(0)
            acc_2.append(0)
            acc_3.append(0)
            for i in range(5):
                X, y = [], []
                
                for j in range(5):
                    if j != i:
                        X.extend(X_c[j])
                        y.extend(y_c[j])
                X_val, y_val = np.array(X_c[i]), np.array(y_c[i])
                X, y = np.array(X), np.array(y)
                X, y, flip_pnts = data.apply_rand_flip(X, y, L)

                trainer = svm.SVMTrainer(kernel, C)
                predictor = trainer.train(X, y, remove_zero=True)
                acc_1[-1] += (1-predictor.error(X_val, y_val))
            #print(acc_1[-1])

    
                trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.1)
                predictor = trainer.train(X, y, remove_zero=True)
                acc_2[-1] += (1-predictor.error(X_val, y_val))
            #print(acc_2[-1])
            
                trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.5 - 1e-4)
                predictor = trainer.train(X, y, remove_zero=True)
                acc_3[-1] += (1-predictor.error(X_val, y_val))
            #print(acc_3[-1])
            acc_1[-1] /= 5
            acc_2[-1] /= 5
            acc_3[-1] /= 5
        acc_1 = np.array(acc_1)
        acc_2 = np.array(acc_2)
        acc_3 = np.array(acc_3)
        flip_ratio = np.linspace(0, 40,acc_1.shape[0])
        plt.ylim((0.3,1))
        print(acc_1)
        ax.plot(flip_ratio,acc_1,color="blue", label='mu=0')
        ax.plot(flip_ratio,acc_2,color="red", label='mu=0.1')
        ax.plot(flip_ratio,acc_3,color="black", label='mu=0.5')
        ax.set_xlabel('% flipped labels')
        ax.set_ylabel('test acc')
        ax.set_title("C = %d" % C)
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        
        ax.spines['left'].set_color('black')
        
        ax.spines['bottom'].set_color('black')
        ax.patch.set_facecolor("white")
        ax.grid(color='r', linestyle='--',linewidth=1, alpha=0.3)
        #if l == 1:
        #    ax.legend(facecolor='white')
        l += 1
    fig.suptitle(dataset, fontsize=12)
    
    
exp = 'else'
if exp == 'proc':
    C = 10.
    kernel = 'linear'
    #如果要从清洗过的训练集读就用下面这句
    #proc_train = np.loadtxt('proc-train-clean', delimiter=' ')
    proc_train = np.loadtxt('proc-train', delimiter=' ')
    X, y = proc_train[:, 1:], proc_train[:, 0]
    X_test = np.loadtxt('proc-test', delimiter=' ')
    
    #trainer = svm.SVMTrainer(kernel, C)
    trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.4)
    
    predictor = trainer.train(X, y, remove_zero=True)
    #读取在训练集上的错误率
    error_rate = predictor.error(X, y)
    print("Training error rate: %f" % error_rate)
    
    
    #根据训练得出的SVM模型，得出训练集X的score，与他们的标签比对
    #如果符号相反且差距太大，则翻转标签
    #如果不需要清洗数据集就把这段代码注释掉，这样不会破坏之前清洗好的数据proc-train-clean
    scores = predictor.score(X)
    error_index = np.where((y * scores < 0) & (np.abs(y+scores) > 10))
    print("Potential dirty data index:")
    print(error_index)
    proc_train[error_index,0] *= -1
    np.savetxt('proc-train-clean', proc_train, delimiter=' ')
    
    y_pred = predictor.predict(X_test)
    
    np.savetxt('y-pred', y_pred, delimiter=' ')

elif exp == 'else':
    #test_on_dataset('australian', 'rbf', 690, 14)
    #test_on_dataset('diabetes', 'rbf', 768, 8)
    #test_on_dataset('heart', 'rbf', 270, 13)
    test_on_dataset()
    plt.show()    

elif exp == 'toy':

    # ori
    n_samples = 100
    kernel = 'linear'#原来是rbf
    seed = 16
    C = 10.
    R = 30
    L = 100 // 10
    beta1 = beta2 = 0.1
    X, y = data.get_toy_data(n_samples=n_samples, seed=seed)
    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y, remove_zero=True)

    print(X.shape)
    plt.figure()
    data.svm_plot(X, y)
    data.boundary_plot(X, predictor)
    plt.savefig('ori.png')
    plt.show()

    # tainted data
    X, y_p, flip_pnts = data.get_adv_data(n_samples=n_samples, seed=seed, C=C, R=R, L=L, beta1=beta1, beta2=beta2)
    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y_p, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y_p)
    data.boundary_plot(X, predictor)
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.savefig('ln.png')
    plt.show()

    trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.1)
    predictor = trainer.train(X, y_p, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y_p)
    data.boundary_plot(X, predictor)
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.savefig('ln.robust.mu.0.1.png')
    #plt.show()

    trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.5)
    predictor = trainer.train(X, y_p, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y_p)
    data.boundary_plot(X, predictor)
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.savefig('ln.robust.mu.0.5.png')
    plt.show()
