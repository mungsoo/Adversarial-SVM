# from lz import *
import logging, numpy as np, matplotlib.pyplot as plt
logging.root.setLevel(logging.ERROR)
import svm, data

exp = 'proc'
if exp == 'proc':
    C = 10.
    kernel = 'linear'
    # If wanna use cleaned dataset, use this line
    #proc_train = np.loadtxt('proc-train-clean', delimiter=' ')
    
    proc_train = np.loadtxt('proc-train', delimiter=' ')
    X, y = proc_train[:, 1:], proc_train[:, 0]
    X_test = np.loadtxt('proc-test', delimiter=' ')
    
    #trainer = svm.SVMTrainer(kernel, C)
    trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.5)
    
    predictor = trainer.train(X, y, remove_zero=True)
    
    # Compute the error_rate in training set
    error_rate = predictor.error(X, y)
    print("Training error rate: %f" % error_rate)
    
    
    # Use the SVM model we have trained to derive scores of each sample in training set X
    # Compare scores with their label, if the sign is opposite and has a huge difference, reverse the label
    # If do not need to wash the data set, comment this code section so that the cleaned proc-train-clean won't be changed
    scores = predictor.score(X)
    error_index = np.where((y * scores < 0) & (np.abs(y+scores) > 10))
    print("Potential dirty data index:")
    print(error_index)
    proc_train[error_index,0] *= -1
    np.savetxt('proc-train-clean', proc_train, delimiter=' ')
    
    
    y_pred = predictor.predict(X_test)
    
    np.savetxt('y-pred', y_pred, delimiter=' ')

elif exp == 'sonar':
    n_samples = 208
    n_features = 60
    C = 1.
    L = 208 // 10
    kernel = 'rbf'
    i = 1;
    fig = plt.figure(figsize=(30,4))
    for C in [0.1, 1, 10, 100]:
        ax = fig.add_axes([0.25*i-0.20, 0.3,0.18,0.4])
        
        acc_1 = []
        acc_2 = []
        acc_3 = []
        for L in [0, 208//10, 208//5, int(208//3.3333), int(208//2.5)]:
            X, y = data.get_sonar_data()
            X, y, X_val, y_val = data.split_train_test(X, y)
            # trainer = svm.SVMTrainer(kernel, C)
            # predictor = trainer.train(X, y, remove_zero=True)
            # print(predictor.error(X_val, y_val))
            
    
            X, y, flip_pnts = data.apply_rand_flip(X, y, L)

            trainer = svm.SVMTrainer(kernel, C)
            predictor = trainer.train(X, y, remove_zero=True)
            acc_1.append(1-predictor.error(X_val, y_val))
            #print(acc_1[-1])

    
            trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.1)
            predictor = trainer.train(X, y, remove_zero=True)
            acc_2.append(1-predictor.error(X_val, y_val))
            #print(acc_2[-1])
            
            trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.5)
            predictor = trainer.train(X, y, remove_zero=True)
            acc_3.append(1-predictor.error(X_val, y_val))
            #print(acc_3[-1])

        acc_1 = np.array(acc_1)
        acc_2 = np.array(acc_2)
        acc_3 = np.array(acc_3)
        flip_ratio = np.linspace(0, 40,acc_1.shape[0])
        plt.ylim((0.3,1))
        
        ax.plot(flip_ratio,acc_1,color="blue", label='mu=0')
        ax.plot(flip_ratio,acc_2,color="red", label='mu=0.1')
        ax.plot(flip_ratio,acc_3,color="black", label='mu=0.5')
        ax.set_xlabel('% flipped labels')
        ax.set_ylabel('test acc')
        ax.set_title("C = %f" % C)
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.patch.set_facecolor("white")
        ax.grid(color='r', linestyle='--',linewidth=1, alpha=0.3)
        if i == 1:
            ax.legend(facecolor='white')
        i += 1
    plt.show()    

elif exp == 'toy':


    n_samples = 100
    kernel = 'linear'
    seed = 16
    C = 1.
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
    #plt.show()

    # tainted data
    X, y_p, flip_pnts = data.get_adv_data(n_samples=n_samples, seed=seed, C=C, R=R, L=L, beta1=beta1, beta2=beta2)
    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y_p, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y_p)
    data.boundary_plot(X, predictor)
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.savefig('ln.png')
    #plt.show()

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
