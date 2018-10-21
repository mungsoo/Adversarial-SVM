# Adversarial-SVM
Implementation of  an adversarial label resistant SVM 

SVM without label noise
-----
The data set choise is toy.

<center>![](https://github.com/mungsoo/Adversarial-SVM/blob/master/images/ori.png?raw=true)</center>

SVM with label noise
---
<center>![](https://github.com/mungsoo/Adversarial-SVM/blob/master/images/ln.png?raw=true)</center>


Ln-Robust-SVM with \mu = 0.1
---
<center>![](https://github.com/mungsoo/Adversarial-SVM/blob/master/images/ln.robust.mu.0.1.png?raw=true)</center>


Ln-Robust-SVM with \mu = 0.5
---
<center>![](https://github.com/mungsoo/Adversarial-SVM/blob/master/images/ln.robust.mu.0.5.png?raw=true)</center>


Training result on sonar_scale
-----
<center>![](https://github.com/mungsoo/Adversarial-SVM/blob/master/images/sonar.jpg?raw=true)</center>

Data Cleansing of proc-train
---
Fristly, I try to use RBF kernel. It shows high accuracy in dirty training set, which means it definetly
overfits. So I decide to use linear kernel. And insert the code below
```
scores = predictor.score(X)
error_index = np.where(y * scores < 0) & (np.abs(y + scores) > 10))
print("Potential dirty data index:")
print(error_index)
proc_train[error_index, 0] *= -1
np.savetxt('proc-train-clean', proc_train, delimiter = ' ')
```

error_index are the training sample which are falsely predicted and the distance to dicision boundry is greater than 9.

<center>![](https://github.com/mungsoo/Adversarial-SVM/blob/master/images/dirty_data.jpg?raw=true)</center>

We can find that the total number of dirty label is 200, which is 20% of the size of training set.
So I reverse these labels and save as proc-train-clean.
