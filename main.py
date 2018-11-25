from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
from mySVM import *

def main():
    np.random.seed(1)
    iris = load_iris()
    N_pos = len(list(filter(lambda x: x == 0, iris.target)))
    tmp_list = np.random.choice(list(range(N_pos, len(iris.target))), N_pos, replace=False)

    print(N_pos)
    label = [1 if i < N_pos else -1 for i in range(N_pos*2)]

    #print(iris.data)
    index_list = list(range(N_pos))
    index_list.extend(tmp_list)
    #print(len(index_list))
    x = iris.data[index_list]
    x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.2, random_state=2)
    
    clf = MySVM()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(pred)
    print(y_test)
    print("Accuray: " + str(accuracy_score(y_test, pred)))
    
    clf = SVC(gamma='auto')
    clf.fit(x_train, y_train)
    print(accuracy_score(y_test, clf.predict(x_test)))
    
if __name__ == "__main__":
    main()
