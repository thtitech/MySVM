from sklearn.datasets import load_iris
import numpy as np
from mySVM import *

def main():
    iris = load_iris()
    N_pos = len(list(filter(lambda x: x == 0, iris.target)))
    tmp_list = np.random.choice(list(range(N_pos, len(iris.target))), N_pos, replace=False)

    print(N_pos)
    label = [1 if i < N_pos else -1 for i in range(N_pos*2)]

    #print(iris.data)
    index_list = list(range(N_pos))
    index_list.extend(tmp_list)
    print(len(index_list))
    x = iris.data[index_list]
    clf = MySVM()
    clf.fit(x, label)
    
if __name__ == "__main__":
    main()
