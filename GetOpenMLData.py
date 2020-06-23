from sklearn import datasets
import numpy as np


class OpenMLData:
    def __init__(self, db_id, positive_class = 1):
        self.db_id = db_id
        self.data_X, self.data_y = datasets.fetch_openml(data_id=db_id, return_X_y=True)

        self.data_y = np.where(self.data_y == positive_class, 1, 0)
        self.data_y = self.data_y.astype(np.int)

def main():
    data = OpenMLData(1460)
    print(data.data)


if __name__ == '__main__':
    main()
