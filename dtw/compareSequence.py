import numpy as np


def euclidean(x, y):
    return np.sqrt((x - y) ** 2)


class dtw:
    '''
    dtw class caculate distance between two univariate time series data

    x : univaritate time series
    y : univaritate time series
    i,j : coordinate of matrix that user want to know information(minimum distance to (i,j) the coordinate of matrix)

    dist_ij : euclidean distance(~cost) between x_i and y_j
    dist_mat : culmulative distance(~cost) to (i,j) coordinate
    path : shortest path to (i,j) coordinate
    dtw_dist : minimum culmulative distance(~cost) to (i,j) coordinate

    '''

    def __init__(self, x, y, i, j):
        self.x = x
        self.y = y
        self.dist_ij = np.zeros((len(self.x), len(self.y)))
        self.dist_mat = np.zeros((len(self.x), len(self.y)))
        self.path = []

        for i in range(len(x)):
            for j in range(len(y)):
                self.dist_ij[i, j] = euclidean(self.x[i], self.y[j])
        self.dtw_dist = self.dtw_cost(i, j)
        self.dtw_path(i, j, self.dist_mat)


    ''' calculate minimum culmulative distance(~cost) to (i,j) coordinate'''
    def dtw_cost(self, i, j):
        result = 0
        if i >= 1 and j >= 1:
            result += euclidean(self.x[i], self.y[j]) + min([ self.dtw_cost(i- 1, j - 1), self.dtw_cost(i - 1, j), self.dtw_cost(i, j - 1)])
            self.dist_mat[i, j] = result
        elif i == 0 and j == 0:
            result = euclidean(self.x[i], self.y[j])
            self.dist_mat[i, j] = result
        elif i == 0 and j != 0:
            result += euclidean(self.x[i], self.y[j]) + self.dtw_cost(i, j - 1)
            self.dist_mat[i, j] = result
        elif j == 0 and i != 0:
            result += euclidean(self.x[i], self.y[j]) + self.dtw_cost(i - 1, j)
            self.dist_mat[i, j] = result

        return result


    ''' extract shortest path to (i,j) coordinate'''
    def dtw_path(self, i, j, dis_mat):
        if i >= 1 and j >= 1:
            self.path.append((i, j))
            idx = np.argmin([dis_mat[i - 1, j - 1], dis_mat[i, j - 1], dis_mat[i - 1, j]])
            if idx == 0:
                self.dtw_path(i - 1, j - 1, dis_mat)
            elif idx == 1:
                self.dtw_path(i, j - 1, dis_mat)
            elif idx == 2:
                self.dtw_path(i - 1, j, dis_mat)
        elif i == 0 and j == 0:
            self.path.append((0, 0))
        elif i == 0 and j != 0:
            self.path.append((0, j))
            self.dtw_path(0, j - 1, dis_mat)
        elif j == 0 and i != 0:
            self.path.append((i, 0))
            self.dtw_path(i - 1, 0, dis_mat)

        return


    def set_location(self,set_i, set_j):
        print('Coordinate : ({}, {})'.format(set_i, set_j))
        self.dtw_dist = self.dtw_cost(set_i, set_j)
        self.path = []
        self.dtw_path(set_i, set_j, self.dist_mat)

        return

