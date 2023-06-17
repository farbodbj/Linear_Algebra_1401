


#####
#####           FIRST PART: NEEDED CODES AND FUNCTIONS
#####

import numpy as np

ROW = 0
COL = 1

#returns a copy of an array with one row and column deleted
def sub_mat(input_mat: np.ndarray, del_row: int, del_col: int)->np.ndarray:
    try:
        output = np.delete(input_mat, (del_row), axis=0)
        output = np.delete(output, (del_col), axis=1)
        return output
    except:
        raise Exception("Given row or column out of bounds!")
        

#A very simple implementaion of the method that
#uses cofactor expansion to calculate determinant
#this method's time complexity is O(n!) :>
det = 0
def naive_det(input_mat: np.ndarray)->float:
    global det
    col_count = input_mat.shape[COL]
    row_count = input_mat.shape[ROW]
    if row_count != col_count:
        raise Exception("Matrix is not square!")
    if input_mat.shape == (1,1):
        return input_mat[0][0]
    if input_mat.shape == (2,2):
        return input_mat[0][0] * input_mat[1][1] - input_mat[1][0] * input_mat[0][1]
    else:
        for ind in range(col_count):
            det += input_mat[0][ind] * ((-1)**(ind%2) * naive_det(sub_mat(input_mat, 0, ind)))
        return det

#produces matrix transpose
def transpose(input_mat: np.ndarray)->np.ndarray:
    col_count = input_mat.shape[COL]
    row_count = input_mat.shape[ROW]
    output = np.zeros((col_count, row_count))
    for i in range(row_count):
        for j in range(col_count):
            output[j][i] = input_mat[i][j]
    
    return output

#calculated dot product using a simple for-loop. 
#Can be replaced with np.dot !
def dot_product(A:np.ndarray,B:np.ndarray)->int:
    dot = 0
    cols = A.shape[ROW]
    for ind in range(cols):
        dot += A[ind]*B[ind]
    
    return dot



#uses dot_product function to calculate matrix multiplication result
#other algorithms could be used for better performance (e.g: strassen's, schonhagen etc)
def naive_multiplication(A:np.ndarray,B:np.ndarray)->np.ndarray:
    if(A.shape[COL] != B.shape[ROW]):
        raise Exception("Matrices are not multipliable!")
    
    A_rows = A.shape[ROW]
    B_cols = B.shape[COL]
    
    result = np.empty((A_rows, B_cols))
    
    for row in range(A_rows):
        for col in range(B_cols):
            result[row][col] = dot_product(A[row],B[:,col])
    
    return result


def strategy_rating(input_strategy: np.ndarray)->float:
    return naive_det(naive_multiplication(transpose(input_strategy), input_strategy))





#####
#####               SECOND PART: ANSWERS TO THE QUESTIONS
#####


#question 1:
print("Question one:\n")
d = [
    [1, 1, -1, -1]
    ]
D = np.asarray(d)
print(strategy_rating(D))

d = [
    [1, -1, 1, -1]
    ]
D = np.asarray(d)
print(strategy_rating(D))

print("\n*************")

#it is observed that for weighting strategies that have matrices of only one row (with values only equal to +1 or -1),
#the determinant of transpose(D) * D is always equal to 0 regardless of values 


#question 2: 
print("Question two:\n")
d = [
     [1, 1, -1, -1],
     [1, -1, 1, -1]
    ]
D = np.asarray(d)
print(strategy_rating(D))

d = [
     [-1, -1, 1, -1],
     [1, -1, -1, 1]
    ]
D = np.asarray(d)
print(strategy_rating(D))

print("\n*************")

#also for matrices of two rows and n columns that only consist of -1 and +1 the determinant is equal to 0


#question 3:
print("Question three:\n**************\n")

'''
when matrices have less rows than columns in a matrix representing the weighting table of jewelry, 
it can be deduced that the weighting strategy is not good and can not help us find the accurate weight for 
the matrix, thus det(DT * D) is equal to zero in this conditions.
'''
print("\n*************")

#question 4:
print("Question four:\n")

#Alef:
d = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
D = np.asarray(d)
print(strategy_rating(D)) #equal to 0



#B:
d = [[-1, 1, 1, 1], [1, -1, -1, 1], [1, 1, 1, -1], [1, 1, 1, -1]]
D = np.asarray(d)
print(strategy_rating(D)) #between 0 and 256

#P:
d = [[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]]
D = np.asarray(d)
print(strategy_rating(D)) #equal to 256

#no the maximum value for det(DT * D) is 256

#for the best results, in the Jth time of weighting (Jth row in matrix) the Ith jewl (Ith column) must be in the left pan of the scale.
#or in other words the items on the main diagonal should be equal to -1 and all others equal to 1

print("\n*************")

#question 5:
print("Question five:\n")
#Alef:
d = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
D = np.asarray(d)
print(strategy_rating(D)) #equal to 0

#B:
d = [[-1, 1, 1, 1 ], [1, -1, -1, 1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]]
D = np.asarray(d)
print(strategy_rating(D)) #between 0 and 512

#P:
d = [[-1, 1, 1, 1 ], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]]
D = np.asarray(d)
print(strategy_rating(D)) #equal to 512
print("\n*************")

#no the maximum value for this determinant is 512

#for best results first we create a 4 by 4 matrix like the previous question and add a non-zero row at the bottom of the matrix which is not equal to any row 
#in the matrix.