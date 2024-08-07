import numpy as np

# #->python code , but commented.
# all other str is text body.

#  "# title" "## subject" ### ..don'now yet.


"""# world of sparse matrix, and numpy array.

author:kiri
2024,08,05.

"""


"## 1. cartesian product"

# list1 = [0, 1]
# list2 = [0, 1]

# "nested loops"
# cartesian_product = []
# for a in list1:
#     for b in list2:
#         cartesian_product.append((a, b))

# "listcomp"
# cartesian_product = [(a, b) for a in list1 for b in list2]
# print(cartesian_product)

# import itertools
# "itertools.product"
# cartesian_product = list(itertools.product(list1, list2))
# print(cartesian_product)


# list1 = [i for i in range(5)]
# list2 = [i for i in range(5)]
# cartesian_product = [(a, b) for a in list1 for b in list2 if a-b<=0]
# print(cartesian_product)


# list1 = [i for i in range(1,5)]
# list2 = [i for i in range(1,5)]
# cartesian_product = [(a, b) for a in list1 for b in list2 if a-b<=0]
# print(cartesian_product)





def cartesian_product(list1,list2):
	return [(a, b) for a in list1 for b in list2]

def get_upper_triangular_idxs(list1,list2):
	"UTM upper triangular matrix"
	return [(a, b) for a in list1 for b in list2 if a-b<=0]




"## 2. creating id_ for ..maybe sparse matrix"

# # traditional way to assign matrix to the idxs.
# a = np.zeros( (5,5) )
# b = np.arange(4).reshape(2,2)
# idx = (1,3)
# idx2d = np.ix_(idx,idx)
# a[idx2d] = b
# print(f"insert 2d array b:\n{b}\n to the array a. result:\n\n{a}")


# a = np.zeros( (5,5) )
# b = np.arange(4).reshape(2,2)
# idx = (1,3)
# # b = b.flatten()  # since it's small array, creating new array acceptable.
# b = b.ravel()  #creates view
# for idx,idx2 in enumerate([(a, b) for a in idx for b in idx]):
# 	x,y = idx2
# 	a[x][y] = b[idx]
# print(f"insert 2d array b:\n{b}\n to the array a. result:\n\n{a}")



# def add_2d(a,b,idx):
# 	b = b.ravel()
# 	for idx,idx2 in enumerate([(a, b) for a in idx for b in idx]):
# 		x,y = idx2
# 		a[x][y] += b[idx]

# a = np.zeros( (5,5) )
# b = np.arange(4).reshape(2,2)
# idx = (1,3)
# add_2d(a,b,idx)
# print(f"insert 2d array b:\n{b}\n to the array a. result:\n\n{a}")


def add_2d(a,b,idx):
	"add b to a, with idx. (1,3)->(1,1),(1,3),(3,1),(3,3)"
	b = b.ravel()  # make view of flatten.
	for idx,idx2 in enumerate( cartesian_product(idx,idx) ):
		x,y = idx2
		a[x][y] += b[idx]


"## 3. sparse matrix.."

from scipy import sparse


# a = np.arange(12).reshape(3,4)
# a[0]=0
# a[:,1]=0
# print(a)

# sparse_matrix = sparse.csr_matrix(a)
# print(sparse_matrix)

# [[ 0  0  0  0]
#  [ 4  0  6  7]
#  [ 8  0 10 11]]
#   (1, 0)	4
#   (1, 2)	6
#   (1, 3)	7
#   (2, 0)	8
#   (2, 2)	10
#   (2, 3)	11

# print(sparse_matrix[1][0],'ha')
#   (0, 0)	4
#   (0, 2)	6
#   (0, 3)	7 ha
# for i in range(4):
# 	sparse_matrix[0,i] = i*2
# Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.

# print(sparse_matrix.todense())



from scipy.sparse import lil_matrix
#nested list..
# Note that inserting a single item can take linear time in the worst case;
# a = lil_matrix((5, 5), dtype=float)
# print('\n'*20)
# a[0,3]=1
# print(a)


from scipy.sparse import dok_matrix
#dict of keys, fine but not use it.









# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
from scipy.sparse import coo_matrix

# does not directly support:
# arithmetic operations
# slicing

# By default when converting to CSR or CSC format,
# duplicate (i,j) entries will be summed together.

# This facilitates efficient construction of
# finite element matrices and the like. (see example)


sparse_matrix = sparse.coo_matrix( (5,5) ,dtype=float)
print(sparse_matrix)

# sparse_matrix[0]=4
# TypeError: 'coo_matrix' object does not support item assignment

#COO is from data,row,col,   shape.  it stores data, very-eff.
#but no item assign.
#but so-fast to the CSR/SCS.

# By default when converting to CSR or CSC format, duplicate (i,j) entries will be summed
#..

# Constructing a matrix with duplicate coordinates
row  = np.array([0, 0, 1, 3, 1, 0, 0])
col  = np.array([0, 2, 1, 3, 1, 0, 0])
data = np.array([1, 1, 1, 1, 1, 1, 1])
coo = coo_matrix((data, (row, col)), shape=(4, 4))
# Duplicate coordinates are maintained until implicitly or explicitly summed
np.max(coo.data) #=1
# print(coo)
  # (0, 0)	1
  # (0, 2)	1
  # (1, 1)	1
  # (3, 3)	1
  # (1, 1)	1
  # (0, 0)	1
  # (0, 0)	1
print(coo.toarray())

#i like it! the way it treats Duplicate coordinates..







# CSR (Compressed Sparse Row) CSC for col.
# indptr : index for each row. (since entire row can be empty)
# csr = csr_matrix((data, indices, indptr), shape=(2, 3))

# Less Efficient for Item Assignment:

# Direct Updates: CSR format does not support efficient
#  direct updates (i.e., modifying individual elements)
#   without a potential performance hit. 
#   This is because the format is optimized for read operations rather than updates.





#dok, so slow matrix multiplication kinds.
# but very eff. item assign.. so incremental creating sparse matrix. ..fast for coo.


#and solver, https://newtonexcelbach.com/2021/10/17/making-finite-element-analysis-go-faster-update-and-pypardiso/



#====
#numpy matrix, was dense matrix. full-data.
#https://docs.nvidia.com/cuda/cusparse/



#https://stackoverflow.com/questions/40454543/symmetrization-of-scipy-sparse-matrices
# 98]: data,rows,cols=[],[],[]
# In [399]: for i in range(10):
#      ...:     for j in range(i,10):
#      ...:         v=np.random.randint(0,10)
#      ...:         if v>5:
#      ...:             if i==j:
#      ...:                 # prevent diagonal duplication
#      ...:                 data.append(v)
#      ...:                 rows.append(i)
#      ...:                 cols.append(j)
#      ...:             else:
#      ...:                 data.extend((v,v))
#      ...:                 rows.extend((i,j))
#      ...:                 cols.extend((j,i))
#      ...:                 
# In [400]: sparse.coo_matrix((data,(rows,cols)),shape=(10,10)).A

#fine, i,j, j,i is not that costly.

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html
#bytes? forget it along with np.writetxt



#https://sparse.tamu.edu/
#some, lots,of,sparse matricies.
#https://sparse.tamu.edu/HB/bcsstk25
#svd appears. why?

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html
#triangular solve.
#i think, coo, with full mat, is ..fine. store UTM, upper tri.mat., if wanted.



#https://caam37830.github.io/book/02_linear_algebra/blas_lapack.html
#blas and lapack. funny but keep this somewhere.

#https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
#some math.


#LU matrix..
# https://velog.io/@claude_ssim/%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98-LU-Factorization
# https://blog.naver.com/cj3024/221124535258
