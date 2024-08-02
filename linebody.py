import numpy as np
from math import dist



#1. node/element data
#=============================================

#units N, m, m^2, N/m^2(pressure)

#150E-3= 0.00150 , 150mm=0.0015m
node_data = {
	1:(0     ,0,0),
	2:(150E-3,0,0),
	3:(300E-3,0,0),
	4:(600E-3,0,0),
}

#250E-6 = 0.00 000 250/0.0000025,, 0 appears 6 times.
#1mm*1mm = 0.001m*0.001m = 1E-3m*1E-3m = 1E-6m^2
#200G pa = 200E9 pa
element_data = [
{ 'E':200E9,'A':250E-6, 'nodes':(1,2) },
{ 'E':200E9,'A':250E-6, 'nodes':(2,3) },
{ 'E':200E9,'A':400E-6, 'nodes':(3,4) },
]












#2. LSM functions
#==========================================

def lsm_1dSpring(K):
	return k*np.array( (1,-1,-1,1) ).reshape(2,2)

def lsm_1dRod(E,A, p1,p2):
	L = dist(p1,p2)
	k = A*E/L
	return k*np.array( (1,-1,-1,1) ).reshape(2,2)






#3. create gsm
#=============================

node_dof = 1
row_size = len(node_data)*node_dof


GSM = np.zeros((row_size,row_size))
for edata in element_data:
	E = edata['E']
	A = edata['A']	
	
	n1,n2 = edata['nodes']
	p1 = node_data[n1]
	p2 = node_data[n2]
	
	lsm = lsm_1dRod(E,A, p1,p2)
	
	nodes = edata['nodes']

	idx = []
	for node in nodes:
		#get indices by node_dof. dof3, node1=> 0,1,2.
		x = [(node-1)*node_dof+i for i in range(node_dof)]
		idx.extend(x)

	idx2d = np.ix_(idx,idx)  #(1,3)-> (1,1),(1,3),(3,1),(3,3)		
	GSM[idx2d] += lsm













#4. solve.
#==========================
F_full = np.zeros(row_size)
U_full = np.zeros(row_size)

#node 2
F_full[1] = 300E3  # =300k.

#node 1,4
fixed_nodes = [0,3]
idx = [i for i in range(row_size) if i not in fixed_nodes]
idx2d = np.ix_(idx,idx)


K = GSM[idx2d]  # is sliced.
F = F_full[idx].T  #.T to verticalize, {}, shape=(N,1)
U = np.linalg.solve(K, F)
U_full[idx] = U











#5.post-process
#=====================
for idx,i in enumerate(U_full):
	print( f"node {idx+1} displacement: {i}")

# node 2 displacement: [0.00062308]
# node 3 displacement: [0.00034615]













#==============appendix

#to make dist
# def dist(p1,p2):
# 	# x1,y1,z1 = 0,0,0
# 	# x2,y2,z2 = 0,0,0
# 	# x1,y1,z1 = p1
# 	# x2,y2,z2 = p2
# 	# return ( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )**0.5
# 	return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))


#more generalized data

# force_data = {2:300E3}
# for node in force_data:
# 	F_full[node-1] = force_data[node]

# fixed_data = {1:(1,0,0),4:(1,0,0)}
# def get_free_idx(fixed_data):
# 	free_idx = []
# 	for node,disp in fixed_data.items():
# 		x = [ (node-1)*node_dof+i for i in range(node_dof) if disp[i]==0 ]
# 		free_idx.extend(x)
# 	return free_idx

