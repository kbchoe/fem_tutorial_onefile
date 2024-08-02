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

def lsm_2dRod(E,A, p1,p2):
    L = dist(p1,p2) 
    k = A*E/L
    #=============
    # T = global->local disp transform matrix.
    # C,S, 0,0 {u1}
    #-S,C, 0,0 {v1}
    # 0,0, C,S {u2}
    # 0,0,-S,C {v2}
    
    # K = local stiffness matrix
    # k,0,-k,0
    # 0,0, 0,0
    #-k,0, k,S
    # 0,0, 0,C
    
    #and K_global = T.T@K@T

    # theta = 0
    # C = cos(theta)
    # S = sin(theta)
    x1,y1,_ = p1
    x2,y2,_ = p2
    C = (x2-x1)/L
    S = (y2-y1)/L
    C2 = C**2
    S2 = S**2
    CS = C*S

    raw=(
         C2,  CS, -C2, -CS,
        
         CS,  S2, -CS, -S2,
        
        -C2, -CS,  C2,  CS,
        
        -CS, -S2,  CS,  S2,
    )
    return k*np.array(raw).reshape(4,4)






#3. create gsm
#=============================

node_dof = 2
row_size = len(node_data)*node_dof


GSM = np.zeros((row_size,row_size))
for edata in element_data:
    E = edata['E']
    A = edata['A']  
    
    n1,n2 = edata['nodes']
    p1 = node_data[n1]
    p2 = node_data[n2]
    
    lsm = lsm_2dRod(E,A, p1,p2)
    
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

#0,1, 2,3, 4,5, 6,7
#node 2
F_full[2] = 300E3  # =300k.

#node 1,4
fixed_nodes = [0,6, 1,3,5,7]
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

# node 1 displacement: 0.0
# node 2 displacement: 0.0
# node 3 displacement: 0.000623076923076923
# node 4 displacement: 0.0
# node 5 displacement: 0.0003461538461538462
# node 6 displacement: 0.0
# node 7 displacement: 0.0
# node 8 displacement: 0.0












#==============appendix

#to make dist
# def dist(p1,p2):
#   # x1,y1,z1 = 0,0,0
#   # x2,y2,z2 = 0,0,0
#   # x1,y1,z1 = p1
#   # x2,y2,z2 = p2
#   # return ( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )**0.5
#   return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))


#more generalized data

# force_data = {2:300E3}
# for node in force_data:
#   F_full[node-1] = force_data[node]

# fixed_data = {1:(1,0,0),4:(1,0,0)}
# def get_free_idx(fixed_data):
#   free_idx = []
#   for node,disp in fixed_data.items():
#       x = [ (node-1)*node_dof+i for i in range(node_dof) if disp[i]==0 ]
#       free_idx.extend(x)
#   return free_idx

