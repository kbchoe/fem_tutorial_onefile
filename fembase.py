#========================import modules

import numpy as np
#note: scipy.sparse.coo_matrix (may)shall prefer tuple/list, for construction of the matrix.

from math import dist
#dist(p1,p2) , p1=(x,y) or p1=(x,y,z).. euclidian dist.

import matplotlib.pyplot as plt

#========================helper function

def get_directional_cosine(p1,p2):
    "len(p1)==3. returns Cx,Cy,Cz. Cx=dx/L"
    L = dist(p1,p2)
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    Cx = (x2-x1)/L
    Cy = (y2-y1)/L
    Cz = (z2-z1)/L
    return Cx,Cy,Cz

#========================create lsm

def get_lsm_spring(dimension, k,p1,p2):
    esm = k*np.array( (1,-1,-1,1) ).reshape(2,2)  #elemental stiffness matrix
    
    if dimension ==1:
        # T = np.array( (Cx,0, 0,Cx) ).reshape(2,2)
        return esm  # no rotation in 1D space!
    
    Cx,Cy,Cz = get_directional_cosine(p1,p2)
    if dimension ==2:
        T = np.array( [Cx,Cy,0,0, 0,0,Cx,Cy] ).reshape(2,4)
    elif dimension ==3:
        # (6,2)@(2,2)@(2,6) => (6,6) -for global space.. x1y1z1,x2y2z1.
        T = np.array( [Cx,Cy,Cz,0,0,0, 0,0,0,Cx,Cy,Cz] ).reshape(2,6)
    
    #F=kU / TF=kTU / F = T.TkTU , K=T.TkT (when K global, k local.)
    lsm_global = T.T@esm@T  # T is global to local transformation matrix.
    return lsm_global

def get_lsm_frame2d(E,A,I, p1,p2):
    "with the rod,beam."
    L = dist(p1,p2)    
    C1 = A*E/L  # k of axial forced element.
    C2 = E*I/L**3  # for beam.
    # Ux, Uy, th
    raw=[C1,   0,       0,       -C1,     0,        0,
         0,    12*C2,   6*L*C2,    0,     -12*C2,   6*L*C2,
         0,    6*L*C2,  4*L**2*C2,  0,     -6*L*C2,  2*L**2*C2,
         -C1,  0,       0,         C1,    0,        0,
         0,    -12*C2,  -6*L*C2,   0,     12*C2,    -6*L*C2,
         0,    6*L*C2,  2*L**2*C2,  0,     -6*L*C2,  4*L**2*C2,
        ]
    #4.53 of 238p
    esm = np.array(raw).reshape(6,6)
    
    Cx,Cy,Cz = get_directional_cosine(p1,p2)
    C,S = Cx,Cy
    T = np.array([
        C,S,0, 0,0,0,
        -S,C,0, 0,0,0,
        0,0,1, 0,0,0,
        0,0,0, C,S,0,
        0,0,0, -S,C,0,
        0,0,0, 0,0,1,
    ]
    ).reshape(6,6) #global to local

    # raw = [Cx,-Cy,0, Cy,Cx,0, 0,0,1]
    # sub = np.array(raw).reshape(3,3)
    # zero = np.zeros( (3,3) )
    # T = np.block([ [sub,zero],[zero,sub] ]).T  # make it global to local.

    #F=kU / TF=kTU / F = T.TkTU , K=T.TkT (when K global, k local.)
    lsm_global = T.T@esm@T  # T is global to local transformation matrix.
    return lsm_global

def get_lsm_frame3d(E,A,Iy,Iz,G,J, p1,p2):    
    L = dist(p1,p2)  
    L2 = L**2
    L3 = L**3
    
    raw = [A*E/L  ,0          ,0           ,0     , 0          , 0,
           0      ,12*E*Iz/L3 ,0           ,0     , 0          , 6*E*Iz/L2,
           0      ,0          ,12*E*Iy/L3  ,0     ,-6*E*Iy/L2  , 0,
           0      ,0          ,0           ,G*J/L , 0          , 0,
           0      ,0          ,-6*E*Iy/L2  ,0     , 4*E*Iy/L    , 0,
           0      ,6*E*Iz/L2  ,0           ,0     , 0          , 4*E*Iz/L]
    
    sub1 = np.array(raw).reshape(6,6)

    raw = [A*E/L  ,0          ,0           ,0     , 0          , 0,
           0      ,12*E*Iz/L3 ,0           ,0     , 0          , -6*E*Iz/L2,
           0      ,0          ,12*E*Iy/L3  ,0     ,6*E*Iy/L2  , 0,
           0      ,0          ,0           ,G*J/L , 0          , 0,
           0      ,0          ,6*E*Iy/L2  ,0     , 4*E*Iy/L    , 0,
           0      ,-6*E*Iz/L2  ,0           ,0     , 0          , 4*E*Iz/L]
    
    sub4 = np.array(raw).reshape(6,6)


    raw = [-A*E/L  ,0          ,0           ,0     , 0          , 0,
           0      ,-12*E*Iz/L3 ,0           ,0     , 0          , 6*E*Iz/L2,
           0      ,0          ,-12*E*Iy/L3  ,0     ,-6*E*Iy/L2  , 0,
           0      ,0          ,0           ,-G*J/L , 0          , 0,
           0      ,0          ,6*E*Iy/L2  ,0     , 2*E*Iy/L    , 0,
           0      ,-6*E*Iz/L2  ,0           ,0     , 0          , 2*E*Iz/L]
    sub2 = np.array(raw).reshape(6,6)

    raw = [-A*E/L  ,0          ,0           ,0     , 0          , 0,
           0      ,-12*E*Iz/L3 ,0           ,0     , 0          , -6*E*Iz/L2,
           0      ,0          ,-12*E*Iy/L3  ,0     ,6*E*Iy/L2  , 0,
           0      ,0          ,0           ,-G*J/L , 0          , 0,
           0      ,0          ,-6*E*Iy/L2  ,0     , 2*E*Iy/L    , 0,
           0      ,6*E*Iz/L2  ,0           ,0     , 0          , 2*E*Iz/L]
    sub3 = np.array(raw).reshape(6,6)

    esm = np.block([ [sub1,sub2],[sub3,sub4] ])

    Cx,Cy,Cz = get_directional_cosine(p1,p2)
    l,m,n = Cx,Cy,Cz
    D = (l**2+m**2)**0.5
    #singularity.. when perpendicular to z-axis..
    try:
        raw = [l,m,n, -m/D, l/D, 0,   -l*n/D, -m*n/D, D]
    except ZeroDivisionError:  #D is zero
        if Cz>0:  #guess it works..
            raw = [0,0,1, 0,1,0, -1,0,0]  #  z
        else:
            raw = [0,0,-1, 0,1,0, 1,0,0]  # -z


    sub = np.array(raw).reshape(3,3)
    zero = np.zeros( (3,3) )

    T = np.block([
        [sub,zero,zero,zero],
        [zero,sub,zero,zero],
        [zero,zero,sub,zero],
        [zero,zero,zero,sub] ])

    
    #F=kU / TF=kTU / F = T.TkTU , K=T.TkT (when K global, k local.)
    lsm_global = T.T@esm@T  # T is global to local transformation matrix.
    return lsm_global



def get_lsm(edata, node_data):
    "element / node data"
    n1,n2 = edata['nodes']  # 1d element has 2 nodes.
    p1 = node_data[n1]  #access via node_id.
    p2 = node_data[n2]

    etype = edata['etype']

    #line-line elements.
    if etype == '1dSpring':
        k = edata['K']
        lsm = get_lsm_spring(1, k,p1,p2)
    elif etype == '1dRod':
        E = edata['E']
        A = edata['A']
        L = dist(p1,p2)
        k = A*E/L
        lsm = get_lsm_spring(1, k,p1,p2)
    
    #Truss is composite of the Rod. free ball-joint.
    #it can be used for rotated 1d rod, too.
    elif etype == '2dTruss':
        E = edata['E']
        A = edata['A']
        L = dist(p1,p2)
        k = A*E/L
        lsm = get_lsm_spring(2, k,p1,p2)
    elif etype == '3dTruss':
        E = edata['E']
        A = edata['A']
        L = dist(p1,p2)
        k = A*E/L
        lsm = get_lsm_spring(3, k,p1,p2)

    #============================
    elif etype == '2dFrame':
        E = edata['E']
        A = edata['A']
        I = edata['I']
        lsm = get_lsm_frame2d(E,A,I, p1,p2)
    elif etype == '3dFrame':
        E = edata['E']
        A = edata['A']
        Iy = edata['Iy']
        Iz = edata['Iz']
        G = edata['G']
        J = edata['J']
        lsm = get_lsm_frame3d(E,A,Iy,Iz,G,J, p1,p2)

    return lsm


#========================

def example_lsm_spring():
    print('by k')
    p1=(0,0,0)
    p2=(1,0,0)
    for k in range(1,4):
        lsm = get_lsm_spring(1, k,p1,p2)
        print(lsm, 'k :',k)

    print('by dimension')
    p1=(0,0,0)
    p2=(1,0,0)
    k=10
    for dim in range(1,4):
        lsm = get_lsm_spring(dim, k,p1,p2)
        print(lsm, 'dim :',dim)







#======================== gsm

def nid_to_idxs(node_dof, node_id):
    "dof3, 1->(0, 1, 2), 2->(3,4,5)"
    idx = node_id-1
    return tuple(idx*node_dof+i for i in range(node_dof) )


def get_gsm(node_dof, node_data, element_data):
    row_size = len(node_data)*node_dof
    GSM = np.zeros((row_size,row_size))
    
    for edata in element_data:
        lsm = get_lsm(edata,node_data)
        
        nodes = edata['nodes']
        
        idx = []
        for node_id in nodes:
            x = nid_to_idxs(node_dof, node_id)
            idx.extend(x)

        idx2d = np.ix_(idx,idx)  #(1,3)-> (1,1),(1,3),(3,1),(3,3)
        GSM[idx2d] += lsm
    return GSM


def get_force(node_dof, row_size, force_data):
    "is moment load?"
    F_full = np.zeros(row_size)

    for node_id,fxfyfz in force_data.items():
        idxs = nid_to_idxs(node_dof, node_id)
        for i,idx in enumerate(idxs):
            F_full[idx] = fxfyfz[i]
    return F_full


def get_free_idx(node_dof, row_size,fixed_data):
    fixed_idxs = []
    for node_id,fxfyfz in fixed_data.items():
        node_ids = nid_to_idxs(node_dof,node_id)
        for idx,i in enumerate(node_ids):
            if fxfyfz[idx] == 1:
                fixed_idxs.append(i)
    
    free_idx = [i for i in range(row_size) if i not in fixed_idxs]    
    return free_idx






#========================= solve

def solve(node_dof, node_data,element_data, force_data,fixed_data):
    # node_dof = int(element_data[0]['etype'][0]) #3dFrame->3..no!
    
    GSM = get_gsm(node_dof, node_data, element_data)
    row_size = GSM.shape[0]

    F_full = get_force(node_dof, row_size, force_data)    

    idx = get_free_idx(node_dof, row_size, fixed_data)
    idx2d = np.ix_(idx,idx)

    K = GSM[idx2d]  # is sliced.
    F = F_full[idx].T  #.T to verticalize, {}, shape=(N,1)
    U = np.linalg.solve(K, F)
    
    U_full = np.zeros(row_size)
    U_full[idx] = U

    return U_full




#============================

def post_process(node_dof, node_data, element_data,force_data,fixed_data, solved_data, scale=30):
    for k,v in enumerate(solved_data):
        print( f"idx {k}  displacement: {v}")
    
    #dirty code for display nodes xyz
    node_disp = {}
    
    iters = len(solved_data)//node_dof
    for i in range(0,iters):
        # x = [i, i + 1, i + 2]
        idxs = [i*node_dof+j for j in range(node_dof)]
        
        x = [solved_data[idx] for idx in idxs]
        node_disp[i+1] = x
    # print(node_disp, iters,node_dof,iters)
    #=================
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each line in the line_data
    # for x,y,z in zip(xx,yy,zz):
    
    for edata in element_data:
        xx,yy,zz = [],[],[]
        dxx,dyy,dzz = [],[],[]
        nidx=0
        for nid in edata['nodes']:
            x,y,z = node_data[nid]
            xx.append(x)
            yy.append(y)
            zz.append(z)

            dx = x
            dy = y
            dz = z
            for didx, disp in enumerate(node_disp[nid]):
                if didx == 0:
                    dx+=disp*scale
                elif didx == 1:
                    dy+=disp*scale
                elif didx == 1:
                    dz+=disp*scale
            
            dxx.append(dx)
            dyy.append(dy)
            dzz.append(dz)
        ax.plot(xx, yy, zz, marker='o', linewidth=1)
        ax.plot(dxx, dyy, dzz, marker='*',linestyle='--', linewidth=2)

    
    #lim not today.
    # x_min, x_max = np.min(xx), np.max(xx)
    # y_min, y_max = np.min(yy), np.max(yy)
    # z_min, z_max = np.min(zz), np.max(zz)
    # ax.set_xlim(x_min-0.1, x_max+0.1)
    # ax.set_ylim(y_min-0.1, y_max+0.1)
    # ax.set_zlim(z_min-0.1, z_max+0.1)
    # dx=x_max-x_min+1
    # dy=y_max-y_min+1
    # dz=z_max-z_min+1
    # ax.set_box_aspect([1,1,1])
    # Set the limits of the axes to visualize the aspect ratio clearly
    
    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Line Plot')

    # Show plot
    plt.show()

#=========================







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
{ 'E':200E9,'A':250E-6, 'etype':'1dRod', 'nodes':(1,2) },
{ 'E':200E9,'A':250E-6, 'etype':'1dRod', 'nodes':(2,3) },
{ 'E':200E9,'A':400E-6, 'etype':'1dRod', 'nodes':(3,4) },
]
force_data = {
    2: (300E3,0,0),
}
fixed_data = {
1:(1,1,1),
4:(1,1,1),
}

node_dof=1
# U_full = solve(node_dof, node_data, element_data,force_data,fixed_data)
# post_process(node_dof, node_data, element_data,force_data,fixed_data, U_full)




node_data = {
    1:(0    ,0     ,0),
    2:(3.6  ,0     ,0),
    3:(1.8  ,3.118 ,0),
    4:(7.2  ,0     ,0),
    5:(5.4  ,3.118 ,0),
    6:(10.8 ,0     ,0),
    7:(9    ,3.118 ,0), 
}
element_data = [
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(1,3) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(1,2) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(2,3) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(2,5) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(2,4) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(4,5) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(4,7) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(4,6) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(6,7) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(5,7) },
{ 'E':200e+9, 'A':3250e-6, 'etype':'2dTruss', 'nodes':(3,5) },
]

force_data = {
    1: (0,-280e3,0),
    2: (0,-210e3,0),
    4: (0,-280e3,0),
    6: (0,-360e3,0),
}
fixed_data = {
1:(1,1,0),
6:(0,1,0),
}
node_dof = 2
# U_full = solve(node_dof, node_data, element_data,force_data,fixed_data)
# post_process(node_dof, node_data, element_data,force_data,fixed_data, U_full)








node_data = {
    1:(0    ,0     ,0),
    2:(3.6  ,0     ,0),
    3:(1.8  ,3.118 ,0),
    4:(7.2  ,0     ,0),
    5:(5.4  ,3.118 ,0),
    6:(10.8 ,0     ,0),
    7:(9    ,3.118 ,0), 
}
element_data = [
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(1,3) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(1,2) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(2,3) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(2,5) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(2,4) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(4,5) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(4,7) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(4,6) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(6,7) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(5,7) },
{ 'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame', 'nodes':(3,5) },
]
#x,y,theta(moment?). force->load??
force_data = {
    1: (0,-280e3,0),
    2: (0,-210e3,0),#moment 10000000, RH rule, CCW rotating force.
    4: (0,-280e3,0),
    6: (0,-360e3,0),
}
fixed_data = {
1:(1,1,1),
6:(0,1,0),
}
node_dof = 3
# U_full = solve(node_dof, node_data, element_data,force_data,fixed_data)
# post_process(node_dof, node_data, element_data,force_data,fixed_data, U_full)





node_data= {
            1: (0    ,0    ,0),
            2: (0.25 ,0    ,0),
            3: (0    ,0.25 ,0),
            4: (0    ,0    ,0.5),
            5: (0.25 ,0    ,0.5),
            6: (0    ,0.25 ,0.5),
            7: (0    ,0    ,1.0),
            8: (0.25 ,0    ,1.0),
            9: (0    ,0.25 ,1.0),
            }
element_data = [
            {'nodes':(1,2), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(1,3), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(2,3), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(1,4), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(2,5), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(3,6), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(2,4), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(3,4), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(2,6), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(4,5), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(4,6), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(5,6), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(4,7), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(5,8), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(6,9), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(5,7), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(6,7), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(5,9), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(7,8), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(7,9), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            {'nodes':(8,9), 'A':900e-6, 'E':200e+9, 'etype':'3dTruss'},
            ]
#x,y,theta
force_data = {
        8: (0,-5000,0),
    }
#truss requires fixed theta.
fixed_data = {
    1:(1,1,1),
    2:(1,1,1),
    3:(1,1,1),
    }

node_dof = 3
U_full = solve(node_dof, node_data, element_data,force_data,fixed_data)
post_process(node_dof, node_data, element_data,force_data,fixed_data, U_full)



section_data = {
    1: {'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'}, 
}

node_data= {
            1: (0    ,0    ,0),
            2: (0.25 ,0    ,0),
            3: (0    ,0.25 ,0),
            4: (0    ,0    ,0.5),
            5: (0.25 ,0    ,0.5),
            6: (0    ,0.25 ,0.5),
            7: (0    ,0    ,1.0),
            8: (0.25 ,0    ,1.0),
            9: (0    ,0.25 ,1.0),
            }
element_data = [
            {'nodes':(1,2), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(1,3), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(2,3), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(1,4), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(2,5), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(3,6), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(2,4), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(3,4), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(2,6), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(4,5), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(4,6), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(5,6), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(4,7), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(5,8), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(6,9), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(5,7), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(6,7), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(5,9), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(7,8), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(7,9), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            {'nodes':(8,9), 'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'},
            ]
#x,y,z, phi for x,y,z
force_data = {
        8: (0,-500000,0,0,0,0),#-500000 to see displacement.
    }
#truss requires fixed theta.. while frame dosen't
fixed_data = {
    1:(1,1,1, 0,0,0),
    2:(1,1,1, 0,0,0),
    3:(1,1,1, 0,0,0),
    }
# fixed_data = {
#     1:(1,1,1, 1,1,1),
#     2:(1,1,1, 1,1,1),
#     3:(1,1,1, 1,1,1),
#     }
node_dof = 6
U_full = solve(node_dof, node_data, element_data,force_data,fixed_data)
post_process(node_dof, node_data, element_data,force_data,fixed_data, U_full)
 
