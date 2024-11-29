#========================import modules

import numpy as np
#note: scipy.sparse.coo_matrix (may)shall prefer tuple/list, for construction of the matrix.

from math import dist
#dist(p1,p2) , p1=(x,y) or p1=(x,y,z).. euclidian dist.

import matplotlib.pyplot as plt

#pretty print
from pprint import pprint









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


def get_dof_data(etype):
    if etype in ('1dSpring','1dRod'):        
        dof_data = ('x')
    elif etype == '2dTruss':        
        dof_data = ('x','y')
    elif etype == '3dTruss':        
        dof_data = ('x','y','z')    
    elif etype == '2dFrame':#xy phi        
        dof_data = ('x','y','rz')
    elif etype == '3dFrame':#xyz phi xyz        
        dof_data = ('x','y','z', 'rx','ry','rz')
    elif etype == '2dQ4':
        dof_data = ('x','y')
    return dof_data


def salt_U(U_full,etype):
    dof_data = get_dof_data(etype)
    node_dof = len(dof_data)
    U_salted = {}

    for idx,U in enumerate(U_full.reshape(-1,node_dof)):            
        U_salted[idx+1] = {i:U[idx] for idx,i in enumerate(dof_data)}
    return U_salted


def plot_elements(post_data, title='3D plot', scale=30,true_ratio=False):
    XYZ = post_data['XYZ']
    DXYZ = post_data['DXYZ']
    TX,TY,TZ = [],[],[] #total
    for X,Y,Z in XYZ:
        TX.extend(X)
        TY.extend(Y)
        TZ.extend(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, [X,Y,Z] in enumerate(XYZ):
        ax.plot(X, Y, Z, marker='o', linewidth=1)
        
        DX,DY,DZ = DXYZ[idx]
        DX = np.array(X)+np.array(DX)*scale
        DY = np.array(Y)+np.array(DY)*scale
        DZ = np.array(Z)+np.array(DZ)*scale
        ax.plot(DX, DY, DZ, marker='*',linestyle='--', linewidth=2)

    
    #ratio ignored when <1.0 ,not bad.
    if true_ratio:
        dx = abs(max(TX)-min(TX))
        dy = abs(max(TY)-min(TY))
        dz = abs(max(TZ)-min(TZ))

        div = min([dx,dy,dz])
        if div == 0:
            div=1

        sx = max(1, dx/div)
        sy = max(1, dy/div)
        sz = max(1, dz/div)
        ax.set_box_aspect([sx,sy,sz])
    # Set the limits of the axes to visualize the aspect ratio clearly
    
    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)

    # Show plot
    plt.show()








#========================base lsm functions

def get_lsm_spring(dimension, k,p1,p2):
    #elemental stiffness matrix for the rod. 1,-1/-1+1 for u1-u2/u2-u1
    esm = np.array( (k,-k,-k,k) ).reshape(2,2)
    
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
            raw = (0,0,1, 0,1,0, -1,0,0)  #  z
        else:
            raw = (0,0,-1, 0,1,0, 1,0,0)  # -z


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



def get_lsm_q4(E, nu, t, coords):
    """
    Computes the stiffness matrix for a Q4 element.

    Parameters:
    E (float): Young's modulus.
    nu (float): Poisson's ratio. greek v 'new',
    t (float): Thickness of the element.
    coords (np.array): Nodal coordinates array of shape (4, 2).

    Returns:
    K (np.array): Stiffness matrix of shape (8, 8).
    """
    # Elasticity matrix for plane stress
    C = E / (1 - nu ** 2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    coords = np.array(coords)[:,:2]#convinient list to -1,2 shaped.

    # Gauss points and weights for 2x2 quadrature
    gp = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)
    weights = np.array([1, 1, 1, 1])

    K = np.zeros((8, 8))

    for i in range(len(gp)):
        xi, eta = gp[i]
        weight = weights[i]

        # Shape functions and derivatives in natural coordinates
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])

        dN_dxi = 0.25 * np.array([
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)]
        ])

        # Jacobian matrix
        J = dN_dxi.T @ coords
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        # Derivatives of shape functions in physical coordinates
        dN_dx = invJ @ dN_dxi.T

        # Strain-displacement matrix B
        B = np.zeros((3, 8))
        for j in range(4):
            B[0, j*2] = dN_dx[0, j]
            B[1, j*2+1] = dN_dx[1, j]
            B[2, j*2] = dN_dx[1, j]
            B[2, j*2+1] = dN_dx[0, j]

        # Element stiffness matrix
        K += B.T @ C @ B * detJ * weight * t

    return K







#========================create lsm

def get_lsm(edata, node_coords):
    "element / node data"
    etype = edata['etype']

    #line-line elements.
    if etype == '1dSpring':
        p1,p2 = node_coords
        k = edata['K']
        lsm = get_lsm_spring(1, k,p1,p2)
    elif etype == '1dRod':
        p1,p2 = node_coords
        E = edata['E']
        A = edata['A']
        L = dist(p1,p2)
        k = A*E/L
        lsm = get_lsm_spring(1, k,p1,p2)
    
    #Truss is composite of the Rod. free ball-joint.
    #it can be used for rotated 1d rod, too.
    elif etype == '2dTruss':
        p1,p2 = node_coords
        E = edata['E']
        A = edata['A']
        L = dist(p1,p2)
        k = A*E/L
        lsm = get_lsm_spring(2, k,p1,p2)
    elif etype == '3dTruss':
        p1,p2 = node_coords
        E = edata['E']
        A = edata['A']
        L = dist(p1,p2)
        k = A*E/L
        lsm = get_lsm_spring(3, k,p1,p2)

    #============================
    elif etype == '2dFrame':
        p1,p2 = node_coords
        E = edata['E']
        A = edata['A']
        I = edata['I']
        lsm = get_lsm_frame2d(E,A,I, p1,p2)
    elif etype == '3dFrame':
        p1,p2 = node_coords
        E = edata['E']
        A = edata['A']
        Iy = edata['Iy']
        Iz = edata['Iz']
        G = edata['G']
        J = edata['J']
        lsm = get_lsm_frame3d(E,A,Iy,Iz,G,J, p1,p2)
    
    elif etype == '2dQ4':
        E = edata['E']
        nu = edata['nu']
        t = edata['t']
        lsm = get_lsm_q4(E,nu,t,node_coords)

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
        plt.spy(lsm)
        plt.show()









#======================== gsm

def nid_to_idxs(node_dof, node_id):
    "dof3, 1->(0, 1, 2), 2->(3,4,5)"
    idx = node_id-1
    return tuple(idx*node_dof+i for i in range(node_dof) )


def get_gsm(node_dof, node_data, element_data,elements):
    row_size = len(node_data)*node_dof
    GSM = np.zeros((row_size,row_size))
    
    for element in elements:
        eid = element['edata']
        edata = element_data[eid]
        
        nodes = element['nodes']
        node_coords = [ node_data[node_id] for node_id in nodes]
        lsm = get_lsm(edata,node_coords)
        
        idx = []
        for node_id in nodes:
            x = nid_to_idxs(node_dof, node_id)
            idx.extend(x)

        idx2d = np.ix_(idx,idx)  #(1,3)-> (1,1),(1,3),(3,1),(3,3)
        GSM[idx2d] += lsm
    return GSM


def get_load(node_dof, row_size, load_data):
    "is moment load?"
    F_full = np.zeros(row_size)
    for node_id,fxfyfz in load_data.items():
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

def solve(mesh_data, load_data,fixed_data):
    node_data = mesh_data['node_data']
    element_data = mesh_data['element_data']
    elements = mesh_data['elements']
    
    # node_dof = int(element_data[0]['etype'][0]) #3dFrame->3..no!
    etype = element_data[elements[0]['edata']]['etype']
    dof_data = get_dof_data(etype)
    node_dof = len(dof_data)

    
    GSM = get_gsm(node_dof, node_data, element_data,elements)
    # print(GSM.shape,node_dof,len(node_data))
    row_size = GSM.shape[0]

    F_full = get_load(node_dof, row_size, load_data)    

    idx = get_free_idx(node_dof, row_size, fixed_data)
    idx2d = np.ix_(idx,idx)

    K = GSM[idx2d]  # is sliced.
    F = F_full[idx].T  #.T to verticalize, {}, shape=(N,1)
    U = np.linalg.solve(K, F)
    
    U_full = np.zeros(row_size)
    U_full[idx] = U

    return U_full


#============================ post process











def post_process(mesh_data, load_data,fixed_data, solved_data):
    node_data = mesh_data['node_data']
    element_data = mesh_data['element_data']
    elements = mesh_data['elements']
    
    for k,v in enumerate(solved_data):
        print( f"idx {k}  displacement: {v}")
    
    etype = element_data[elements[0]['edata']]['etype']
    salted_data = salt_U(solved_data,etype)
    # pprint(salted_data)

    #=================
    XYZ = []
    DXYZ = []

    # TX,TY,TZ = [],[],[] #total
    for element in elements:
        X,Y,Z = [],[],[]
        DX,DY,DZ = [],[],[]

        coords = [node_data[nid] for nid in element['nodes']]
        for nid in element['nodes']:
            x,y,z = node_data[nid]
            X.append(x)
            Y.append(y)
            Z.append(z)
            DX.append(salted_data[nid].get('x',0))
            DY.append(salted_data[nid].get('y',0))
            DZ.append(salted_data[nid].get('z',0))

        XYZ.append([X,Y,Z])
        DXYZ.append([DX,DY,DZ])
    
    post_data = {
    'XYZ':XYZ,
    'DXYZ':DXYZ
    }
    return post_data



























#=========================


if __name__ == '__main__':    




    #units N, m, m^2, N/m^2(pressure)
    #150E-3= 0.00150 , 150mm=0.0015m
    node_data = {
        1:(0     ,0,0),
        2:(150E-3,0,0),
        3:(300E-3,0,0),
        4:(600E-3,0,0),
    }

    element_data = {
        1:{'E':200E9,'A':250E-6, 'etype':'1dRod'},
    }

    #250E-6 = 0.00 000 250/0.0000025,, 0 appears 6 times.
    #1mm*1mm = 0.001m*0.001m = 1E-3m*1E-3m = 1E-6m^2
    #200G pa = 200E9 pa
    elements = [
    { 'edata':1,'nodes':(1,2) },
    { 'edata':1,'nodes':(2,3) },
    { 'edata':1,'nodes':(3,4) },
    ]

    mesh_data = {
        'node_data':node_data,
        'element_data':element_data,
        'elements':elements,
    }

    load_data = {
        2: (300E3,0,0),
    }
    fixed_data = {
    1:(1,1,1),
    4:(1,1,1),
    }

    solve_data = solve( mesh_data,load_data,fixed_data)
    post_data = post_process( mesh_data,load_data,fixed_data, solve_data)
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}')









    node_data = {
        1:(0    ,0     ,0),
        2:(3.6  ,0     ,0),
        3:(1.8  ,3.118 ,0),
        4:(7.2  ,0     ,0),
        5:(5.4  ,3.118 ,0),
        6:(10.8 ,0     ,0),
        7:(9    ,3.118 ,0), 
    }
    element_data = {
        1:{'E':200e+9, 'A':3250e-6, 'etype':'2dTruss'},
    }

    elements = [
    { 'edata':1, 'nodes':(1,3) },
    { 'edata':1, 'nodes':(1,2) },
    { 'edata':1, 'nodes':(2,3) },
    { 'edata':1, 'nodes':(2,5) },
    { 'edata':1, 'nodes':(2,4) },
    { 'edata':1, 'nodes':(4,5) },
    { 'edata':1, 'nodes':(4,7) },
    { 'edata':1, 'nodes':(4,6) },
    { 'edata':1, 'nodes':(6,7) },
    { 'edata':1, 'nodes':(5,7) },
    { 'edata':1, 'nodes':(3,5) },
    ]
    mesh_data = {
        'node_data':node_data,
        'element_data':element_data,
        'elements':elements,
    }

    load_data = {
        1: (0,-280e3,0),
        2: (0,-210e3,0),
        4: (0,-280e3,0),
        6: (0,-360e3,0),
    }
    fixed_data = {
    1:(1,1,0),
    6:(0,1,0),
    }
    solve_data = solve( mesh_data,load_data,fixed_data)
    post_data = post_process( mesh_data,load_data,fixed_data, solve_data)
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}')










    node_data = {
        1:(0    ,0     ,0),
        2:(3.6  ,0     ,0),
        3:(1.8  ,3.118 ,0),
        4:(7.2  ,0     ,0),
        5:(5.4  ,3.118 ,0),
        6:(10.8 ,0     ,0),
        7:(9    ,3.118 ,0), 
    }
    element_data = {
        1:{'E':200e+9, 'A':3250e-6, 'I':2e-4, 'etype':'2dFrame'}
    }
    elements = [
    { 'edata':1, 'nodes':(1,3) },
    { 'edata':1, 'nodes':(1,2) },
    { 'edata':1, 'nodes':(2,3) },
    { 'edata':1, 'nodes':(2,5) },
    { 'edata':1, 'nodes':(2,4) },
    { 'edata':1, 'nodes':(4,5) },
    { 'edata':1, 'nodes':(4,7) },
    { 'edata':1, 'nodes':(4,6) },
    { 'edata':1, 'nodes':(6,7) },
    { 'edata':1, 'nodes':(5,7) },
    { 'edata':1, 'nodes':(3,5) },
    ]
    mesh_data = {
        'node_data':node_data,
        'element_data':element_data,
        'elements':elements,
    }
    #x,y,theta(moment?). force->load??
    load_data = {
        1: (0,-280e3,0),
        2: (0,-210e3,0),#moment 10000000, RH rule, CCW rotating force.
        4: (0,-280e3,0),
        6: (0,-360e3,0),
    }
    fixed_data = {
    1:(1,1,1),
    6:(0,1,0),
    }
    solve_data = solve( mesh_data,load_data,fixed_data)
    post_data = post_process( mesh_data,load_data,fixed_data, solve_data)
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}')
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}',true_ratio=True)









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
    element_data = {
        1:{'A':900e-6, 'E':200e+9, 'etype':'3dTruss'}
    }
    elements = [
    {'nodes':(1,2), 'edata':1},
    {'nodes':(1,3), 'edata':1},
    {'nodes':(2,3), 'edata':1},
    {'nodes':(1,4), 'edata':1},
    {'nodes':(2,5), 'edata':1},
    {'nodes':(3,6), 'edata':1},
    {'nodes':(2,4), 'edata':1},
    {'nodes':(3,4), 'edata':1},
    {'nodes':(2,6), 'edata':1},
    {'nodes':(4,5), 'edata':1},
    {'nodes':(4,6), 'edata':1},
    {'nodes':(5,6), 'edata':1},
    {'nodes':(4,7), 'edata':1},
    {'nodes':(5,8), 'edata':1},
    {'nodes':(6,9), 'edata':1},
    {'nodes':(5,7), 'edata':1},
    {'nodes':(6,7), 'edata':1},
    {'nodes':(5,9), 'edata':1},
    {'nodes':(7,8), 'edata':1},
    {'nodes':(7,9), 'edata':1},
    {'nodes':(8,9), 'edata':1},
    ]

    mesh_data = {
        'node_data':node_data,
        'element_data':element_data,
        'elements':elements,
    }
    #x,y,theta
    load_data = {
            8: (0,-5000,0),
        }
    fixed_data = {
        1:(1,1,1),
        2:(1,1,1),
        3:(1,1,1),
        }

    solve_data = solve( mesh_data,load_data,fixed_data)
    post_data = post_process( mesh_data,load_data,fixed_data, solve_data)
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}')
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}',true_ratio=True)










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
    element_data = {
        1:{'A':900e-6, 'E':200e+9, 'Iy':2e-4, 'Iz':2e-4, 'G':80e+9, 'J':0, 'etype':'3dFrame'}
    }
    elements = [
        {'nodes':(1,2), 'edata':1},
        {'nodes':(1,3), 'edata':1},
        {'nodes':(2,3), 'edata':1},
        {'nodes':(1,4), 'edata':1},
        {'nodes':(2,5), 'edata':1},
        {'nodes':(3,6), 'edata':1},
        {'nodes':(2,4), 'edata':1},
        {'nodes':(3,4), 'edata':1},
        {'nodes':(2,6), 'edata':1},
        {'nodes':(4,5), 'edata':1},
        {'nodes':(4,6), 'edata':1},
        {'nodes':(5,6), 'edata':1},
        {'nodes':(4,7), 'edata':1},
        {'nodes':(5,8), 'edata':1},
        {'nodes':(6,9), 'edata':1},
        {'nodes':(5,7), 'edata':1},
        {'nodes':(6,7), 'edata':1},
        {'nodes':(5,9), 'edata':1},
        {'nodes':(7,8), 'edata':1},
        {'nodes':(7,9), 'edata':1},
        {'nodes':(8,9), 'edata':1},
    ]
    mesh_data = {
        'node_data':node_data,
        'element_data':element_data,
        'elements':elements,
    }

    #x,y,z, phi for x,y,z
    load_data = {
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
    solve_data = solve( mesh_data,load_data,fixed_data)
    post_data = post_process( mesh_data,load_data,fixed_data, solve_data)
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}')
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}',scale=50,true_ratio=True)








    node_data = {
        1:(3,2,0),
        2:(5,2,0),
        3:(5,4,0),
        4:(3,4,0),
        5:(7,2,0),
        6:(7,4,0),    
    }

    element_data = {
        1:{'E':30e6,'nu':0.25,'t':1, 'etype':'2dQ4'},
    }
    elements = [
    { 'edata':1,'nodes':(1,2,3,4) },
    { 'edata':1,'nodes':(2,5,6,3) },
    ]

    mesh_data = {
        'node_data':node_data,
        'element_data':element_data,
        'elements':elements,
    }

    load_data = {
        5: (0,10000,0),
    }
    fixed_data = {
    1:(1,1,1),
    4:(1,1,1),
    }

    solve_data = solve( mesh_data,load_data,fixed_data)
    post_data = post_process( mesh_data,load_data,fixed_data, solve_data)
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}')









    node_data = {
            1 : (   2.665846445E-02,    -2.429309196E-02     ,0.000000000E+00),
            2 : (   1.910056381E-02,    -2.781068473E-02     ,0.000000000E+00),
            3 : (   2.757787033E-02,    -3.285615412E-02     ,0.000000000E+00),
            4 : (   1.892646128E-02,    -3.412359215E-02     ,0.000000000E+00),
            5 : (   3.685549030E-02,    -3.162841396E-02     ,0.000000000E+00),
            6 : (   3.625732252E-02,    -2.204547664E-02     ,0.000000000E+00),
            7 : (   4.605848486E-02,    -2.091198696E-02     ,0.000000000E+00),
            8 : (   4.638304947E-02,    -3.083040555E-02     ,0.000000000E+00),
            9 : (   5.576492586E-02,    -2.036832789E-02     ,0.000000000E+00),
           10: (    5.595039144E-02,    -3.038448976E-02     ,0.000000000E+00),
           11: (    6.548125261E-02,    -3.012011515E-02     ,0.000000000E+00),
           12: (    6.539707048E-02,    -2.011817555E-02     ,0.000000000E+00),
           13: (    5.617867522E-02,    -4.020133572E-02     ,0.000000000E+00),
           14: (    6.562915990E-02,    -4.007241204E-02     ,0.000000000E+00),
           15: (    4.670551188E-02,    -4.042668645E-02     ,0.000000000E+00),
           16: (    3.726522046E-02,    -4.080290924E-02     ,0.000000000E+00),
           17: (    2.794232309E-02,    -4.132705140E-02     ,0.000000000E+00),
           18: (    1.880731545E-02,    -4.180840832E-02     ,0.000000000E+00),
           19: (    1.011865872E-02,    -3.341606272E-02     ,0.000000000E+00),
           20: (    9.596725580E-03,    -4.182446076E-02     ,0.000000000E+00),
           21: (    1.194217511E-02,    -2.390264269E-02     ,0.000000000E+00),
           22: (    2.402667543E-02,    -1.381114917E-02     ,0.000000000E+00),
           23: (    3.576028442E-02,    -1.148416783E-02     ,0.000000000E+00),
           24: (    4.595863570E-02,    -1.059936790E-02     ,0.000000000E+00),
           25: (    5.573528261E-02,    -1.021622944E-02     ,0.000000000E+00),
           26: (    6.538354656E-02,    -1.006023238E-02     ,0.000000000E+00),
           27: (    7.500000000E-02,    -2.000000030E-02     ,0.000000000E+00),
           28: (    7.500000000E-02,    -1.000000015E-02     ,0.000000000E+00),
           29: (    7.500000000E-02,    -3.000000045E-02     ,0.000000000E+00),
           30: (    7.500000000E-02,    -4.000000060E-02     ,0.000000000E+00),
           31: (    1.604662371E-02,    -9.206294994E-03     ,0.000000000E+00),
           32: (    9.341717358E-03,    -1.596816573E-02     ,0.000000000E+00),
           33: (    6.587775262E-02,    -5.000000000E-02     ,0.000000000E+00),
           34: (    7.500000298E-02,    -5.000000075E-02     ,0.000000000E+00),
           35: (    5.646667724E-02,    -5.000000000E-02     ,0.000000000E+00),
           36: (    4.705552687E-02,    -5.000000000E-02     ,0.000000000E+00),
           37: (    3.764445150E-02,    -5.000000000E-02     ,0.000000000E+00),
           38: (    2.823330112E-02,    -5.000000000E-02     ,0.000000000E+00),
           39: (    1.882222575E-02,    -5.000000000E-02     ,0.000000000E+00),
           40: (    9.411075374E-03,    -5.000000000E-02     ,0.000000000E+00),
           41:(     2.734326767E-02,     0.000000000E+00     ,0.000000000E+00),
           42:(     1.850000024E-02,     0.000000000E+00     ,0.000000000E+00),
           43:(     3.680747013E-02,     0.000000000E+00     ,0.000000000E+00),
           44:(     4.635557509E-02,     0.000000000E+00     ,0.000000000E+00),
           45:(     5.590373655E-02,     0.000000000E+00     ,0.000000000E+00),
           46:(     6.545184152E-02,     0.000000000E+00     ,0.000000000E+00),
           47:(     7.500000298E-02,     0.000000000E+00     ,0.000000000E+00),
           48:(    -8.411400452E-19,    -4.220340412E-02     ,0.000000000E+00),
           49:(     0.000000000E+00,    -5.000000075E-02     ,0.000000000E+00),
           50:(    -1.707526847E-18,    -3.417279399E-02     ,0.000000000E+00),
           51:(    -2.557363465E-18,    -2.629558886E-02     ,0.000000000E+00),
           52:(     0.000000000E+00,    -1.850000024E-02     ,0.000000000E+00),
    }

    element_data = {
        1:{'E':200e9,'nu':0.25,'t':0.025, 'etype':'2dQ4'},
    }
    elements = [
    {'edata':1, 'nodes':  ( 1,2,4,3),},
    {'edata':1, 'nodes':  ( 3,5,6,1),},
    {'edata':1, 'nodes':  ( 7,6,5,8),},
    {'edata':1, 'nodes':  ( 9,7,8,10),},
    {'edata':1, 'nodes': (11,12,9,10),},
    {'edata':1, 'nodes': (13,14,11,10),},
    {'edata':1, 'nodes': (15,13,10,8),},
    {'edata':1, 'nodes': (16,15,8,5),},
    {'edata':1, 'nodes': (17,16,5,3),},
    {'edata':1, 'nodes': (18,17,3,4),},
    {'edata':1, 'nodes': (19,20,18,4),},
    {'edata':1, 'nodes': (21,19,4,2),},
    {'edata':1, 'nodes': (22,21,2,1),},
    {'edata':1, 'nodes': (23,22,1,6),},
    {'edata':1, 'nodes': (24,23,6,7),},
    {'edata':1, 'nodes': (25,24,7,9),},
    {'edata':1, 'nodes': (26,25,9,12),},
    {'edata':1, 'nodes': (27,28,26,12),},
    {'edata':1, 'nodes': (29,27,12,11),},
    {'edata':1, 'nodes': (30,29,11,14),},
    {'edata':1, 'nodes': (31,32,21,22),},
    {'edata':1, 'nodes': (33,34,30,14),},
    {'edata':1, 'nodes': (35,33,14,13),},
    {'edata':1, 'nodes': (36,35,13,15),},
    {'edata':1, 'nodes': (37,36,15,16),},
    {'edata':1, 'nodes': (38,37,16,17),},
    {'edata':1, 'nodes': (39,38,17,18),},
    {'edata':1, 'nodes': (40,39,18,20),},
    {'edata':1, 'nodes': (41,42,31,22),},
    {'edata':1, 'nodes': (43,41,22,23),},
    {'edata':1, 'nodes': (44,43,23,24),},
    {'edata':1, 'nodes': (45,44,24,25),},
    {'edata':1, 'nodes': (46,45,25,26),},
    {'edata':1, 'nodes': (47,46,26,28),},
    {'edata':1, 'nodes': (48,49,40,20),},
    {'edata':1, 'nodes': (50,48,20,19),},
    {'edata':1, 'nodes': (51,50,19,21),},
    {'edata':1, 'nodes': (52,51,21,32),},
    ]
    mesh_data = {
        'node_data':node_data,
        'element_data':element_data,
        'elements':elements,
    }
    # CMBLOCK,_FIXEDSU,NODE,        5
    # (8i10)
    #         48        49        50        51        52

    fixed_data = {
    48:(1,1,1),
    49:(1,1,1),
    50:(1,1,1),
    51:(1,1,1),
    52:(1,1,1),
    }
    f=100000
    load_data = {
    27:(f,0,0),
    28:(f,0,0),
    29:(f,0,0),
    30:(f,0,0),
    34:(f,0,0),
    47:(f,0,0),
    }
           # 39        2        2        2       12       28       27
           # 40        2        2        2       12       27       29
           # 41        2        2        2       12       29       30
           # 42        2        2        2       12       30       34 ->
           # 43        2        2        2       12       47       28 ->(47,28 to a line)

    solve_data = solve( mesh_data,load_data,fixed_data)
    post_data = post_process( mesh_data,load_data,fixed_data, solve_data)
    plot_elements(post_data, title= f'3D Line plot for {element_data[1]['etype']}')
