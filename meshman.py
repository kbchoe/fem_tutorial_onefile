from fembase import solve, post_process, plot_elements


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
