import numpy as np
import matplotlib.pyplot as plt
import random
import pygmsh
import os
import torch
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Returns a referance polygon
def ReferancePolygon(n_edges):
    polygon = []
    for i in range(n_edges):
        polygon.append([np.cos(2 * np.pi * i / n_edges),np.sin(2 * np.pi * i / n_edges)])
    return np.array(polygon)

# Preformes procrustes superimposition on the input polygon
def ProcrustesSuperimposition(P):                    
    n = P.shape[0]                               
    Q = ReferancePolygon(n)                          
    P_mean = np.mean(P, 0)                  
    P_centered = P - P_mean  

    P_norm = np.sqrt(np.sum(P_centered**2))                
    Q_mean = np.mean(Q, 0)                  
    Q_centered = Q - Q_mean            
    Q_norm = np.sqrt(np.sum(Q_centered**2))
    
    P_scaled = (P / P_norm)  
    Q_scaled = (Q/Q_norm)                 
    A = Q_scaled.T @ P_scaled                                               
    U, C, V = np.linalg.svd(A)
    R = U @ V.T
    S = Q_norm * np.sum(C)
    P_transformed = (S*P_scaled) @ R + Q_mean
    return P_transformed

# Returns a radius weighted towards larger numbers 
def WeightedRadius():
    list_random = [random.uniform(0.2,0.3), random.uniform(0.3,0.4), random.uniform(0.4,0.6),random.uniform(0.6,0.8),random.uniform(0.8,1.0)]
    weights = (5,5,25,30,35)
    return random.choices(list_random, weights, k=1)[0]

# Returns a random polygon
def GetRandomContour(n_edges):
    contour = []
    one_rad = np.pi/180
    rad = (2*np.pi)/n_edges
    theta = 0
    for i in range(n_edges):
        theta_ = random.uniform(theta+one_rad*5, theta + rad-one_rad*5)
        radius_ = WeightedRadius()
        contour.append([radius_*np.cos(theta_),radius_*np.sin(theta_)])
        theta += rad
    return contour
    
# Returns a distance field
def CreateDistanceFunction(pts):
    distances = np.sqrt((pts[:,0][:,None] - X.ravel())**2 + (pts[:,1][:,None] - Y.ravel())**2)
    min_dist = distances.min(axis=0)
    return min_dist.reshape([dim,dim])
    #return min_dist

# Create contours, mesh and distance field
def CreateData(dataType,i):
    P = GetRandomContour(n_edges)
    P_t = ProcrustesSuperimposition(np.array(P))

    path = f'./data/{dataType}/{i}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    max_dist = 0
    for a, b in combinations(np.array(P_t),2):
        cur_dist = np.linalg.norm(a-b)
        if cur_dist > max_dist:
            max_dist = cur_dist
    trans_vec = np.mean(P_t,axis=0)       
    P_t = P_t - trans_vec               
    P_t /= max_dist                  
    
    mesh_size = 0.4 
    with pygmsh.geo.Geometry() as geom:
        #gmsh.option.setNumber('Mesh.RecombineAll', 1)
        geom.add_polygon(P_t, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
        mesh_pts = mesh.points
        mesh.write(f'{path}/mesh.vtk')

    df = CreateDistanceFunction(mesh_pts)
    print(df.shape)
    Ni = len(mesh_pts)   

    data = {
        "Pc": P_t,
        "mesh_pts": mesh_pts,
        "df": df, 
        "Ni": Ni,
        "x": X,
        "y": Y
        }

    torch.save(data,f'{path}/data.pth')
    return P_t, mesh, df
        
# create training data for testing, training and validation
def CreateDataMain():
    for i in range(training_samples):
        if i < testing_samples:
            CreateData("test",i)
        if i < training_samples:
            P, mesh, df = CreateData("train",i)
            P_list.append(P), df_list.append(df), mesh_list.append(mesh)
        if i < validation_samples:
            CreateData("validation",i)
    print('Data sets created!')

# Returns the inner points in the mesh
def GetInnerPts(mesh_pts, contour_pts):
    inner_pts = []
    for pt in mesh_pts:
        contour_pts = torch.DoubleTensor(contour_pts)
        pt = torch.DoubleTensor(pt)
        res = torch.isclose(pt[:2], contour_pts)
        res = res[:,0] & res[:,1]
        if not (res.sum()>0):
            inner_pts.append(pt)
    return inner_pts

def format_ax(ax, pc,fig, loc='bottom'):
    ax.axis('scaled')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size='5%', pad=0.4)
    fig.colorbar(pc, cax=cax, orientation='horizontal')

# Plots the distance field
def PlotDistanceField(P, M, df):
    contour = P
    P_list = contour.tolist()
    P_list.append(P_list[0])
    xs, ys = zip(*P_list)
 
    fig, axs = plt.subplots(1,2, figsize=(12,8))
    
    for pt in contour:
        axs[0].plot(pt[0],pt[1], "b.")
    mesh = M
    inner_pts = GetInnerPts(mesh.points.astype(np.float32), contour)
    for pt in inner_pts:
        axs[0].plot(pt[0],pt[1], "r.")
    axs[0].set_title('Mesh vertices and contour')
    axs[0].axis('off')
    axs[0].plot(xs,ys)

    pc = axs[1].pcolormesh(X,Y, df, cmap='terrain')
    format_ax(axs[1], pc, fig)
    axs[1].set_title('Distance field')

    plt.show()


# Grid size
dim = 64
min_xy, max_xy = -0.8, 0.8
step = (max_xy - min_xy)/dim
xs = np.linspace(min_xy,max_xy,dim)
ys = np.linspace(min_xy,max_xy,dim)
X,Y = np.meshgrid(xs,ys)
#grid_pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten())])



# Hyperparameters for data
testing_samples = 2#3000
training_samples = 2#10000
validation_samples= 2#500
n_edges = 4

P_list = []
mesh_list = []
df_list = []


if __name__ == "__main__":
    CreateDataMain()
    #PlotContour(P_list[0], mesh_list[0])
    PlotDistanceField(P_list[0], mesh_list[0], df_list[0])

   
  
  


