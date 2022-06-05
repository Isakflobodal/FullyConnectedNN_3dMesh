from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch
from itertools import combinations
import pyvista as pv
import trimesh
from pysdf import SDF

# Returns a scaled and centered object
def ScaleAndSenter(contour):
    max_dist = 0
    for a, b in combinations(np.array(contour),2):
        cur_dist = np.linalg.norm(a-b)
        if cur_dist > max_dist:
            max_dist = cur_dist
   
    center = np.mean(contour,axis=0)      
    contour = contour - center                 
    contour /= max_dist                     
    return contour

# Returns contour pts, mesh pts, volumetric mesh and surface mesh
def GetRandomContour(N):
    l = 1.0
    b = random.uniform(0.5,1.0)
    h = random.uniform(0.5,1.0)

    #b = h = l = 1.0

    contour = np.array([[-l/2,-b/2,-h/2],[l/2,-b/2,-h/2],[l/2,b/2,-h/2],[-l/2,b/2,-h/2],[-l/2,-b/2,h/2],[l/2,-b/2,h/2],[l/2,b/2,h/2],[-l/2,b/2,h/2]])
    contour = ScaleAndSenter(contour=contour)
    l = abs(contour[0][0])*2
    b = abs(contour[0][1])*2
    h = abs(contour[0][2])*2

    target_len = np.cbrt(b*h*l/N)
    nl = max(round(l/target_len)+1,2)
    nb = max(round(b/target_len)+1,2)
    nh = max(round(h/target_len)+1,2)

    ls = np.linspace(-l/2,l/2,nl)
    bs = np.linspace(-b/2,b/2,nb)
    hs = np.linspace(-h/2,h/2,nh)

    X,Y = np.meshgrid(ls,bs)
    pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten(), np.full(nl*nb,0))])

    levels = []
    for l in hs:
        level = pts.copy()
        level[:,-1] = l
        levels.append(level)
    
    mesh = pv.StructuredGrid()
    pts = np.vstack(levels)
    mesh.points = pts
    mesh.dimensions = [nl,nb,nh]
    surf_mesh = trimesh.primitives.Box((l,b,h))
    #mesh.plot(show_edges=True, show_grid=False, cpos = 'xy', color="#009ee5", background="white")

    return pts, contour, mesh, surf_mesh

# Returns two distance fields (df). A scalar distance field and a vector distance field
def GetDF(pts):
    x = (pts[:,0][:,None] - BB[:,0])
    y = (pts[:,1][:,None] - BB[:,1])
    z = (pts[:,2][:,None] - BB[:,2])
    #print(x.shape)
    vec = np.vstack([x[None],y[None],z[None]]).swapaxes(0,2) 
    vec_length = np.linalg.norm(vec, axis=-1)
    a = np.arange(0, vec_length.shape[0])
    
    min_vec_length_idx = np.argmin(vec_length, axis=1) 
    min_length = vec_length[a, min_vec_length_idx]
    min_vec = vec[a,min_vec_length_idx]
  
    return min_length.reshape([dim,dim,dim]), min_vec.reshape([dim,dim,dim,3])

# Returns a signed distance field (sdf)
def Sdf3D(surf_mesh):
    f = SDF(surf_mesh.vertices, surf_mesh.faces)
    sdf = f(BB.tolist())
    return sdf

# Returns a sdf, df, mesh pts and contour pts
def CreateData(dataType,i, N):
    path = f'./data/{dataType}/{i}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)     

    mesh_pts, P, mesh, surf_mesh = GetRandomContour(N=N)       
    mesh_pts = np.array(mesh_pts) 
    pv.save_meshio(f'{path}/mesh.vtk', mesh=mesh)

    sdf = Sdf3D(surf_mesh)
    sdf = torch.from_numpy(np.array(sdf)).view(dim,dim,dim).float()
    df, df_vec = GetDF(mesh_pts)
    df = torch.from_numpy(np.array(df)).float()
    df_vec = torch.from_numpy(np.array(df_vec)).float()
    
    data = {
        "Pc": P,
        "mesh_pts": mesh_pts,
        "df": df,
        "df_vec": df_vec,
        'sdf':sdf
        }

    torch.save(data,f'{path}/data.pth')
    return sdf, df, mesh_pts, P

# creates data files and append plot data
def CreateDataMain(N):
    for i in range(training_samples):
        if i < testing_samples:
            CreateData("test",i,N)
        if i < training_samples:
            sdf, df, mesh_pts, P = CreateData("train",i, N)
            df_list.append(df)
            mesh_pts_list.append(mesh_pts)
            sdf_list.append(sdf)
            P_list.append(P)
        if i < validation_samples:
            CreateData("validation",i, N)
    print('Data sets created!')


# Plots df and sdf
def PlotDistanceField(df, sdf, mesh_pts, P):

    x, y, z = BB[:,0], BB[:,1], BB[:,2]
    max, min = x.max().item(), x.min().item()
    fig = plt.figure(figsize=plt.figaspect(0.5))

        # 1,2,1
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.scatter(x,y,z, s=30, c = df, cmap='RdBu', alpha=0.005)
    ax.set_title('Distance function')
    #ax.scatter(x,y,z, s=30, c='blue', alpha=0.02) # del
    #ax.scatter(P[:,0],P[:,1],P[:,2], s=200, c='red' ) # del
    #ax.scatter(mesh_pts[:,0],mesh_pts[:,1],mesh_pts[:,2], c='green' )
    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)
        # 1,2,2
    # ax = fig.add_subplot(1,2,2, projection = '3d')
    # ax.scatter(x,y,z, s=30, c = sdf, cmap='hsv', alpha=0.03)
    # ax.scatter(P[:,0],P[:,1],P[:,2], c='red' )
    # ax.set_title('Signed distance field')
    # ax.set_xlim(min,max)
    # ax.set_ylim(min,max)
    # ax.set_zlim(min,max)
    # fig.tight_layout()
    plt.axis('off')
    plt.show()


# Grid properties
dim = 40
min_xy, max_xy = -0.5, 0.5
step = (max_xy - min_xy)/dim
xs = np.linspace(min_xy,max_xy,dim)
ys = np.linspace(min_xy,max_xy,dim)
zs = np.linspace(min_xy, max_xy,dim)
X,Y = np.meshgrid(xs,ys)

pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten(), np.full(dim*dim,0))])

BB = []
for l in zs:
    level = pts.copy()
    level[:,-1] = l
    BB.append(level)
BB = np.array(BB)
BB = BB.reshape(-1,3)

# Hyperparameters for data
testing_samples = 1#500
training_samples = 1#10000
validation_samples= 1#1000
N = 10

df_list = []
sdf_list = []
mesh_pts_list = []
P_list = []

if __name__ == "__main__":
    CreateDataMain(N=N)
    PlotDistanceField(df_list[0], sdf_list[0], mesh_pts_list[0], P_list[0])