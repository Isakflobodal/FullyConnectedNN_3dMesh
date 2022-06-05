from turtle import color
import pygmsh
from dataset import ContourDataset
import numpy as np
import meshplot as mp
import matplotlib.pyplot as plt
import pyvista as pv
import random
import os

import math
import gmsh
from createdata import N


train_data = ContourDataset(split="train")

def GetRandomContour():
    num = random.randint(0,99)
    num2 = random.randint(0,99)
    A, _, _, _ = train_data[num]
    B, _, _, _ = train_data[num2]

    A = A.numpy()
    B = B.numpy()

    return A, B 

def mainAtoB(num=3):

    
    for i in range(num):

        path = f'./gifs/gifs_AtoB'
        isExist2 = os.path.exists(path)
        if not isExist2:
            os.makedirs(path)

        A, B = GetRandomContour()
        t = np.linspace(0,1,100)

        p = pv.Plotter(notebook=False, off_screen=True)
        p.open_gif(f'./gifs/gifs_AtoB/{i}.gif')

        cnt = 0
        for phase in t:

            # interpolate between contour A and B
            s = A*(1-phase) + B*phase

            l = abs(s[0][0])*(2)
            b = abs(s[0][1])*(2)

            nl = max(round(math.sqrt(N*l/b))+1,2)
            nb = max(round(N/nl)+1,2)

            ls = np.linspace(-l/2,l/2,nl)
            bs = np.linspace(-b/2,b/2,nb)

            X,Y = np.meshgrid(ls,bs)
            pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten(), np.full(nl*nb,0))])
    
            # create mesh for current contour
            mesh = pv.StructuredGrid()
            mesh.points = pts
            mesh.dimensions = [nl,nb,1]

            edges = mesh.extract_all_edges()
            p.clear()
            p.add_mesh(mesh, color='green', lighting=False)
            p.add_mesh(edges, color='black', line_width=3, lighting=False)
            p.camera_position = 'xy'
            p.add_text(f'Step: {cnt}', position='upper_left', font_size=12)
            p.add_text(f'Vertices: {len(mesh.points)}', position='upper_right', font_size=12)
            p.write_frame()
           
            cnt += 1
        p.close()
     

def mainRot(num=3):

    for i in range(num):

        path = f'./gifs/gifs_rot'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        A, _ = GetRandomContour()
        l = abs(A[0][0])*(2)
        b = abs(A[0][1])*(2)
        nl = max(round(math.sqrt(N*l/b))+1,2)
        nb = max(round(N/nl)+1,2)
        ls = np.linspace(-l/2,l/2,nl)
        bs = np.linspace(-b/2,b/2,nb)
        X,Y = np.meshgrid(ls,bs)
        pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten())])

        p = pv.Plotter(notebook=False, off_screen=True)
        p.open_gif(f'./gifs/gifs_rot/{i}.gif')

        cnt = 0
        for phi in np.linspace(0,2*np.pi,100):
        
            rot_mat = np.array([[math.cos(phi), -math.sin(phi)],[math.sin(phi), math.cos(phi)]])
            pts_rot = pts @ rot_mat.T

            A = np.zeros([len(pts_rot),1])
            pts_rot = np.concatenate([pts_rot,A], axis=-1)
            
            # create mesh for current contour
            mesh = pv.StructuredGrid()
            mesh.points = pts_rot
            mesh.dimensions = [nl,nb,1]

            edges = mesh.extract_all_edges()
            p.clear()
            p.add_mesh(mesh, color='green', lighting=False)
            p.add_mesh(edges, color='black', line_width=3, lighting=False)
            p.camera_position = 'xy'
            p.add_text(f'Step: {cnt}', position='upper_left', font_size=12)
            p.add_text(f'Vertices: {len(mesh.points)}', position='upper_right', font_size=12)
            p.write_frame()
            p.write_frame()            

            cnt += 1
        p.close()

if __name__ == "__main__":
    mainAtoB()
    mainRot()