import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Function to convert global positions to local positions
def global_to_local(global_position, reference_position):
    local_positions = (global_position - reference_position[:2]) 
    #+random_variances
    return local_positions

# Function to convert global positions to local positions
def local_to_global(local_position, reference_position):
    global_position = (local_position + reference_position[:2]) 
    #+ random_variances
    return global_position

# distance Calculation
def dist(x, y, pos):
    return math.sqrt(((pos[0]-x)**2) + ((pos[1]-y)**2))
 
# partitionFinder
def partitionFinder_robotariumm(ax, robotsPositions, envSize_X, envSize_Y, resolution, densityFlag, densityArray,alpha, partitionMarkerSize,globalFrame):
    hull_figHandles = []
    x_global_values = np.arange(envSize_X[0], envSize_X[1]+resolution, resolution)
    y_global_values = np.arange(envSize_Y[0], envSize_Y[1]+resolution, resolution)    
    distArray = np.zeros(robotsPositions.shape[0])
    colorList = ["red","green","blue","black","grey","orange"]
    locations = [[] for _ in range(robotsPositions.shape[0])]
    robotDensity = [[] for _ in range(robotsPositions.shape[0])]
    locationsIdx = [[] for _ in range(robotsPositions.shape[0])]
    text_handles = []
    if densityFlag:
        if(densityArray.shape[0]>x_global_values.shape[0] or densityArray.shape[1]>y_global_values.shape[0]):
            densityArray = densityArray[:x_global_values.shape[0],:y_global_values.shape[0]]
        if(densityArray.shape[0]<x_global_values.shape[0] or densityArray.shape[1]<y_global_values.shape[0]):
            x_global_values = x_global_values[:densityArray.shape[0]]
            y_global_values = x_global_values[:densityArray.shape[1]]

    
    for i, x_pos in enumerate(x_global_values):
        for j, y_pos in enumerate(y_global_values):
            for r in range(robotsPositions.shape[0]):    
                distanceSq = (robotsPositions[r, 0] - x_pos) ** 2 + (robotsPositions[r, 1] - y_pos) ** 2
                distArray[r] = abs(math.sqrt(distanceSq))
            minValue = np.min(distArray)
            minIndices = np.where(distArray == minValue)[0]
            for r in minIndices:
                locations[r].append([x_pos, y_pos])
                if(densityFlag):
                    robotDensity[r].append(densityArray[i,j])   

    for r in range(robotsPositions.shape[0]):
        robotsLocation = np.array(locations[r])
        if not densityFlag:
            if(robotsLocation.shape[0]>0 and robotsLocation.shape[1]>0):
                #ax.scatter(robotsLocation[:,0], robotsLocation[:,1], color = colorList[r],marker="v",linewidths=partitionMarkerSize, alpha=alpha)
                if globalFrame:
                    lineHandle, = ax.plot(robotsLocation[:,0], robotsLocation[:,1], color = colorList[r],marker="v",markersize=partitionMarkerSize, linestyle="none", alpha=alpha) 
                    hull_figHandles.append(lineHandle)
                    text_handle =  ax.text((robotsPositions[r])[0]+0.09,(robotsPositions[r])[1]+0.05,str(r+1),color="black",fontsize=18)
                    text_handles.append(text_handle)
        else:
            if(globalFrame):
                hull = ConvexHull(robotsLocation)
                # Get the vertices of the convex hull
                boundary_points = robotsLocation[hull.vertices]
                # Extract x and y coordinates
                x, y = boundary_points[:, 0], boundary_points[:, 1]
                hullHandle, =  ( ax.plot(x, y, marker='None', linestyle='-', color="black", markersize=6, linewidth =8))
                hull_figHandles.append(hullHandle)
                text_handle =  ax.text((robotsPositions[r])[0]+0.09,(robotsPositions[r])[1]+0.05,str(r+1),color="black",fontsize=18)
                text_handles.append(text_handle)
    #ax.set(xlim=(envSize_X[0], envSize_X[1]), ylim=(envSize_Y[0], envSize_Y[1]))
    Mass = np.zeros(robotsPositions.shape[0])
    C_x = np.zeros(robotsPositions.shape[0])
    C_y = np.zeros(robotsPositions.shape[0])
    locationalCost = 0
    
    for r in range(robotsPositions.shape[0]):
        Cx_r = 0
        Cy_r = 0
        Mass_r = 0
        locationInRobotRegion = np.array(locations[r])
        currentrobotLoc = robotsPositions[r]
        r_dens = robotDensity[r]    
        for pos in range(locationInRobotRegion.shape[0]):
            if densityFlag:
                dens = resolution * resolution * r_dens[pos]
            else:
                dens = resolution * resolution     
            Mass_r += dens
            Cx_r += dens * locationInRobotRegion[pos, 0]
            Cy_r += dens * locationInRobotRegion[pos, 1]
            positionDiffSq = (locationInRobotRegion[pos, 0] - currentrobotLoc[0]) ** 2 + (locationInRobotRegion[pos, 1] - currentrobotLoc[1]) ** 2  
            locationalCost += dens * (positionDiffSq)
        if(Mass_r!=0):
            Cx_r /= Mass_r
            Cy_r /= Mass_r
            C_x[r] = Cx_r
            C_y[r] = Cy_r
            Mass[r] = Mass_r
    return C_x, C_y, locationalCost, Mass,hull_figHandles,text_handles
    
    
def getDensityArray_robotarium(ax,x_min,x_max,y_min,y_max, resolution,u1, sigmaValue,displayFlag):
    x = np.arange(x_min, x_max + resolution, resolution)
    y = np.arange(y_min, y_max + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    sigma = sigmaValue * np.eye(2)
    covInv = np.linalg.inv(sigma)
    densityArray = np.zeros(X.shape)
    constant_term = 4/ (1 * np.pi * np.sqrt(np.linalg.det(sigma)))
    q = np.column_stack((X.ravel(), Y.ravel()))
    diff1 = q - u1
    handle = ""
    for i in range(q.shape[0]):
        exponent1 = -0.5 * np.dot(diff1[i], np.dot(covInv, diff1[i]))
        densityArray.ravel()[i] = constant_term * (np.exp(exponent1))
    density = densityArray.reshape(X.shape)
    if displayFlag:
        handle = ax.pcolor(X, Y, density,shading="auto")
    return density,handle
