from executeCoverageScript import executeCoverageScript_function
from executeCoverageScript_dynamicAP import executeCoverageScript_dynamicAP
import numpy as np
import random
N = 4
densFlag = False
targetTrackingFlag = False
dynamicAP = True
envRadius_diff = np.array([random.uniform(0, 0.2) for _ in range(N)])
# For better understanding of dynamicAP, it is recommended to comment line 121,122 (sets x and y lim for workspace) in installed rps/robotarium_abc.py
# targetTracking and dynamicAP scenarios sets the densFlag "True" in the executeCoverageScript_function
if dynamicAP:
    executeCoverageScript_dynamicAP(N ,resolution=0.02,coverageIterations=80,noise_level=0.05,densityFlag=densFlag,showLocalFOR=True, envRadius_diff = envRadius_diff)     
else:
    executeCoverageScript_function(N ,resolution=0.02,coverageIterations=80,noise_level=0.05,densityFlag=densFlag, targetTrackingFlag=targetTrackingFlag,showLocalFOR=True, envRadius_diff = envRadius_diff)     
