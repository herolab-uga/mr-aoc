from executeCoverageScript import executeCoverageScript_function
from executeCoverageScript_dynamicAP import executeCoverageScript_dynamicAP
N = 4
noiseLevel = [0.1,0.2,0.3,0.4,0]
densFlag = True
targetTrackingFlag = True
dynamicAP = True

# For better understanding of dynamicAP, it is recommended to comment line 121,122 (sets x and y lim for workspace) in robotarium_abc.py
# densFlag shoud be "True" for targetTracking scenario
if dynamicAP:
    executeCoverageScript_dynamicAP(N ,resolution=0.02,coverageIterations=200,noise_level=0.1,densityFlag=densFlag)     
else:
    executeCoverageScript_function(N ,resolution=0.02,coverageIterations=200,noise_level=0.1,densityFlag=densFlag, targetTrackingFlag=targetTrackingFlag)     
