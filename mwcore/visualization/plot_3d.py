import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from .graph_utilities import eulerRot, getBoxArcs, getBoxArcs2D ,getBoxLines, getSquareLines

# Different methods to color the points 
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'


class Plot3D():
    def __init__(self):
        # Create plot
        self.plot_3d = gl.GLViewWidget()
        # Sets background to a pastel grey
        self.plot_3d.setBackgroundColor(70, 72, 79)
        # Create the background grid
        gz = gl.GLGridItem()
        self.plot_3d.addItem(gz)

        # Create scatter plot for point cloud
        self.scatter = gl.GLScatterPlotItem(size=5)
        self.scatter.setData(pos=np.zeros((1,3)))
        self.plot_3d.addItem(self.scatter)
        self.boundaryBoxList = []

        # Create scatter plot for clusters
        self.scatterClusters = gl.GLScatterPlotItem(size=10)
        self.scatterClusters.setData(pos=np.zeros((1,3)))
        self.plot_3d.addItem(self.scatterClusters)

        # Sensor position
        self.xOffset = 0
        self.yOffset = 0
        self.sensorHeight = 0
        self.az_tilt = 0
        self.elev_tilt = 0
    
        # Create box to represent EVM
        evmSizeX = 0.0625
        evmSizeZ = 0.125
        verts = np.empty((2,3,3))
        verts[0,0,:] = [-evmSizeX, 0, evmSizeZ]
        verts[0,1,:] = [-evmSizeX,0,-evmSizeZ]
        verts[0,2,:] = [evmSizeX,0,-evmSizeZ]
        verts[1,0,:] = [-evmSizeX, 0, evmSizeZ]
        verts[1,1,:] = [evmSizeX, 0, evmSizeZ]
        verts[1,2,:] = [evmSizeX, 0, -evmSizeZ]
        self.evmBox = gl.GLMeshItem(vertexes=verts,smooth=False,drawEdges=True,edgeColor=pg.glColor('r'),drawFaces=False)
        self.plot_3d.addItem(self.evmBox)

        # Initialize other elements
        self.boundaryBoxViz = []
        self.coordStr = []
        self.classifierStr = []
        self.ellipsoids = []
        self.plotComplete = 1

        self.zRange = [-3, 3]

        # Persistent point cloud
        self.previousClouds = []
        self.numPersistentFrames = int(3)

        self.mpdZoneType = None
        self.snapTo2D = None
        self.modeSwitchLabel = None

    def updatePointCloud(self, outputDict):
        if ('pointCloud' in outputDict and 'numDetectedPoints' in outputDict):
            pointCloud = outputDict['pointCloud']
            pointCloud = np.asarray(pointCloud)

            # Rotate point cloud and tracks to account for elevation and azimuth tilt
            if (self.elev_tilt != 0 or self.az_tilt != 0):
                for i in range(outputDict['numDetectedPoints']):
                    rotX, rotY, rotZ = eulerRot (pointCloud[i,0], pointCloud[i,1], pointCloud[i,2], self.elev_tilt, self.az_tilt)
                    pointCloud[i,0] = rotX
                    pointCloud[i,1] = rotY
                    pointCloud[i,2] = rotZ

            # Shift points to account for sensor height
            if (self.sensorHeight != 0):
                pointCloud[:,2] = pointCloud[:,2] + self.sensorHeight

            # Add current point cloud to the cumulative cloud if it's not empty
            self.previousClouds.append(outputDict['pointCloud'])
        else:
            # if there is no point cloud, append an empty array
            self.previousClouds.append([])

        # If we have more point clouds than needed, stated by numPersistentFrames, delete the oldest ones 
        while(len(self.previousClouds) > self.numPersistentFrames):
            self.previousClouds.pop(0)
            
    # Add a boundary box to the boundary boxes tab
    def addBoundBox(self, name, minX=0, maxX=0, minY=0, maxY=0, minZ=0, maxZ=0):
        newBox = gl.GLLinePlotItem()
        newBox.setVisible(True)
        self.plot_3d.addItem(newBox)     

        if ('mpdBoundaryArc' in name):
            try:
                if(self.snapTo2D.checkState() == 0):
                    boxLines = getBoxArcs(minX,minY,minZ,maxX,maxY,maxZ)
                elif(self.snapTo2D.checkState() == 2):
                    boxLines = getBoxArcs2D(minX,minY,0,maxX,maxY,0)
 
                boxColor = pg.glColor('b')
                newBox.setData(pos=boxLines,color=boxColor,width=2,antialias=True,mode='lines')

                # TODO add point boundary back into visualizer

                boundaryBoxItem = {
                    'plot': newBox,
                    'name': name,
                    'boxLines': boxLines,
                    'minX': float(minX),
                    'maxX': float(maxX),
                    'minY': float(minY),
                    'maxY': float(maxY),
                    'minZ': float(minZ),
                    'maxZ': float(maxZ)
                }   

                self.boundaryBoxViz.append(boundaryBoxItem) 
                self.plot_3d.addItem(newBox)
                self.boundaryBoxList.append(newBox)
            except:
                # You get here if you enter an invalid number
                # When you enter a minus sign for a negative value, you will end up here before you type the full number
                pass
        else:
            try:
                if(self.snapTo2D.checkState() == 0):
                    boxLines = getBoxLines(minX,minY,minZ,maxX,maxY,maxZ)
                elif(self.snapTo2D.checkState() == 2):
                    boxLines = getSquareLines(minX,minY,0,maxX,maxY,0)
 
                boxColor = pg.glColor('b')
                newBox.setData(pos=boxLines,color=boxColor,width=2,antialias=True,mode='lines')

                # TODO add point boundary back into visualizer

                boundaryBoxItem = {
                    'plot': newBox,
                    'name': name,
                    'boxLines': boxLines,
                    'minX': float(minX),
                    'maxX': float(maxX),
                    'minY': float(minY),
                    'maxY': float(maxY),
                    'minZ': float(minZ),
                    'maxZ': float(maxZ)
                }   

                self.boundaryBoxViz.append(boundaryBoxItem) 
                self.plot_3d.addItem(newBox)
                self.boundaryBoxList.append(newBox)
            except:
                # You get here if you enter an invalid number
                # When you enter a minus sign for a negative value, you will end up here before you type the full number
                pass

    def removeAllBoundBoxes(self):
        for item in self.boundaryBoxList:
            item.setVisible(False)
        if(self.snapTo2D is not None):
            self.snapTo2D.setEnabled(1)
        if(self.modeSwitchLabel is not None):
            self.modeSwitchLabel.setText('Two Pass Mode Disabled')
            self.modeSwitchLabel.setStyleSheet("background-color: lightgrey; border: 1px solid black;")
        self.boundaryBoxList.clear()

    def changeBoundaryBoxColor(self, box, color):
        box['plot'].setData(pos=box['boxLines'], color=pg.glColor(color),width=2,antialias=True,mode='lines')
        
    def parseTrackingCfg(self, args):
        self.maxTracks = int(args[4])

    def parseBoundaryBox(self, args):
        self.snapTo2D.setEnabled(0)

        if (args[0] == 'SceneryParam' or args[0] == 'boundaryBox'):
            leftX = float(args[1])
            rightX = float(args[2])
            nearY = float(args[3])
            farY = float(args[4])
            bottomZ = float(args[5])
            topZ = float(args[6])
                        
            self.addBoundBox('trackerBounds', leftX, rightX, nearY, farY, bottomZ, topZ)
        elif (args[0] == 'zoneDef'):
            zoneIdx = int(args[1])
            minX = float(args[2])
            maxX = float(args[3])
            minY = float(args[4])
            maxY = float(args[5])
            # Offset by 3 so it is in center of screen
            minZ = float(args[6]) + self.sensorHeight
            maxZ = float(args[7]) + self.sensorHeight

            name = 'occZone' + str(zoneIdx)
            self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)
        elif (args[0] == 'mpdBoundaryBox'):
            zoneIdx = int(args[1])
            minX = float(args[2])
            maxX = float(args[3])
            minY = float(args[4])
            maxY = float(args[5])
            minZ = float(args[6])
            maxZ = float(args[7])
            name = 'mpdBoundaryBox' + str(zoneIdx)
            self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)
        elif (args[0] == 'mpdBoundaryArc'):
            zoneIdx = int(args[1])
            minR = float(args[2])
            maxR = float(args[3])
            minTheta = float(args[4])
            maxTheta = float(args[5])
            minZ = float(args[6])
            maxZ = float(args[7])
            name = 'mpdBoundaryArc' + str(zoneIdx)
            self.addBoundBox(name, minR, maxR, minTheta, maxTheta, minZ, maxZ)

        # TODO print out somewhere these boundary boxes

    def parseSensorPosition(self, args, is_x843):
        if (is_x843):
            self.sensorHeight = float(args[1])
            self.az_tilt = float(args[2])
            self.elev_tilt = float(args[3])
        else:
            self.xOffset = float(args[1])
            self.yOffset = float(args[2])
            self.sensorHeight = float(args[3])
            self.az_tilt = float(args[4])
            self.elev_tilt = float(args[5])

        self.evmBox.resetTransform()
        self.evmBox.rotate(-1 * self.elev_tilt, 1, 0, 0)
        self.evmBox.rotate(-1 * self.az_tilt, 0, 0, 1)
        self.evmBox.translate(0, 0, self.sensorHeight)

        # TODO update text showing sensor position text?
