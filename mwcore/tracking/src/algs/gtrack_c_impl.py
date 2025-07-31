from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Union, Tuple
from ..algs.gtrack_c.gtrack_interface import *
from ctypes import CDLL, c_void_p, c_float, c_uint16, c_uint8, c_int32, byref, POINTER, addressof, c_uint32, c_int, c_byte, c_void_p, pointer, memmove
import os
import time
import numpy as np

@dataclass
class GTrackCConfig:
    max_units: int
    is_ceiling_mounted: bool
    frame_rate_ms: float
    stateVectorType: GTRACK_STATE_VECTOR_TYPE
    maxNumPoints: int
    maxNumTracks: int
    initialRadialVelocity: float
    maxRadialVelocity: float
    radialVelocityResolution: float
    maxAcceleration: Tuple[float, float, float]
    deltaT: float
    verbose: GTRACK_VERBOSE_TYPE
    gating_limits: List[float] = field(default_factory=list)
    allocation_params: Dict[str, Union[float, int]] = field(default_factory=dict)
    scenery_params: Dict[str, Any] = field(default_factory=dict)
    state_params: Dict[str, int] = field(default_factory=dict)

class GTrackCBuffer:
    def __init__(self, config: GTrackCConfig):
        self.config = config
        try:
            self._load_library()
            self._create_tracker()
            if not self.handle:
                raise RuntimeError("Tracker handle not created")
        except Exception as e:
            print(f"Tracker initialization failed: {str(e)}")
            raise
        self.last_time = time.time()

    def _load_library(self):
        """Load the compiled GTrack library with proper function bindings"""
        path_to_lib = os.path.join("src", "tracking", "gtrack", "gtrack.dll")
        print(f"\n=== Loading GTrack Library ===")
        print(f"Library path: {os.path.abspath(path_to_lib)}")
        try:
            # Load the library
            self.lib = CDLL(path_to_lib)
            print("Library loaded successfully")
            
            # Verify critical functions exist (correct way)
            required_funcs = {
                'gtrack_create': [POINTER(GTRACK_moduleConfig), POINTER(c_int32)],
                'gtrack_step': None,  # We'll define this later
                'gtrack_delete': [c_void_p]
            }
            
            for func_name, argtypes in required_funcs.items():
                if not hasattr(self.lib, func_name):
                    raise RuntimeError(f"Missing required function: {func_name}")
                
                func = getattr(self.lib, func_name)
                if argtypes is not None:
                    func.argtypes = argtypes
                print(f"Found {func_name} at {hex(addressof(func))}")
            
            # sys.stdout = open('output.log', 'w')
            # sys.stderr = open('error.log', 'w')
            print("=== Library Verified ===\n")
        
        except Exception as e:
            print(f"Failed to load library: {str(e)}")
            print(f"Current PATH: {os.environ.get('PATH', '')}")
            raise
        # gtrack_create
        self.lib.gtrack_create.argtypes = [
            POINTER(GTRACK_moduleConfig),
            POINTER(c_int32)
        ]
        self.lib.gtrack_create.restype = c_void_p
        
        # gtrack_step
        self.lib.gtrack_step.argtypes = [
            c_void_p,
            POINTER(GTRACK_measurementPoint_3D),
            POINTER(c_float),  # variances
            c_uint16,
            POINTER(GTRACK_targetDesc),
            POINTER(c_uint16),
            POINTER(c_uint8),
            POINTER(c_uint8),
            POINTER(c_uint8),
            POINTER(c_uint32)
        ]
        self.lib.gtrack_step.restype = None
        
        # gtrack_delete
        self.lib.gtrack_delete.argtypes = [c_void_p]
        self.lib.gtrack_delete.restype = None


    def _create_tracker(self):
        """Initialize the GTrack module instance with proper configuration"""
        # Create advanced parameters
        gating_params = GTRACK_gatingParams()
        gating_params.gain = 2.0
        gating_params.limits = (c_float * 4)(*self.config.gating_limits)
        
        alloc_params = GTRACK_allocationParams()
        alloc_params.snrThre = self.config.allocation_params['snrThre']
        alloc_params.snrThreObscured = self.config.allocation_params['snrThreObscured']
        alloc_params.velocityThre = self.config.allocation_params['velocityThre']
        alloc_params.pointsThre = self.config.allocation_params['pointsThre']
        alloc_params.maxDistanceThre = self.config.allocation_params['maxDistanceThre']
        alloc_params.maxVelThre = self.config.allocation_params['maxVelThre']
        
        state_params = GTRACK_stateParams()
        state_params.det2actThre = self.config.state_params['det2actThre']
        state_params.det2freeThre = self.config.state_params['det2freeThre']
        state_params.active2freeThre = self.config.state_params['active2freeThre']
        state_params.static2freeThre = self.config.state_params['static2freeThre']
        state_params.exit2freeThre = self.config.state_params['exit2freeThre']
        state_params.sleep2freeThre = self.config.state_params['sleep2freeThre']
        
        # scenery_params = GTRACK_sceneryParams()
        # scenery_params.sensorPosition.x = self.config.scenery_params['sensor_position'][0]
        # scenery_params.sensorPosition.y = self.config.scenery_params['sensor_position'][1]
        # scenery_params.sensorPosition.z = self.config.scenery_params['sensor_position'][2]
        # scenery_params.sensorOrientation.azimTilt = self.config.scenery_params['sensor_orientation'][0]
        # scenery_params.sensorOrientation.elevTilt = self.config.scenery_params['sensor_orientation'][1]
        
        # Set boundary box
        # scenery_params.numBoundaryBoxes = 1
        # scenery_params.boundaryBox[0].x1 = self.config.scenery_params['boundary_box'][0]
        # scenery_params.boundaryBox[0].x2 = self.config.scenery_params['boundary_box'][1]
        # scenery_params.boundaryBox[0].y1 = self.config.scenery_params['boundary_box'][2]
        # scenery_params.boundaryBox[0].y2 = self.config.scenery_params['boundary_box'][3]
        # scenery_params.boundaryBox[0].z1 = self.config.scenery_params['boundary_box'][4]
        # scenery_params.boundaryBox[0].z2 = self.config.scenery_params['boundary_box'][5]
        
        presence_params = GTRACK_presenceParams()  # Defaults
        
        # Create advanced parameters structure
        adv_params = GTRACK_advancedParameters()
        adv_params.gatingParams = pointer(gating_params)
        adv_params.allocationParams = pointer(alloc_params)
        adv_params.stateParams = pointer(state_params)
        # adv_params.sceneryParams = pointer(scenery_params)
        adv_params.presenceParams = pointer(presence_params)
        
        # Create main config
        config = GTRACK_moduleConfig()
        config.stateVectorType = self.config.stateVectorType
        config.verbose = GTRACK_VERBOSE_TYPE.GTRACK_VERBOSE_DEBUG
        config.maxNumPoints = 1000  
        config.maxNumTracks = self.config.max_units
        # config.initialRadialVelocity = 0.0
        # config.maxRadialVelocity = 20.0  
        # config.radialVelocityResolution = 0.1
        # config.maxAcceleration = (c_float * 3)(5.0, 5.0, 2.0)  # x,y,z max acceleration
        config.deltaT = self.config.frame_rate_ms
        config.advParams = pointer(adv_params)
    
    # Explicitly set log level
        if hasattr(self.lib, 'gtrack_set_log_level'):
            print("Setting log level to DEBUG")
            self.lib.gtrack_set_log_level(config.verbose)
        # Call create function
        err_code = c_int32(0)
        handle_ptr = c_void_p()
    
        print(f"Calling gtrack_create with config at {hex(id(config))}")
        print(f"Max units: {self.config.max_units}")
        
        self.handle = self.lib.gtrack_create(
            byref(config),
            byref(err_code)
        )
        
        print(f"gtrack_create returned handle: {hex(self.handle) if self.handle else 'NULL'}")
        print(f"Error code: {err_code.value}")
        
        if err_code.value != 0:
            raise RuntimeError(f"GTrack initialization failed with error code {err_code.value}")
        
        if not self.handle or self.handle == 0xFFFFFFFFFFFFFFFF:
            raise RuntimeError("Invalid tracker handle received")
        
        print("=== Tracker Created Successfully ===\n")

    def track(self, pointcloud: np.ndarray) -> Optional[List[Dict]]:
        """
        Perform tracking on the input pointcloud.
        
        Args:
            pointcloud: Input pointcloud as numpy array (N x 11 if keepRadial else 8)
            # Columns:
            # - x-coordinate
            # - y-coordinate
            # - z-coordinate
            # - Cartesian velocity along the x-axis
            # - Cartesian velocity along the y-axis
            # - Cartesian velocity along the z-axis
            # - doppler
            # - peakval
            # - range (radial distance)
            # - theta (angle in polar coordinates)
            # - radial velocity (ṙ)
            
        Returns:
            List of track dictionaries or None if no tracks
        """
        if not hasattr(self, 'handle') or not self.handle:
            raise RuntimeError("Tracker not initialized or already destroyed")
    
        if pointcloud.shape[0] == 0:
            return None
        
        is_3d = self.config.stateVectorType in [
            GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DV,
            GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DA
        ]
        # print(f"Tracking with {'3D' if is_3d else '2D'} because stateVectorType is {self.config.stateVectorType.value}")
        try:
            # Convert point cloud to C format
            num_points = pointcloud.shape[0]
            measurements = (GTRACK_measurementPoint_2D * num_points)() if not is_3d else (GTRACK_measurementPoint_3D * num_points)()
            
            for i in range(num_points):
                if is_3d:
                    # Process 3D data
                    mv = MeasurementVector(is_3d=True)
                    mv.range = pointcloud[i, 8]  # range_val
                    mv.azimuth = pointcloud[i, 9]  # azimuth_val
                    mv.elev = pointcloud[i, 2]  # z
                    mv.doppler = pointcloud[i, 6]  # doppler_val
                else:
                    # Process 2D data
                    mv = MeasurementVector(is_3d=False)
                    mv.range = pointcloud[i, 8]  # range_val
                    mv.angle = pointcloud[i, 9]  # angle_val
                    mv.doppler = pointcloud[i, 6]  # doppler_val
                
                # Copy the binary representation into the measurement point
                memmove(
                    addressof(measurements[i].vector),
                    mv.to_bytes(),
                    len(mv.to_bytes())
                )
                measurements[i].snr = pointcloud[i, 7]  # peak_val
                
                
            # Prepare output buffers
            max_tracks = self.config.max_units
            targets = (GTRACK_targetDesc * max_tracks)()
            num_tracks = c_uint16(0)
            
            # Call gtrack_step with NULL for optional parameters
            try:
                self.lib.gtrack_step(
                    self.handle,
                    measurements,
                    None,  # variances
                    num_points,
                    targets,
                    byref(num_tracks),
                    None,  # mIndex
                    None,  # uIndex
                    None,  # presence
                    None   # bench
                )
            except Exception as e:
                print(f"gtrack_step failed: {str(e)}")
                if hasattr(e, 'winerror') and e.winerror:
                    print(f"Windows error code: {e.winerror}")
                return None

            # Process results
            tracks = []
            is_3d = self.config.stateVectorType in (GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DA, GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DV)
            is_accel = self.config.stateVectorType in (GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_2DA, GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DA)

            print(f"Number of tracks detected: {num_tracks.value}")
            for i in range(num_tracks.value):
                track = {
                    'uid': targets[i].uid,
                    'tid': targets[i].tid,
                    'confidence': targets[i].confidenceLevel,
                    'dimensions': list(targets[i].dim[:3])  # Only first 3 dimensions
                }

                # Position always has 3 components (z=0 in 2D)
                track['position'] = [targets[i].S[0], targets[i].S[1], targets[i].S[2] if is_3d else 0.0]

                # Velocity handling
                vel_offset = 3 if is_3d else 2
                track['velocity'] = [
                    targets[i].S[vel_offset],
                    targets[i].S[vel_offset+1],
                    targets[i].S[vel_offset+2] if is_3d else 0.0
                ]

                # Acceleration if in *A mode
                if is_accel:
                    accel_offset = 6 if is_3d else 4
                    track['acceleration'] = [
                        targets[i].S[accel_offset],
                        targets[i].S[accel_offset+1],
                        targets[i].S[accel_offset+2] if is_3d else 0.0
                    ]

                tracks.append(track)

            return tracks if tracks else None
        except Exception as e:
            print(f"Tracking failed: {str(e)}")
            return None

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'handle') and self.handle:
            self.lib.gtrack_delete(self.handle)
    
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'handle') and self.handle:
            self.lib.gtrack_delete(self.handle)