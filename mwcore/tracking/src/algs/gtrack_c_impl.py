from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Union, Tuple
from ..algs.gtrack_c.gtrack_interface import *
from ctypes import CDLL, c_void_p, c_float, c_uint16, c_uint8, c_int32, byref, POINTER, addressof, c_uint32, c_int, c_byte, c_void_p, pointer, memmove
import os
import time
import numpy as np
    
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
            log.error(f"Failed to initialize GTrackCBuffer: {str(e)}")
            raise
        self.last_time = time.time()
        self.effective_tracks = []
        # Prepare output buffers
        self.max_tracks = self.config.max_units
        self.targets = (GTRACK_targetDesc * self.max_tracks)()

        log.info(f"Memory allocated for {self.max_tracks} targets at {hex(addressof(self.targets))}")
        self.num_tracks = c_uint16(0)

    def _load_library(self):
        """Load the compiled GTrack library with proper function bindings"""
        path_to_lib = os.path.join("mwcore","tracking","src", "algs", "gtrack_c", "gtrack.dll")
        log.info(f"\n=== Loading GTrack Library ===")
        log.info(f"Library path: {os.path.abspath(path_to_lib)}")
        try:
            # Load the library
            self.lib = CDLL(path_to_lib)
            log.info("Library loaded successfully")
            
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
                log.info(f"Found {func_name} at {hex(addressof(func))}")

            # sys.stdout = open('output.log', 'w')
            # sys.stderr = open('error.log', 'w')
            log.info("=== Library Verified ===\n")

        except Exception as e:
            log.error(f"Failed to load library: {str(e)}")
            log.error(f"Current PATH: {os.environ.get('PATH', '')}")
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
        gating_params.gain = 5.0
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
        
        scenery_params = GTRACK_sceneryParams()

        # Configure boundary boxes to prevent tracks from being killed due to "outside boundary" logic
        if self.config.scenery_params.get('num_boundary_boxes', 0) > 0:
            # Set number of boundary boxes
            scenery_params.numBoundaryBoxes = self.config.scenery_params['num_boundary_boxes']
            
            # Configure each boundary box
            for i, box_config in enumerate(self.config.scenery_params['boundary_boxes']):
                if i < 4:  # GTRACK_MAX_BOUNDARY_BOXES limit
                    bbox = GTRACK_boundaryBox()
                    bbox.x1 = box_config['x1']
                    bbox.x2 = box_config['x2'] 
                    bbox.y1 = box_config['y1']
                    bbox.y2 = box_config['y2']
                    bbox.z1 = box_config['z1']
                    bbox.z2 = box_config['z2']
                    scenery_params.boundaryBox[i] = bbox
        else:
            # No boundary boxes - this will cause tracks to be considered "outside" and die
            # Better to have at least one large boundary box
            scenery_params.numBoundaryBoxes = 0

        # Configure static boxes (for sleep logic)
        if self.config.scenery_params.get('num_static_boxes', 0) > 0:
            scenery_params.numStaticBoxes = self.config.scenery_params['num_static_boxes']
            
            for i, box_config in enumerate(self.config.scenery_params['static_boxes']):
                if i < 4:  # GTRACK_MAX_STATIC_BOXES limit  
                    sbox = GTRACK_boundaryBox()
                    sbox.x1 = box_config['x1']
                    sbox.x2 = box_config['x2']
                    sbox.y1 = box_config['y1'] 
                    sbox.y2 = box_config['y2']
                    sbox.z1 = box_config['z1']
                    sbox.z2 = box_config['z2']
                    scenery_params.staticBox[i] = sbox
        else:
            scenery_params.numStaticBoxes = 0

        # Configure sensor position and orientation
        sensor_pos = self.config.scenery_params.get('sensor_position', [0, 0, 1.0])
        sensor_orient = self.config.scenery_params.get('sensor_orientation', [0, 0])

        # scenery_params.sensorPosition.x = sensor_pos[0]
        # scenery_params.sensorPosition.y = sensor_pos[1] 
        # scenery_params.sensorPosition.z = sensor_pos[2] if len(sensor_pos) > 2 else 0.0

        # scenery_params.sensorOrientation.azimTilt = sensor_orient[0]
        # scenery_params.sensorOrientation.elevTilt = sensor_orient[1]
       
        # presence_params = GTRACK_presenceParams()  # Defaults
        
        # Create advanced parameters structure
        adv_params = GTRACK_advancedParameters()
        adv_params.gatingParams = pointer(gating_params)
        adv_params.allocationParams = pointer(alloc_params)
        adv_params.stateParams = pointer(state_params)
        adv_params.sceneryParams = pointer(scenery_params)
        # adv_params.presenceParams = pointer(presence_params)
        
        # Create main config
        config = GTRACK_moduleConfig()
        config.stateVectorType = self.config.stateVectorType
        config.verbose = self.config.verbose
        config.maxNumPoints = self.config.maxNumPoints
        config.maxNumTracks = self.config.maxNumTracks
        # config.initialRadialVelocity = 0.0
        # config.maxRadialVelocity = 20.0  
        # config.radialVelocityResolution = 0.1
        # config.maxAcceleration = (c_float * 3)(5.0, 5.0, 2.0)  # x,y,z max acceleration
        config.deltaT = self.config.deltaT
        config.advParams = pointer(adv_params)
    
    # Explicitly set log level
        # if hasattr(self.lib, 'gtrack_set_log_level'):
        #     log.info("Setting log level to DEBUG")
        #     self.lib.gtrack_set_log_level(config.verbose)
        # Call create function
        err_code = c_int32(0)
        handle_ptr = c_void_p()

        log.info(f"Calling gtrack_create with config at {hex(id(config))}")
        log.info(f"Max units: {self.config.max_units}")

        self.handle = self.lib.gtrack_create(
            byref(config),
            byref(err_code)
        )

        log.info(f"gtrack_create returned handle: {hex(self.handle) if self.handle else 'NULL'}")
        log.info(f"Error code: {err_code.value}")

        if err_code.value != 0:
            raise RuntimeError(f"GTrack initialization failed with error code {err_code.value}")
        
        if not self.handle or self.handle == 0xFFFFFFFFFFFFFFFF:
            raise RuntimeError("Invalid tracker handle received")

        log.info("=== Tracker Created Successfully ===\n")

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
        toPrint=""
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
                    toPrint+=f"Range: {mv.range}, Azimuth: {mv.azimuth}, Elevation: {mv.elev}, Doppler: {mv.doppler}"
                else:
                    # Process 2D data
                    mv = MeasurementVector(is_3d=False)
                    mv.range = pointcloud[i, 8]  # range_val
                    mv.angle = pointcloud[i, 9]  # angle_val
                    mv.doppler = pointcloud[i, 6]  # doppler_val
                    toPrint+=f"Range: {mv.range}, Angle: {mv.angle}, Doppler: {mv.doppler}"
                
                # Copy the binary representation into the measurement point
                memmove(
                    addressof(measurements[i].vector),
                    mv.to_bytes(),
                    len(mv.to_bytes())
                )
                measurements[i].snr = pointcloud[i, 7]  # peak_val
                toPrint+=f", SNR: {measurements[i].snr}\n"
                
                
            
            
            # Call gtrack_step with NULL for optional parameters
            try:
                # log.info("Calling gtrack_step with measurements: %s", toPrint)
                self.lib.gtrack_step(
                    self.handle,
                    measurements,
                    None,  # variances
                    num_points,
                    self.targets,
                    byref(self.num_tracks),
                    None,  # mIndex
                    None,  # uIndex
                    None,  # presence
                    None   # bench
                )
            except Exception as e:
                log.error(f"gtrack_step failed: {str(e)}")
                if hasattr(e, 'winerror') and e.winerror:
                    log.error(f"Windows error code: {e.winerror}")
                return None

            # Process results
            tracks = []
            is_3d = self.config.stateVectorType in (GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DA, GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DV)
            is_accel = self.config.stateVectorType in (GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_2DA, GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DA)

            for i in range(self.num_tracks.value):
                try:
                    # Get the target descriptor
                    target = self.targets[i]
                    
                    uid = int(target.uid)
                    tid = int(target.tid)  
                    confidence = float(target.confidenceLevel)

                    dimensions = []
                    for j in range(4 if is_3d else 3):
                        dim_val = float(target.dim[j])
                        dimensions.append(dim_val)
                    
                    
                    # Extract state vector - be very careful with indexing
                    state_values = []
                    for j in range(9):  # Maximum state vector size
                        try:
                            state_val = float(target.S[j])
                            state_values.append(state_val)
                        except (IndexError, ValueError) as e:
                            log.warning(f"Track {i}: Error reading S[{j}]: {e}")
                            state_values.append(0.0)
                    
                    
                    # Build track dictionary with safe indexing
                    track = {
                        'uid': uid,
                        'tid': tid,
                        'confidence': confidence,
                        'dimensions': dimensions
                    }

                    # Position - always 3D (z=0 for 2D)
                    if is_3d:
                        track['position'] = [state_values[0], state_values[1], state_values[2]]
                    else:
                        track['position'] = [state_values[0], state_values[1], 0.0]

                    # Velocity
                    vel_offset = 3 if is_3d else 2
                    if is_3d:
                        track['velocity'] = [
                            state_values[vel_offset], 
                            state_values[vel_offset+1], 
                            state_values[vel_offset+2]
                        ]
                    else:
                        track['velocity'] = [
                            state_values[vel_offset], 
                            state_values[vel_offset+1], 
                            0.0
                        ]

                    # Acceleration if available
                    if is_accel:
                        accel_offset = 6 if is_3d else 4
                        if is_3d:
                            track['acceleration'] = [
                                state_values[accel_offset],
                                state_values[accel_offset+1], 
                                state_values[accel_offset+2]
                            ]
                        else:
                            track['acceleration'] = [
                                state_values[accel_offset],
                                state_values[accel_offset+1],
                                0.0
                            ]
               
                    tracks.append(track)
                    
                except Exception as e:
                    log.error(f"Error processing track {i}: {e}")
                    continue
            
            self.effective_tracks = tracks
            # return tracks if tracks else None
        except Exception as e:
            log.error(f"Tracking failed: {str(e)}")
            return None

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'handle') and self.handle:
            self.lib.gtrack_delete(self.handle)
    
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'handle') and self.handle:
            self.lib.gtrack_delete(self.handle)