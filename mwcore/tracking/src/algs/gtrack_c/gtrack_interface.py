from ctypes import Structure, c_float, c_uint8, c_uint16, c_uint32, POINTER, c_int, sizeof, c_byte

class GTRACK_sensorPosition(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float)
    ]

class GTRACK_sensorOrientation(Structure):
    _fields_ = [
        ("azimTilt", c_float),
        ("elevTilt", c_float)
    ]

class GTRACK_boundaryBox(Structure):
    _fields_ = [
        ("x1", c_float),
        ("x2", c_float),
        ("y1", c_float),
        ("y2", c_float),
        ("z1", c_float),
        ("z2", c_float)
    ]

class GTRACK_gateLimits(Structure):
    _fields_ = [
        ("depth", c_float),
        ("width", c_float),
        ("height", c_float),
        ("vel", c_float)
    ]

class GTRACK_gatingParams(Structure):
    _fields_ = [
        ("gain", c_float),
        ("limits", c_float * 4)  # Using array instead of union
    ]

class GTRACK_stateParams(Structure):
    _fields_ = [
        ("det2actThre", c_uint16),
        ("det2freeThre", c_uint16),
        ("active2freeThre", c_uint16),
        ("static2freeThre", c_uint16),
        ("exit2freeThre", c_uint16),
        ("sleep2freeThre", c_uint16)
    ]

class GTRACK_allocationParams(Structure):
    _fields_ = [
        ("snrThre", c_float),
        ("snrThreObscured", c_float),
        ("velocityThre", c_float),
        ("pointsThre", c_uint16),
        ("maxDistanceThre", c_float),
        ("maxVelThre", c_float)
    ]

class GTRACK_sceneryParams(Structure):
    _fields_ = [
        ("sensorPosition", GTRACK_sensorPosition),
        ("sensorOrientation", GTRACK_sensorOrientation),
        ("numBoundaryBoxes", c_uint8),
        ("boundaryBox", GTRACK_boundaryBox * 2),  # GTRACK_MAX_BOUNDARY_BOXES
        ("numStaticBoxes", c_uint8),
        ("staticBox", GTRACK_boundaryBox * 2)     # GTRACK_MAX_STATIC_BOXES
    ]

class GTRACK_presenceParams(Structure):
    _fields_ = [
        ("pointsThre", c_uint16),
        ("velocityThre", c_float),
        ("on2offThre", c_uint16),
        ("numOccupancyBoxes", c_uint8),
        ("occupancyBox", GTRACK_boundaryBox * 2)  # GTRACK_MAX_OCCUPANCY_BOXES
    ]

class GTRACK_advancedParameters(Structure):
    _fields_ = [
        ("gatingParams", POINTER(GTRACK_gatingParams)),
        ("allocationParams", POINTER(GTRACK_allocationParams)),
        ("stateParams", POINTER(GTRACK_stateParams)),
        ("sceneryParams", POINTER(GTRACK_sceneryParams)),
        ("presenceParams", POINTER(GTRACK_presenceParams))
    ]

class GTRACK_STATE_VECTOR_TYPE(c_int):
    GTRACK_STATE_VECTORS_2DV = 0
    GTRACK_STATE_VECTORS_2DA = 1
    GTRACK_STATE_VECTORS_3DV = 2
    GTRACK_STATE_VECTORS_3DA = 3

class GTRACK_VERBOSE_TYPE(c_int):
    GTRACK_VERBOSE_NONE = 0
    GTRACK_VERBOSE_ERROR = 1
    GTRACK_VERBOSE_WARNING = 2
    GTRACK_VERBOSE_DEBUG = 3
    GTRACK_VERBOSE_MATRIX = 4
    GTRACK_VERBOSE_MAXIMUM = 5

class GTRACK_moduleConfig(Structure):
    _fields_ = [
        ("stateVectorType", GTRACK_STATE_VECTOR_TYPE),
        ("verbose", GTRACK_VERBOSE_TYPE),
        ("maxNumPoints", c_uint16),
        ("maxNumTracks", c_uint16),
        ("initialRadialVelocity", c_float),
        ("maxRadialVelocity", c_float),
        ("radialVelocityResolution", c_float),
        ("maxAcceleration", c_float * 3),
        ("deltaT", c_float),
        ("advParams", POINTER(GTRACK_advancedParameters))
    ]

class GTRACK_measurement_vector_2D(Structure):
    """For 2D configurations (range, angle, doppler)"""
    _fields_ = [
        ("range", c_float),
        ("angle", c_float),
        ("doppler", c_float)
    ]
    _pack_ = 1

class GTRACK_measurement_vector_3D(Structure):
    """For 3D configurations (range, azimuth, elev, doppler)"""
    _fields_ = [
        ("range", c_float),
        ("azimuth", c_float),
        ("elev", c_float),
        ("doppler", c_float)
    ]
    _pack_ = 1

class GTRACK_measurementPoint_2D(Structure):
    _fields_ = [
        ("vector", c_byte * sizeof(GTRACK_measurement_vector_2D)),
        ("snr", c_float)
    ]
    _pack_ = 1

class GTRACK_measurementPoint_3D(Structure):
    _fields_ = [
        ("vector", c_byte * sizeof(GTRACK_measurement_vector_3D)),
        ("snr", c_float)
    ]
    _pack_ = 1

GTRACK_STATE_VECTOR_SIZE = 9  # 6 for 2D, 9 for 3D
GTRACK_MEASUREMENT_VECTOR_SIZE = 4  # 3 for 2D, 4 for 3D

class GTRACK_targetDesc(Structure):
    _fields_ = [
        ("uid", c_uint8),
        ("tid", c_uint32),
        ("S", c_float * GTRACK_STATE_VECTOR_SIZE),
        ("EC", c_float * (GTRACK_MEASUREMENT_VECTOR_SIZE * GTRACK_MEASUREMENT_VECTOR_SIZE)),
        ("G", c_float),
        ("dim", c_float * GTRACK_MEASUREMENT_VECTOR_SIZE),
        ("uCenter", c_float * GTRACK_MEASUREMENT_VECTOR_SIZE),
        ("confidenceLevel", c_float)
    ]


class StateVector:
    def __init__(self, config):
        self.config = config
        self.size = {
            GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_2DV: 4,
            GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_2DA: 6,
            GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DV: 6,
            GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DA: 9
        }[config.stateVectorType]
        
    def unpack(self, target: GTRACK_targetDesc) -> dict:
        s = target.S
        if self.config.stateVectorType == GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_2DV:
            return {'pos': (s[0], s[1], 0), 'vel': (s[2], s[3], 0)}
        elif self.config.stateVectorType == GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_2DA:
            return {'pos': (s[0], s[1], 0), 'vel': (s[2], s[3], 0), 'acc': (s[4], s[5], 0)}
        elif self.config.stateVectorType == GTRACK_STATE_VECTOR_TYPE.GTRACK_STATE_VECTORS_3DV:
            return {'pos': (s[0], s[1], s[2]), 'vel': (s[3], s[4], s[5])}
        else:  # 3DA
            return {'pos': (s[0], s[1], s[2]), 'vel': (s[3], s[4], s[5]), 'acc': (s[6], s[7], s[8])}
        
class MeasurementVector:
    def __init__(self, is_3d=False):
        self._is_3d = is_3d
        if is_3d:
            self._struct = GTRACK_measurement_vector_3D()
        else:
            self._struct = GTRACK_measurement_vector_2D()
    
    @property
    def range(self):
        return self._struct.range
    
    @range.setter
    def range(self, value):
        self._struct.range = value
        
    # Similarly implement angle/azimuth, elev, doppler properties
    # with appropriate checks for mode
    @property
    def angle(self):
        if self._is_3d:
            raise ValueError("Angle is not applicable for 3D measurement vector")
        return self._struct.angle
    
    @angle.setter
    def angle(self, value):
        if self._is_3d:
            raise ValueError("Angle is not applicable for 3D measurement vector")
        self._struct.angle = value

    @property
    def azimuth(self):
        if not self._is_3d:
            raise ValueError("Azimuth is only applicable for 3D measurement vector")
        return self._struct.azimuth
    
    @azimuth.setter
    def azimuth(self, value):
        if not self._is_3d:
            raise ValueError("Azimuth is only applicable for 3D measurement vector")
        self._struct.azimuth = value

    @property
    def elev(self):
        if not self._is_3d:
            raise ValueError("Elevation is only applicable for 3D measurement vector")
        return self._struct.elev
    
    @elev.setter
    def elev(self, value):
        if not self._is_3d:
            raise ValueError("Elevation is only applicable for 3D measurement vector")
        self._struct.elev = value

    @property
    def doppler(self):
        return self._struct.doppler
    @doppler.setter
    def doppler(self, value):
        self._struct.doppler = value

    
    
    def to_bytes(self):
        return bytes(self._struct)