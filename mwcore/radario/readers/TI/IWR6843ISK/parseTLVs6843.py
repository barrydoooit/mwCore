
from enum import Enum
import struct
import logging
from typing import Callable, Dict
import numpy as np

log = logging.getLogger(__name__)


class TLVTYPES(Enum):
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
    MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS = 1000
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO = 7
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST = 1010
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX = 1011
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT = 1012
    MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS = 1020
    MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION = 1021

class BaseParser:
    def parse(self, tlv_data, tlv_length, output_dict):
        raise NotImplementedError

class PointCloudTLVParser(BaseParser):
    @staticmethod
    def parse(tlv_data, tlv_length, output_dict):
        point_cloud = output_dict['pointCloud']
        point_struct = '4f'
        point_size = struct.calcsize(point_struct)
        num_points = int(tlv_length / point_size)
        for i in range(num_points):
            try:
                (x, y, z, doppler) = struct.unpack(point_struct, tlv_data[:point_size])
            except:
                num_points = i
                log.error("Point Cloud TLV Parser Failed")
                break
            tlv_data = tlv_data[point_size:]
            point_cloud[i, 0] = x
            point_cloud[i, 1] = y
            point_cloud[i, 2] = z
            point_cloud[i, 3] = doppler
        output_dict['numDetectedPoints'], output_dict['pointCloud'] = num_points, point_cloud

class SideInfoTLVParser(BaseParser):
    @staticmethod
    def parse(tlv_data, tlv_length, output_dict):
        point_cloud = output_dict['pointCloud']
        point_struct = '2H'
        point_struct_size = struct.calcsize(point_struct)
        num_points = int(tlv_length / point_struct_size)
        
        for i in range(num_points):
            try:
                snr, noise = struct.unpack(point_struct, tlv_data[:point_struct_size])
            except:
                num_points = i
                log.error("Side Info TLV Parser Failed")
                break
            tlv_data = tlv_data[point_struct_size:]
            point_cloud[i, 4] = snr * 0.1
            point_cloud[i, 5] = noise * 0.1
        output_dict['pointCloud'] = point_cloud

class SphericalPointCloudTLVParser(BaseParser):
    @staticmethod
    def spherical2cartersian(spherical_point_cloud: "np.ndarray[tuple[int, 3]]"):
        shape = spherical_point_cloud.shape
        cartesian_point_cloud = spherical_point_cloud.copy()
        if shape[1] < 3:
            log.error('Error: Failed to convert spherical point cloud to cartesian due to numpy array with too few dimensions')
            return cartesian_point_cloud
        
        # X = Range * sin (azimuth) * cos (elevation)
        cartesian_point_cloud[:, 0] =\
            spherical_point_cloud[:,0] * np.sin(spherical_point_cloud[:,1]) * np.cos(spherical_point_cloud[:,2])
        # Y = Range * cos (azimuth) * cos (elevation)
        cartesian_point_cloud[:, 1] =\
            spherical_point_cloud[:,0] * np.cos(spherical_point_cloud[:,1]) * np.cos(spherical_point_cloud[:,2])
        # Z = Range * sin (elevation)
        cartesian_point_cloud[:, 2] =\
            spherical_point_cloud[:,0] * np.sin(spherical_point_cloud[:,2])
        return cartesian_point_cloud
    
    @staticmethod
    def parse(tlv_data, tlv_length, output_dict):
        point_cloud = output_dict['pointCloud']
        point_struct = '4f'  # Range, Azimuth, Elevation, and Doppler
        point_struct_size = struct.calcsize(point_struct)
        num_points = int(tlv_length / point_struct_size)
        
        for i in range(num_points):
            try:
                rng, azimuth, elevation, doppler = struct.unpack(point_struct, tlv_data[:point_struct_size])
            except:
                num_points = i
                log.error("Spherical Point Cloud TLV Parser Failed")
                break
            tlv_data = tlv_data[point_struct_size:]
            point_cloud[i, 0] = rng
            point_cloud[i, 1] = azimuth
            point_cloud[i, 2] = elevation
            point_cloud[i, 3] = doppler
        
        point_cloud[:, :3] = SphericalPointCloudTLVParser.spherical2cartersian(point_cloud[:, :3])
        output_dict['numDetectedPoints'], output_dict['pointCloud'] = num_points, point_cloud
        
class CompressedSphericalPointCloudTLVParser(SphericalPointCloudTLVParser):
    @staticmethod
    def parse(tlv_data, tlv_length, output_dict):
        point_cloud = output_dict['pointCloud']
        p_unit_struct = '5f'
        point_struct = '2bh2H'
        p_unit_size = struct.calcsize(p_unit_struct)
        point_size = struct.calcsize(point_struct)
        
        try:
            p_unit = struct.unpack(p_unit_struct, tlv_data[:p_unit_size])
        except:
            log.error("Point Cloud TLV Parser Failed")
            output_dict['numDetectedPoints'], output_dict['pointCloud'] = 0, point_cloud
        tlv_data = tlv_data[p_unit_size:]
        
        num_points = int((tlv_length - p_unit_size) / point_size)
        for i in range(num_points):
            try:
                (elevation, 
                 azimuth,
                 doppler,
                 rng,
                 snr) = struct.unpack(point_struct, tlv_data[:point_size])
            except:
                num_points = i
                log.error("Point Cloud TLV Parser Failed")
                break
                
            tlv_data = tlv_data[point_size:]
            if azimuth >= 128:
                log.error("Azimuth greater than 127")
                azimuth -= 256
            if elevation >= 128:
                log.error("Elevation greater than 127")
                elevation -= 256
            if doppler >= 32768:
                log.error("Doppler greater than 32767")
                doppler -= 65536

            point_cloud[i, 0] = rng * p_unit[3]
            point_cloud[i, 1] = azimuth * p_unit[1]
            point_cloud[i, 2] = elevation * p_unit[0]
            point_cloud[i, 3] = doppler * p_unit[2]
            point_cloud[i, 4] = snr * p_unit[4]
        
        point_cloud[:, :3] = CompressedSphericalPointCloudTLVParser.spherical2cartersian(
            point_cloud[:, :3])
        output_dict['numDetectedPoints'], output_dict['pointCloud'] = num_points, point_cloud
        
    
        
TLV2PARSER: Dict[TLVTYPES, BaseParser] = {
    TLVTYPES.MMWDEMO_OUTPUT_MSG_DETECTED_POINTS: PointCloudTLVParser,
    TLVTYPES.MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS: SphericalPointCloudTLVParser,
    TLVTYPES.MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO: SideInfoTLVParser,
    TLVTYPES.MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS: CompressedSphericalPointCloudTLVParser,
}

def tlv2parser(tlv_type: int) -> Callable:
    try:
        return TLV2PARSER[TLVTYPES(tlv_type)].parse
    except ValueError:
        # log.error(f"{tlv_type} is not a valid TLV type")
        return None
    except KeyError:
        log.info(f"TLV type {tlv_type} not found in TLV2PARSER")
        return None