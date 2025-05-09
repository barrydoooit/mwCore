import numpy as np

from .AsteriosTracking import AsteriosConfig, ConstAccModel
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise



def make_config_asterios(raw: dict) -> AsteriosConfig:
    def KF_F(dt):
        return np.array([
            [1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0, 0],
            [0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0],
            [0, 0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2)],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
    
    def KF_Q_DISCR(dt):
        return block_diag(
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
        )
    
    def STATE_VEC(init):
        return [*init[:6], 0, 0, 0]
    
    motion_model = ConstAccModel(
        KF_DIM=[9, 6],
        KF_H=np.eye(6, 9),
        KF_F=KF_F,
        KF_Q_DISCR=KF_Q_DISCR,
        STATE_VEC=STATE_VEC,
    )
    
    return AsteriosConfig(
        motion_model=motion_model,
        FB_FRAMES_BATCH=raw["FB_FRAMES_BATCH"],
        FB_FRAMES_BATCH_STATIC=raw.get("FB_FRAMES_BATCH_STATIC"),
        DB_POINTS_THRES=raw.get("DB_POINTS_THRES"),
        DB_SPREAD_THRES=raw.get("DB_SPREAD_THRES"),
        DB_EPS=raw.get("DB_EPS"),
        DB_RANGE_WEIGHT=raw.get("DB_RANGE_WEIGHT"),
        DB_Z_WEIGHT=raw.get("DB_Z_WEIGHT"),
        DB_MIN_SAMPLES_MIN=raw.get("DB_MIN_SAMPLES_MIN"),
        KF_R_STD=raw.get("KF_R_STD"),
        KF_P_INIT=raw.get("KF_P_INIT"),
        KF_GROUP_DISP_EST_INIT=raw.get("KF_GROUP_DISP_EST_INIT"),
        KF_ENABLE_EST=raw.get("KF_ENABLE_EST"),
        KF_A_N=raw.get("KF_A_N"),
        KF_EST_POINTNUM=raw.get("KF_EST_POINTNUM"),
        KF_SPREAD_LIM=raw.get("KF_SPREAD_LIM"),
        KF_A_SPR=raw.get("KF_A_SPR"),
        # MODEL_MIN_INPUT=raw.get("MODEL_MIN_INPUT"),
        # MODEL_DEFAULT_POSTURE=np.zeros(57),
        TR_LIFETIME_DYNAMIC=raw.get("TR_LIFETIME_DYNAMIC"),
        TR_LIFETIME_STATIC=raw.get("TR_LIFETIME_STATIC"),
        TR_GATE=raw.get("TR_GATE"),
        TR_MAX_TRACKS=raw.get("TR_MAX_TRACKS"),
        TR_VEL_THRES=raw.get("TR_VEL_THRES"),
    )

from .GTrack import ConstVelModel, GTrackConfig

def make_config_gtrack(raw: dict) -> GTrackConfig:
    def KF_F(dt):
        return np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

    def KF_Q_DISCR(dt):
        return block_diag(
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
            Q_discrete_white_noise(3, dt, var=raw['KF_Q_STD']),
        )

    const_vel_model = ConstVelModel(
        KF_DIM=[6, 6],
        KF_H=np.eye(6),
        KF_F=KF_F,
        KF_Q_DISCR=KF_Q_DISCR,
    )

    return GTrackConfig(
        const_vel_model=const_vel_model,
        KF_GROUP_DISP_EST_INIT=raw.get("KF_GROUP_DISP_EST_INIT"),
        KF_ENABLE_EST=raw.get("KF_ENABLE_EST"),
        KF_A_N=raw.get("KF_A_N"),
        KF_EST_POINTNUM=raw.get("KF_EST_POINTNUM"),
        KF_SPREAD_LIM=raw.get("KF_SPREAD_LIM"),
        KF_A_SPR=raw.get("KF_A_SPR"),
        DB_POINTS_THRES=raw.get("DB_POINTS_THRES"),
        DB_SPREAD_THRES=raw.get("DB_SPREAD_THRES"),
        FB_FRAMES_BATCH_STATIC=raw.get("FB_FRAMES_BATCH_STATIC"),
        FB_FRAMES_BATCH=raw.get("FB_FRAMES_BATCH"),
        DB_INNER_EPS=raw.get("DB_INNER_EPS"),
        DB_EPS=raw.get("DB_EPS"),
        DB_RANGE_WEIGHT=raw.get("DB_RANGE_WEIGHT"),
        DB_Z_WEIGHT=raw.get("DB_Z_WEIGHT"),
        DB_MIN_SAMPLES_MIN=raw.get("DB_MIN_SAMPLES_MIN"),
        KF_R_STD=raw.get("KF_R_STD"),
        KF_P_INIT=raw.get("KF_P_INIT"),
        # MODEL_DEFAULT_POSTURE=np.zeros(57),
        TR_LIFETIME_DYNAMIC=raw.get("TR_LIFETIME_DYNAMIC"),
        TR_LIFETIME_STATIC=raw.get("TR_LIFETIME_STATIC"),
        TR_GATE=raw.get("TR_GATE"),
        TR_MAX_TRACKS=raw.get("TR_MAX_TRACKS"),
        TR_VEL_THRES=raw.get("TR_VEL_THRES"),
    )

from .RKFTracking import RKFConfig

def make_config_rkf(raw: dict) -> RKFConfig:
    def RKF_F(dt):
        return np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1],
        ])

    def RKF_Q(dt):
        return np.array([
            [0.2, 0, 0, 0],
            [0, 0.2, 0, 0],
            [0, 0, 0.2, 0],
            [0, 0, 0, 0.2]
        ])

    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])

    return RKFConfig(
        RKF_F=RKF_F,
        RKF_Q=RKF_Q,
        H=H,
        KF_GROUP_DISP_EST_INIT=raw.get("KF_GROUP_DISP_EST_INIT"),
        KF_ENABLE_EST=raw.get("KF_ENABLE_EST"),
        KF_A_N=raw.get("KF_A_N"),
        KF_EST_POINTNUM=raw.get("KF_EST_POINTNUM"),
        KF_SPREAD_LIM=raw.get("KF_SPREAD_LIM"),
        KF_A_SPR=raw.get("KF_A_SPR"),
        DB_POINTS_THRES=raw.get("DB_POINTS_THRES"),
        DB_SPREAD_THRES=raw.get("DB_SPREAD_THRES"),
        DB_EPS=raw.get("DB_EPS"),
        DB_RANGE_WEIGHT=raw.get("DB_RANGE_WEIGHT"),
        DB_Z_WEIGHT=raw.get("DB_Z_WEIGHT"),
        DB_MIN_SAMPLES_MIN=raw.get("DB_MIN_SAMPLES_MIN"),
        FB_FRAMES_BATCH=raw.get("FB_FRAMES_BATCH"),
        KF_R_STD=raw.get("KF_R_STD"),
        # MODEL_DEFAULT_POSTURE=np.zeros(57),
        TR_LIFETIME_DYNAMIC=raw.get("TR_LIFETIME_DYNAMIC"),
        TR_LIFETIME_STATIC=raw.get("TR_LIFETIME_STATIC"),
        TR_GATE=raw.get("TR_GATE"),
        TR_MAX_TRACKS=raw.get("TR_MAX_TRACKS"),
        TR_VEL_THRES=raw.get("TR_VEL_THRES"),
    )