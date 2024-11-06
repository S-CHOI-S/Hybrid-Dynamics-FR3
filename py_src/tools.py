from typing import Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import torch

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0
def orientation_6d_to_euler(r_6d) -> np.ndarray:
    v1 = r_6d[:3]
    v2 = r_6d[3:]
    # Normalize vectors
    col1 = v1 / np.linalg.norm(v1)
    col2 = v2 / np.linalg.norm(v2)

    # Find their orthogonal vector via cross product
    col3 = np.cross(col1, col2)
    euler = R.from_matrix(np.array([col1, col2, col3]).T).as_euler('xyz', degrees=False)

    return euler

def orientation_6d_to_euler_tensor(r_6d,batch_size=256) -> torch.tensor:
    v1 = r_6d[:,:3]
    v2 = r_6d[:,3:]
    # Normalize vectors
    col1 = v1 / torch.linalg.norm(v1)
    col2 = v2 / torch.linalg.norm(v2)

    # Find their orthogonal vector via cross product
    col3 = torch.cross(col1, col2)

    # Stack into rotation matrix as columns, convert to quaternion and return
    Ro = torch.stack((col1.unsqueeze(2), col2.unsqueeze(2), col3.unsqueeze(2)), dim=1)
    Ro = Ro.reshape(batch_size, 3, 3)
    Ro = Ro.transpose(1, 2)

    # Stack into rotation matrix as columns, convert to quaternion and return
    euler = R.from_matrix(Ro.cpu()).as_euler('xyz', degrees=False)
    return torch.tensor(euler)


def orientation_6d_to_matrix(r_6d) -> np.ndarray:
    v1 = r_6d[:3]
    v2 = r_6d[3:]
    # Normalize vectors
    col1 = v1 / np.linalg.norm(v1)
    col2 = v2 / np.linalg.norm(v2)

    # Find their orthogonal vector via cross product
    col3 = np.cross(col1, col2)

    # Stack into rotation matrix as columns, convert to quaternion and return
    return R.from_matrix(np.array([col1, col2, col3]).T)

def orientation_6d_to_quat(
    r_6d: Union[np.ndarray, Tuple[float, float, float, float]], use: str
) -> np.ndarray:
    v1 = r_6d[:3]
    v2 = r_6d[3:]
    # Normalize vectors
    col1 = v1 / np.linalg.norm(v1)
    col2 = v2 / np.linalg.norm(v2)

    # Find their orthogonal vector via cross product
    col3 = np.cross(col1, col2)

    # Stack into rotation matrix as columns, convert to quaternion and return
    quat_xyzw = R.from_matrix(np.array([col1, col2, col3]).T).as_quat()

    if use == 'mujoco':
        return xyzw2quat(quat_xyzw)
    elif use == 'scipy':
        return quat_xyzw
    else:
        return None

def orientation_quat_to_6d(
    quat: Tuple[float, float, float, float], use: str
) -> np.ndarray:

    # Convert quaternion into rotation matrix
    if use == 'mujoco':
        rot_mat = R.from_quat(quat2xyzw(quat)).as_matrix()
    elif use == 'scipy':
        rot_mat = R.from_quat(quat).as_matrix()
    # Return first two columns (already normalised)
    return np.concatenate([rot_mat[:,0], rot_mat[:,1]])

def orientation_euler_to_6d(rpy):
    r = R.from_euler('xyz', rpy, degrees=False)
    r6d = orientation_quat_to_6d(r.as_quat(), 'scipy')
    return r6d
def xyzw2quat(
    xyzw: Union[np.ndarray, Tuple[float, float, float, float]]
) -> np.ndarray:

    if isinstance(xyzw, tuple):
        return (xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    return xyzw[[3, 0, 1, 2]]


def quat2xyzw(
    quat: Union[np.ndarray, Tuple[float, float, float, float]]
) -> np.ndarray:

    # if isinstance(quat, tuple):
    #     return [quat[1], quat[2], quat[3], quat[0]]

    return [quat[1], quat[2], quat[3], quat[0]]
def quat2xyzw_high(quat) -> np.ndarray:

    # if isinstance(quat, tuple):
    #     return [quat[1], quat[2], quat[3], quat[0]]
    xyzw = quat.copy()  # Create a copy to avoid modifying the original array
    xyzw[:, 0], xyzw[:, 1] ,xyzw[:, 2], xyzw[:, 3]= quat[:, 1], quat[:, 2],quat[:, 3], quat[:, 0]

    return xyzw


def name2id(m, type, name_list):
    id_list = []
    for name in name_list:
        id_list.append(mujoco.mj_name2id(m, type, name))
    return id_list

def detect_contact(contact_data, desired_contact_bid):
    geom1 = contact_data.geom1
    geom2 = contact_data.geom2

    contact_list = []
    for i in range(len(geom1)):
        if geom1[i] in desired_contact_bid and geom2[i] in desired_contact_bid:
            contact_list.append(1)
        else:
            contact_list.append(-1)
    return contact_list
def detect_grasp(contact_data, finger, obj):
    geom1 = contact_data.geom1
    geom2 = contact_data.geom2
    grasp_list = []
    pairs = list(zip(geom1, geom2))

    pair_counts = {}
    for idx, pair in enumerate(pairs):
        if pair in pair_counts:
            pass
        else:
            pair_counts[pair] = idx
    non_duplicated_indices = list(pair_counts.values())
    geom1 = geom1[non_duplicated_indices]
    geom2 = geom2[non_duplicated_indices]
    for i in range(len(geom1)):
        if geom1[i] in finger:
            if geom2[i] in obj:
                grasp_list.append(1)
        if geom2[i] in finger:
            if geom1[i] in obj:
                grasp_list.append(1)
    return grasp_list

def detect_q_operation(qpos, q_range):
    q_operation_list = []
    for i in range(7):
        if q_range[i][0] <= qpos[i] <= q_range[i][1]:
            q_operation_list.append(1)
        else:
            # out of bound
            q_operation_list.append(-1)
    return q_operation_list

def calc_manipulability(jacobian):
    # A = np.linalg.svd(jacobian @ jacobian.T, compute_uv=False)
    A = jacobian @ jacobian.T
    detA = np.linalg.det(A)
    return np.sqrt(detA)

def scaling_minmax(unscaled_v, range_from, range_to):
    if isinstance(range_from, list):
        # range_from = np.reshape(range_from, (1,2))
        range_from = np.array([range_from,range_from,range_from,range_from,range_from,range_from,range_from])
    if isinstance(range_to, list):
        # range_from = np.reshape(range_from, (1,2))
        range_to = np.array([range_to,range_to,range_to,range_to,range_to,range_to,range_to])
    # if not isinstance(range_to, tuple):
    #     range_to = tuple(range_to)


    # scaled_v = (unscaled_v - range_from[:, 0]) / (range_from[:, 1] - range_from[:, 0]) * (range_to[1] - range_to[0]) + range_to[0]
    scaled_v = (unscaled_v - range_from[:, 0]) / (range_from[:, 1] - range_from[:, 0]) * (range_to[:,1] - range_to[:,0]) + \
               range_to[:,0]

    # q = (q_unscaled - self.q_range[:, 0]) / (self.q_range[:, 1] - self.q_range[:, 0]) * (1 - (-1)) - 1
    return scaled_v

def range_margin(range, margin):
    new_range = np.zeros(range.shape)
    margin_ = (range[:,1] - range[:,0]) * margin
    new_range[:,1] = range[:,1] - margin_
    new_range[:,0] = range[:,0] + margin_

    return new_range


def camera():
    scn = mujoco.MjvScene
    cam = mujoco.MjvCamera
    opt = mujoco.MjvOption

def get_T_ab(T_a, T_b):
    """
    Calculate the transformation matrix of T_b relative to T_a.

    Parameters:
        T_a: numpy array, transformation matrix T_a
        T_b: numpy array, transformation matrix T_b

    Returns:
        T_rel: numpy array, transformation matrix of T_b relative to T_a
    """
    # Calculate the inverse of T_a
    T_a_inv = np.linalg.inv(T_a)

    # Calculate the relative transformation matrix
    T_rel = np.dot(T_a_inv, T_b)

    return T_rel