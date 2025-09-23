import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def rotation_transfer_matrix_to_6D_batch(rotate_matrix):

    # rotate_matrix.shape = (B, 9) or (B x 3, 3) rotation transpose (i.e., row vectors instead of column vectors)
    # return shape = (B, 6)

    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(-1, 9)
    rotate_matrix = rotate_matrix.reshape(-1, 9)

    return rotate_matrix[:,:6]

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def rotation_transfer_6D_to_matrix_batch(orient):

    # orient shape = (B, 6)
    # return shape = (3, B * 3)

    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)
    
    assert orient.shape[-1] == 6

    orient = orient.reshape(-1, 2, 3)
    a1 = orient[:,0]
    a2 = orient[:,1]

    b1 = a1 / np.linalg.norm(a1, axis=-1).reshape(-1,1)
    b2 = a2 - (np.sum(a2*b1, axis=-1).reshape(-1,1) * b1)
    b2 = b2 / np.linalg.norm(b2, axis=-1).reshape(-1,1)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.hstack((b1, b2, b3))
    rotate_matrix = rotate_matrix.reshape(-1, 3).T

    return rotate_matrix

original_gripper_pcd = np.array([[ 0.10432111,  0.00228697,  0.8474241 ],
       [ 0.12816067, -0.04368229,  0.8114649 ],
       [ 0.08953098,  0.0484529 ,  0.80711854],
       [ 0.11198021,  0.00245327,  0.7828771 ]])
original_gripper_pos = np.array([0.1119802 , 0.00245327, 0.78287711])
original_gripper_orn = np.array([0.97841681, 0.19802945, 0.0581003 , 0.01045192])

def compute_plane_normal(gripper_pcd):
    x1 = gripper_pcd[0]
    x2 = gripper_pcd[1]
    x4 = gripper_pcd[3]
    v1 = x2 - x1
    v2 = x4 - x1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

original_gripper_normal = compute_plane_normal(original_gripper_pcd)

def quaternion_to_rotation_matrix(quat):
    rotation = R.from_quat(quat)
    return rotation.as_matrix()

def rotation_matrix_to_quaternion(R_opt):
    rotation = R.from_matrix(R_opt)
    return rotation.as_quat()

def rotation_matrix_from_vectors(v1, v2):
    """
    Find the rotation matrix that aligns v1 to v2
    :param v1: A 3d "source" vector
    :param v2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to v1, aligns it with v2.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0:
        axis = axis / axis_len
    dot_prod = np.clip(np.dot(v1,v2),-1,1)
    angle = np.arccos(dot_prod)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def get_gripper_pos_orient_from_4_points(gripper_pcd):
    normal = compute_plane_normal(gripper_pcd)
    R1 = rotation_matrix_from_vectors(original_gripper_normal, normal)
    v1 = original_gripper_pcd[3] - original_gripper_pcd[0]
    v2 = gripper_pcd[3] - gripper_pcd[0]
    v1_prime = np.dot(R1, v1)
    R2 = rotation_matrix_from_vectors(v1_prime, v2)
    R = np.dot(R2, R1)
    gripper_pos = original_gripper_pos + gripper_pcd[3] - original_gripper_pcd[3]
    original_R = quaternion_to_rotation_matrix(original_gripper_orn)
    R = np.dot(R, original_R)
    gripper_orn = rotation_matrix_to_quaternion(R)
    return gripper_pos, gripper_orn

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn):
    goal_R = R.from_quat(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd

closed_45448_gripper_pcd = np.array([[ 0.6510935 , -0.14546424,  0.6576897 ],
       [ 0.6809345 , -0.15275753,  0.62872416],
       [ 0.6931994 , -0.13693176,  0.6597541 ],
       [ 0.7114738 , -0.14507635,  0.6336259 ]])
closed_45448_gripper_pos = np.array([ 0.71147382, -0.14507634,  0.63362595])
closed_45448_gripper_orn = np.array([ 0.70043311, -0.44102076,  0.47293441, -0.30203838])

def get_4_points_from_closed_45448_gripper_pos_orient(gripper_pos, gripper_orn):
    goal_R = R.from_quat(gripper_orn)
    original_R = R.from_quat(closed_45448_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = closed_45448_gripper_pcd - closed_45448_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd


# now I want to make get_gripper_pos_orient_from_4_points support the torch tensor with gripper_pcd (B, 4, 3) as input
def compute_plane_normal_torch(gripper_pcd):
    x1 = gripper_pcd[:, 0]
    x2 = gripper_pcd[:, 1]
    x4 = gripper_pcd[:, 3]
    v1 = x2 - x1
    v2 = x4 - x1
    normal = torch.cross(v1, v2)
    return normal / torch.norm(normal, dim=1, keepdim=True)

def rotation_matrix_from_vectors_torch(v1, v2):
    v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
    
    # Compute the axis of rotation
    axis = torch.cross(v1, v2, dim=1)
    axis_len = torch.norm(axis, dim=1, keepdim=True)
    
    # Avoid division by zero
    axis = axis / torch.clamp(axis_len, min=1e-9)
    
    # Compute the angle of rotation
    angle = torch.acos(torch.clamp(torch.sum(v1 * v2, dim=1), -1.0, 1.0))

    # Create the skew-symmetric cross-product matrix for the axis
    K = torch.zeros((v1.shape[0], 3, 3), device=v1.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    eye = torch.eye(3, device=v1.device).unsqueeze(0).repeat(v1.shape[0], 1, 1)
    angle = angle.view(-1, 1, 1)  # Reshape for broadcasting
    K_dot_K = torch.matmul(K, K)
    R = eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * K_dot_K
    
    return R

def quaternion_to_rotation_matrix_torch(quat):
    rotation = R.from_quat(quat)
    return torch.tensor(rotation.as_matrix())

def rotation_matrix_to_quaternion_torch(R_opt):
    ret_ = []
    for i in range(R_opt.shape[0]):
        rotation = R.from_matrix(R_opt[i].cpu().numpy())
        ret_.append(rotation.as_quat())
    ret_ = np.array(ret_)
    return torch.tensor(ret_).cuda().float()

def get_gripper_pos_orient_from_4_points_torch(gripper_pcd):
    normal = compute_plane_normal_torch(gripper_pcd).float()
    original_gripper_normal = compute_plane_normal(original_gripper_pcd)
    original_gripper_normal = torch.tensor(original_gripper_normal).unsqueeze(0).repeat(gripper_pcd.shape[0], 1).cuda().float()
    R1 = rotation_matrix_from_vectors_torch(original_gripper_normal, normal.cuda())
    v1 = gripper_pcd[:, 3] - gripper_pcd[:, 0]
    v2 = original_gripper_pcd[3] - original_gripper_pcd[0]
    v2 = torch.tensor(v2).unsqueeze(0).repeat(gripper_pcd.shape[0], 1).cuda().float()
    v1_prime = torch.matmul(R1, v1.unsqueeze(-1).cuda()).squeeze(-1)
    R2 = rotation_matrix_from_vectors_torch(v1_prime, v2)
    R = torch.matmul(R2, R1)
    gripper_pos = torch.tensor(original_gripper_pos).unsqueeze(0).repeat(gripper_pcd.shape[0], 1).cuda().float() + gripper_pcd[:, 3].cuda() - torch.tensor(original_gripper_pcd[3]).unsqueeze(0).repeat(gripper_pcd.shape[0], 1).cuda().float()
    original_R = quaternion_to_rotation_matrix_torch(original_gripper_orn).cuda().float()
    R = torch.matmul(R, original_R)
    gripper_orn = rotation_matrix_to_quaternion_torch(R)
    return gripper_pos, gripper_orn

def gripper_pcd_to_10d_vector_torch(gripper_pcd, is_open=False):
    device = gripper_pcd.device
    gripper_pcd = gripper_pcd.cpu()
    all_representations = []
    for pcd in gripper_pcd:
        gripper_pos, gripper_orn = get_gripper_pos_orient_from_4_points_torch(pcd)
        vec_shape = tuple(gripper_pos.shape)
        vec_shape = (*vec_shape[:-1], 1)
        if is_open:
            grip_state = torch.zeros(vec_shape, device=device)
        else: 
            grip_state = torch.ones(vec_shape, device=device)
        gripper_rot_matrix = quaternion_to_rotation_matrix_torch(gripper_orn.cpu()).cuda()
        gripper_6d_pose = rotation_transfer_matrix_to_6D_batch(gripper_rot_matrix)
        representation = torch.concatenate([gripper_pos, gripper_6d_pose, grip_state], axis=-1)
        all_representations.append(representation)
    #all_representations = torch.stack(all_representations, device=device)
    all_representations = torch.stack(all_representations).to(device)
    return all_representations

def gripper_pcd_to_10d_vector(gripper_pcd, is_open=False):
    all_representations = []
    for pcd in gripper_pcd:
        gripper_pos, gripper_orn = get_gripper_pos_orient_from_4_points(pcd)
        vec_shape = gripper_pos.shape
        vec_shape = (*vec_shape[:-1], 1)
        if is_open:
            grip_state = np.zeros(vec_shape)
        else: 
            grip_state = np.ones(vec_shape)
        gripper_rot_matrix = quaternion_to_rotation_matrix(gripper_orn)
        gripper_6d_pose = rotation_transfer_matrix_to_6D(gripper_rot_matrix)
        representation = np.concatenate([gripper_pos, gripper_6d_pose, grip_state], axis=-1)
        all_representations.append(representation)
    all_representations = np.stack(all_representations).astype(np.float32)
    return all_representations
