import torch
import torch.nn.functional as F
import numpy as np


def adjacent_faces(face):
    """
    Find the adjacent two faces for each edge
    
    Inputs:
    - face: mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - adj_faces: indices of two adjacent two faces
    for each edge, (|E|, 2) torch.LongTensor
    
    """
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], axis=0)  # (2|E|, 2)
    nf = face.shape[1]
    # map the edge to its belonging face
    fid = torch.arange(nf).to(face.device)
    adj_faces = torch.cat([fid]*3)  # (3|F|)

    edge = edge.cpu().numpy()
    # sort the edge such that v_i < v_j
    edge = np.sort(edge, axis=-1)
    # sort the edge to find the correspondence 
    # between e_ij and e_ji
    eid = np.lexsort((edge[:,1], edge[:,0]))  # (2|E|)

    # map edge to its adjacent two faces
    adj_faces = adj_faces[eid].reshape(-1,2)  # (|E|, 2)
    return adj_faces


def vert_normal(vert, face):
    """
    Compute the normal vector of each vertex.
    
    This function is retrieved from pytorch3d.
    For original code please see: 
    _compute_vertex_normals function in
    https://pytorch3d.readthedocs.io/en/latest/
    _modules/pytorch3d/structures/meshes.html
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - v_normal: vertex normals, (1,|V|,3) torch.Tensor
    """
    
    v_normal = torch.zeros_like(vert)   # normals of vertices
    v_f = vert[:, face[0]]   # vertices of each face

    # compute normals of faces
    f_normal_0 = torch.cross(v_f[:,:,1]-v_f[:,:,0], v_f[:,:,2]-v_f[:,:,0], dim=2) 
    f_normal_1 = torch.cross(v_f[:,:,2]-v_f[:,:,1], v_f[:,:,0]-v_f[:,:,1], dim=2) 
    f_normal_2 = torch.cross(v_f[:,:,0]-v_f[:,:,2], v_f[:,:,1]-v_f[:,:,2], dim=2) 

    # sum the faces normals
    v_normal = v_normal.index_add(1, face[0,:,0], f_normal_0)
    v_normal = v_normal.index_add(1, face[0,:,1], f_normal_1)
    v_normal = v_normal.index_add(1, face[0,:,2], f_normal_2)

    v_normal = v_normal / (torch.norm(v_normal, dim=-1).unsqueeze(-1) + 1e-12)
    return v_normal


def face_normal(vert, face):
    """
    Compute the normal vector of each face.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - f_normal: face normals, (1,|F|,3) torch.Tensor
    """
    
    v_f = vert[:, face[0]]
    # compute normals of faces
    f_normal = torch.cross(v_f[:,:,1]-v_f[:,:,0],
                           v_f[:,:,2]-v_f[:,:,0], dim=2) 
    f_normal = f_normal / (torch.norm(f_normal, dim=-1).unsqueeze(-1) + 1e-12)

    return f_normal
    
    
def mesh_area(vert, face):
    """
    Compute the total area of the mesh

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area: mesh area, float
    """

    v0 = vert[:,face[0,:,0]]
    v1 = vert[:,face[0,:,1]]
    v2 = vert[:,face[0,:,2]]
    area = 0.5*torch.norm(torch.cross(v1-v0, v2-v0), dim=-1)
    return area.sum()

    
def laplacian(face):
    """
    Compute Laplacian matrix.
    
    Inputs:
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - L: Laplacian matrix, (1,|V|,|V|) torch.sparse.Tensor
    """
    nv = face.max().item()+1
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], dim=0).T
    # adjacency matrix A
    A = torch.sparse_coo_tensor(
        edge, torch.ones_like(edge[0]).float(), (nv, nv)).unsqueeze(0)

    # number of neighbors for each vertex
    degree = torch.sparse.sum(A, dim=-1).to_dense()[0]
    weight = 1./degree[edge[0]]
    # normalized adjacency matrix
    A_hat = torch.sparse_coo_tensor(
        edge, weight, (nv, nv)).unsqueeze(0)
    
    # normalized degree matrix, i.e., identity matrix
    # set the diagonal entries to one
    self_edge = torch.arange(nv)[None].repeat([2,1]).to(face.device)
    D_hat = torch.sparse_coo_tensor(
        self_edge, torch.ones_like(self_edge[0]).float(), (nv, nv)).unsqueeze(0)
    L = D_hat - A_hat
    return L


def laplacian_smooth(vert, face, lambd=1., n_iters=1):
    """
    Laplacian mesh smoothing.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - lambd: strength of mesh smoothing [0,1]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - lambd * L.bmm(vert)
    return vert


def taubin_smooth(vert, face, lambd=0.5, mu=-0.53, n_iters=1):
    """
    Taubin mesh smoothing.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - lambd: strength of mesh smoothing [0,1]
    - mu: strength of mesh smoothing [-1,0]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - mu * L.bmm(vert)
        vert = vert - lambd * L.bmm(vert)
    return vert


def deformable(vert, face, sdf, n_iters=200, h=0.05, gamma=5.0, verbose=False):
    '''
    A deformable model for explicit mesh deformation
    based on implicit surface representation.
    
    The initial surface deforms along the normal direction
    scaled by the signed distance function (SDF).
    
    Args:
    vert: initial vertices of the surface mesh
    face: initial faces of the surface mesh
    sdf: signed distance function volume
    niter: number of interations
    h: step size
    gamma: weight of the Laplacian smoothing / spring force
    
    Output:
    vert_deform: vertices of deformed mesh
    face_deform: faces of deformed mesh
    '''
    
    vert_deform = vert.clone()
    face_deform = face.clone()
    
    D1,D2,D3 = sdf[0,0].shape
    scale = torch.Tensor([D3-1, D2-1, D1-1]).to(vert.device)
    L = laplacian(face)

    for i in range(n_iters):
        normal_deform = vert_normal(vert_deform, face_deform)
        grid = vert_deform.unsqueeze(dim=-2).unsqueeze(dim=-2).flip(-1)
        grid = 2 * grid / scale - 1
        dist = F.grid_sample(
            sdf, grid, mode='bilinear', padding_mode='border',
            align_corners=True)[...,0,0].transpose(2,1)
        
        # deform along the normal direction scaled by the SDF value
        vert_deform -= h * (dist * normal_deform + gamma * L.bmm(vert_deform))
        if verbose:
            print('Distance error:', (dist**2).mean())
    return vert_deform, face_deform


# def curvature(vert, face, L):
#     normal = vert_normal(vert, face)
#     # \Laplacian x = -2Hn
#     curv = (L.bmm(vert) * normal).sum(-1) / 2
#     return curv

def curvature(vert, face, L):
    normal = vert_normal(vert, face)  # (B,N,3)
    # (B,N,3) --> (N,3,B) --> (1,N,3B)
    B, N, D = vert.shape
    vert_ = vert.permute(1,2,0).reshape(1,N,3*B)
    # (1,N,3B) --> (N,3,B) --> (B,N,3)
    lap = L.bmm(vert_).reshape(N,3,B).permute(2,0,1)
    # \laplacian x = -2Hn
    curv = (lap * normal).sum(-1) / 2
    return curv  # (B,N)
