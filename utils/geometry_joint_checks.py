"""Strict geometry gate for the new joint model; existing FFD/metrics unchanged.

Neighbor-difference bounds certify each shared spline map is injective in the
convex support box (||D displacement||_2 < 1). With a checked nonintersecting
source template, compositions cannot introduce within-chamber self crossings.
Actual output triangles and sampled composed Jacobians are checked separately.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree
from net.geometry_heartffdnet import control_lipschitz_bound
from utils.ffd_physical import bspline_ffd_physical
from data.mask_transform import CHAMBERS


def triangle_self_intersections(vertices, faces):
    """Broad phase KD-tree/AABB and triangle separating-axis test.

    Excludes topological neighbors. Includes coplanar overlap axes, treats
    non-neighbor touching as intersection, and never swallows failures.
    """
    triangles = np.asarray(vertices,dtype=np.float64)[np.asarray(faces)]
    centers = triangles.mean(1)
    radius = np.linalg.norm(triangles-centers[:,None],axis=-1).max(1)
    pairs = cKDTree(centers).query_pairs(2*radius.max(),output_type='ndarray')
    count = 0
    for piece in np.array_split(pairs,max(1,len(pairs)//20000+1)):
        if not len(piece):
            continue
        a,b = triangles[piece[:,0]],triangles[piece[:,1]]
        indices_a,indices_b = faces[piece[:,0]],faces[piece[:,1]]
        mask = ~(indices_a[:,:,None] == indices_b[:,None,:]).any((1,2))
        mask &= (a.min(1) <= b.max(1)+1e-9).all(1) & (b.min(1) <= a.max(1)+1e-9).all(1)
        a,b = a[mask],b[mask]
        if not len(a):
            continue
        ea,eb = np.roll(a,-1,axis=1)-a,np.roll(b,-1,axis=1)-b
        na,nb = np.cross(ea[:,0],ea[:,1]),np.cross(eb[:,0],eb[:,1])
        axes = np.concatenate((na[:,None],nb[:,None],
            np.cross(ea[:,:,None],eb[:,None,:]).reshape(-1,9,3),
            np.cross(na[:,None],ea),np.cross(nb[:,None],eb)),1)
        norms = np.linalg.norm(axes,axis=-1)
        axes = axes/np.maximum(norms[:,:,None],1e-20)
        pa,pb = np.einsum('nvc,nac->nav',a,axes),np.einsum('nvc,nac->nav',b,axes)
        separate = ((pa.max(-1) < pb.min(-1)-1e-9) | (pb.max(-1) < pa.min(-1)-1e-9)) & (norms>1e-12)
        count += int((~separate.any(1)).sum())
    return count


def validate_template(model):
    result = {}
    for c,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist()):
        v = model.template_vertices_mm[a:b].detach().cpu().numpy()
        f = model.faces[c].cpu().numpy()
        crossings = triangle_self_intersections(v,f)
        if crossings:
            raise ValueError(f'Source template self intersects in {c}: {crossings} non-neighbor pairs')
        result[c] = dict(self_intersecting_non_neighbor_pairs=crossings)
    model._template_self_intersection_checked = True
    return result


@torch.no_grad()
def validate_prediction(model, output, *, sample_jacobian=False, check_intersections=False,
                        reference_only=False):
    """Legacy24-phase audit, or explicit T=1 reference audit using the same core.

    Reference mode rejects dynamic controls/trajectories rather than truncating
    them. It audits exactly coarse -> medium -> fine0, without padded zero maps.
    """
    if not getattr(model,'_template_self_intersection_checked',False):
        raise ValueError('Source template intersection check required before geometry certification')
    vertices = output['vertices_4d'].detach()
    frames = 1 if reference_only else 24
    if reference_only and (vertices.ndim != 4 or vertices.shape[1] != 1
                           or output['control_grids']['fine'].ndim != 6
                           or output['control_grids']['fine'].shape[1] != 1):
        raise ValueError('Reference-only audit requires T=1 vertices and fine0 only')
    if (not torch.isfinite(vertices).all() or ((vertices < model.lower_mm) | (vertices > model.upper_mm)).any()):
        raise ValueError('Invalid finite/support geometry')
    degenerates = 0
    minimum_area = float('inf')
    for c,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist()):
        tri = vertices[:,:,a:b][:,:,model.faces[c]]
        area = torch.cross(tri[...,1,:]-tri[...,0,:],tri[...,2,:]-tri[...,0,:],dim=-1).norm(dim=-1)
        degenerates += int((area < 1e-8).sum())
        minimum_area = min(minimum_area,float(area.min()))
    if degenerates:
        raise ValueError('Degenerate output mesh triangles')
    controls = output['control_grids']
    certificate = getattr(model,'geometry_certificate','independent-global-columns-v1')
    certificate_details = {}
    if certificate == 'independent-global-columns-v1':
        bounds = [control_lipschitz_bound(controls[k],model.control_spacings_mm[i])
                  for i,k in enumerate(('coarse','medium','fine'))]
    elif certificate == 'bernstein-frobenius-reference-v1':
        # Explicit new-stage contract. The old estimator remains visible; the
        # declared blended GLOBAL bound is audited independently in float64.
        from utils.bernstein_certificate import bernstein_frobenius_bound
        alpha = getattr(model.heads,'certificate_alpha',float('nan'))
        if (not np.isfinite(alpha) or not 0<=alpha<=1 or model.heads.max_lipschitz != .45
                or vertices.ndim!=4 or vertices.shape[1]!=1 or controls['fine'].shape[1]!=1):
            raise ValueError('Bernstein certificate requires declared alpha,0.45 and Shape-only output')
        old = [control_lipschitz_bound(controls[k].double(),model.control_spacings_mm[i].double())
               for i,k in enumerate(('coarse','medium','fine'))]
        bernstein = [bernstein_frobenius_bound(controls[k].double(),model.control_spacings_mm[i].double())
                     for i,k in enumerate(('coarse','medium','fine'))]
        bounds = [(1-alpha)*a+alpha*b for a,b in zip(old,bernstein)]
        certificate_details = dict(certificate=certificate,certificate_alpha=alpha,
            max_legacy_global_bound=max(float(x.max()) for x in old),
            max_bernstein_bound=max(float(x.max()) for x in bernstein),
            certificate_arithmetic='float64, original stored physical lattice')
        if any(float(value.abs().max())>model.heads.reference_limit_mm+1e-5 for value in controls.values()):
            raise ValueError('Bernstein reference displacement limit exceeded')
    else:
        raise ValueError('Unknown geometry certificate contract: '+str(certificate))
    maximum = max(float(bound.max()) for bound in bounds)
    if not np.isfinite(maximum) or maximum > model.heads.max_lipschitz+1e-5:
        raise ValueError('Spline injectivity certificate failed')
    result = dict(status='pass',degenerate_triangles=degenerates,min_twice_triangle_area_mm2=minimum_area,
        max_displacement_lipschitz_bound=maximum,
        self_intersection_policy='Within-chamber injectivity certificate composed from checked source template')
    result.update(certificate_details)
    if check_intersections:
        crossings = 0
        for patient in range(len(vertices)):
            for phase in range(frames):
                for c,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist()):
                    crossings += triangle_self_intersections(vertices[patient,phase,a:b].cpu().numpy(),
                                                              model.faces[c].cpu().numpy())
        if crossings:
            raise ValueError(f'Predicted triangle mesh self intersects: {crossings} non-neighbor pairs')
        result['actual_mesh_self_intersecting_pairs'] = crossings
    if sample_jacobian:
        indices = torch.linspace(0,len(model.template_vertices_mm)-1,64,device=vertices.device).long()
        p = model.template_vertices_mm[indices][None].expand(len(vertices),-1,-1).clone()
        jac = torch.eye(3,device=p.device,dtype=p.dtype).expand(len(p),len(indices),3,3).clone()
        minimum = float('inf')
        fields = [(controls['coarse'],0),(controls['medium'],1)]+[(controls['fine'][:,t],2) for t in range(frames)]
        epsilon = .02
        for field,level in fields:
            columns = []
            for axis in range(3):
                delta = torch.zeros_like(p);delta[...,axis] = epsilon
                plus = bspline_ffd_physical(p+delta,field,model.control_origins_mm[level],model.control_spacings_mm[level])
                minus = bspline_ffd_physical(p-delta,field,model.control_origins_mm[level],model.control_spacings_mm[level])
                columns.append((plus-minus)/(2*epsilon))
            derivative = torch.stack(columns,-1)+torch.eye(3,device=p.device,dtype=p.dtype)
            jac = derivative@jac
            minimum = min(minimum,float(torch.linalg.det(jac).min()))
            p = p+bspline_ffd_physical(p,field,model.control_origins_mm[level],model.control_spacings_mm[level])
        if minimum <= 0 or not np.isfinite(minimum):
            raise ValueError('Sampled composed Jacobian nonpositive/nonfinite')
        result['min_sampled_composed_jacobian'] = minimum
    return result
