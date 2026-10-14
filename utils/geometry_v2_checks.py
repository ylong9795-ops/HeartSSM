"""Independent four-chamber geometry audit; frozen v1 checks stay unchanged.

Surface contacts are reported separately from enclosed vertices with positive
boundary distance. Shared valve-cap conventions require an explicit contract;
this module never automatically exempts an intersecting chamber pair.
VTK native OBB collision/enclosed-point APIs, no copied third-party code.
"""
from itertools import combinations
import numpy as np
import trimesh
from data.mask_transform import CHAMBERS


class GeometrySafetyError(ValueError):
    def __init__(self,message,report):
        super().__init__(message)
        self.report=report


def polydata(vertices,faces=None):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.asarray(vertices,dtype=np.float64),deep=True))
    result=vtk.vtkPolyData();result.SetPoints(points)
    if faces is not None:
        f=np.asarray(faces,dtype=np.int64)
        cells=vtk.vtkCellArray()
        cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(f),3),f)).astype(np.int64).ravel(),deep=True))
        result.SetPolys(cells)
    return result


def chamber_pair_audit(vertices_a,faces_a,vertices_b,faces_b,*,boundary_tolerance_mm=1e-4):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    a,b=np.asarray(vertices_a),np.asarray(vertices_b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite chamber geometry')
    if ((a.max(0)<b.min(0))|(b.max(0)<a.min(0))).any():
        return dict(surface_contact_pairs=0,penetrating_vertices=0,maximum_penetration_mm=0.,directions={})
    pa,pb=polydata(a,faces_a),polydata(b,faces_b)
    collision=vtk.vtkCollisionDetectionFilter();collision.SetInputData(0,pa);collision.SetInputData(1,pb)
    collision.SetTransform(0,vtk.vtkTransform());collision.SetTransform(1,vtk.vtkTransform())
    collision.SetBoxTolerance(0.);collision.SetCellTolerance(1e-12);collision.SetCollisionModeToAllContacts();collision.Update()
    result=dict(surface_contact_pairs=int(collision.GetNumberOfContacts()),directions={})
    if result['surface_contact_pairs']:
        result['contact_face_pairs']=np.column_stack((
            vtk_to_numpy(collision.GetContactCells(0)),
            vtk_to_numpy(collision.GetContactCells(1)))).astype(int).tolist()
    else:
        result['contact_face_pairs']=[]
    for label,source,target in (('a_inside_b',a,pb),('b_inside_a',b,pa)):
        enclosure=vtk.vtkSelectEnclosedPoints();enclosure.SetInputData(polydata(source));enclosure.SetSurfaceData(target)
        enclosure.SetTolerance(1e-8);enclosure.CheckSurfaceOn();enclosure.Update()
        value=enclosure.GetOutput().GetPointData().GetArray('SelectedPoints')
        if value is None:raise ValueError('Enclosed-point check failed; no geometry accepted')
        inside=vtk_to_numpy(value).astype(bool)
        distance=vtk.vtkImplicitPolyDataDistance();distance.SetInput(target)
        depths=np.array([abs(distance.EvaluateFunction(point)) for point in source[inside]])
        indices=np.flatnonzero(inside)
        penetrating=depths>boundary_tolerance_mm
        result['directions'][label]=dict(enclosed_vertices=int(inside.sum()),
            penetrating_vertices=int(penetrating.sum()),indices=indices[penetrating].tolist(),
            maximum_penetration_mm=float(depths.max()) if len(depths) else 0.,
            median_depth_mm=float(np.median(depths)) if len(depths) else 0.)
    result['penetrating_vertices']=sum(x['penetrating_vertices'] for x in result['directions'].values())
    result['maximum_penetration_mm']=max(x['maximum_penetration_mm'] for x in result['directions'].values())
    result['boundary_tolerance_mm']=boundary_tolerance_mm
    return result


def four_chamber_audit(vertices,ranges,faces):
    vertices=np.asarray(vertices)
    regions={c:vertices[a:b] for c,(a,b) in zip(CHAMBERS,ranges)}
    topology={}
    for c in CHAMBERS:
        mesh=trimesh.Trimesh(vertices=regions[c],faces=np.asarray(faces[c]),process=False)
        topology[c]=dict(watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
                        connected_components=len(mesh.split(only_watertight=False)))
    pairs={a+'_'+b:chamber_pair_audit(regions[a],faces[a],regions[b],faces[b]) for a,b in combinations(CHAMBERS,2)}
    penetration=sum(x['penetrating_vertices'] for x in pairs.values())
    contacts=sum(x['surface_contact_pairs'] for x in pairs.values())
    topology_valid=all(x['watertight'] and x['winding_consistent'] and x['connected_components']==1 for x in topology.values())
    return dict(topology=topology,pairs=pairs,penetrating_vertices=penetration,surface_contact_pairs=contacts,
                maximum_penetration_mm=max(x['maximum_penetration_mm'] for x in pairs.values()),
                strict_disjoint_cavity_safe=bool(topology_valid and penetration==0 and contacts==0),
                contact_convention='No implicit valve-cap exemption')


def require_strict_template_safety(model):
    report=four_chamber_audit(model.template_vertices_mm.detach().cpu().numpy(),
        model.chamber_ranges.cpu().tolist(),{c:model.faces[c].cpu().numpy() for c in CHAMBERS})
    if not report['strict_disjoint_cavity_safe']:
        raise GeometrySafetyError('Frozen template violates strict inter-chamber geometry: '
            f'{report["penetrating_vertices"]} enclosed vertices, maximum depth '
            f'{report["maximum_penetration_mm"]:.6f} mm; no automatic contact exemption',report)
    return report


def validate_four_chamber_prediction(model,output):
    """Inspect actual triangle meshes in every batch member and all 24 phases.

    A spline certificate alone does not certify the piecewise planar triangle
    approximation. No phase or chamber pair is silently dropped.
    """
    vertices=output['vertices_4d'].detach().cpu().numpy()
    if vertices.ndim!=4 or vertices.shape[1]!=24 or not np.isfinite(vertices).all():
        raise ValueError('Finite complete [B,24,N,3] trajectory required')
    ranges=model.chamber_ranges.cpu().tolist()
    faces={c:model.faces[c].cpu().numpy() for c in CHAMBERS}
    records=[]
    for patient in range(len(vertices)):
        for phase in range(24):
            report=four_chamber_audit(vertices[patient,phase],ranges,faces)
            record=dict(patient=patient,phase=phase,**report)
            if not report['strict_disjoint_cavity_safe']:
                raise GeometrySafetyError(f'Unsafe predicted chamber geometry at patient {patient}, phase {phase}',record)
            records.append(record)
    return dict(status='pass',patients=len(vertices),phases_checked=len(records),records=records)


def validate_observed_sections(model,output,observations):
    from net.multiview_encoder import observation_validity
    vertices=output['vertices_4d'].detach().cpu().numpy()
    truth=observations['vertices_gt'].detach().cpu().numpy()
    valid=observation_validity(observations,len(vertices),24,observations['masks'].device).cpu().numpy()
    rotations=observations['pose_gt']['probe_R'].cpu().numpy()
    translations=observations['pose_gt']['probe_t'].cpu().numpy()*model.norm_to_mm
    counts={}
    for patient,phase,view,chamber in np.argwhere(valid):
        left,right=model.chamber_ranges[chamber].tolist()
        report=compare_section_topology(vertices[patient,phase,left:right],truth[patient,phase,left:right],
            model.faces[CHAMBERS[chamber]].cpu().numpy(),rotations[patient,phase,view],translations[patient,phase,view])
        classification=report['classification'];counts[classification]=counts.get(classification,0)+1
        if not report['valid']:
            raise GeometrySafetyError(f'Illegal observed contour at patient {patient}, phase {phase}, '
                f'view {view}, chamber {CHAMBERS[chamber]}',report)
    return dict(valid_cells=int(valid.sum()),classification_counts=counts,status='pass')


def section_topology(vertices,faces,rotation,translation_mm):
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    section=mesh.section(plane_origin=np.asarray(translation_mm),plane_normal=np.asarray(rotation)[:,2])
    if section is None:return dict(closed_components=0,open_or_branch_vertices=0,empty=True)
    graph=section.vertex_graph
    abnormal=sum(degree!=2 for _,degree in graph.degree())
    return dict(closed_components=len(section.paths),open_or_branch_vertices=abnormal,empty=False)


def compare_section_topology(predicted,truth,faces,rotation,translation_mm):
    p=section_topology(predicted,faces,rotation,translation_mm)
    g=section_topology(truth,faces,rotation,translation_mm)
    if p['open_or_branch_vertices']:
        classification='illegal_open_or_branching_contour'
    elif g['closed_components']==1 and p['closed_components']>1:
        classification='illegal_extra_closed_components'
    elif g['empty'] and p['empty']:classification='legal_empty'
    elif g['empty'] != p['empty']:classification='segmentation_false_positive_or_missing'
    else:classification='matching_component_count' if p['closed_components']==g['closed_components'] else 'component_count_disagreement'
    return dict(prediction=p,truth=g,classification=classification,
                valid=not classification.startswith('illegal_'))
