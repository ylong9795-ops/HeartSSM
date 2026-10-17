"""Observable relative chamber translation hints; never final mesh offsets."""
import numpy as np
from .centres import huber_location

MODE = 'initial_candidates_only'


def relative_translation_hints(rows, pose, *, huber_delta_mm=3., ridge=1e-3,
                               singular_ratio=1e-3, max_hint_mm=10.):
    """Solve projected pred->observation hints with sum_c delta_c=0.

    Each chamber is restricted to the SVD-observable subspace. In particular,
    a single A4C observation has zero normal component even AFTER removing the
    shared-translation gauge. Hints are not 3D labels or accepted corrections.
    """
    systems, bases, raw, records = [], [], [], []
    for ci in range(4):
        matrices, targets, observed_views = [], [], []
        for vi in range(3):
            group = [r for r in rows if r['chamber']==ci and r['view']==vi and
                     r['fit_weight']>0 and r['centroid_error_xy_mm'] is not None and not r['renderer_error']]
            if not group: continue
            e=huber_location([r['centroid_error_xy_mm'] for r in group], [r['fit_weight'] for r in group],huber_delta_mm)
            matrices.append(np.asarray(pose['R'][vi])[:,:2].T)
            targets.append(-e)  # Moving the MESH has the opposite sign to moving a plane.
            observed_views.append(vi)
        a=np.concatenate(matrices) if matrices else np.zeros((0,3))
        b=np.concatenate(targets) if targets else np.zeros(0)
        if len(a):
            _, singular, vh=np.linalg.svd(a,full_matrices=True)
            rank=int((singular>max(singular[0]*singular_ratio,1e-10)).sum())
            basis=vh[:rank].T
            unconstrained=np.linalg.lstsq(a,b,rcond=singular_ratio)[0]
        else:
            singular=np.zeros(0); rank=0; basis=np.zeros((3,0)); unconstrained=np.zeros(3)
        systems.append((a,b)); bases.append(basis); raw.append(unconstrained)
        records.append(dict(chamber=ci, observed_views=len(matrices), rank=rank,
            observed_view_indices=observed_views,
            singular_values=singular, observable_basis=basis, nullity=3-rank,
            depth_identifiability='NOT_IDENTIFIED' if rank<3 else 'MULTIVIEW_CONDITIONED'))
    sizes=[b.shape[1] for b in bases]; offsets=np.r_[0,np.cumsum(sizes)]
    total=int(offsets[-1]); delta=np.zeros((4,3)); shared=np.zeros(3)
    if total:
        # Joint nuisance translation absorbs the shared component BEFORE it is
        # discarded. Merely constraining sum(delta)=0 can manufacture relative
        # hints from a pure common shift when chambers have unequal view counts.
        h=np.eye(total+3)*1e-12; h[3:,3:]+=np.eye(total)*ridge
        rhs=np.zeros(total+3); common=np.zeros((3,total+3))
        for ci,((a,b),basis) in enumerate(zip(systems,bases)):
            sl=slice(3+offsets[ci],3+offsets[ci+1]); block=np.zeros((len(a),total+3))
            block[:,:3]=a;block[:,sl]=a@basis
            h+=block.T@block;rhs+=block.T@b;common[:,sl]=basis
        kkt=np.block([[h,common.T],[common,np.zeros((3,3))]])
        fitted=np.linalg.lstsq(kkt,np.r_[rhs,np.zeros(3)],rcond=1e-10)[0][:total+3]
        shared=fitted[:3];solution=fitted[3:]
        delta=np.stack([basis@solution[offsets[ci]:offsets[ci+1]] for ci,basis in enumerate(bases)])
        largest=np.linalg.norm(delta,axis=1).max()
        if largest>max_hint_mm: delta*=max_hint_mm/largest
    for ci,record in enumerate(records):
        record['unobservable_component_norm_mm']=float(np.linalg.norm(delta[ci]-bases[ci]@bases[ci].T@delta[ci]))
        if record['observed_views']==1:
            vi=record['observed_view_indices'][0]
            record['single_view_normal_component_mm']=float(np.asarray(pose['R'][vi])[:,2]@delta[ci])
    return dict(displacements_mm=delta, common_translation_mm=delta.mean(0),
        unconstrained_displacements_mm=np.array(raw), removed_shared_component_mm=shared,
        chambers=records, contract='INITIALIZATION_HINT_ONLY; not 3D GT; never added to final meshes',
        sign='A_c delta_c = -(predicted_centroid-observed_centroid)_mm',
        gauge='sum of four relative hints = 0, with per-chamber null directions fixed to zero')


def generate_candidates(prior, observations, state, *, rows, config):
    if state.observation_pose is None: raise ValueError('S2 requires S1 shared-view poses')
    pose={k:v.detach().cpu().numpy() for k,v in state.observation_pose.items()}
    return relative_translation_hints(rows,pose,huber_delta_mm=config['huber_delta_mm'],
        ridge=config['hint_ridge'],singular_ratio=config['observable_singular_ratio'],
        max_hint_mm=config['max_hint_mm'])
