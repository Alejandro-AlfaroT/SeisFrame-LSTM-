"""Paired physical action of explicitly supplied full ABD/web material matrices."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from Design.SMRF_Paired_Arithmetic import add_pair,paired_dense_action,paired_sparse_action
from Design.SMRF_Paired_Assembly import paired_congruence,paired_triplet_matrix
from Design.SMRF_Paired_Physical_Equilibrium import remove_rigid_pair
from Design.SMRF_Material_Floor import common_shell_strain_matrix

METHOD='paired_full_ABD_and_web_floor_v2_typed_joint_loads_unqualified'


class PairedMaterialFloorOperator:
    def __init__(self,reference_model,*,input_identity):
        if not input_identity or not all(hasattr(reference_model,k) for k in ('floor','element_matrices','slab','shear')):
            raise ValueError('Complete explicit physical material floor required')
        self.reference=reference_model;self.floor=floor=reference_model.floor;self.stiffness=reference_model.stiffness
        self.element_matrices=reference_model.element_matrices;self.slab=reference_model.slab;self.shear=reference_model.shear
        if set(self.element_matrices)!={r['id'] for r in floor.elements}:raise ValueError('Complete material element inventory required')
        rr=[];cc=[];hh=[];ll=[]
        for row in floor.elements:
            h,l=paired_congruence(self.element_matrices[row['id']],row['transform']);ri,ci=np.nonzero((h!=0)|(l!=0))
            rr.append(row['indices'][ri]);cc.append(row['indices'][ci]);hh.append(h[ri,ci]);ll.append(l[ri,ci])
        self.high,self.low=paired_triplet_matrix(np.concatenate(hh),np.concatenate(ll),np.concatenate(rr),np.concatenate(cc),shape=self.stiffness.shape)
        self.metadata={**reference_model.metadata,'method':METHOD,'reference_model':reference_model.metadata,
            'input_identity':input_identity,'stiffness_role':'preconditioner_only','coefficient_pairs_retained':True,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'qualification_asserted':False}

    def apply_pair(self,high,low):
        h,l,_=remove_rigid_pair(self.floor,high,low)
        return add_pair(*paired_sparse_action(self.high,h,l),*paired_sparse_action(self.low,h,l))

    def recover(self,*args,**kwargs):raise ValueError('Physical paired material recovery requires both displacement parts')

    def recover_pair(self,high,low,load_record):
        floor=self.floor;uh,ul=np.asarray(high,float),np.asarray(low,float);dh,dl,rigid=remove_rigid_pair(floor,uh,ul)
        if set(load_record.get('element_loads',{}))!=set(self.element_matrices):raise ValueError('Complete physical element loads required')
        rows=[];highs=[];lows=[];applied_high=[];applied_low=[];elements=[]
        for row in floor.elements:
            t=row['transform'];ix=row['indices'];lh,ll=paired_dense_action(t,dh[ix],dl[ix])
            eh,el=paired_dense_action(self.element_matrices[row['id']],lh,ll)
            applied=np.asarray(load_record['element_loads'][row['id']],float)
            if applied.shape!=eh.shape or not np.isfinite(applied).all():raise ValueError('Finite actual element loads required')
            fh,fl=add_pair(eh,el,-applied);gh,gl=paired_dense_action(t.T,fh,fl);ph,pl=paired_dense_action(t.T,applied,np.zeros_like(applied))
            rows.append(ix);highs.append(gh);lows.append(gl);applied_high.append(ph);applied_low.append(pl)
            rawh,rawl=paired_dense_action(t,uh[ix],ul[ix])
            out=dict(id=row['id'],kind=row['kind'],positions_in=row['points'].tolist(),displacements=(rawh+rawl).tolist(),
                global_resisting_force=(fh+fl).tolist(),global_elastic_force=(eh+el).tolist())
            if row['kind']=='slab':
                strain=common_shell_strain_matrix(row);q=[];resultants=[]
                for gp in range(4):
                    qh,ql=paired_dense_action(strain[gp],lh,ll);rh,rl=paired_dense_action(self.slab[row['id']][gp],qh,ql)
                    q.append(qh+ql);resultants.append(rh+rl)
                th,tl=paired_dense_action(floor.ec*row['native']['raw_resultant'],lh,ll)
                transverse=(th+tl).reshape(4,8)[:,6:]*self.shear[row['id']]
                out.update(cell=row['cell'],area_in2=row['area_in2'],raw_resultants=np.c_[resultants,transverse].tolist(),common_generalized_strains=np.array(q).tolist())
            else:out.update(member_id=row['member_id'],axis=row['axis'],station_in=row['station_in'],section=row['section'])
            elements.append(out)
        rr=np.concatenate(rows);cc=np.zeros(len(rr),int);shape=(len(uh),1)
        ah,al=(v.toarray()[:,0] for v in paired_triplet_matrix(np.concatenate(highs),np.concatenate(lows),rr,cc,shape=shape))
        ph,pl=(v.toarray()[:,0] for v in paired_triplet_matrix(np.concatenate(applied_high),np.concatenate(applied_low),rr,cc,shape=shape))
        fh,fl=self.apply_pair(uh,ul);eh,el=add_pair(ah,al,ph,pl);eh,el=add_pair(eh,el,-fh,-fl)
        error=float(np.max(abs(eh+el),initial=0.))/max(1.,float(np.max(abs(fh+fl),initial=0.)))
        if error>1e-9:raise ValueError('Paired material local forces fail original partition criterion')
        return dict(method=METHOD,elements=elements,dof_ids=floor.dof_ids,boundary_dofs=floor.boundary_dofs,
            assembled_element_resistance=(ah+al).tolist(),assembled_element_resistance_hi=ah.tolist(),assembled_element_resistance_lo=al.tolist(),
            applied_element_forces_hi=ph.tolist(),applied_element_forces_lo=pl.tolist(),
            joint_wrenches=[dict(node_id=tag,wrench_kip_kipin=list(wrench)) for tag,wrench in load_record.get('joint_wrenches',{}).items()],
            force_partition_relative_error=error,stiffness=self.metadata,retained_displacements=uh.tolist(),retained_displacements_lo=ul.tolist(),
            removed_rigid_translation_in=rigid['translation'],removed_rigid_rotation_rad=rigid['rotation'],rigid_motion_reference_xyz_in=rigid['reference'],
            qualification_asserted=False,equilibrium_inferred_from_recovery=False)
