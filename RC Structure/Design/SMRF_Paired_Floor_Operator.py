"""Physical paired floor action/recovery; condensed stiffness is preconditioner.

This adapter uses the supplied model's exact local component operators and
material matrices. It does not select or qualify stiffness/material premises.
A coupled solver must balance interior AND joint forces with apply_pair before
accepting a response; recover_pair alone does not establish equilibrium.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from Design.SMRF_Paired_Arithmetic import paired_dense_action,add_pair
from Design.SMRF_Paired_Assembly import paired_congruence,paired_triplet_matrix
from Design.SMRF_Paired_Physical_Equilibrium import PairedMembraneEnergy,remove_rigid_pair

METHOD='paired_physical_floor_action_recovery_v2_typed_joint_loads_unqualified'


def paired_floor_model(model,*,input_identity):
    """Dispatch explicit supported constitutive families; never fall back to K*u."""
    from Design.SMRF_Paired_Material_Floor import PairedMaterialFloorOperator
    from Design.SMRF_Material_Floor import MaterialFloorOperator
    from Design.SMRF_Floor_Substructure import GeneralizedFloorStiffness
    if isinstance(model,(PairedFloorOperator,PairedMaterialFloorOperator)):return model
    if isinstance(model,GeneralizedFloorStiffness):return PairedFloorOperator(model,input_identity=input_identity)
    if isinstance(model,MaterialFloorOperator):return PairedMaterialFloorOperator(model,input_identity=input_identity)
    raise ValueError('Unsupported physical floor operator: explicit paired force and recovery required')

class PairedFloorOperator:
    def __init__(self,reference_model,*,input_identity):
        if not input_identity or not all(hasattr(reference_model,k) for k in ('floor','beta','torsion','_slab_K','_slab_D','alpha','_relaxation_factor','_relaxation_free')):
            raise ValueError('Complete explicit generalized physical floor model required')
        self.reference=reference_model;self.floor=floor=reference_model.floor
        self.stiffness=reference_model.stiffness;self.beta=reference_model.beta;self.torsion=reference_model.torsion
        self._slab_K=reference_model._slab_K;self._slab_D=reference_model._slab_D;self.alpha=reference_model.alpha
        if set(self._slab_K)!={r['id'] for r in floor.elements if r['kind']=='slab'}:raise ValueError('Complete actual slab operator inventory required')
        matrices={}
        for family in ('membrane','other'):
            rr=[];cc=[];hh=[];ll=[]
            for row in floor.elements:
                for part in self._parts(row,family):
                    high,low=paired_congruence(part,row['transform']);ri,ci=np.nonzero((high!=0)|(low!=0))
                    rr.append(row['indices'][ri]);cc.append(row['indices'][ci]);hh.append(high[ri,ci]);ll.append(low[ri,ci])
            matrices[family]=paired_triplet_matrix(np.concatenate(hh),np.concatenate(ll),np.concatenate(rr),np.concatenate(cc),shape=self.stiffness.shape)
        coords=np.asarray([floor.node_positions[n] for n in floor.node_ids]);m=floor.membrane_dofs;n=3*len(coords)
        modes=np.zeros((len(m),3));modes[:n:3,0]=1;modes[1:n:3,1]=1;modes[:n:3,2]=-coords[:,1];modes[1:n:3,2]=coords[:,0];modes[2:n:3,2]=1;modes[n:,2]=1
        first=int(np.lexsort((coords[:,1],coords[:,0]))[0]);far=int(np.argmax(coords[:,0]));gauges=[3*first,3*first+1,3*far+1]
        if not np.array_equal(np.setdiff1d(np.arange(len(m)),gauges),reference_model._relaxation_free):raise ValueError('Reused membrane factor gauge order differs')
        self.energy=PairedMembraneEnergy(*matrices['membrane'],*matrices['other'],m,beam_bending_factor=self.beta,
            rigid_modes=modes,gauge_indices=gauges,input_identity=input_identity,membrane_factor=reference_model._relaxation_factor)
        self.metadata=dict(method=METHOD,input_identity=input_identity,reference_model=reference_model.metadata,
            stiffness_role='preconditioner_only',coefficient_pairs_retained=True,additional_membrane_factorizations=0,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification_asserted=False)

    def _parts(self,row,family):
        if family=='membrane':return [row['parts']['membrane']]
        if row['kind']=='slab':return [self._slab_K[row['id']],row['parts']['shear']]
        return [self.beta*row['parts']['web_bending'],self.torsion[row['member_id']]*row['parts']['web_torsion']]

    def apply_pair(self,high,low):
        dh,dl,_=remove_rigid_pair(self.floor,high,low)
        return self.energy.apply_pair(dh,dl)

    def recover(self,*args,**kwargs):
        raise ValueError('Paired floor requires explicit paired equilibrium and recover_pair')

    def recover_pair(self,high,low,load_record):
        floor=self.floor;uh,ul=np.asarray(high,float),np.asarray(low,float)
        dh,dl,rigid=remove_rigid_pair(floor,uh,ul);fields=self.energy.paired_fields(dh,dl);wh,wl=fields['weighted_hi'],fields['weighted_lo']
        ids={r['id'] for r in floor.elements}
        if set(load_record.get('element_loads',{}))!=ids:raise ValueError('Complete physical element load inventory required')
        rows=[];highs=[];lows=[];applied_high=[];applied_low=[];elements=[]
        for row in floor.elements:
            t=row['transform'];ind=row['indices'];mh,ml=paired_dense_action(t,wh[ind],wl[ind]);ah,al=paired_dense_action(t,dh[ind],dl[ind])
            fh,fl=paired_dense_action(row['parts']['membrane'],mh,ml)
            for part in self._parts(row,'other'):fh,fl=add_pair(fh,fl,*paired_dense_action(part,ah,al))
            elastic=fh+fl;applied=np.asarray(load_record['element_loads'][row['id']],float)
            if applied.shape!=fh.shape or not np.isfinite(applied).all():raise ValueError('Finite actual element load vector required')
            fh,fl=add_pair(fh,fl,-applied);gh,gl=paired_dense_action(t.T,fh,fl)
            ph,pl=paired_dense_action(t.T,applied,np.zeros_like(applied));rows.append(ind);highs.append(gh);lows.append(gl);applied_high.append(ph);applied_low.append(pl)
            disp_h,disp_l=paired_dense_action(t,uh[ind],ul[ind])
            out=dict(id=row['id'],kind=row['kind'],positions_in=row['points'].tolist(),displacements=(disp_h+disp_l).tolist(),
                global_resisting_force=(fh+fl).tolist(),global_elastic_force=elastic.tolist())
            if row['kind']=='slab':
                sh,sl=paired_dense_action(floor.ec*row['native']['raw_resultant'],mh,ml)
                bh,bl=paired_dense_action(floor.ec*row['native']['raw_resultant'],ah,al);sm=(sh+sl).reshape(4,8);base=(bh+bl).reshape(4,8)
                if self.alpha is not None:sm[:,3:6]=self.alpha[row['id']]*base[:,3:6]
                else:
                    strain=-row['native']['strain'].reshape(4,8,24)[:,3:6,:]
                    for gp in range(4):
                        ch,cl=paired_dense_action(strain[gp],ah,al);ph,pl=paired_dense_action(self._slab_D[row['id']][gp],ch,cl);sm[gp,3:6]=ph+pl
                sm[:,6:]=base[:,6:];out.update(cell=row['cell'],area_in2=row['area_in2'],raw_resultants=sm.tolist())
            else:out.update(member_id=row['member_id'],axis=row['axis'],station_in=row['station_in'],section=row['section'])
            elements.append(out)
        rr=np.concatenate(rows);cc=np.zeros(len(rr),dtype=int);shape=(len(uh),1)
        assembled=paired_triplet_matrix(np.concatenate(highs),np.concatenate(lows),rr,cc,shape=shape)
        element_loads=paired_triplet_matrix(np.concatenate(applied_high),np.concatenate(applied_low),rr,cc,shape=shape)
        ah,al=(v.toarray()[:,0] for v in assembled);ph,pl=(v.toarray()[:,0] for v in element_loads)
        expected_h,expected_l=self.energy.apply_pair(dh,dl)
        eh,el=add_pair(ah,al,ph,pl);eh,el=add_pair(eh,el,-expected_h,-expected_l);error=eh+el
        scale=max(1.,float(np.max(abs(expected_h+expected_l),initial=0.)));relative=float(np.max(abs(error),initial=0.))/scale
        if relative>1e-9:raise ValueError('Paired local actions fail original physical force-partition criterion')
        return dict(method=METHOD,elements=elements,dof_ids=floor.dof_ids,boundary_dofs=floor.boundary_dofs,
            assembled_element_resistance=(ah+al).tolist(),assembled_element_resistance_hi=ah.tolist(),assembled_element_resistance_lo=al.tolist(),
            force_partition_relative_error=relative,stiffness=self.metadata,physical_relaxation=self.energy.last_relaxation,
            retained_displacements=uh.tolist(),retained_displacements_lo=ul.tolist(),
            removed_rigid_translation_in=rigid['translation'],removed_rigid_rotation_rad=rigid['rotation'],rigid_motion_reference_xyz_in=rigid['reference'],
            applied_element_forces_hi=ph.tolist(),applied_element_forces_lo=pl.tolist(),
            joint_wrenches=[dict(node_id=tag,wrench_kip_kipin=list(wrench)) for tag,wrench in load_record.get('joint_wrenches',{}).items()],
            qualification_asserted=False,equilibrium_inferred_from_recovery=False)
