"""Physical full-T elastic incremental fields, with explicit common axes.

inch/kip/ksi; tensile normal force positive; z upward from slab midplane.
M_z=int(z*sigma_axial dA) denotes the STRONG-axis force first moment, not
the vector moment about global z. M_s=int(s*sigma_axial dA) is the transverse
first moment. Caller maps these scalar cut signs to its global nodal wrench.
s is the transverse distance from the declared beam reference line. Flange/
web intervals retain actual global transverse coordinates.

This is the primary effective-beam incremental idealization. It includes the
flange's own I, web own I and axial couple once. It never adds gravity C0,
changes reinforcement or reduces Mn/Mpr. Physical floor compatibility and
spatial qualification remain upstream. An asymmetric L flange can generate
M_s under the usual2x2 uncoupled beam projection; that is reported explicitly.
The optional3x3 projection closes a supplied M_s with transverse curvature.
"""
from __future__ import annotations

import math
import numpy as np

METHOD='full_T_increment_physical_partition_v1'


def _finite(x,name):
    x=float(x)
    if not math.isfinite(x):raise ValueError(f'Finite {name} required')
    return x


def _interval(values,name):
    if len(values)!=2:raise ValueError(f'Two {name} endpoints required')
    lo,hi=(_finite(v,name) for v in values)
    if hi<=lo:raise ValueError(f'Positive {name} length required')
    return (lo,hi)


def _integrals(interval):
    a,b=interval
    return np.array([b-a,(b*b-a*a)/2,(b**3-a**3)/3])


def released_increment(loaded,gravity,*,increment_id):
    """Subtract exactly the same released gravity state, not a continuous model.

    Both rows require model_fingerprint,state_identity,cut_id,gravity_load_id,
    analysis_id,reference_z_in,n_tension_kip,m_z_kip_in. Optional m_s_kip_in
    must be present in both or neither. The common model fingerprint must
    include releases, stiffness, constraints and geometry. Original analysis
    identities are retained. Reference transport is explicit before subtraction.
    """
    if not isinstance(increment_id,str) or not increment_id:raise ValueError('Increment identity required')
    for key in ('model_fingerprint','state_identity','cut_id','gravity_load_id'):
        if loaded.get(key) is None or loaded.get(key)!=gravity.get(key):
            raise ValueError(f'Released increment requires identical {key}')
    if not loaded.get('analysis_id') or not gravity.get('analysis_id'):
        raise ValueError('Both original solved analysis identities required')
    def at_mid(row):
        n=_finite(row['n_tension_kip'],'tension-positive N')
        z=_finite(row['reference_z_in'],'reference z')
        return np.array([n,_finite(row['m_z_kip_in'],'strong-axis first moment')+z*n])
    values=at_mid(loaded)-at_mid(gravity)
    result=dict(increment_id=increment_id,model_fingerprint=loaded['model_fingerprint'],
        state_identity=loaded['state_identity'],cut_id=loaded['cut_id'],
        gravity_load_id=loaded['gravity_load_id'],loaded_analysis_id=loaded['analysis_id'],
        gravity_analysis_id=gravity['analysis_id'],reference_z_in=0.,
        n_tension_kip=float(values[0]),m_z_kip_in=float(values[1]),
        source_operation='released_loaded_minus_identical_released_gravity')
    if ('m_s_kip_in' in loaded)!=('m_s_kip_in' in gravity):
        raise ValueError('Weak-axis first moment must be supplied in both states or neither')
    if 'm_s_kip_in' in loaded:
        if loaded.get('reference_s_in')!=gravity.get('reference_s_in'):
            raise ValueError('Transverse first-moment references must match')
        s=_finite(loaded.get('reference_s_in',0.),'transverse reference')
        result['m_s_kip_in']=(_finite(loaded['m_s_kip_in'],'loaded weak moment')-
            _finite(gravity['m_s_kip_in'],'gravity weak moment'))
        result['reference_s_in']=s
    return result


def normal_cut_from_global_wrench(wrench,*,axis):
    """Global internal cut wrench ON THE POSITIVE global beam-axis cut face.

    For X: r=(0,s,z),t=(sigma,0,0), so My=Mz_scalar,Mz=-Ms_scalar.
    For Y: r=(s,0,z),t=(0,sigma,0), so Mx=-Mz_scalar,Mz=Ms_scalar.
    Caller must orient element end forces to this actual face before calling.
    No torsion/transverse shear is silently added to normal-section bending.
    """
    f=np.asarray(wrench,dtype=float)
    if f.shape!=(6,) or not np.isfinite(f).all():raise ValueError('Finite global Fx,Fy,Fz,Mx,My,Mz required')
    if axis=='x':n,m,ms=f[0],f[4],-f[5]
    elif axis=='y':n,m,ms=f[1],-f[3],f[5]
    else:raise ValueError('Global beam direction x or y required')
    return dict(n_tension_kip=float(n),m_z_kip_in=float(m),m_s_kip_in=float(ms),
                cut_face='positive_global_beam_axis',axis=axis)


class FullTIncrementProjector:
    def __init__(self,*,slab_thickness_in,beam_depth_in,flange_interval_in,
                 web_interval_in,slab_E_ksi,web_E_ksi,axis,transverse_reference_in=None):
        self.t=_finite(slab_thickness_in,'slab thickness')
        self.h=_finite(beam_depth_in,'total beam depth')
        self.ef=_finite(slab_E_ksi,'slab E');self.ew=_finite(web_E_ksi,'web E')
        if min(self.t,self.ef,self.ew)<=0 or self.h<=self.t:
            raise ValueError('Positive moduli, slab thickness and actual downstand required')
        if axis not in ('x','y'):raise ValueError('Beam global direction x or y required')
        self.axis=axis;self.j=0 if axis=='x' else 1
        self.flange=_interval(flange_interval_in,'flange interval')
        self.web=_interval(web_interval_in,'web interval')
        self.sref=sum(self.web)/2 if transverse_reference_in is None else _finite(transverse_reference_in,'beam transverse reference')
        if self.flange[0]>self.web[0]+1e-10 or self.flange[1]<self.web[1]-1e-10:
            raise ValueError('Full-T slab portion must include the physical rectangle above the web')
        self.regions=[dict(id='slab',s=self.flange,z=(-self.t/2,self.t/2),E=self.ef),
                      dict(id='web',s=self.web,z=(self.t/2-self.h,-self.t/2),E=self.ew)]
        self.K=np.zeros((3,3))
        for region in self.regions:self.K+=self._region_matrix(region,region['s'])
        if np.linalg.eigvalsh(self.K).min()<=0:raise ValueError('Physical section has no positive elastic stiffness')
        a,b,d=self.K[0,0],self.K[0,1],self.K[1,1]
        self.ABD=np.array([[a,b],[b,d]])

    def _region_matrix(self,region,interval):
        s,z=_integrals((interval[0]-self.sref,interval[1]-self.sref)),_integrals(region['z']);e=region['E']
        return e*np.array([[s[0]*z[0],s[0]*z[1],s[1]*z[0]],
                           [s[0]*z[1],s[0]*z[2],s[1]*z[1]],
                           [s[1]*z[0],s[1]*z[1],s[2]*z[0]]])

    def properties(self):
        a,b,d=self.ABD[0,0],self.ABD[0,1],self.ABD[1,1]
        return dict(method=METHOD,axis=self.axis,flange_interval_in=list(self.flange),
            web_interval_in=list(self.web),slab_thickness_in=self.t,total_beam_depth_in=self.h,
            full_ABD=self.ABD.tolist(),full_three_resultant_matrix=self.K.tolist(),
            axial_EA_kip=float(a),first_moment_EB_kip_in=float(b),
            full_ED_kip_in2=float(d),zero_N_EI_kip_in2=float(d-b*b/a),
            flange_own_EI_kip_in2=self.ef*(self.flange[1]-self.flange[0])*self.t**3/12,
            elastic_centroid_z_in=float(b/a),elastic_centroid_s_in=float(self.K[0,2]/a),
            reference_z_in=0.,reference_s_in=self.sref)

    def project(self,*,n_tension_kip,m_z_kip_in,increment_id,m_s_kip_in=None,
                allow_unclosed_transverse_diagnostic=False):
        if not isinstance(increment_id,str) or not increment_id:raise ValueError('Atomic incremental action identity required')
        n=_finite(n_tension_kip,'N');m=_finite(m_z_kip_in,'M_z')
        asymmetric=abs(sum(self.flange)/2-self.sref)>1e-9 or abs(sum(self.web)/2-self.sref)>1e-9
        if asymmetric and m_s_kip_in is None and not allow_unclosed_transverse_diagnostic:
            raise ValueError('Asymmetric flange requires the actual weak-axis first moment, including explicit zero')
        if m_s_kip_in is None:
            q=np.r_[np.linalg.solve(self.ABD,[n,m]),0.]
            specified=[n,m,None];mode='uncoupled_primary_beam_2x2'
        else:
            ms=_finite(m_s_kip_in,'M_s');q=np.linalg.solve(self.K,[n,m,ms])
            specified=[n,m,ms];mode='complete_normal_section_3x3'
        regions=[];total=np.zeros(3)
        for r in self.regions:
            forces=self._region_matrix(r,r['s'])@q;total+=forces
            zc=sum(r['z'])/2;sc=sum(r['s'])/2-self.sref
            regions.append(dict(id=r['id'],interval_in=list(r['s']),z_interval_in=list(r['z']),
                n_tension_kip=float(forces[0]),m_z_kip_in=float(forces[1]),m_s_kip_in=float(forces[2]),
                m_about_own_z_centroid_kip_in=float(forces[1]-zc*forces[0]),
                m_z_axial_couple_kip_in=float(zc*forces[0]),
                m_about_own_s_centroid_kip_in=float(forces[2]-sc*forces[0]),
                m_s_offset_couple_kip_in=float(sc*forces[0])))
        target=np.array([n,m,total[2] if m_s_kip_in is None else m_s_kip_in])
        error=total-target
        tolerance=1e-9+1e-11*np.maximum(abs(total),abs(target))
        if not np.all(abs(error)<=tolerance):raise ValueError('Physical full-T projection failed section equilibrium')
        ms_unclosed=float(total[2]) if m_s_kip_in is None else 0.
        return dict(method=METHOD,increment_id=increment_id,mode=mode,axis=self.axis,
            generalized_normal_strain=q.tolist(),strain_convention='epsilon=epsilon0+kappa_z*z+kappa_s*(global_transverse-reference_s)',
            requested_resultants=specified,reconstructed_resultants=total.tolist(),
            equilibrium_residual=error.tolist(),equilibrium_verified=True,
            incidental_transverse_first_moment_kip_in=ms_unclosed,
            transverse_action_specified=m_s_kip_in is not None,
            requires_transverse_action_accounting=asymmetric and m_s_kip_in is None,
            regions=regions,properties=self.properties(),
            physical_floor_compatibility_verified=False,
            nominal_probable_capacity_inventory_modified=False)

    def slab_field(self,projection,transverse_coordinate_in):
        """Six membrane/plate resultants per inch at the actual plan coordinate."""
        s=_finite(transverse_coordinate_in,'transverse position')
        q=np.asarray(projection['generalized_normal_strain'],dtype=float)
        if projection.get('properties')!=self.properties():raise ValueError('Projection differs from this physical section')
        out=np.zeros(6)
        if self.flange[0]-1e-12<=s<=self.flange[1]+1e-12:
            out[self.j]=self.ef*self.t*(q[0]+q[2]*(s-self.sref))
            out[self.j+3]=self.ef*self.t**3/12*q[1]
        return out.tolist()

    def integrated_shear_partition(self,*,resultant_derivatives,vertical_shear_kip,source_id):
        """Whole-flange/web shear from normal derivatives and interface transfer.

        Q=-dM/dx is the physical floor cut convention. For a traction-free
        slab top, tau_bottom=dNf/dx, hence Qf=-dMf/dx-(t/2)dNf/dx.
        Qw=-dMw_global/dx+(t/2)dNf/dx. This INTEGRATED identity does not
        establish local flange shear flow or its variation across the width.
        Nonzero total axial derivative needs exterior load-depth information
        and is deliberately outside this traction-free incremental contract.
        """
        d=np.asarray(resultant_derivatives,float);v=float(vertical_shear_kip)
        if d.shape!=(3,) or not np.isfinite(d).all() or not math.isfinite(v) or not source_id:
            raise ValueError('Actual N/Mstrong/Mweak derivatives, signed shear and source identity required')
        if abs(d[0])>1e-9:raise ValueError('Nonzero total axial derivative needs actual distributed axial-load depth')
        if abs(d[1]+v)>1e-8+1e-9*max(abs(d[1]),abs(v)):
            raise ValueError('Actual normal moment derivative and vertical shear disagree')
        qp=np.linalg.solve(self.K,d)
        flange=self._region_matrix(self.regions[0],self.flange)@qp
        web=self._region_matrix(self.regions[1],self.web)@qp
        qf=-flange[1]-self.t/2*flange[0]
        qw=-web[1]+self.t/2*flange[0]
        return dict(source_id=str(source_id),generalized_normal_strain_derivative=qp.tolist(),
            normal_resultant_derivatives=d.tolist(),flange_resultant_derivatives=flange.tolist(),
            web_resultant_derivatives=web.tolist(),slab_integrated_Q_kip=float(qf),web_integrated_Q_kip=float(qw),
            slab_integrated_Q_diagram_kip=float(qf),slab_integrated_native_raw_Q_kip=float(-qf),
            shear_sign_basis='Q_diagram=-positive_face_Fz; ShellMITC4_raw_Q=positive_face_Fz; legacy slab_integrated_Q_kip is diagram Q',
            interface_axial_transfer_kip_per_in=float(flange[0]),
            shear_reconstruction_residual_kip=float(qf+qw-v),
            local_transverse_shear_field_established=False,
            basis='integrated_axial_equilibrium_and_first_moment_integration_by_parts')

    def partition(self,projection,strips,*,require_whole_web=True):
        """Integrate disjoint physical intervals and reconstruct ALL first moments.

        strips=[{id,interval_in:[lo,hi]},...]. Cover the entire actual flange
        exactly once; strips may extend beyond it with zero incremental field.
        For common-strength use the complete web belongs to one strip, so no
        beam bar is assigned several incompatible strip strains.
        """
        if projection.get('properties')!=self.properties():raise ValueError('Projection differs from this physical section')
        q=np.asarray(projection['generalized_normal_strain'],dtype=float)
        if q.shape!=(3,) or not np.isfinite(q).all():raise ValueError('Physical normal strain field required')
        records=[];seen=set()
        for row in strips:
            if not row.get('id') or row['id'] in seen:raise ValueError('Unique physical strip IDs required')
            seen.add(row['id']);interval=_interval(row['interval_in'],'strip')
            records.append((interval,row['id']))
        records.sort()
        cursor=self.flange[0];outputs=[];total=np.zeros(3);web_owners=0
        for (lo,hi),name in records:
            flo,fhi=max(lo,self.flange[0]),min(hi,self.flange[1])
            if flo<fhi:
                if abs(flo-cursor)>1e-9:raise ValueError('Physical partition has a flange gap or overlap')
                cursor=fhi
            forces=np.zeros(3);parts={}
            for r in self.regions:
                a,b=max(lo,r['s'][0]),min(hi,r['s'][1])
                force=np.zeros(3) if b<=a else self._region_matrix(r,(a,b))@q
                parts[r['id']]=force.tolist();forces+=force
                if r['id']=='web' and b>a:
                    web_owners+=1
                    if require_whole_web and (abs(a-self.web[0])>1e-9 or abs(b-self.web[1])>1e-9):
                        raise ValueError('A common-strength strip cannot split the actual beam web/cage')
            six=np.zeros(6);six[self.j]=forces[0];six[self.j+3]=forces[1]
            total+=forces
            outputs.append(dict(id=name,interval_in=[lo,hi],physical_width_in=hi-lo,
                integrated_six_resultants=six.tolist(),average_six_resultants_per_in=(six/(hi-lo)).tolist(),
                n_mz_ms=forces.tolist(),region_n_mz_ms=parts,
                transverse_first_moment_kip_in=float(forces[2]),
                transverse_normal_strain_variation=abs(q[2])*(hi-lo)))
        if abs(cursor-self.flange[1])>1e-9:raise ValueError('Physical strips do not cover complete flange')
        if require_whole_web and web_owners!=1:raise ValueError('Exactly one physical strip must own the web')
        reference=np.asarray(projection['reconstructed_resultants'])
        err=total-reference
        if not np.all(abs(err)<=1e-8+1e-10*np.maximum(abs(total),abs(reference))):
            raise ValueError('Partitioned field does not reconstruct original projection')
        return dict(method=METHOD,increment_id=projection['increment_id'],strips=outputs,
            sum_n_mz_ms=total.tolist(),reconstruction_residual=err.tolist(),
            complete_disjoint_coverage_verified=True,whole_web_owner_verified=web_owners==1,
            physical_strength_spatial_projection_verified=False)
