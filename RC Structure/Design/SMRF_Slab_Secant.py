"""Actual-layer slab A/B/D secants for an explicit restricted constitutive analysis.

Polak/Vecchio1993 pp3443-3445 Eq6-9: principal Ei=fi/epsi, cracked nu=0,
G12=E1*E2/(E1+E2), then transform and integrate. Uses the named no-concrete-
tension/no-steel-hardening strength law already exposed by LayeredSlabSection.
This is a secant iteration operator, not its consistent Newton tangent.
It is NOT a bound on forces: stiffness changes can redistribute either way.
When used to claim nonlinear response, full floor equilibrium iteration and
convergence remain necessary. A frozen secant is instead an explicit linear
stiffness model under the declared ACI6.3 analysis assumptions; it needs its
reference-state and response comparisons. K(q)q=R(q) alone proves neither.

The elastic uncracked branch is an explicit initial/service predictor under
ACI24.2.3.1/R24.2.3.1 and Polak Eq24-26. It tests actual principal tensile
stress against the published cracking stress and actual steel yield. Once a
cracked analysis is selected, no automatic uncracking/history reset occurs.
No arbitrary shear retention or positive eigenvalue floor is introduced.

Units inch,ksi; q=[ex,ey,gxy,kx,ky,kxy], tension positive, slabmidplane z=0.
Output R=[Nxx,Nyy,Nxy,Mxx,Myy,Mxy]; engineering shear/twist conventions.
"""
from __future__ import annotations
import math
import numpy as np
from Design.SMRF_Layered_Slab_Section import (
    LayeredSlabSection, concrete_response, strain_tensors, design_concrete_peak_strain,
)

METHOD='actual_layer_Polak1993_restricted_slab_secant_v3_stable_membrane_relaxation'


def principal_secant_matrix(strain,fc_ksi,*,peak_strain,compression_strain_limit):
    """Symmetric PSD in-plane secant, reproducing the restricted stress exactly.

    At exactly zero principal strain choose the compressive initial modulus;
    either one-sided modulus gives zero stress there. This declared selection
    initializes a missing direction; it is not a tensile stiffness floor.
    Positive tensile strain has exactly zero concrete modulus.
    """
    e=np.asarray(strain,dtype=float)
    if e.shape!=(2,2):raise ValueError('One symmetric2x2 strain tensor required')
    result=principal_secant_matrices(e[None,:,:],fc_ksi,peak_strain=peak_strain,
                                    compression_strain_limit=compression_strain_limit)
    return {key:(value[0] if isinstance(value,np.ndarray) else value) for key,value in result.items()}


def principal_secant_matrices(strains,fc_ksi,*,peak_strain,compression_strain_limit):
    """Batched form of the identical published principal secant equations."""
    e=np.asarray(strains,dtype=float)
    if e.ndim!=3 or e.shape[1:]!=(2,2):raise ValueError('A batch of symmetric2x2 strain tensors required')
    response=concrete_response(e,fc_ksi,peak_strain=peak_strain,
        compression_strain_limit=compression_strain_limit,model='polak1993_thorenfeldt')
    eps,v=np.linalg.eigh(e)
    sigma=np.einsum('nki,nkl,nli->ni',v,response['stress_ksi'],v)
    effective=np.asarray(response['effective_strength_ksi'])
    n=.8+effective*6.894757293168/17
    initial=effective*np.maximum(n,1+1e-9)/(peak_strain*(np.maximum(n,1+1e-9)-1))
    mod=np.divide(sigma,eps,out=initial.copy(),where=eps!=0)
    mod=np.maximum(mod,0.)  # negative roundoff in a nominally zero principal stress
    total=mod.sum(axis=1);g=np.divide(mod[:,0]*mod[:,1],total,out=np.zeros(len(e)),where=total>0)
    a,b=v[:,:,0],v[:,:,1]
    transform=np.stack([np.c_[a[:,0]**2,a[:,1]**2,a[:,0]*a[:,1]],
                        np.c_[b[:,0]**2,b[:,1]**2,b[:,0]*b[:,1]],
                        np.c_[2*a[:,0]*b[:,0],2*a[:,1]*b[:,1],a[:,0]*b[:,1]+a[:,1]*b[:,0]]],axis=1)
    matrix=np.einsum('nki,nk,nkj->nij',transform,np.c_[mod,g],transform)
    vector=np.c_[e[:,0,0],e[:,1,1],2*e[:,0,1]]
    stress=np.c_[response['stress_ksi'][:,0,0],response['stress_ksi'][:,1,1],response['stress_ksi'][:,0,1]]
    error=np.einsum('nij,nj->ni',matrix,vector)-stress
    if not np.all(abs(error)<=1e-10+1e-9*abs(stress)):
        raise ValueError('Principal secant does not reconstruct restricted concrete stress')
    return dict(matrix=matrix,energy_factors=np.sqrt(np.c_[mod,g])[:,:,None]*transform,
        stress=stress,principal_secants=mod,
        principal_shear_secant=g,material_domain_valid=response['prepeak_valid'] and response['backbone_valid'])


class LayeredSlabSecant:
    def __init__(self,thickness_in,fc_ksi,layers,*,ec_ksi,poisson=.2,
                 concrete_peak_strain=None,steel_strain_limit=.01,quadrature_order=16,
                 void_only_layers=(),concrete_void_profile=None):
        self.ec=float(ec_ksi);self.nu=float(poisson)
        if not math.isfinite(self.ec) or self.ec<=0 or not math.isfinite(self.nu) or not 0<=self.nu<.5:
            raise ValueError('Positive Ec and physical uncracked Poisson ratio required')
        peak=(design_concrete_peak_strain(fc_ksi,self.ec)['peak_strain'] if concrete_peak_strain is None
              else float(concrete_peak_strain))
        self.void_only_ids={r['id'] for r in void_only_layers}
        self.void_profile=concrete_void_profile
        if concrete_void_profile is not None and (not callable(getattr(concrete_void_profile,'occupied',None)) or
                not getattr(concrete_void_profile,'identity',None)):
            raise ValueError('Explicit finite-cell void profile and source identity required')
        if len(self.void_only_ids)!=len(void_only_layers):raise ValueError('Unique externally owned concrete void identities required')
        self.kernel=LayeredSlabSection(thickness_in,fc_ksi,list(layers)+list(void_only_layers),concrete_peak_strain=peak,
            concrete_model='polak1993_thorenfeldt',compression_strain_limit=peak,
            steel_strain_limit=steel_strain_limit,quadrature_order=quadrature_order,allow_unreinforced=True,
            actual_cell_homogenization=concrete_void_profile is not None)
        if any(row['concrete_owner']!='slab' for row in self.kernel.layers):
            raise ValueError('This slab-own operator cannot also own beam web concrete/bars')
        self.elastic=self.ec/(1-self.nu**2)*np.array([[1,self.nu,0],[self.nu,1,0],[0,0,(1-self.nu)/2]])
        # Source uses MPa: direct tensile cracking stress, not beam rupture fr.
        self.fcr=.33*math.sqrt(self.kernel.fc*6.894757293168)/6.894757293168

    def _assemble(self,q,branch,order):
        k=self.kernel;z,w=k._points(q,order,extra_breaks=() if self.void_profile is None else self.void_profile.breaks);eps=strain_tensors(q,z)
        occupied=np.zeros(len(z))
        for row in k.layers:
            occupied+=2*np.sqrt(np.maximum(0,row['radius']**2-(z-row['z_in'])**2))*row['count_per_in']
        if self.void_profile is not None:occupied=np.asarray(self.void_profile.occupied(z),dtype=float)
        if occupied.shape!=z.shape or not np.isfinite(occupied).all() or np.any(occupied<0):
            raise ValueError('Finite nonnegative actual concrete void profile required')
        if np.any(occupied>1+1e-10):raise ValueError('Actual finite bar voids overlap or exceed local concrete volume')
        occupied=np.minimum(occupied,1.) # exact all-steel slices have zero concrete
        K=np.zeros((6,6));R=np.zeros(6);steel=[];maximum_tension=-math.inf;minimum_compression=math.inf
        domains=True;minimum_g=math.inf
        if branch=='restricted_cracked':
            state=principal_secant_matrices(eps,k.fc,peak_strain=k.peak_strain,compression_strain_limit=k.compression_limit)
            d=state['matrix'];sigma=state['stress'];domains=state['material_domain_valid']
            minimum_g=float(np.min(state['principal_shear_secant']))
        else:
            d=np.broadcast_to(self.elastic,(len(z),3,3));ev=np.c_[eps[:,0,0],eps[:,1,1],2*eps[:,0,1]]
            sigma=ev@self.elastic.T
            st=np.empty((len(z),2,2));st[:,0,0]=sigma[:,0];st[:,1,1]=sigma[:,1];st[:,0,1]=st[:,1,0]=sigma[:,2]
            eig=np.linalg.eigvalsh(st);maximum_tension=float(eig[:,-1].max());minimum_compression=float(eig[:,0].min())
        weight=(1-occupied)*w
        blocks=[np.einsum('n,nij->ij',weight*z**i,d) for i in range(3)]
        K=np.block([[blocks[0],blocks[1]],[blocks[1],blocks[2]]])
        R=np.r_[weight@sigma,(weight*z)@sigma]
        for row in k.layers:
            if row['id'] in self.void_only_ids:continue
            j=0 if row['axis']=='x' else 1
            b=np.zeros(6);b[j]=1;b[j+3]=row['z_in'];e=float(b@q)
            trial=row['es_ksi']*e
            stress=float(np.clip(trial,-row['fy_ksi'],row['fy_ksi'])) if branch=='restricted_cracked' else trial
            es=stress/e if e!=0 else row['es_ksi']
            K+=row['area_in2_per_in']*es*np.outer(b,b);R+=row['area_in2_per_in']*stress*b
            steel.append(dict(id=row['id'],strain=e,stress_ksi=stress,secant_ksi=es,
                              yield_utilization=abs(trial)/row['fy_ksi']))
        return K,R,dict(steel=steel,material_domain_valid=bool(domains),
            minimum_principal_concrete_shear_secant_ksi=minimum_g if branch=='restricted_cracked' else self.ec/(2*(1+self.nu)),
            max_concrete_tensile_stress_ksi=maximum_tension if branch=='uncracked_elastic' else 0.,
            min_concrete_compressive_stress_ksi=minimum_compression if branch=='uncracked_elastic' else None)

    def _relaxed_energy(self,q,branch,order,membrane_response):
        """Integrate the SAME zero-incremental-N energy without D-B'A^-1B.

        At depth z, an incremental curvature has strain (M+zI) kappa, where
        M=-A^+B is the already computed stationary membrane response. Each
        concrete principal mode and actual steel bar contributes a positive
        weighted square. This avoids cancellation of large A/B/D terms in
        nearly unreinforced cracked edge cells. No eigenvalue is projected,
        no positive stiffness floor is supplied, and true zero modes remain.
        """
        k=self.kernel;z,w=k._points(q,order,extra_breaks=() if self.void_profile is None else self.void_profile.breaks)
        occupied=np.zeros(len(z))
        for row in k.layers:occupied+=2*np.sqrt(np.maximum(0,row['radius']**2-(z-row['z_in'])**2))*row['count_per_in']
        if self.void_profile is not None:occupied=np.asarray(self.void_profile.occupied(z),float)
        if occupied.shape!=z.shape or not np.isfinite(occupied).all() or np.any(occupied<0) or np.any(occupied>1+1e-10):
            raise ValueError('Valid actual concrete void fractions required for relaxed energy')
        if branch=='restricted_cracked':
            factors=principal_secant_matrices(strain_tensors(q,z),k.fc,peak_strain=k.peak_strain,
                compression_strain_limit=k.compression_limit)['energy_factors']
        else:factors=np.broadcast_to(np.linalg.cholesky(self.elastic).T,(len(z),3,3))
        strain_map=np.asarray(membrane_response,float)[None,:,:]+z[:,None,None]*np.eye(3)
        modes=np.einsum('nki,nij->nkj',factors,strain_map)
        weighted=(np.sqrt((1-np.minimum(occupied,1.))*w)[:,None,None]*modes).reshape(-1,3)
        result=weighted.T@weighted
        for row in k.layers:
            if row['id'] in self.void_only_ids:continue
            j=0 if row['axis']=='x' else 1;e=q[j]+row['z_in']*q[j+3]
            trial=row['es_ksi']*e
            stress=float(np.clip(trial,-row['fy_ksi'],row['fy_ksi'])) if branch=='restricted_cracked' else trial
            secant=stress/e if e!=0 else row['es_ksi']
            vector=np.asarray(membrane_response[j],float).copy();vector[j]+=row['z_in']
            result+=row['area_in2_per_in']*secant*np.outer(vector,vector)
        return result

    def evaluate(self,generalized_strain,*,branch,verify_quadrature=True,maximum_quadrature_order=64):
        if branch not in ('restricted_cracked','uncracked_elastic'):
            raise ValueError('Declare restricted_cracked or uncracked_elastic constitutive analysis')
        q=np.asarray(generalized_strain,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Six finite common strains required')
        K,R,state=self._assemble(q,branch,self.kernel.order)
        check=None;order=self.kernel.order
        if verify_quadrature:
            if type(maximum_quadrature_order) is not int or not self.kernel.order<=maximum_quadrature_order<=64:
                raise ValueError('Explicit maximum coarse quadrature order between initial order and64 required')
            order=self.kernel.order;attempts=[]
            while True:
                fine,finer,_=self._assemble(q,branch,order*2)
                error=np.abs(fine-K);force_error=np.abs(finer-R)
                passed=bool(np.all(error<=1e-6+1e-5*np.maximum(abs(fine),abs(K))) and
                            np.all(force_error<=1e-7+1e-5*np.maximum(abs(finer),abs(R))))
                check=dict(passed=passed,max_matrix_absolute_error=float(error.max()),
                    max_force_absolute_error=float(force_error.max()),orders=[order,order*2])
                attempts.append(dict(check))
                if passed or order==maximum_quadrature_order:break
                order=min(order*2,maximum_quadrature_order)
                K,R,state=self._assemble(q,branch,order)
            check['attempts']=attempts;check['maximum_coarse_order']=maximum_quadrature_order
        reconstruction=K@q-R
        reasons=[]
        if branch=='restricted_cracked':
            if self.void_profile is None:
                independent=self.kernel.state(q)
                target=np.array(independent['nominal_resultants'])
                for row in independent['steel']:
                    if row['id'] in self.void_only_ids:
                        j=0 if row['axis']=='x' else 1
                        target[j]-=row['force_kip_per_in'];target[j+3]-=row['z_in']*row['force_kip_per_in']
                if not np.all(abs(R-target)<=1e-7+1e-6*np.maximum(abs(R),abs(target))):
                    reasons.append('secant_force_differs_from_independent_named_section_response')
                reasons.extend(v for v in independent['material_domain_errors'] if v!='reinforcement_strain_exceeds_declared_limit')
            else:
                # Actual partial-cell circles are not the uniform layer-density
                # geometry of the strength strip. Check the same named law at
                # depth endpoints (minimum eigenstrain is concave in depth),
                # and retain the direct stress/quadrature reconstruction above.
                endpoints=concrete_response(strain_tensors(q,[-self.kernel.h/2,self.kernel.h/2]),
                    self.kernel.fc,peak_strain=self.kernel.peak_strain,compression_strain_limit=self.kernel.compression_limit,
                    model='polak1993_thorenfeldt')
                if not endpoints['prepeak_valid']:reasons.append('concrete_compression_exceeds_declared_prepeak_limit')
                if not endpoints['backbone_valid']:reasons.append('concrete_backbone_n_not_above_one')
            if max((abs(v['strain']) for v in state['steel']),default=0)>self.kernel.steel_limit+1e-12:
                reasons.append('reinforcement_strain_exceeds_declared_limit')
        else:
            if state['max_concrete_tensile_stress_ksi']>self.fcr*(1+1e-10):reasons.append('elastic_predictor_exceeds_published_concrete_cracking_stress')
            if state['min_concrete_compressive_stress_ksi'] < -self.kernel.fc*(1+1e-10):reasons.append('elastic_predictor_exceeds_concrete_compressive_strength')
            if max((s['yield_utilization'] for s in state['steel']),default=0)>1+1e-10:reasons.append('elastic_predictor_exceeds_actual_steel_yield')
        if not np.all(abs(reconstruction)<=1e-7+1e-6*abs(R)):reasons.append('secant_reconstruction_failed')
        # The dimensional curvature scale makes eigenvalue/rank inspection
        # independent of selecting inches versus another consistent length unit.
        scale=np.diag([1,1,1,2/self.kernel.h,2/self.kernel.h,2/self.kernel.h])
        eig=np.linalg.eigvalsh(scale@K@scale);threshold=1e-11*max(1.,max(abs(eig)))
        if min(eig)<-threshold:reasons.append('negative_secant_energy')
        rank=int(np.sum(eig>threshold))
        A=K[:3,:3];B=K[:3,3:];D=K[3:,3:]
        ae,av=np.linalg.eigh(A);atol=1e-11*max(1.,max(abs(ae)))
        inverse=(av*np.array([1/v if v>atol else 0 for v in ae]))@av.T
        relaxed=D-B.T@inverse@B
        membrane_response=-inverse@B
        range_tolerance=1e-7+1e-8*max(1.,float(np.max(abs(B))))
        original_range_error=float(np.max(abs(A@inverse@B-B)))
        response_range_error=float(np.max(abs(A@membrane_response+B)))
        membrane_recovery=dict(applied=False,method='ordinary_retained_eigen_inverse',
            original_left_associated_range_error=original_range_error,
            original_response_range_error=response_range_error,tolerance=range_tolerance)
        if max(original_range_error,response_range_error)>range_tolerance:
            # Solve for the required action directly instead of constructing an
            # ill-conditioned explicit inverse. Keep the original physical rank
            # and nullspace; no small mode is added or discarded by this repair.
            original_response=membrane_response.copy();basis=av[:,ae>atol]
            if basis.shape[1]==3:
                membrane_response=-np.linalg.solve(A,B)
            elif basis.shape[1]:
                membrane_response=-basis@np.linalg.solve(basis.T@A@basis,basis.T@B)
            else:membrane_response=np.zeros((3,3))
            membrane_recovery.update(applied=True,method='direct_retained_range_linear_solve',
                retained_rank=int(basis.shape[1]),original_response=original_response.tolist(),
                maximum_response_change=float(np.max(abs(membrane_response-original_response))),
                artificial_stiffness_added=False,rank_threshold_changed=False)
        # Use exactly the downstream operator's existing energy criterion to
        # detect cancellation. Unaffected matrices retain their original
        # arithmetic; the repair is an equivalent material-energy integral.
        rscale=max(1.,float(np.max(abs(relaxed))));raw_relaxed=relaxed.copy()
        energy_recovery=dict(applied=False,method='ordinary_schur_subtraction')
        if (membrane_recovery['applied'] or np.max(abs(relaxed-relaxed.T))>1e-9*rscale or
                np.linalg.eigvalsh((relaxed+relaxed.T)/2).min() < -1e-9*rscale):
            relaxed=self._relaxed_energy(q,branch,order,membrane_response)
            finer_energy=self._relaxed_energy(q,branch,2*order,membrane_response)
            eerror=abs(relaxed-finer_energy)
            epass=bool(np.all(eerror<=1e-6+1e-5*np.maximum(abs(relaxed),abs(finer_energy))))
            if not epass:reasons.append('relaxed_bending_energy_quadrature_not_converged')
            energy_recovery=dict(applied=True,method='integrated_stationary_material_energy',
                original_D=raw_relaxed.tolist(),maximum_absolute_change=float(np.max(abs(relaxed-raw_relaxed))),
                quadrature=dict(passed=epass,orders=[order,2*order],maximum_absolute_error=float(eerror.max())),
                eigenvalue_projection_applied=False,stiffness_floor_added=False)
        rscale=max(1.,float(np.max(abs(relaxed))))
        if (np.max(abs(relaxed-relaxed.T))>1e-9*rscale or
                np.linalg.eigvalsh((relaxed+relaxed.T)/2).min() < -1e-9*rscale):
            reasons.append('relaxed_bending_negative_or_nonsymmetric_energy')
        range_error=float(np.max(abs(A@membrane_response+B)))
        membrane_recovery.update(final_response_range_error=range_error,passed=range_error<=range_tolerance)
        if range_error>range_tolerance:reasons.append('membrane_gauge_range_inconsistent')
        if check is not None and not check['passed']:reasons.append('secant_quadrature_not_converged')
        return dict(method=METHOD,branch=branch,generalized_strain=q.tolist(),
            ABD=K.tolist(),A=A.tolist(),B=B.tolist(),D=D.tolist(),
            zero_incremental_N_relaxed_D=relaxed.tolist(),
            relaxed_bending_energy_recovery=energy_recovery,
            membrane_relaxation_recovery=membrane_recovery,
            zero_incremental_N_membrane_response=membrane_response.tolist(),
            nominal_resultants=R.tolist(),secant_reconstruction_residual=reconstruction.tolist(),
            constitutive_state_verified=not reasons,domain_reasons=reasons,
            rank=rank,rank_deficient=rank<6,scaled_eigenvalues=eig.tolist(),
            quadrature=check,material_state=state,
            concrete_void_only_layer_ids=sorted(self.void_only_ids),
            concrete_void_profile_identity=None if self.void_profile is None else self.void_profile.identity,
            slab_steel_layer_ids=[r['id'] for r in self.kernel.layers if r['id'] not in self.void_only_ids],
            fcr_ksi=self.fcr,poisson=self.nu if branch=='uncracked_elastic' else 0.,
            constitutive_iteration_required=branch=='restricted_cracked',
            full_floor_equilibrium_converged=False,force_bound_claimed=False,
            strength_phi_applied=False,transverse_shear_model_included=False)
