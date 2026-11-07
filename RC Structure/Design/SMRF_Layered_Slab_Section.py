"""Six-resultant, common-strain layered slab research kernel.

Units: inches, ksi, N=[Nxx,Nyy,Nxy] kip/in and M=[Mxx,Myy,Mxy]
kip-in/in. Tension is positive; z is upward from the slab midplane.
q=[ex,ey,gxy,kx,ky,kxy], with ENGINEERING shear strain/twist:
epsilon(z)=[[ex+z*kx,(gxy+z*kxy)/2],[...,ey+z*ky]].
Mij=int(z*sigmaij dz), so Mxy is not doubled in the output.

Concrete uses the prepeak Vecchio/Collins (1986) Eq.18a/b compression law:
fc_eff=fc/max(1,.8+.34*e1/peak); sigma2=-fc_eff*(2*r-r*r), r=-e2/peak.
The positive backbone peak magnitude is supplied explicitly; the often-used
170 coefficient is only the specialization peak=.002. The primary test
program used measured peak magnitudes from .0018 to .0028.
Zero concrete tension, no tension stiffening, no steel hardening or biaxial
compression enhancement. This is a restricted compression-field adaptation,
NOT the full MCFT/CMM model. The finite compression region follows the actual
strain field, not a chosen sandwich depth. Actual finite circular bar volume
is removed from concrete; each bar inventory enters only once.

This module exposes equilibrium, quadrature and material-domain checks
separately. It NEVER declares ACI code qualification. ACI 318-19 22.2.2.3
requires comprehensive-test agreement for an alternative stress distribution;
22.2.2.4 is the rectangular block's deemed compliance, not this model's.
Transverse shear, anchorage, bond/crack slip and cyclic behaviour are outside
this kernel. See action_reference/LAYERED_SECTION_METHOD.txt.

The named polak1993_thorenfeldt option replaces the compression backbone by
the directly published layered-shell Eq17-23/Fig3(a), including simultaneous
principal compression without a confinement bonus. Tension/hardening remain
omitted; this is its restricted strength formulation, not the full APECS
load-deformation law. Source-backed specimen comparisons are saved separately.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

METHOD = 'common_strain_prepeak_compression_field_v2'
SOURCE = 'Vecchio and Collins (1986), ACI Journal 83(2), p.224, Eq.18a/b'
_C0 = .002
_MPA_PER_KSI = 6.894757293168
_CONCRETE_MODELS = ('vecchio1986_parabola','polak1993_thorenfeldt')


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name} requires {size} finite values')
    return a


def strain_tensors(q, z):
    q = _vector(q, 6, 'Generalized strain')
    z = np.atleast_1d(np.asarray(z, dtype=float))
    e = q[:3] + z[:, None]*q[3:]
    tensor = np.zeros((len(z), 2, 2))
    tensor[:, 0, 0], tensor[:, 1, 1] = e[:, 0], e[:, 1]
    tensor[:, 0, 1] = tensor[:, 1, 0] = e[:, 2]/2
    return tensor


def concrete_response(strain, fc_ksi, *, peak_strain=.002,
                      model='vecchio1986_parabola',compression_strain_limit=None):
    """Vectorized spectral response, plus explicit restricted-domain flags.

    Beyond peak, the numerical continuation holds the peak stress ONLY so a
    nonlinear residual can be evaluated. Such a state is inadmissible, never
    a plateau-capacity credit. Independent compression in both principal
    directions is evaluated for diagnostics but is unqualified.
    """
    e = np.asarray(strain, dtype=float)
    if e.shape[-2:] != (2, 2) or not np.isfinite(e).all():
        raise ValueError('Finite symmetric 2x2 strain tensors required')
    if not np.allclose(e, np.swapaxes(e, -1, -2), atol=1e-15, rtol=0):
        raise ValueError('Strain tensors must be symmetric')
    fc = float(fc_ksi)
    if not math.isfinite(fc) or fc <= 0:
        raise ValueError('Positive finite concrete strength required')
    peak_strain=float(peak_strain)
    if not math.isfinite(peak_strain) or not 0<peak_strain<=.003:
        raise ValueError('Explicit compression backbone peak strain must be in (0,.003]')
    if model not in _CONCRETE_MODELS:raise ValueError('Unknown concrete constitutive model')
    limit=min(_C0,peak_strain) if compression_strain_limit is None else float(compression_strain_limit)
    if not math.isfinite(limit) or not 0<limit<=min(peak_strain,.003):
        raise ValueError('Compression limit must be positive and at most measured/design peak and .003')
    principal, axes = np.linalg.eigh(e)
    transverse_tension = np.maximum(principal[..., ::-1], 0.)
    peak = fc/np.maximum(1., .8+.34*transverse_tension/peak_strain)
    ratio = np.clip(-principal/peak_strain, 0., 1.)
    n=.8+peak*_MPA_PER_KSI/17.
    backbone_valid=bool(np.all(n>1.)) if model=='polak1993_thorenfeldt' else True
    if model=='polak1993_thorenfeldt':
        # Polak/Vecchio1993 pp3445-3447 Eq17-23 and Fig3(a): prepeak k=1.
        # n uses the softened principal peak strength, expressed in MPa.
        # n<=1 is outside the monotonically rising prepeak form; numerical
        # evaluation remains finite but such states cannot be accepted.
        safe_n=np.maximum(n,1.+1e-9)
        principal_stress=-peak*safe_n*ratio/(safe_n-1+ratio**safe_n)
    else:
        principal_stress = -peak*(2*ratio-ratio**2)
    stress = np.einsum('...ik,...k,...jk->...ij', axes, principal_stress, axes)
    return dict(stress_ksi=stress, principal_strain=principal,
                effective_strength_ksi=peak,
                prepeak_valid=bool(np.all(principal >= -limit-1e-12)),
                backbone_valid=backbone_valid,model=model,
                biaxial_compression_formulation_supported=model=='polak1993_thorenfeldt',
                biaxial_compression=bool(np.any(principal[..., 1] < -1e-12)))


def design_concrete_peak_strain(fc_ksi,ec_ksi):
    """Derived Thorenfeldt peak for an explicitly chosen uncracked modulus.

    FHWA-HRT-05-058 (2006), Chapter3 Eq5/6 explicitly uses this relation.
    Polak/Vecchio1993 Eq17/18 gives n=.8+fc(MPa)/17. Its derivative at
    zero compression is fc*n/((n-1)*peak). Equating that slope to the actual
    model's declared Ec gives peak=fc/Ec*n/(n-1). This is a transparent
    constitutive calibration to Ec, not a claim of a measured cylinder peak.
    ACI318-19 19.2.2.1 supplies Ec when applicable; caller owns its density/
    strength assumptions. Published-specimen comparisons use measured peaks.
    """
    fc,ec=float(fc_ksi),float(ec_ksi)
    if not all(math.isfinite(v) and v>0 for v in (fc,ec)):
        raise ValueError('Positive finite concrete strength and elastic modulus required')
    n=.8+fc*_MPA_PER_KSI/17.
    if n<=1:raise ValueError('Thorenfeldt prepeak requires n>1')
    peak=fc/ec*n/(n-1)
    if not 0<peak<=.003:raise ValueError('Derived peak exceeds declared ACI strain domain')
    return dict(peak_strain=peak,fc_ksi=fc,ec_ksi=ec,n=n,
                provenance='FHWA_HRT_05_058_Chapter3_Eq5_6',measured=False)


def compose_strain_fields(*fields):
    """Add compatible kinematic fields, never separately computed strengths.

    Each field has the same reference axis and q convention. A caller adding
    global beam curvature must first REMOVE its existing contribution from
    the plate field; this helper cannot establish that physical projection.
    """
    if not fields:
        raise ValueError('At least one strain field required')
    return np.sum([_vector(q, 6, 'Strain field') for q in fields], axis=0)


class LayeredSlabSection:
    def __init__(self, thickness_in, fc_ksi, layers, *, phi=.65,
                 steel_strain_limit=.01, quadrature_order=16,
                 cut_width_in=1., drop_webs=(), concrete_peak_strain=.002,
                 concrete_model='vecchio1986_parabola',compression_strain_limit=None,
                 allow_unreinforced=False,actual_cell_homogenization=False):
        """Layers are {id,z_in,axis:'x'|'y',area_in2_per_in,bar_area_in2,
        fy_ksi,es_ksi(optional)}; actual x/y mats occupy distinct elevations.

        phi is a deliberately conservative constant for N/M (<=.65). The
        applicable seismic diaphragm shear phi must ALSO be supplied to solve;
        a .60 requirement cannot be satisfied by calling .65 conservative.
        steel_strain_limit is a declared restriction, not a rupture prediction.
        """
        self.h, self.fc, self.phi = map(float, (thickness_in, fc_ksi, phi))
        self.steel_limit = float(steel_strain_limit)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.h, self.fc, self.phi, self.steel_limit)) or self.phi > .65:
            raise ValueError('Positive finite inputs and phi <= .65 required')
        if not isinstance(quadrature_order, int) or not 4 <= quadrature_order <= 64:
            raise ValueError('Quadrature order must be an integer from 4 to 64')
        self.order = quadrature_order
        self.peak_strain=float(concrete_peak_strain)
        if not math.isfinite(self.peak_strain) or not 0<self.peak_strain<=.003:
            raise ValueError('Explicit compression backbone peak strain must be in (0,.003]')
        if concrete_model not in _CONCRETE_MODELS:raise ValueError('Unknown concrete model')
        self.concrete_model=concrete_model
        self.compression_limit=(min(_C0,self.peak_strain) if compression_strain_limit is None
                                else float(compression_strain_limit))
        if not math.isfinite(self.compression_limit) or not 0<self.compression_limit<=min(self.peak_strain,.003):
            raise ValueError('Compression limit must be positive and at most backbone peak and .003')
        self.cut_width=float(cut_width_in)
        if not math.isfinite(self.cut_width) or self.cut_width<=0:
            raise ValueError('Positive physical strip cut width required')
        self.webs=[]
        for item in drop_webs:
            web=dict(item)
            if not web.get('id') or web['id']=='slab' or any(v['id']==web['id'] for v in self.webs):
                raise ValueError('Distinct non-slab web identity required')
            if web.get('axis') not in ('x','y'):
                raise ValueError('Drop web requires its longitudinal x or y axis')
            for key in ('z_bottom_in','z_top_in','width_in'):
                web[key]=float(web[key])
                if not math.isfinite(web[key]):raise ValueError('Finite web geometry required')
            if not web['z_bottom_in']<web['z_top_in']<=-self.h/2+1e-12:
                raise ValueError('Drop web concrete must lie entirely below slab concrete')
            if not 0<web['width_in']<=self.cut_width:
                raise ValueError('Web width must fit declared physical strip width')
            # A crossing beam joint requires its own three-dimensional treatment.
            # Two concrete webs cannot occupy the same scalar integration region.
            if any(min(web['z_top_in'],v['z_top_in'])>max(web['z_bottom_in'],v['z_bottom_in'])+1e-12 for v in self.webs):
                raise ValueError('Overlapping web regions or beam intersections unsupported')
            self.webs.append(web)
        self.layers = []
        ids = set()
        for row in layers:
            item = dict(row)
            if not str(item.get('id', '')) or item['id'] in ids:
                raise ValueError('Every physical steel layer needs a unique id')
            ids.add(item['id'])
            if item.get('axis') not in ('x', 'y'):
                raise ValueError('Only orthogonal actual reinforcement supported')
            for key in ('z_in', 'area_in2_per_in', 'bar_area_in2', 'fy_ksi'):
                item[key] = float(item[key])
                if not math.isfinite(item[key]):
                    raise ValueError('Finite steel geometry/materials required')
            item['es_ksi'] = float(item.get('es_ksi', 29000.))
            if not math.isfinite(item['es_ksi']):
                raise ValueError('Finite steel elastic modulus required')
            if min(item[k] for k in ('area_in2_per_in', 'bar_area_in2',
                                    'fy_ksi', 'es_ksi')) <= 0:
                raise ValueError('Positive steel areas/materials required')
            item['radius'] = math.sqrt(item['bar_area_in2']/math.pi)
            item['physical_radius']=float(item.get('db_in',2*item['radius']))/2
            if not math.isfinite(item['physical_radius']) or item['physical_radius']<=0:
                raise ValueError('Positive finite actual physical bar diameter required')
            item['count_per_in'] = item['area_in2_per_in']/item['bar_area_in2']
            item['concrete_owner']=item.get('concrete_owner','slab')
            if item['concrete_owner']=='slab':
                lo,hi=-self.h/2,self.h/2
            else:
                owners=[v for v in self.webs if v['id']==item['concrete_owner']]
                if not owners:raise ValueError('Bar concrete owner is absent')
                owner=owners[0]
                if item['axis']!=owner['axis']:raise ValueError('Web bar direction differs from web axis')
                lo,hi=owner['z_bottom_in'],owner['z_top_in']
            if item['z_in']-item['physical_radius']<lo-1e-12 or item['z_in']+item['physical_radius']>hi+1e-12:
                raise ValueError('Physical reinforcement extends outside its concrete region')
            if not actual_cell_homogenization and 2*item['physical_radius']*item['count_per_in'] >= 1:
                raise ValueError('Bar spacing is not physically available')
            for old in self.layers:
                if old['axis'] != item['axis'] and abs(old['z_in']-item['z_in']) < old['physical_radius']+item['physical_radius']-1e-12:
                    raise ValueError('Orthogonal bars intersect physically; use actual stacked elevations')
            self.layers.append(item)
        if not self.layers and not allow_unreinforced:
            raise ValueError('Actual reinforcement inventory is required')
        if sum(v['area_in2_per_in'] for v in self.layers) >= self.h:
            raise ValueError('Steel occupies the entire slab')

    def _points(self, q, order, *, region=None,extra_breaks=()):
        # Split all material edges and roots det(epsilon(z))=0: this accurately
        # integrates finite tension/compression zones instead of missing a thin
        # compressed layer between fixed through-thickness integration points.
        low,high=(-self.h/2,self.h/2) if region is None else (region['z_bottom_in'],region['z_top_in'])
        owner='slab' if region is None else region['id']
        breaks = [low,high]
        breaks.extend(float(v) for v in extra_breaks if low<float(v)<high)
        for row in self.layers:
            if row['concrete_owner']!=owner:continue
            breaks.extend(v for v in (row['z_in']-row['radius'],row['z_in']+row['radius']) if low<v<high)
        # Eigenstrain zero locations must not depend on the magnitude of q.
        # Directly thresholding the quadratic coefficients loses all roots
        # for roundoff-sized but nonzero rigid-region curvature fields.
        depth_scale=max(abs(low),abs(high),self.h/2)
        normalized=np.asarray(q,float)*np.array([1,1,1,depth_scale,depth_scale,depth_scale])
        amplitude=float(np.max(abs(normalized)))
        if amplitude>0:normalized=normalized/amplitude
        a,b,c,d,e,f = normalized
        polynomial = [d*e-f*f/4, a*e+b*d-c*f/2, a*b-c*c/4]
        nz = np.flatnonzero(np.asarray(polynomial)!=0)
        roots = np.roots(polynomial[nz[0]:]) if len(nz) else []
        # Rank-one uniaxial states have det(epsilon)==0 over the entire depth;
        # their nonzero eigenvalue crosses zero with the trace instead.
        roots = list(roots)
        if d+e!=0:
            roots.append(complex(-(a+b)/(d+e)))
        # The affine shear-zero point also resolves a repeated determinant
        # root in pure shear; generic polynomial eigensolvers can give that
        # mathematically real double root a tiny imaginary roundoff part.
        for intercept,slope in ((a,d),(b,e),(c,f)):
            if slope!=0:roots.append(complex(-intercept/slope))
        for value in roots:
            value=value*depth_scale
            if abs(value.imag) < 1e-10 and low < value.real < high:
                breaks.append(float(value.real))
        breaks = np.unique(breaks)
        x, w = leggauss(order)
        z, weights = [], []
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            # Cosine transform removes circular bar-void endpoint square-root
            # singularities; quadrature order is checked by an independent rerun.
            t=(x+1)*math.pi/2
            z.extend((lo+hi)/2-(hi-lo)*np.cos(t)/2)
            weights.extend(w*(hi-lo)*math.pi*np.sin(t)/4)
        return np.asarray(z), np.asarray(weights)

    def state(self, generalized_strain, *, quadrature_order=None):
        q = _vector(generalized_strain, 6, 'Generalized strain')
        order = self.order if quadrature_order is None else int(quadrature_order)
        if not 4 <= order <= 128:
            raise ValueError('Invalid integration order')
        z,w = self._points(q,order)
        concrete = concrete_response(strain_tensors(q,z),self.fc,peak_strain=self.peak_strain,
                                     model=self.concrete_model,compression_strain_limit=self.compression_limit)
        occupied = np.zeros(len(z))
        steel = np.zeros(6)
        bar_states = []
        for row in self.layers:
            if row['concrete_owner']=='slab':
                occupied += 2*np.sqrt(np.maximum(0.,row['radius']**2-(z-row['z_in'])**2))*row['count_per_in']
            axis = 0 if row['axis']=='x' else 1
            strain = q[axis]+row['z_in']*q[axis+3]
            stress = float(np.clip(row['es_ksi']*strain,-row['fy_ksi'],row['fy_ksi']))
            force = row['area_in2_per_in']*stress
            steel[axis] += force
            steel[axis+3] += row['z_in']*force
            bar_states.append(dict(id=row['id'],axis=row['axis'],z_in=row['z_in'],
                                   concrete_owner=row['concrete_owner'],
                                   strain=float(strain),stress_ksi=stress,
                                   force_kip_per_in=float(force),
                                   yielded=bool(abs(stress)>=row['fy_ksi']*(1-1e-12))))
        if np.max(occupied,initial=0.)>=1:
            raise ValueError('Combined physical bar volume exceeds available concrete')
        stresses = concrete['stress_ksi'][:,(0,1,0),(0,1,1)]
        density = stresses*(1-occupied[:,None])
        nc = np.sum(density*w[:,None],axis=0)
        mc = np.sum(density*(w*z)[:,None],axis=0)
        rc = np.r_[nc,mc]
        slab_concrete=rc.copy()
        web_states=[]
        web_prepeak=True
        web_backbone=True
        for web in self.webs:
            wz,ww=self._points(q,order,region=web)
            axis=0 if web['axis']=='x' else 1
            we=q[axis]+wz*q[axis+3]
            web_tensor=np.zeros((len(wz),2,2));web_tensor[:,0,0]=we
            wr=concrete_response(web_tensor,self.fc,peak_strain=self.peak_strain,
                                 model=self.concrete_model,compression_strain_limit=self.compression_limit)
            ws=wr['stress_ksi'][:,0,0]
            web_backbone=web_backbone and wr['backbone_valid']
            net=np.full(len(wz),web['width_in']/self.cut_width)
            for row in self.layers:
                if row['concrete_owner']==web['id']:
                    net-=2*np.sqrt(np.maximum(0.,row['radius']**2-(wz-row['z_in'])**2))*row['count_per_in']
            if np.min(net)<=0:raise ValueError('Web bars exhaust local concrete area')
            contribution=np.zeros(6)
            contribution[axis]=np.sum(ws*net*ww)
            contribution[axis+3]=np.sum(ws*net*ww*wz)
            rc+=contribution
            endpoint=q[axis]+np.array([web['z_bottom_in'],web['z_top_in']])*q[axis+3]
            valid=bool(np.min(endpoint)>=-self.compression_limit-1e-12)
            web_prepeak=web_prepeak and valid
            web_states.append(dict(id=web['id'],axis=web['axis'],
                                   concrete_nominal_resultants=contribution.tolist(),
                                   concrete_net_volume_per_unit_area_in=float(np.sum(net*ww)),
                                   minimum_axial_strain=float(min(endpoint)),prepeak_valid=valid))
        # Smallest eigenvalue is concave in z: its minimum is at an endpoint.
        # Largest eigenvalue is convex: its compression may occur internally;
        # use integration points plus endpoints for scope reporting.
        extremes = concrete_response(strain_tensors(q,np.r_[z,-self.h/2,self.h/2]),self.fc,peak_strain=self.peak_strain,
                                     model=self.concrete_model,compression_strain_limit=self.compression_limit)
        reasons=[]
        if not extremes['prepeak_valid'] or not web_prepeak: reasons.append('concrete_compression_exceeds_declared_prepeak_limit')
        if extremes['biaxial_compression'] and not extremes['biaxial_compression_formulation_supported']:
            reasons.append('biaxial_compression_constitutive_qualification_required')
        if not extremes['backbone_valid'] or not web_backbone:reasons.append('concrete_backbone_n_not_above_one')
        if max((abs(v['strain']) for v in bar_states),default=0.)>self.steel_limit+1e-12:
            reasons.append('reinforcement_strain_exceeds_declared_limit')
        nominal=rc+steel
        return dict(method=METHOD,generalized_strain=q.tolist(),
                    nominal_resultants=nominal.tolist(),design_resultants=(self.phi*nominal).tolist(),
                    concrete_nominal_resultants=rc.tolist(),steel_nominal_resultants=steel.tolist(),
                    slab_concrete_nominal_resultants=slab_concrete.tolist(),
                    drop_webs=web_states,cut_width_in=self.cut_width,
                    full_strip_nominal_resultants=(self.cut_width*nominal).tolist(),
                    phi=self.phi,steel=bar_states,
                    concrete_backbone_peak_strain=self.peak_strain,
                    concrete_compression_strain_limit=self.compression_limit,
                    concrete_model=self.concrete_model,
                    concrete_min_principal_strain=float(extremes['principal_strain'].min()),
                    concrete_max_principal_strain=float(extremes['principal_strain'].max()),
                    concrete_min_softened_strength_ksi=float(extremes['effective_strength_ksi'].min()),
                    concrete_net_volume_per_unit_area_in=float(np.sum((1-occupied)*w)),
                    material_domain_valid=not reasons,material_domain_errors=reasons,
                    quadrature_order=order,quadrature_points=len(z),
                    code_qualified=False,transverse_shear_verified=False)

    def verify_quadrature(self,q,*,relative_tolerance=1e-5,absolute_tolerance=1e-7):
        coarse=self.state(q)
        fine=self.state(q,quadrature_order=2*self.order)
        a,b=np.asarray(coarse['nominal_resultants']),np.asarray(fine['nominal_resultants'])
        error=np.abs(a-b)
        passed=bool(np.all(error<=absolute_tolerance+relative_tolerance*np.maximum(abs(a),abs(b))))
        return dict(passed=passed,absolute_error=error.tolist(),
                    coarse_order=self.order,fine_order=2*self.order,
                    relative_tolerance=relative_tolerance,absolute_tolerance=absolute_tolerance)

    def solve(self, factored_resultants, *, required_shear_phi,
              initial_strain=None, max_evaluations=250,
              relative_tolerance=1e-6, absolute_tolerance=1e-6):
        """Find one simultaneous compatible equilibrium state, not a capacity ray.

        A failed numerical solve is not proof of physical inadequacy. A returned
        equilibrated state is not code qualification. Caller must provide the
        applicable Chapter21 diaphragm shear phi, physical common-action mapping,
        development and transverse-shear qualifications independently.
        """
        target=_vector(factored_resultants,6,'Factored resultants')
        shear_phi=float(required_shear_phi)
        if not math.isfinite(shear_phi) or not 0<shear_phi<=1:
            raise ValueError('Applicable positive finite shear phi required')
        if not isinstance(max_evaluations,int) or not 1<=max_evaluations<=2000:
            raise ValueError('Bounded integer evaluation count required')
        # Dimensionless solver coordinates: membrane strain and surface strain
        # contributed by curvature, each scaled by 0.001.
        factor=np.array([1,1,1,2/self.h,2/self.h,2/self.h])*1e-3
        scale=np.array([self.fc*self.h]*3+[self.fc*self.h**2/4]*3)
        q0=np.zeros(6) if initial_strain is None else _vector(initial_strain,6,'Initial strain')
        def residual(x):
            return (np.asarray(self.state(x*factor)['design_resultants'])-target)/scale
        fit=least_squares(residual,q0/factor,max_nfev=max_evaluations,
                          ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale='jac')
        q=fit.x*factor
        state=self.state(q)
        error=np.asarray(state['design_resultants'])-target
        equal=bool(np.all(abs(error)<=absolute_tolerance+relative_tolerance*abs(target)))
        quadrature=self.verify_quadrature(q)
        phi_valid=self.phi<=shear_phi+1e-12 or np.max(abs(target[[2,5]]),initial=0.)<=absolute_tolerance
        state.update(factored_resultants=target.tolist(),equilibrium_residual=error.tolist(),
                     equilibrium_verified=equal,quadrature=quadrature,
                     shear_phi_valid=bool(phi_valid),required_shear_phi=shear_phi,
                     numerical_solver_success=bool(fit.success),
                     numerical_solver_message=str(fit.message),evaluations=int(fit.nfev),
                     restricted_model_state_verified=bool(equal and quadrature['passed'] and
                                                          state['material_domain_valid'] and phi_valid))
        return state

    def solve_load_path(self, factored_resultants, *, required_shear_phi,
                        steps=12,max_evaluations_per_step=250):
        """Bounded diagnostic of a proportional force path from the origin.

        Each step independently passes equilibrium, quadrature and material
        limits using the preceding compatible strain as the numerical seed.
        This detects isolated-root/branch concerns at the specified sampling;
        it is not a proof of continuous global stability or cyclic resistance.
        Failure remains a numerical/domain diagnostic, never proof of physical
        inadequacy. Use on governing witnesses; a complete path is recorded.
        """
        target=_vector(factored_resultants,6,'Factored resultants')
        if not isinstance(steps,int) or not 2<=steps<=128:
            raise ValueError('Path requires a bounded integer step count from2 to128')
        q=np.zeros(6);previous=np.zeros(6);path=[];final=None
        for factor in np.linspace(1/steps,1,steps):
            final=self.solve(factor*target,required_shear_phi=required_shear_phi,
                             initial_strain=q,max_evaluations=max_evaluations_per_step)
            next_q=np.array(final['generalized_strain'])
            # Conjugate virtual work uses engineering gamma/kxy with Nxy/Mxy.
            # Report its incremental slope sign; no unsourced tangent-eigenvalue
            # threshold is substituted for the actual constitutive/domain checks.
            work=float(np.dot(factor*target-previous,next_q-q))
            path.append(dict(load_fraction=float(factor),generalized_strain=next_q.tolist(),
                             equilibrium_verified=final['equilibrium_verified'],
                             material_domain_valid=final['material_domain_valid'],
                             quadrature_verified=final['quadrature']['passed'],
                             shear_phi_valid=final['shear_phi_valid'],
                             max_absolute_equilibrium_residual=max(abs(v) for v in final['equilibrium_residual']),
                             incremental_work_slope=work,evaluations=final['evaluations']))
            if not final['restricted_model_state_verified']:break
            q=next_q;previous=factor*target
        complete=len(path)==steps and bool(final['restricted_model_state_verified'])
        return dict(proportional_force_path_verified=complete,steps_requested=steps,
                    steps_completed=len(path),path=path,final_state=final,
                    continuous_stability_proven=False,code_qualified=False)


def common_field_ownership(section, generalized_strain, *, plane_reference_id,
                           action_projection_id, material_owner_id):
    """Expose a single inventory's shared stresses to the action integrator.

    The three provenance identities must refer to a verified upstream physical
    projection, not independently factored strain states. The caller may derive
    partial generalized forces for audit, but may NOT add multiple evaluations
    of this same concrete/steel to beam and slab capacities.
    """
    if not all(isinstance(v,str) and v.strip() for v in
               (plane_reference_id,action_projection_id,material_owner_id)):
        raise ValueError('Explicit common plane, action projection and ownership identities required')
    result=section.state(generalized_strain)
    result['ownership']=dict(plane_reference_id=plane_reference_id,
                             action_projection_id=action_projection_id,
                             material_owner_id=material_owner_id,
                             physical_steel_ids=[v['id'] for v in section.layers],
                             action_projection_verified=False)
    return result
