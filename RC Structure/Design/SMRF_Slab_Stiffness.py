"""Actual-mat cracked-strip stiffness and ACI318-19 effective inertia.

Units: inch,kip,ksi; section properties and moments are per inch of strip.
z is up from slabmidplane; M=int(z*sigma dz), tensionpositive. Consequently
positive M compresses the BOTTOM face. Caller must transform its own signs.

ACI24.2.3.1/.3 require cracking/reinforcement and actual two-way support effects;
R24.2.3.3 explicitly permits Ie24.2.3.5 for two-way slab immediate deflections.
This computes a ZERO-AXIAL-FORCE, uniaxial cracked reference for each axis/sign,
not a full biaxial/membrane stiffness law. It does not replace floor support/
restraint modeling. For factored force analysis, using this response-dependent
stiffness is an explicit detailed-analysis assumption under6.6.3.1.1/.2;
it is not a universal application of the .25Ig flat-plate table entry.

Concrete in tension is absent in Icr. Finite circles of EVERY physical bar
displace concrete; only bars parallel to the requested strip carry its axial
steel stress. Bars use centroid axial strain, as in standard transformed-section
analysis. Concrete void integrals are analytic, including partial circles at
the neutral axis. No concrete/steel strength or extra flange area is credited.
"""
from __future__ import annotations
import math
from scipy.optimize import brentq

from Design.SMRF_Layered_Slab_Section import LayeredSlabSection

METHOD='actual_mat_transformed_Icr_ACI31819_24_2_3_5_v1'


def _circle_integrals(radius,cut_from_center):
    """Integral of1,t,t² over circle area below t=cut, in its depth axis."""
    r=float(radius);t=float(cut_from_center)
    if t<=-r:return (0.,0.,0.)
    if t>=r:return (math.pi*r*r,0.,math.pi*r**4/4)
    root=math.sqrt(max(0.,r*r-t*t));angle=math.asin(t/r)+math.pi/2
    return (r*r*angle+t*root,-2*root**3/3,
            (r**4*angle+t*root*(2*t*t-r*r))/4)


class SlabStripStiffness:
    def __init__(self,thickness_in,fc_ksi,layers,*,ec_ksi,lambda_concrete=1.):
        # Share only the already-tested physical-inventory validation. This
        # helper does not invoke the nonlinear strength model for stiffness.
        physical=LayeredSlabSection(thickness_in,fc_ksi,layers)
        self.h,self.fc=physical.h,physical.fc
        self.ec=float(ec_ksi);self.lam=float(lambda_concrete)
        if not math.isfinite(self.ec) or self.ec<=0:
            raise ValueError('Explicit positive finite actual concrete modulus required')
        if not math.isfinite(self.lam) or not 0<self.lam<=1:
            raise ValueError('Concrete lambda must be positive and no greater than1')
        if any(r['concrete_owner']!='slab' for r in physical.layers):
            raise ValueError('Slab-own stiffness excludes drop-web concrete and bars')
        self.rows=physical.layers;self.cache={}
        self.ig=self.h**3/12
        self.fr=7.5*self.lam*math.sqrt(1000*self.fc)/1000
        self.mcr=self.fr*self.ig/(self.h/2)

    def properties(self,axis,moment_sign):
        """Finite-void zero-N cracked section for one axis and bending sign."""
        if axis not in ('x','y'):raise ValueError('Strip axis must be x or y')
        if moment_sign not in (-1,1):raise ValueError('Moment sign must be -1 or1')
        key=(axis,moment_sign)
        if key in self.cache:return dict(self.cache[key])
        longitudinal=[r for r in self.rows if r['axis']==axis]
        if not longitudinal:raise ValueError('Actual reinforcement for requested strip axis is absent')
        rows=[dict(r,depth=self.h/2+moment_sign*r['z_in']) for r in self.rows]
        def integral(c,power):
            value=c**(power+1)/(power+1)
            for row in rows:
                a,t,t2=_circle_integrals(row['radius'],c-row['depth'])
                lever=c-row['depth']
                removed=lever*a-t if power==1 else lever*lever*a-2*lever*t+t2
                value-=row['count_per_in']*removed
                if row['axis']==axis:
                    value+=row['es_ksi']/self.ec*row['area_in2_per_in']*lever**power
            return value
        c=brentq(lambda v:integral(v,1),0.,self.h,xtol=1e-12,rtol=1e-14,maxiter=100)
        icr=integral(c,2)
        if not math.isfinite(icr) or icr<=0:
            raise ValueError('Physical cracked reference has nonpositive inertia')
        # Very heavily reinforced transformed sections can exceed gross
        # concrete Ig. Keep the raw value and diagnose the code-interpolation
        # domain instead of silently reducing steel or altering its geometry.
        result=dict(method=METHOD,axis=axis,moment_sign=moment_sign,
                    compression_face='bottom' if moment_sign>0 else 'top',
                    neutral_depth_from_compression_face_in=c,
                    neutral_axis_z_in=moment_sign*(c-self.h/2),
                    Ig_in4_per_in=self.ig,Icr_in4_per_in=icr,Icr_over_Ig=icr/self.ig,
                    Ec_ksi=self.ec,fc_ksi=self.fc,fr_ksi=self.fr,
                    Mcr_kip_in_per_in=self.mcr,
                    zero_N_equilibrium_residual_in2=integral(c,1),
                    interpolation_domain_valid=icr<=self.ig*(1+1e-12),
                    physical_steel_ids=[r['id'] for r in self.rows],
                    axial_reference='zero_total_axial_force',
                    steel_area_model='centroid_fibers_with_finite_concrete_circle_subtraction')
        self.cache[key]=result
        return dict(result)

    def evaluate(self,axis,moment_kip_in_per_in,*,purpose):
        """Return Ie plus explicit validity of the elastic steel premise.

        purpose='service_deflection' is direct24.2.3 applicability;
        purpose='factored_action_stiffness' is a declared detailed force-analysis
        stiffness choice requiring integration validation. Neither declares the
        complete floor code-qualified. Above steel yield, Icr remains a useful
        ELASTIC reference but is not a proved lower bound on postyield secant
        stiffness. Such a point needs a secant/nonlinear analysis; it is not
        automatically a physically failed design.
        """
        if purpose not in ('service_deflection','factored_action_stiffness'):
            raise ValueError('Declare service_deflection or factored_action_stiffness')
        moment=float(moment_kip_in_per_in)
        if not math.isfinite(moment):raise ValueError('Finite signed strip moment required')
        sign=1 if moment>=0 else -1;ma=abs(moment)
        result=self.properties(axis,sign);icr=result['Icr_in4_per_in']
        cracked=ma>2*self.mcr/3
        if not result['interpolation_domain_valid']:
            ie=None
        elif not cracked:ie=self.ig
        else:ie=icr/(1-(2*self.mcr/(3*ma))**2*(1-icr/self.ig))
        # Ie is a member-average stiffness. Use LOCAL fully-cracked curvature
        # when cracked to check steel, rather than understating its stress by
        # substituting the larger Ie in this local material-domain check.
        local_i=icr if cracked else self.ig
        c=result['neutral_depth_from_compression_face_in'] if cracked else self.h/2
        curvature=ma/(self.ec*local_i)
        bars=[]
        for r in self.rows:
            if r['axis']!=axis:continue
            depth=self.h/2+sign*r['z_in']
            strain=curvature*(depth-c);stress=r['es_ksi']*strain
            bars.append(dict(id=r['id'],strain=strain,stress_ksi=stress,fy_ksi=r['fy_ksi'],
                             utilization=abs(stress)/r['fy_ksi']))
        maximum=max(r['utilization'] for r in bars)
        concrete_stress=self.ec*curvature*c
        steel_ok=maximum<=1+1e-10
        concrete_ok=concrete_stress<=self.fc*(1+1e-10)
        reasons=[]
        if not result['interpolation_domain_valid']:reasons.append('transformed_Icr_exceeds_code_gross_Ig_interpolation_domain')
        if not steel_ok:reasons.append('local_elastic_steel_estimate_exceeds_actual_yield')
        if not concrete_ok:reasons.append('local_linear_compression_estimate_exceeds_fc')
        result.update(purpose=purpose,signed_moment_kip_in_per_in=moment,
                      cracked_at_reduced_Mcr=cracked,Ie_in4_per_in=ie,
                      alpha_Ie_over_Ig=ie/self.ig if ie is not None else None,
                      local_reference_curvature_per_in=sign*curvature,
                      local_concrete_compression_ksi=concrete_stress,steel=bars,
                      max_steel_yield_utilization=maximum,elastic_steel_domain_valid=steel_ok,
                      local_linear_concrete_domain_valid=concrete_ok,
                      elastic_reference_applicable=not reasons,
                      applicability_reasons=reasons,
                      needs_secant_or_nonlinear_analysis=not (steel_ok and concrete_ok),
                      direct_code_service_formula=purpose=='service_deflection',
                      factored_analysis_requires_model_validation=purpose=='factored_action_stiffness',
                      complete_floor_qualified=False)
        return result

    def both_axes(self,signed_moments,*,purpose):
        if set(signed_moments)!={'x','y'}:raise ValueError('Both actual x and y strip moments required')
        return {a:self.evaluate(a,signed_moments[a],purpose=purpose) for a in ('x','y')}
