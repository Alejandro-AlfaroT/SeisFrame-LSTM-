"""Fully cracked, linear-elastic, pure-bending section diagnostic.

Concrete carries compression only; steel stays elastic. This is the classical
transformed-section Icr calculation, not a first-yield secant, nonlinear
moment-curvature curve, empirically calibrated member stiffness or IMK input.
No tension stiffening, slip, shear, creep or cyclic damage is represented.
The matrix retains axial/flexural coupling at the explicit reference.
"""
from __future__ import annotations

import math
from .SMRF_Section_Interaction import _section, _number, ES_KSI

METHOD_VERSION = 'fully_cracked_elastic_pure_bending_v1'


def cracked_flexural_rigidity(section, *, compression_face, reference_y_in,
                             ec_ksi=None, es_ksi=ES_KSI, subtract_displaced_concrete=True):
    """Return K for [N, M] = K [epsilon_reference, kappa], tension/sagging positive.

    Strain = epsilon_reference + kappa*(y-reference_y). N=0 determines the
    cracked neutral axis, with either physical top or bottom in compression.
    EI at constant axial force is K22-K12**2/K11; it is reference invariant.
    K22 alone corresponds to fixing strain at that reference and is different.
    """
    fc, fy, h, bands, layers = _section(section)
    ref = _number(reference_y_in, 'reference_y_in')
    ec = 57.*math.sqrt(fc*1000.) if ec_ksi is None else _number(ec_ksi, 'ec_ksi', True)
    es = _number(es_ksi, 'es_ksi', True)
    if es <= ec:
        raise ValueError('This RC transformed-section calculation requires Es > Ec')
    if compression_face not in ('top', 'bottom') or type(subtract_displaced_concrete) is not bool:
        raise ValueError('Explicit compression face and boolean concrete-displacement policy required')

    def matrix(c, reference):
        terms = []
        for low, high, b in bands:
            lo, hi = (low, min(high, c)) if compression_face == 'top' else (max(low, c), high)
            if hi > lo:
                a = b*(hi-lo); y = (hi+lo)/2-reference
                terms.append((ec*a, ec*a*y, ec*(b*(hi-lo)**3/12+a*y*y)))
        for _, y, area in layers:
            compressed = y < c if compression_face == 'top' else y > c
            ea = (es-(ec if subtract_displaced_concrete and compressed else 0.))*area
            d = y-reference
            terms.append((ea, ea*d, ea*d*d))
        return tuple(math.fsum(v[i] for v in terms) for i in range(3))

    # Unit curvature is algebraic: no material is evaluated at unit strain.
    # At reference=c, B is the axial resultant divided by curvature.
    lo, hi = 0., h
    for _ in range(100):
        c = (lo+hi)/2
        _, unbalance, _ = matrix(c, c)
        if unbalance > 0: lo = c
        else: hi = c
    c = (lo+hi)/2
    A, B, D = matrix(c, ref)
    EI = D-B*B/A
    scale = max(1., A*h)
    residual = matrix(c, c)[1]/scale
    if A <= 0 or EI <= 0 or abs(residual) > 1e-10:
        raise ValueError('Cracked section has invalid stiffness or axial equilibrium')
    Ag = math.fsum((hi-lo)*b for lo, hi, b in bands)
    yg = math.fsum((hi-lo)*b*(lo+hi)/2 for lo, hi, b in bands)/Ag
    Ig = math.fsum(b*(hi-lo)**3/12+b*(hi-lo)*((lo+hi)/2-yg)**2 for lo, hi, b in bands)
    return dict(method=METHOD_VERSION, section_id=section['id'], source_sha256=section.get('source_sha256'),
                compression_face=compression_face, reference_y_in=ref, ec_ksi=ec, es_ksi=es,
                neutral_axis_from_top_in=c, neutral_axis_from_compression_face_in=c if compression_face=='top' else h-c,
                tangent_matrix=[[A, B], [B, D]], matrix_units=[['kip', 'kip-in'], ['kip-in', 'kip-in2']],
                ei_constant_axial_force_kip_in2=EI, i_cracked_transformed_in4=EI/ec,
                gross_concrete_area_in2=Ag, gross_concrete_centroid_from_top_in=yg, gross_concrete_inertia_in4=Ig,
                ei_over_ec_ig=EI/(ec*Ig), normalized_axial_equilibrium_residual=residual,
                subtract_displaced_concrete=subtract_displaced_concrete,
                model_axial_force_kip=0., engineering_verified=False, production_enabled=False,
                scope='Fully cracked local elastic flexure at N=0; not a member seismic modifier or first-yield stiffness')
