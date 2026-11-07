"""Actual beam-section N-M demand checks, separate from full Mn/Mpr demand generation.

Compression is positive; moments are about the beam gross-depth midplane.
The supplied N is EXTERNAL section force, never the self-equilibrated web /
flange couple already represented by a composite flexural section. Each
end's development decision applies to both mats and both bending signs.
"""
from __future__ import annotations

import numpy as np

from Design.SMRF_Uniaxial_Section import LayeredSection, METHOD as SECTION_METHOD

METHOD = 'member_developed_t_l_external_axial_flexure_v1'


def section_family(*, depth_in, web_width_in, fc_ksi, fy_ksi, beam_bar_area_in2,
                   bar_rows, slab_thickness_in=0., flange_width_in=None,
                   slab_layers=(), slab_fy_ksi=60., slab_bar_areas=None,
                   undeveloped_ends=()):
    """Construct physical layer inventory once per developed/undeveloped family.

    ``bar_rows`` entries contain bars/elevation_from_face_in; slab layers
    contain depth_in/area_in2/face. Areas represent distinct physical bars.
    Arrangement/spacing and end anchorage are validated by their own callers.
    """
    h, bw, t = float(depth_in), float(web_width_in), float(slab_thickness_in)
    if not 0 <= t < h:
        raise ValueError('Slab thickness must be below total beam depth')
    regions = ([(0., t, float(flange_width_in), fc_ksi), (t, h, bw, fc_ksi)]
               if t else [(0., h, bw, fc_ksi)])
    layers = []
    for face in ('top', 'bottom'):
        for row in bar_rows[face]:
            y = float(row['elevation_from_face_in'])
            if face == 'bottom':
                y = h-y
            layers.append((y, row['bars']*beam_bar_area_in2, beam_bar_area_in2, fy_ksi))
    bare = LayeredSection(h, regions, layers)
    mat = []
    for layer in slab_layers:
        if not slab_bar_areas or layer['face'] not in slab_bar_areas:
            raise ValueError('Individual slab bar area is required for concrete displacement')
        mat.append((layer['depth_in'], layer['area_in2'], slab_bar_areas[layer['face']], slab_fy_ksi))
    full = LayeredSection(h, regions, layers+mat) if mat else bare
    return {where: bare if where in undeveloped_ends else full for where in ('i', 'j', 'span')}


def installed_member_sections(member, rows):
    """Resolve the same final installed cage/flange/mats as nominal/probable consumers."""
    import Structure_Parameters as sp
    from Model import Member_Properties as mp
    design = member.design
    if sp.SLAB_THICKNESS_IN is None:
        # The canonical arrangement stores (count, offset) pairs in offsets_in.
        from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
        normalized = {face: [{'bars': n, 'elevation_from_face_in': o}
                             for n, o in validate_bar_rows(rows[face], getattr(design, 'top_bars' if face == 'top' else 'bot_bars'))]
                      for face in ('top', 'bottom')}
        return section_family(depth_in=design.h_in, web_width_in=design.b_in, fc_ksi=design.fc_ksi,
                              fy_ksi=sp.FY_KSI, beam_bar_area_in2=sp.rebar_area(design.bar_size), bar_rows=normalized)
    resolved = mp.beam_strengths(member)
    family = resolved['family']
    layout = (sp.SLAB_REINFORCEMENT or {}).get('layout')
    by_end = resolved['basis']['exterior_anchorage_by_end']
    axis = mp.beam_axis(member)
    return section_family(depth_in=design.h_in, web_width_in=design.b_in, fc_ksi=design.fc_ksi,
                          fy_ksi=sp.FY_KSI, beam_bar_area_in2=sp.rebar_area(design.bar_size),
                          bar_rows=family['bar_rows'], slab_thickness_in=sp.SLAB_THICKNESS_IN,
                          flange_width_in=family['effective_flange_width_in'],
                          slab_layers=family['slab_layers_from_top'], slab_fy_ksi=sp.SLAB_FY_KSI,
                          slab_bar_areas={face: layout['layers'][f'{axis}_{face}']['bar_area_in2']
                                          for face in ('top', 'bottom')} if layout else None,
                          undeveloped_ends=[end for end, data in by_end.items() if not data['developed_both']])


_INTERVALS = {}                       # (section inputs, axial values) -> the arrays of its moment interval
_INTERVAL_BYTES = [0]
INTERVAL_CACHE_LIMIT_BYTES = 400_000_000


def _moment_interval(section, values):
    """section.moment_interval(values), not solved twice for identical section inputs and identical axial values. The key is
    the whole input of the solve: depth, modulus, concrete regions, steel layers and the exact axial values."""
    key = (section.h, section.es, section.regions.tobytes(), section.layers.tobytes(), values.tobytes())
    found = _INTERVALS.get(key)
    if found is not None:
        return found
    solved = section.moment_interval(values)
    kept = {'minimum': solved['minimum'], 'maximum': solved['maximum'],
            'positive': {'eps_t': solved['positive']['eps_t'], 'phi': solved['positive']['phi']},
            'negative': {'eps_t': solved['negative']['eps_t'], 'phi': solved['negative']['phi']}}
    size = 6 * values.nbytes + len(key[2]) + len(key[3]) + len(key[4]) + 400
    if _INTERVAL_BYTES[0] + size > INTERVAL_CACHE_LIMIT_BYTES:
        _INTERVALS.clear()
        _INTERVAL_BYTES[0] = 0
    _INTERVALS[key] = kept
    _INTERVAL_BYTES[0] += size
    return kept


def capacity_at_axial(sections, axial_kip):
    """Weakest end/span moment interval at each external N, with explicit axial failures.

    Array evaluation preserves every requested N; a failing end cannot be
    hidden by the stronger interior section. A moment interval displaced
    entirely from zero is refused for a full-span +/- envelope containing
    zero. This conservative domain refusal does not fabricate a capacity.
    """
    p = np.atleast_1d(np.asarray(axial_kip, dtype=float))
    if not np.isfinite(p).all():
        raise ValueError('Beam external axial actions must be finite')
    lower = np.full(p.shape, -np.inf); upper = np.full(p.shape, np.inf)
    failure = np.zeros(p.shape, dtype=bool)
    tension_controlled = np.ones(p.shape, dtype=bool)
    governing = {'positive': np.full(p.shape, '', dtype=object), 'negative': np.full(p.shape, '', dtype=object)}
    phis = {'positive': np.full(p.shape, np.nan), 'negative': np.full(p.shape, np.nan)}
    for where, section in sections.items():
        inside = (p >= -.9*section.steel_force) & (p <= .65*.8*section.p0)
        failure |= ~inside
        if not inside.any():
            continue
        interval = _moment_interval(section, p[inside])
        required = p[inside] < .10*section.fc*section.ag
        eps_required = max(section.layers[:, 3])/section.es+.003
        satisfied = ((interval['positive']['eps_t'] >= eps_required-1e-12)
                     & (interval['negative']['eps_t'] >= eps_required-1e-12))
        tension_controlled[inside] &= ~required | satisfied
        for sign, name, old, candidate, comparison in (
                ('positive', 'positive', upper, interval['maximum'], np.less),
                ('negative', 'negative', lower, interval['minimum'], np.greater)):
            replace = comparison(candidate, old[inside])
            indices = np.flatnonzero(inside)[replace]
            old[indices] = candidate[replace]
            governing[sign][indices] = where
            phis[sign][indices] = interval[name]['phi'][replace]
    failure |= (lower > 1e-8) | (upper < -1e-8) | ~np.isfinite(lower) | ~np.isfinite(upper)
    return {'positive_kip_in': np.maximum(upper, 0.), 'negative_kip_in': np.maximum(-lower, 0.),
            'domain_failure': failure, 'tension_controlled': tension_controlled,
            'section': governing, 'phi': phis,
            'method': METHOD, 'section_method': SECTION_METHOD}
