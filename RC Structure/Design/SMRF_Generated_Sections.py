"""Read the current automatic design into diagnostic section mechanics.

No dimension or reinforcement is selected here. No slab layout is invented.
These are local, uniaxial sections: neither a floor force allocation nor a
development, biaxial-strength, torsion, joint, or production acceptance check.
"""
from __future__ import annotations

import hashlib
import json
import math

from .SMRF_Section_Interaction import _section

METHOD_VERSION = 'generated_cage_section_v1'
BAR = {3: (.375, .11), 4: (.5, .20), 5: (.625, .31), 6: (.75, .44),
       7: (.875, .60), 8: (1., .79), 9: (1.128, 1.), 10: (1.27, 1.27), 11: (1.41, 1.56),
       14: (1.693, 2.25), 18: (2.257, 4.00)}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _cage(record, prefix):
    r = record['reinforcement']
    size = r[f'{prefix}_bar_size']
    hoop = r[f'{prefix}_stirrup_bar_size']
    if type(size) is not int or size not in BAR or type(hoop) is not int or hoop not in BAR:
        raise ValueError('Unsupported longitudinal or hoop bar size')
    db, area = BAR[size]
    if r.get('cover_basis') != 'clear_cover_outside_hoops':
        raise ValueError('Explicit current clear-cover convention is required')
    cover = _positive(r[f'{prefix}_clear_cover_in'], 'clear cover')
    offset = cover + BAR[hoop][0] + db / 2
    declared = _positive(r[f'{prefix}_longitudinal_centroid_offset_in'], 'centroid offset')
    if not math.isclose(offset, declared, rel_tol=0., abs_tol=1e-10):
        raise ValueError('Saved bar centroid disagrees with installed hoop and longitudinal diameters')
    for suffix, expected in (('bar_diameter_in', db), ('bar_area_in2', area), ('stirrup_diameter_in', BAR[hoop][0])):
        if not math.isclose(_positive(r[f'{prefix}_{suffix}'], suffix), expected, rel_tol=0., abs_tol=1e-10):
            raise ValueError(f'Saved {prefix}_{suffix} disagrees with its bar size')
    return dict(area=area, diameter=db, offset=offset, top=_count(r[f'{prefix}_top_bars'], 'top bars', 2),
                bottom=_count(r[f'{prefix}_bot_bars'], 'bottom bars', 2),
                side=_count(r[f'{prefix}_side_bars'], 'bars per side face'))


def _bars(cage, width, depth, staggered=None):
    """Bar rows of a cage. ``staggered`` = the record's beam_bar_stacking.layers[axis] (2026-09-27): each
    face's rows at their own elevations, the orthogonal cages stacked at the joints and a second layer
    where the column lanes force it, so the generated section agrees with the record's strengths; a
    record without it keeps the nominal row at the centroid offset."""
    o, area = cage['offset'], cage['area']
    if 2 * o >= min(width, depth):
        raise ValueError('Longitudinal bar centroids do not fit inside the section')
    bars = []
    rows = []
    for face, n, y in (('top', cage['top'], o), ('bottom', cage['bottom'], depth - o)):
        if staggered:
            from Design.SMRF_Beam_Slab_Strength import validate_bar_rows
            # validated row by row: lengths, counts, declared layers and centroid, elevations inside the section
            for k, (count, elevation) in enumerate(validate_bar_rows(staggered[face], n, f'{face} bars', depth)):
                rows.append((f'{face}_L{k+1}', count, elevation if face == 'top' else depth - elevation))
        else:
            rows.append((face, n, y))
    for face, n, y in rows:
        for i in range(n):
            bars.append(dict(id=f'{face}_{i+1}', depth_in=y,
                             across_width_in=(width / 2 if n == 1 else o + i*(width-2*o)/(n-1)),
                             area_in2=area, bar_diameter_in=cage['diameter']))
    for side, x in (('left', o), ('right', width-o)):
        for i in range(cage['side']):
            bars.append(dict(id=f'{side}_{i+1}', depth_in=o+(i+1)*(depth-2*o)/(cage['side']+1),
                             across_width_in=x, area_in2=area, bar_diameter_in=cage['diameter']))
    return bars


def _finish(section, provenance):
    _section(section)
    identity = dict(section=section, provenance=provenance, method=METHOD_VERSION)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(section, source_sha256=digest, provenance=provenance, method=METHOD_VERSION,
                engineering_verified=False, production_enabled=False, development_assessed=False)


def beam_section_from_design(record, *, axis, position, scope='composite', slab_steel='require_layout'):
    """Use the installed beam cage and, when available, the actual slab mats.

    ``beam_rectangle`` isolates the beam cage. ``composite`` includes the
    effective flange. Missing slab steel raises by default; the explicitly
    incomplete ``omit_for_sensitivity`` mode does not authorize a capacity.
    A layout supplies physical bars, never an implicit assertion of development.
    L-section results are about one depth axis only, not biaxial/torsional.
    """
    if axis not in ('x', 'y') or position not in ('edge', 'interior'):
        raise ValueError('Beam axis and edge/interior position are required')
    if scope not in ('beam_rectangle', 'composite') or slab_steel not in ('require_layout', 'omit_for_sensitivity'):
        raise ValueError('Unsupported section scope or slab-steel policy')
    s, g, m = record['sections'], record['geometry'], record['materials']
    b, h, fc, fy = [_positive(v, k) for k, v in (('b', s['b_beam_in']), ('h', s['h_beam_in']),
                                               ('fc', s['fc_beam_ksi']), ('fy', m['fy_ksi']))]
    cage = _cage(record, 'beam')
    staggered = ((record['reinforcement'].get('beam_bar_stacking') or {}).get('layers') or {}).get(axis)
    layers = _bars(cage, b, h, staggered)
    bands = [dict(top_in=0., bottom_in=h, width_in=b)]
    provenance = dict(scope=scope, axis=axis, position=position, slab_steel_policy=slab_steel,
                      beam_steel_area_in2=sum(v['area_in2'] for v in layers),
                      slab_steel_area_in2=0., composite_reinforcement_complete=False,
                      source='Automatic design record; no sizing or manual section presets')
    if scope == 'composite':
        from Structure_Parameters import effective_flange_width_in
        slab = record['slab']
        t = _positive(slab['thickness_in'], 'slab thickness')
        if not 0 < t < h:
            raise ValueError('Slab thickness must be less than beam depth')
        if not math.isclose(_positive(slab['concrete_fc_ksi'], 'slab fc'), fc, rel_tol=0., abs_tol=1e-10):
            raise ValueError('Different slab and beam concrete strengths need a multi-material section')
        span = _positive(g[f'bay_{axis}_in'], 'bay span') - _positive(s['h_col_in' if axis == 'x' else 'b_col_in'], 'column depth')
        adjacent = _positive(g['bay_y_in' if axis == 'x' else 'bay_x_in'], 'transverse bay') - b
        if min(span, adjacent) <= 0:
            raise ValueError('Clear span and adjacent-web clearance must be positive')
        bf, overhang = effective_flange_width_in(b, t, span, adjacent, 2 if position == 'interior' else 1)
        bands = [dict(top_in=0., bottom_in=t, width_in=bf), dict(top_in=t, bottom_in=h, width_in=b)]
        provenance.update(effective_flange_width_in=bf, flange_overhang_in=overhang,
                          clear_span_in=span, slab_thickness_in=t,
                          width_basis='Existing ACI 318-19 6.3.2.1 helper; not a floor shell ownership allocation')
        layout = (record.get('slab_reinforcement') or {}).get('layout')
        if slab_steel == 'require_layout':
            if layout is None:
                raise ValueError('Slab reinforcement is unresolved; full composite section cannot be constructed')
            if layout.get('uniform_all_panels_and_floors') is not True:
                raise ValueError('This adapter requires the current uniform slab layout')
            for face in ('top', 'bottom'):
                layer = layout['layers'][f'{axis}_{face}']
                area = _positive(layer['bar_area_in2'], 'slab bar area')
                diameter = _positive(layer['bar_diameter_in'], 'slab bar diameter')
                spacing = _positive(layer['spacing_in'], 'slab spacing')
                d = _positive(layer['effective_depth_in'], 'slab effective depth')
                y = t-d if face == 'top' else d
                if not 0 < y < t:
                    raise ValueError('Slab bar centroid must lie inside the slab thickness')
                a = area*bf/spacing
                layers.append(dict(id=f'slab_{axis}_{face}', depth_in=y, area_in2=a, bar_diameter_in=diameter,
                                   area_basis='Uniform smeared mat over effective width, separate from beam bars'))
                provenance['slab_steel_area_in2'] += a
            provenance['composite_reinforcement_complete'] = True
        else:
            provenance['limitation'] = 'All slab mats intentionally omitted in this sensitivity; not the complete member'
    return _finish(dict(id=f'generated_beam_{axis}_{position}_{scope}', depth_in=h, fc_ksi=fc, fy_ksi=fy,
                        concrete_bands=bands, steel_layers=layers), provenance)


def column_section_from_design(record, *, bending_depth):
    """Project every installed perimeter bar onto the h or b bending depth."""
    if bending_depth not in ('h', 'b'):
        raise ValueError('Choose the physical h or b depth; no inferred global/local axis')
    s, m = record['sections'], record['materials']
    b = _positive(s['b_col_in'], 'column b'); h = _positive(s['h_col_in'], 'column h')
    layers = _bars(_cage(record, 'col'), b, h)
    if bending_depth == 'b':
        layers = [dict(v, depth_in=v['across_width_in'], across_width_in=v['depth_in']) for v in layers]
    depth, width = (h, b) if bending_depth == 'h' else (b, h)
    return _finish(dict(id=f'generated_column_depth_{bending_depth}', depth_in=depth,
                        fc_ksi=_positive(s['fc_col_ksi'], 'fc'), fy_ksi=_positive(m['fy_ksi'], 'fy'),
                        concrete_bands=[dict(top_in=0., bottom_in=depth, width_in=width)], steel_layers=layers),
                   dict(scope='column_rectangle', bending_depth=bending_depth,
                        steel_area_in2=sum(v['area_in2'] for v in layers),
                        source='Every actual perimeter bar; side bars counted on both faces'))
