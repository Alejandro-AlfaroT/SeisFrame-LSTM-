"""Physical-face and pure-sagging recovery from rectangular MITC4 resultants.

Supports uniform or nonuniform axis-aligned rectangular cells. Recover the raw
tensor within the cell containing the physical face, on its clear-span side.
Never average between elements or extend a result beyond its owning cell.
This is a numerical recovery rule, not an engineering verification assertion.
"""
from __future__ import annotations

import math
import heapq
from collections import defaultdict

METHOD_VERSION = "mitc4_physical_face_clear_span_cell_v1"
FIELDS = ("mx", "my", "mxy_raw", "qx_raw", "qy_raw")
SAGGING_RECOVERY_VERSION = "mitc4_bilinear_pure_sagging_shear_bound_v1"


def recover_sagging_shear(panel, geometry, face_widths, *, relative_tolerance=5e-4,
                          max_subdivisions=10000):
    """Bound |Q| over the pure-sagging part of each clear-span cell.

    The old GP-only mask changes abruptly when a GP crosses the zero-moment
    boundary. Here the same bilinear raw-tensor recovery used at physical faces
    is searched within each owning cell. Corner extrema bound bilinear fields;
    interval bounds on the Wood-Armer conditions discard non-sagging boxes.
    Subdivision closes the gap between an actual sagging witness and the upper
    shear bound. Return the conservative upper bound, never an averaged field.
    A search that cannot close its numerical bound fails explicitly.
    The closure gap is 0.05% of the witness (user decision 2026-10-03; it was 0.01%, at which
    2 of 400 random bilinear fields in the review did not close in 10,000 subdivisions): two
    orders below the 5% refinement screen it feeds.
    """
    from Design.SMRF_Slab_Actions import wood_armer
    if not 0 < relative_tolerance < 1 or type(max_subdivisions) is not int or max_subdivisions < 1:
        raise ValueError("Invalid sagging recovery tolerance or subdivision budget")
    grouped = defaultdict(dict)
    for point in panel["gauss_point_resultants"]:
        element, gp = point["element"], point["gauss_point"]
        if gp in grouped[element]:
            raise ValueError("Duplicate element Gauss point")
        grouped[element][gp] = point
    cells = {tag: _cell(points) for tag, points in grouped.items()}
    bounds = {}
    for axis, index in (("x", panel["i"]), ("y", panel["j"])):
        length = geometry[f"bay_{axis}_in"]
        bounds[axis] = (index * length + face_widths[axis][0] / 2.,
                        (index + 1) * length - face_widths[axis][1] / 2.)
        if bounds[axis][0] >= bounds[axis][1]:
            raise ValueError("Beam widths must leave a positive clear slab panel")
    result = {}
    for axis, bottom_index, top_index in (("x", 0, 2), ("y", 1, 3)):
        other = "y" if axis == "x" else "x"
        queue, counter, best, witness = [], 0, 0., None

        def offer(tag, box):
            nonlocal counter, best, witness
            x0, x1, y0, y1 = box
            cell = cells[tag]
            positions = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            raw = [_recover(cell, x, y) for x, y in positions]
            ma = [r[f"m{axis}"] for r in raw]
            mb = [r[f"m{other}"] for r in raw]
            twist = [r["mxy_raw"] for r in raw]
            tmin = 0. if min(twist) <= 0 <= max(twist) else min(map(abs, twist))
            # Top-axis WA demand can be zero only in this union. These are
            # necessary interval conditions, so no feasible point is discarded.
            if max(ma) <= 0 or (max(ma) < tmin and
                    (max(mb) <= tmin or max(ma) * max(mb) < tmin * tmin)):
                return
            upper = max(abs(r[f"q{axis}_raw"]) for r in raw)
            if upper <= best:
                return
            points = list(zip(positions, raw))
            center = ((x0 + x1) / 2., (y0 + y1) / 2.)
            points.append((center, _recover(cell, *center)))
            for (x, y), r in points:
                wa = wood_armer(r["mx"], r["my"], r["mxy_raw"])
                q = abs(r[f"q{axis}_raw"])
                if wa[bottom_index] > 0 and wa[top_index] == 0 and (witness is None or q > best):
                    best = max(best, q)
                    witness = {"element": tag, "x_in": x, "y_in": y,
                               "raw_resultants": r, "bottom_demand": wa[bottom_index],
                               "top_demand": wa[top_index]}
            if upper > best:
                counter += 1
                heapq.heappush(queue, (-upper, counter, tag, box))

        for tag, cell in cells.items():
            box = (max(bounds["x"][0], cell["x_lo"]), min(bounds["x"][1], cell["x_hi"]),
                   max(bounds["y"][0], cell["y_lo"]), min(bounds["y"][1], cell["y_hi"]))
            if box[1] > box[0] + 1e-8 and box[3] > box[2] + 1e-8:
                offer(tag, box)
        subdivisions = 0
        while queue and -queue[0][0] > best * (1. + relative_tolerance) + 1e-10:
            if subdivisions >= max_subdivisions:
                raise ValueError(f"Sagging shear recovery did not close its bound at {panel['panel_id']}/{axis}")
            _, _, tag, (x0, x1, y0, y1) = heapq.heappop(queue)
            xm, ym = (x0 + x1) / 2., (y0 + y1) / 2.
            corners = [_recover(cells[tag], x, y) for x, y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
            def variation(pairs):
                return sum(max(abs(corners[a][f]-corners[b][f]) for a,b in pairs)
                           / max(1e-12, max(abs(r[f]) for r in corners))
                           for f in ("mx", "my", "mxy_raw", f"q{axis}_raw"))
            # Split the varying direction. Four-way splitting duplicates a
            # straight zero-moment boundary into exponentially many identical boxes.
            boxes = (((x0,xm,y0,y1),(xm,x1,y0,y1)) if variation(((0,1),(3,2))) >= variation(((0,3),(1,2)))
                     else ((x0,x1,y0,ym),(x0,x1,ym,y1)))
            for box in boxes:
                offer(tag, box)
            subdivisions += 1
        upper = max(best, -queue[0][0] if queue else best)
        result[axis] = {"method": SAGGING_RECOVERY_VERSION, "shear_upper_kip_per_in": upper,
                        "shear_witness_kip_per_in": best, "witness": witness,
                        "relative_tolerance": relative_tolerance, "subdivisions": subdivisions,
                        "bound_gap_kip_per_in": upper - best}
    return result


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _cell(points):
    if set(points) != {1, 2, 3, 4}:
        raise ValueError("Physical-face recovery requires all four distinct Gauss points per element")
    values = {k: {field: _finite(p[field], field) for field in ("x_in", "y_in", *FIELDS)}
              for k, p in points.items()}
    x1, x2 = values[1]["x_in"], values[2]["x_in"]
    y1, y2 = values[1]["y_in"], values[4]["y_in"]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Expected CCW axis-aligned rectangular Gauss-point ordering")
    for k, x, y in ((2,x2,y1),(3,x2,y2),(4,x1,y2)):
        if not math.isclose(values[k]["x_in"],x,rel_tol=0,abs_tol=1e-8) or not math.isclose(values[k]["y_in"],y,rel_tol=0,abs_tol=1e-8):
            raise ValueError("Nonrectangular or inconsistent Gauss-point geometry")
    dx, dy = math.sqrt(3)*(x2-x1), math.sqrt(3)*(y2-y1)
    return {"points":values, "x_lo":(x1+x2-dx)/2, "x_hi":(x1+x2+dx)/2,
            "y_lo":(y1+y2-dy)/2, "y_hi":(y1+y2+dy)/2}


def _recover(cell, x, y):
    tol=1e-8
    if not cell["x_lo"]-tol <= x <= cell["x_hi"]+tol or not cell["y_lo"]-tol <= y <= cell["y_hi"]+tol:
        raise ValueError("Refusing to extrapolate outside the owning element")
    p=cell["points"]
    tx=(x-p[1]["x_in"])/(p[2]["x_in"]-p[1]["x_in"])
    ty=(y-p[1]["y_in"])/(p[4]["y_in"]-p[1]["y_in"])
    weights={1:(1-tx)*(1-ty),2:tx*(1-ty),3:tx*ty,4:(1-tx)*ty}
    return {field: math.fsum(weights[k]*p[k][field] for k in weights) for field in FIELDS}


def recover_panel_faces(panel, geometry, beam_width_in, face_widths=None):
    """Return both normal faces of both axes, with separate cell-side samples.

    At each face, retain each clipped cell's transverse endpoints and Gauss
    coordinates. A shared transverse endpoint appears once for each owning
    cell: neither side of a stress discontinuity is discarded or averaged.
    Samples exclude perpendicular beam widths. All four faces must be covered.

    ``face_widths`` = {"x": (lower, upper), "y": (lower, upper)} gives the widths of the four beams
    bounding this panel when they differ (Design.SMRF_Floor_Sections.panel_face_widths): "x" the two
    beams whose faces are normal to X. Without it every face is half of ``beam_width_in`` from its line.
    """
    lx=_finite(geometry["bay_x_in"],"bay_x_in"); ly=_finite(geometry["bay_y_in"],"bay_y_in")
    if face_widths is None:
        face_widths={"x":(beam_width_in,beam_width_in),"y":(beam_width_in,beam_width_in)}
    halves={axis:tuple(_finite(w,"beam_width_in")/2 for w in face_widths[axis]) for axis in ("x","y")}
    if any(h<=0 for pair in halves.values() for h in pair) or sum(halves["x"])>=lx or sum(halves["y"])>=ly:
        raise ValueError("Beam width must leave a positive clear slab panel")
    pi,pj=panel["i"],panel["j"]
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in (pi,pj)):
        raise ValueError("Panel indices must be nonnegative integers")
    if pi>=geometry["num_bay_x"] or pj>=geometry["num_bay_y"]:
        raise ValueError("Panel is outside floor geometry")
    grouped=defaultdict(dict)
    for point in panel["gauss_point_resultants"]:
        element,gp=point["element"],point["gauss_point"]
        if isinstance(element, bool) or not isinstance(element, int) or element <= 0:
            raise ValueError("Element tags must be positive integers")
        if isinstance(gp, bool) or not isinstance(gp, int) or gp not in (1, 2, 3, 4):
            raise ValueError("Gauss-point numbers must be integers 1 through 4")
        if gp in grouped[element]:
            raise ValueError("Duplicate element Gauss point")
        grouped[element][gp]=point
    cells={element:_cell(points) for element,points in grouped.items()}
    samples=[];coverage=[];tol=1e-8
    for axis,origin,length,tangent,t_origin,t_length in (("x",pi*lx,lx,"y",pj*ly,ly),("y",pj*ly,ly,"x",pi*lx,lx)):
        t_lower,t_upper=halves[tangent]
        for support,position in (("lower",origin+halves[axis][0]),("upper",origin+length-halves[axis][1])):
            intervals=[]
            for element,cell in cells.items():
                lo,hi=cell[axis+"_lo"],cell[axis+"_hi"]
                # Half-open ownership selects the clear slab, including when a
                # face coincides with a mesh boundary. Never select the first
                # row merely because it is adjacent to the beam centerline.
                owned=(lo-tol<=position<hi-tol) if support=="lower" else (lo+tol<position<=hi+tol)
                if not owned:
                    continue
                start=max(cell[tangent+"_lo"],t_origin+t_lower)
                end=min(cell[tangent+"_hi"],t_origin+t_length-t_upper)
                if end-start<=tol:
                    continue
                intervals.append((start,end))
                coords={start,end}
                coords.update(p[tangent+"_in"] for p in cell["points"].values()
                              if start+tol<p[tangent+"_in"]<end-tol)
                for t in sorted(coords):
                    x,y=(position,t) if axis=="x" else (t,position)
                    samples.append({"panel_id":panel["panel_id"],"axis":axis,"support":support,
                        "element":element,"x_in":x,"y_in":y,"face_position_in":position,
                        "clear_span_side":"increasing" if support=="lower" else "decreasing",
                        "cell_bounds_in":{k:cell[k] for k in ("x_lo","x_hi","y_lo","y_hi")},
                        "raw_resultants":_recover(cell,x,y)})
            reached=t_origin+t_lower
            for start,end in sorted(intervals):
                if start>reached+tol:
                    raise ValueError(f"Incomplete physical-face coverage at {panel['panel_id']}/{axis}/{support}")
                if start<reached-tol:
                    raise ValueError(f"Overlapping physical-face cells at {panel['panel_id']}/{axis}/{support}")
                reached=end
            if not intervals or reached<t_origin+t_length-t_upper-tol:
                raise ValueError(f"Missing physical-face coverage at {panel['panel_id']}/{axis}/{support}")
            coverage.append({"axis":axis,"support":support,"position_in":position,
                             "transverse_clear_interval_in":[t_origin+t_lower,t_origin+t_length-t_upper],
                             "owning_cell_count":len(intervals)})
    return {"method":METHOD_VERSION,"samples":samples,"coverage":coverage,
            "complete":True,"engineering_verified":False}
