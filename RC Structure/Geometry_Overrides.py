"""
Geometry_Overrides.py
=====================

Runtime geometry overrides for RC Structure analyses.

This keeps Structure_Parameters.py as the baseline configuration while allowing
NTHA/hybrid dataset runs to use alternate bay/story/section geometry without
editing the baseline file for every generated structure.
"""

import os

import Structure_Parameters as sp


GEOMETRY_FIELDS = (
    ("num_bay_x", "NUM_BAY_X", int, "RC_NUM_BAY_X"),
    ("num_bay_y", "NUM_BAY_Y", int, "RC_NUM_BAY_Y"),
    ("num_floor", "NUM_FLOOR", int, "RC_NUM_FLOOR"),
    ("bay_x", "BAY_X", float, "RC_BAY_X"),
    ("bay_y", "BAY_Y", float, "RC_BAY_Y"),
    ("story_h", "STORY_H", float, "RC_STORY_H"),
    ("b_col", "B_COL", float, "RC_B_COL"),
    ("h_col", "H_COL", float, "RC_H_COL"),
    ("b_beam", "B_BEAM", float, "RC_B_BEAM"),
    ("h_beam", "H_BEAM", float, "RC_H_BEAM"),
)


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _format_number(value):
    return f"{float(value):g}"


def _validate_overrides(overrides):
    for name in ("NUM_BAY_X", "NUM_BAY_Y", "NUM_FLOOR"):
        if name in overrides and int(overrides[name]) < 1:
            raise ValueError(f"{name} must be at least 1.")

    for name in ("BAY_X", "BAY_Y", "STORY_H", "B_COL", "H_COL", "B_BEAM", "H_BEAM"):
        if name in overrides and float(overrides[name]) <= 0.0:
            raise ValueError(f"{name} must be positive.")


def geometry_overrides_from_args(args):
    overrides = {}
    for attr, sp_name, cast, _env_name in GEOMETRY_FIELDS:
        value = getattr(args, attr, None)
        if value is not None:
            overrides[sp_name] = cast(value)
    return overrides


def geometry_overrides_from_environment():
    overrides = {}
    for _attr, sp_name, cast, env_name in GEOMETRY_FIELDS:
        value = os.environ.get(env_name)
        if value not in (None, ""):
            overrides[sp_name] = cast(value)
    return overrides


def geometry_name_from_overrides(overrides, explicit_name=None):
    if explicit_name:
        return _safe_name(explicit_name)

    if not overrides:
        return None

    num_bay_x = overrides.get("NUM_BAY_X", sp.NUM_BAY_X)
    num_bay_y = overrides.get("NUM_BAY_Y", sp.NUM_BAY_Y)
    num_floor = overrides.get("NUM_FLOOR", sp.NUM_FLOOR)
    bay_x = overrides.get("BAY_X", sp.BAY_X)
    bay_y = overrides.get("BAY_Y", sp.BAY_Y)
    story_h = overrides.get("STORY_H", sp.STORY_H)

    return _safe_name(
        "frame_"
        f"{int(num_bay_x)}x{int(num_bay_y)}_"
        f"{int(num_floor)}floor_"
        f"{_format_number(bay_x)}x{_format_number(bay_y)}bay_"
        f"{_format_number(story_h)}story"
    )


def geometry_name_from_args(args):
    return geometry_name_from_overrides(
        geometry_overrides_from_args(args),
        explicit_name=getattr(args, "geometry_name", None),
    )


def current_geometry_summary():
    return {
        "geometry_variant": getattr(sp, "GEOMETRY_VARIANT_NAME", "baseline"),
        "num_bay_x": sp.NUM_BAY_X,
        "num_bay_y": sp.NUM_BAY_Y,
        "num_floor": sp.NUM_FLOOR,
        "bay_x_in": sp.BAY_X,
        "bay_y_in": sp.BAY_Y,
        "story_h_in": sp.STORY_H,
        "total_height_in": sp.NUM_FLOOR * sp.STORY_H,
        "b_col_in": sp.B_COL,
        "h_col_in": sp.H_COL,
        "b_beam_in": sp.B_BEAM,
        "h_beam_in": sp.H_BEAM,
    }


def apply_geometry_overrides(overrides, variant_name=None, emit=True):
    _validate_overrides(overrides)

    if any(getattr(sp, name) != value for name, value in overrides.items()):
        # The selected thickness and its slab-to-frame transfer were checked
        # against the previous geometry; never carry them silently into a
        # different candidate building.
        sp.SLAB_THICKNESS_IN = None
        sp.FLOOR_TRANSFER = None
        sp.SLAB_REINFORCEMENT = None
        sp.SLAB_ACTIONS = None

    for sp_name, value in overrides.items():
        setattr(sp, sp_name, value)

    # Derived parameters that depend directly on geometry.
    sp.NUM_MODES = sp.NUM_FLOOR + 2
    # The longitudinal bar grade follows the story height (user decision 2026-10-17): ASTM A706 Grade 80 at 16 ft and
    # more, Grade 60 below; the specification is locked to the grade (Structure_Parameters.apply_reinforcement_grade).
    sp.apply_reinforcement_grade(story_h_in=sp.STORY_H)

    name = geometry_name_from_overrides(overrides, explicit_name=variant_name)
    sp.GEOMETRY_VARIANT_NAME = name or "baseline"

    if emit and overrides:
        summary = current_geometry_summary()
        print("\nApplied geometry variant:")
        print(f"  Name: {summary['geometry_variant']}")
        print(f"  Longitudinal bars: {sp.REINFORCEMENT_SPECIFICATION} (fy {sp.FY_KSI:g} ksi; Grade 80 at story heights of "
              f"{sp.LONGITUDINAL_GRADE_STORY_HEIGHT_FT:g} ft and more); transverse {sp.TRANSVERSE_SPECIFICATION}")
        print(
            f"  Plan: {summary['num_bay_x']} x {summary['num_bay_y']} bays "
            f"@ {summary['bay_x_in']:g} in x {summary['bay_y_in']:g} in"
        )
        print(
            f"  Height: {summary['num_floor']} stories "
            f"@ {summary['story_h_in']:g} in = {summary['total_height_in']:g} in"
        )
        print(
            f"  Columns: {summary['b_col_in']:g} in x {summary['h_col_in']:g} in; "
            f"beams: {summary['b_beam_in']:g} in x {summary['h_beam_in']:g} in"
        )

    return sp.GEOMETRY_VARIANT_NAME


def apply_seismic_site_from_args(args, emit=True):
    """Apply the per-case seismic hazard, if one was requested.

    The hazard is a named entry rather than an override tuple because SDS,
    SD1 and S1 are not independent: they describe one design spectrum and
    have to move together. Kept in this module so every entry point that
    already calls the geometry helpers picks it up without new plumbing.
    """
    label = getattr(args, "seismic_site", None)
    if not label:
        return sp.SEISMIC_SITE_LABEL

    applied = sp.apply_seismic_site(label)
    if emit:
        print(
            f"  Seismic hazard: {applied} "
            f"(SDS={sp.ASCE_SDS:g}, SD1={sp.ASCE_SD1:g}, S1={sp.ASCE_S1:g})"
        )
    return applied


def apply_seismic_site_from_environment(emit=True):
    label = os.environ.get("RC_SEISMIC_SITE")
    if not label:
        return sp.SEISMIC_SITE_LABEL
    applied = sp.apply_seismic_site(label)
    if emit:
        print(f"  Seismic hazard: {applied}")
    return applied


def apply_geometry_from_args(args, emit=True):
    name = apply_geometry_overrides(
        geometry_overrides_from_args(args),
        variant_name=getattr(args, "geometry_name", None),
        emit=emit,
    )
    apply_seismic_site_from_args(args, emit=emit)
    return name


def apply_geometry_from_environment(emit=True):
    name = apply_geometry_overrides(
        geometry_overrides_from_environment(),
        variant_name=os.environ.get("RC_GEOMETRY_NAME"),
        emit=emit,
    )
    apply_seismic_site_from_environment(emit=emit)
    return name


def add_geometry_arguments(parser):
    parser.add_argument("--geometry-name", default=None, help="Name for this geometry variant/output folder.")
    parser.add_argument("--num-bay-x", type=int, default=None, help="Number of bays in global X.")
    parser.add_argument("--num-bay-y", type=int, default=None, help="Number of bays in global Y.")
    parser.add_argument("--num-floor", type=int, default=None, help="Number of elevated floors/stories.")
    parser.add_argument("--bay-x", type=float, default=None, help="Bay width in X, inches.")
    parser.add_argument("--bay-y", type=float, default=None, help="Bay width in Y, inches.")
    parser.add_argument("--story-h", type=float, default=None, help="Story height, inches.")
    parser.add_argument("--b-col", type=float, default=None, help="Column width, inches.")
    parser.add_argument("--h-col", type=float, default=None, help="Column depth, inches.")
    parser.add_argument("--b-beam", type=float, default=None, help="Beam width, inches.")
    parser.add_argument("--h-beam", type=float, default=None, help="Beam depth, inches.")
    parser.add_argument(
        "--seismic-site",
        default=None,
        help=(
            "Named ASCE 7 design hazard for this case, one of "
            + ", ".join(entry[0] for entry in sp.SEISMIC_SITE_OPTIONS)
            + ". Sets SDS, SD1 and S1 together."
        ),
    )


def geometry_cli_args_for_command(args):
    command_args = []
    geometry_name = getattr(args, "geometry_name", None)
    if geometry_name:
        command_args.extend(["--geometry-name", str(geometry_name)])

    for attr, _sp_name, _cast, _env_name in GEOMETRY_FIELDS:
        value = getattr(args, attr, None)
        if value is not None:
            command_args.extend([f"--{attr.replace('_', '-')}", str(value)])

    seismic_site = getattr(args, "seismic_site", None)
    if seismic_site:
        command_args.extend(["--seismic-site", str(seismic_site)])

    return command_args
