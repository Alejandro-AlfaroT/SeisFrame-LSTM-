# ==================================================
# Sections.py
# ==================================================
# Defines OpenSees materials, RC fiber sections,
# and beam integration rules.
# ==================================================

import Structure_Parameters as sp
import openseespy.opensees as ops


def define_materials():
    ops.uniaxialMaterial(
        "Steel02",
        sp.STEEL_TAG,
        sp.FY_KSI,
        sp.ES_KSI,
        sp.STEEL_B,
    )

    ops.uniaxialMaterial(
        "Concrete02",
        sp.COVER_COL_TAG,
        -sp.FC_COL_KSI,
        -0.002,
        -0.20 * sp.FC_COL_KSI,
        -0.006,
        0.1,
        0.0,
        0.0,
    )

    ops.uniaxialMaterial(
        "Concrete02",
        sp.CORE_COL_TAG,
        -1.15 * sp.FC_COL_KSI,
        -0.0025,
        -0.30 * sp.FC_COL_KSI,
        -0.020,
        0.1,
        0.0,
        0.0,
    )

    ops.uniaxialMaterial(
        "Concrete02",
        sp.COVER_BEAM_TAG,
        -sp.FC_BEAM_KSI,
        -0.002,
        -0.20 * sp.FC_BEAM_KSI,
        -0.006,
        0.1,
        0.0,
        0.0,
    )

    ops.uniaxialMaterial(
        "Concrete02",
        sp.CORE_BEAM_TAG,
        -1.10 * sp.FC_BEAM_KSI,
        -0.0025,
        -0.30 * sp.FC_BEAM_KSI,
        -0.015,
        0.1,
        0.0,
        0.0,
    )


def compute_gj(fc_ksi, b, h):
    ec = sp.concrete_ec_ksi(fc_ksi)
    gc = sp.concrete_shear_modulus_ksi(ec)
    j = sp.approx_rect_j(b, h)
    return gc * j


def make_rc_rect_section(
    sec_tag,
    b,
    h,
    cover,
    core_mat,
    cover_mat,
    steel_mat,
    top_bars,
    bot_bars,
    bar_area,
    side_bars=0,
    gj=1.0e8,
    longitudinal_cover=None,
    bar_positions=True,
):
    """Build a rectangular fiber section with independent core and bar offsets.

    ``cover`` is the face-to-core boundary, at the outside of hoops for new
    designs. ``longitudinal_cover`` locates longitudinal-bar centroids. Omitting
    it preserves the original shared-offset fiber layout for legacy callers.
    With a slab selected the bars are placed one fiber each on the bar position
    grid (design-basis rule 2026-10-14); ``bar_positions=False`` keeps the even
    straight-layer placement.
    """
    if cover <= 0:
        raise ValueError("cover must be positive.")

    if 2.0 * cover >= min(b, h):
        raise ValueError("cover is too large for the section dimensions.")

    longitudinal_cover = cover if longitudinal_cover is None else longitudinal_cover
    if not (cover <= longitudinal_cover < min(b, h) / 2.0):
        raise ValueError("Longitudinal centroid offset must be inside the core and section.")

    if top_bars <= 0 or bot_bars <= 0:
        raise ValueError("top_bars and bot_bars must be positive.")

    if side_bars < 0:
        raise ValueError("side_bars cannot be negative.")

    if bar_area <= 0:
        raise ValueError("bar_area must be positive.")

    if gj <= 0:
        raise ValueError("gj must be positive.")

    y1 = -b / 2.0
    y2 = b / 2.0
    z1 = -h / 2.0
    z2 = h / 2.0

    yc1 = y1 + cover
    yc2 = y2 - cover
    zc1 = z1 + cover
    zc2 = z2 - cover
    ys1 = y1 + longitudinal_cover
    ys2 = y2 - longitudinal_cover
    zs1 = z1 + longitudinal_cover
    zs2 = z2 - longitudinal_cover

    ops.section("Fiber", sec_tag, "-GJ", gj)

    ops.patch(
        "rect",
        core_mat,
        sp.CORE_PATCH_NY,
        sp.CORE_PATCH_NZ,
        yc1,
        zc1,
        yc2,
        zc2,
    )

    ops.patch(
        "rect",
        cover_mat,
        sp.COVER_PATCH_N_LONG,
        sp.COVER_PATCH_N_SHORT,
        y1,
        z1,
        y2,
        zc1,
    )

    ops.patch(
        "rect",
        cover_mat,
        sp.COVER_PATCH_N_LONG,
        sp.COVER_PATCH_N_SHORT,
        y1,
        zc2,
        y2,
        z2,
    )

    ops.patch(
        "rect",
        cover_mat,
        sp.COVER_PATCH_N_SHORT,
        sp.COVER_PATCH_N_LONG,
        y1,
        zc1,
        yc1,
        zc2,
    )

    ops.patch(
        "rect",
        cover_mat,
        sp.COVER_PATCH_N_SHORT,
        sp.COVER_PATCH_N_LONG,
        yc2,
        zc1,
        y2,
        zc2,
    )

    # Bars where the cage puts them: the first bars in the hoop corners and the
    # interior bars on the bar position grid (Structure_Parameters.face_bar_positions_in),
    # one fiber per bar. Legacy sections (no slab) keep their even single-offset layout.
    if sp.SLAB_THICKNESS_IN is None or bar_positions is False:
        ops.layer("straight", steel_mat, top_bars, bar_area, ys1, zs2, ys2, zs2)
        ops.layer("straight", steel_mat, bot_bars, bar_area, ys1, zs1, ys2, zs1)
        if side_bars > 0:
            z_side_bot = zs1 + (zs2 - zs1) / (side_bars + 1)
            z_side_top = zs2 - (zs2 - zs1) / (side_bars + 1)
            ops.layer("straight", steel_mat, side_bars, bar_area, ys1, z_side_bot, ys1, z_side_top)
            ops.layer("straight", steel_mat, side_bars, bar_area, ys2, z_side_bot, ys2, z_side_top)
        return
    for y in sp.face_bar_positions_in(b, longitudinal_cover, int(top_bars)):
        ops.fiber(y1 + y, zs2, bar_area, steel_mat)
    for y in sp.face_bar_positions_in(b, longitudinal_cover, int(bot_bars)):
        ops.fiber(y1 + y, zs1, bar_area, steel_mat)
    if side_bars > 0:
        for z in sp.face_bar_positions_in(h, longitudinal_cover, int(side_bars) + 2)[1:-1]:
            ops.fiber(ys1, z1 + z, bar_area, steel_mat)
            ops.fiber(ys2, z1 + z, bar_area, steel_mat)


def define_column_section():
    gj_col = compute_gj(sp.FC_COL_KSI, sp.B_COL, sp.H_COL)

    make_rc_rect_section(
        sec_tag=sp.COL_SEC_TAG,
        b=sp.B_COL,
        h=sp.H_COL,
        cover=sp.core_cover_in("column"),
        longitudinal_cover=sp.longitudinal_cover_in("column"),
        core_mat=sp.CORE_COL_TAG,
        cover_mat=sp.COVER_COL_TAG,
        steel_mat=sp.STEEL_TAG,
        top_bars=sp.COL_TOP_BARS,
        bot_bars=sp.COL_BOT_BARS,
        side_bars=sp.COL_SIDE_BARS,
        bar_area=sp.COL_BAR_AREA,
        gj=gj_col,
    )


def define_beam_section():
    gj_beam = compute_gj(sp.FC_BEAM_KSI, sp.B_BEAM, sp.H_BEAM)

    make_rc_rect_section(
        sec_tag=sp.BEAM_SEC_TAG,
        b=sp.B_BEAM,
        h=sp.H_BEAM,
        cover=sp.core_cover_in("beam"),
        longitudinal_cover=sp.longitudinal_cover_in("beam"),
        core_mat=sp.CORE_BEAM_TAG,
        cover_mat=sp.COVER_BEAM_TAG,
        steel_mat=sp.STEEL_TAG,
        top_bars=sp.BEAM_TOP_BARS,
        bot_bars=sp.BEAM_BOT_BARS,
        side_bars=sp.BEAM_SIDE_BARS,
        bar_area=sp.BEAM_BAR_AREA,
        gj=gj_beam,
    )


def define_beam_integrations():
    ops.beamIntegration(
        "Lobatto",
        sp.COL_INTEG_TAG,
        sp.COL_SEC_TAG,
        sp.NUM_INT_PTS,
    )

    ops.beamIntegration(
        "Lobatto",
        sp.BEAM_INTEG_TAG,
        sp.BEAM_SEC_TAG,
        sp.NUM_INT_PTS,
    )


def define_sections():
    define_materials()
    define_column_section()
    define_beam_section()
    define_beam_integrations()
