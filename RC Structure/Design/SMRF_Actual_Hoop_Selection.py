"""Grouped transverse selection on the fabricated, curved 3D hoop geometry.

The catalog and shear envelopes are unchanged. Only steel actually crossing
the shear plane is credited; a wound perimeter leg has its horizontal
projection, and crossties have their full area. Rounded outer hoop dimensions
and area determine confinement. No added, unrepresented legs are credited.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import Structure_Parameters as sp
from Design import SMRF_Capacity_Design as cd
from Design import SMRF_Transition_Physical as ph
from Design.SMRF_Transitions import bar_coordinates
from Design.SMRF_Splice_Physical import beam_cage_geometry


def _summary(cage):
    return {**cage.get("layout",{}),"constructible":bool(cage["passes"]),
            "physical_geometry":{k:cage.get(k) for k in
                ("actual_cover_in","core_outside_dimensions_in","core_area_in2","hoop_form",
                 "hoop_leg_horizontal_projection_factor","total_length_in","set_height_in")},
            "checks":[*cage.get("layout",{}).get("checks",[]),*cage.get("checks",[])]}


def _periodic_fit(cage,spacing):
    a=ph._positioned_cage_segments(cage,0.,0)
    b=ph._positioned_cage_segments(cage,spacing,1)
    return not ph.transverse_collisions(a,b)


def beam_selection(design,coordinates,cover,result):
    """Replace a nominal beam selection using the same demanded Vs and ladder."""
    selected,cage=None,None
    d=result["effective_depth_in"]
    vs=result["vs_required_kip"]
    failures=[]
    for bar,legs in cd.STIRRUP_LADDER:
        trial=replace(design,stirrup_bar_size=bar,stirrup_legs=legs)
        physical=beam_cage_geometry(trial,coordinates,cover,legs=legs)
        if not physical["passes"]:
            failures.append({"bar_size":bar,"legs":legs,"checks":[c for c in physical["checks"] if not c["passes"]]})
            continue
        av=physical["vertical_leg_equivalent"]*sp.rebar_area(bar)
        s_shear=av*sp.FYT_KSI*d/vs if vs>0 else float("inf")
        spacing=cd._grid_down(min(s_shear,*result["spacing_bounds_in"].values()))
        if spacing is None or not _periodic_fit(physical,spacing):
            continue
        selected={"bar_size":bar,"legs":legs,"spacing_in":spacing,"av_in2":av,
                  "nominal_av_in2":legs*sp.rebar_area(bar),
                  "hoop_leg_projection_factor":physical["hoop_leg_horizontal_projection_factor"],
                  "spacing_from_shear_in":s_shear,"phi_vs_kip":cd.PHI_SHEAR*av*sp.FYT_KSI*d/spacing,
                  "phi_vn_kip":cd.PHI_SHEAR*(result["vc_kip"]+av*sp.FYT_KSI*d/spacing),"crossties":legs-2,
                  "physical_area_basis":"actual curved 3D hoop projected legs plus engaged crossties"}
        cage=_summary(physical)
        break
    if cage is None:
        cage={**result.get("cage",{}),"constructible":False,"legs":None,
              "checks":[{"rule":"actual_beam_transverse_geometry_and_projected_area","passes":False,
                         "detail":"No existing hoop catalog entry satisfies physical engagement, cover, self/adjacent-set clearance and projected Av/s."}],
              "physical_trial_failures":failures}
    return {**result,"hoops":selected,"cage":cage,"transverse_geometry_method":"actual_3d_hoop_projected_area_v1"}


@lru_cache(maxsize=256)
def _column_cage(design,cover,high,grid,layout_hoop_db):
    coordinates=bar_coordinates(design,cover)
    return ph.rectangular_cage_geometry(design,coordinates,cover,high_axial=high)


def column_selection(design,cover,result,offset):
    """Reuse each column's actual envelope, selecting actual Ash and Av."""
    worst=result["worst_by_direction"]
    directions=result["directions"]
    axial=max((v["axial_max_kip"] for per in result["stories"].values() for v in per.values()),default=0.)
    ag=design.b_in*design.h_in
    high=axial>.3*ag*design.fc_ksi or design.fc_ksi>10.
    selected,cage,confinement,bounds=None,None,None,result["spacing_bounds_in"]
    for bar in cd.COLUMN_HOOP_BARS:
        priced=[]
        for nb in range(2,design.top_bars+1):
            for nh in range(2,design.side_bars+3):
                trial=replace(design,stirrup_bar_size=bar,stirrup_legs=min(nb,nh),stirrup_legs_by_direction=(nb,nh))
                physical=_column_cage(trial,cover,high,sp.BAR_POSITION_GRID_IN,sp.BAR_LAYOUT_HOOP_DB_IN)
                if not physical["passes"]:
                    continue
                ach=physical["core_area_in2"]
                bc=physical["core_outside_dimensions_in"]
                ratio=max(.3*(ag/ach-1)*design.fc_ksi/sp.FYT_KSI,.09*design.fc_ksi/sp.FYT_KSI)
                if high:
                    count=design.top_bars+design.bot_bars+2*design.side_bars
                    ratio=max(ratio,.2*max(1.,design.fc_ksi/25+.6)*count/max(1,count-2)*max(0.,axial)/(sp.FYT_KSI*ach))
                hx=physical["layout"]["hx_in"]
                limits={"quarter_min_dimension":min(design.b_in,design.h_in)/4,
                        "bar_diameters":cd._hoop_bar_factor(sp.FY_KSI)*sp.rebar_diameter(design.bar_size),
                        "so":max(4.,min(6.,4.+(14.-hx)/3)),"conservative_so_cap":4.}
                projection=physical["hoop_leg_horizontal_projection_factor"]
                legs={"across_b_face":nb,"across_h_face":nh}
                per={}
                for i,axis in enumerate(("x","y")):
                    spec=directions[axis]
                    n=legs[spec["legs_key"]]
                    av=(n-2+2*projection)*sp.rebar_area(bar)
                    depth=spec["depth_in"]-offset
                    demand=worst[axis]["vs_required_kip"] if axis in worst else 0.
                    per[axis]={"legs_key":spec["legs_key"],"legs":n,"av_in2":av,"nominal_av_in2":n*sp.rebar_area(bar),
                               "d_in":depth,"bc_in":bc[i],"vs_required_kip":demand,
                               "spacing_from_shear_in":av*sp.FYT_KSI*depth/demand if demand>0 else float("inf"),
                               "spacing_from_confinement_in":av/(ratio*bc[i]),"ash_required_per_in":ratio*bc[i]}
                ss=min(v["spacing_from_shear_in"] for v in per.values())
                sc=min(v["spacing_from_confinement_in"] for v in per.values())
                spacing=cd._grid_down(min(ss,sc,*limits.values()))
                if spacing is None or not _periodic_fit(physical,spacing):
                    continue
                conf={"ach_in2":ach,"bc_in":max(bc),"bc_by_direction_in":dict(zip(("x","y"),bc)),
                      "ash_ratio_required":ratio,"high_axial":high,"hx_in":hx,"so_in":limits["so"],
                      "hx_within_8in_when_required":not high or hx<=8.,"hoop_leg_projection_factor":projection}
                priced.append((spacing,legs,physical,per,limits,ss,sc,conf))
        if not priced:
            continue
        spacing,legs,physical,per,bounds,ss,sc,confinement=max(priced,key=lambda p:(p[0],-sum(p[1].values()),-max(p[1].values())))
        for axis,v in per.items():
            v["ash_provided_per_in"]=v["av_in2"]/spacing
            v["phi_vn_kip"]=cd.PHI_SHEAR*((worst[axis]["vc_kip"] if axis in worst else 0.)+v["av_in2"]*sp.FYT_KSI*v["d_in"]/spacing)
        selected={"bar_size":bar,"legs":legs,"spacing_in":spacing,"legs_model":min(legs.values()),
                  "av_in2":min(v["av_in2"] for v in per.values()),"spacing_from_shear_in":ss,"spacing_from_confinement_in":sc,
                  "ash_provided_per_in":min(v["ash_provided_per_in"] for v in per.values()),
                  "ash_required_per_in":max(v["ash_required_per_in"] for v in per.values()),
                  "phi_vn_kip":min(v["phi_vn_kip"] for v in per.values()),"crossties_per_direction":{k:n-2 for k,n in legs.items()},
                  "by_direction":per,"physical_area_basis":"actual rounded core and projected 3D perimeter legs"}
        cage=_summary(physical)
        break
    if cage is None:
        cage={**result["cage"],"constructible":False,"legs":None,
              "checks":[{"rule":"actual_column_transverse_geometry_and_projected_area","passes":False,
                         "detail":"No existing hoop catalog entry satisfies actual rounded-core confinement, projected shear steel and physical periodic cage clearance."}]}
        confinement={**result["confinement"],"physical_geometry_available":False}
    return {**result,"hoops":selected,"cage":cage,"confinement":confinement,"spacing_bounds_in":bounds,
            "screen":cd.column_shear_screen(result["stories"],selected,result["vs_limit_by_direction_kip"],sp.FYT_KSI),
            "transverse_geometry_method":"actual_3d_hoop_projected_area_v1"}
