# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2026-07-08"

Created with xrtQook


None

"""

import numpy as np
import sys
# sys.path.append(r"/home/jhanrahan/Code/blop-dev/.pixi/envs/default/lib/python3.12/site-packages")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials.elemental as rmatsel
import xrt.backends.raycing.materials.compounds as rmatsco
import xrt.backends.raycing.materials.crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.figure_error as rfe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

def build_histRGB(lb, gb, limits=None, isScreen=False, shape=None):
    if shape is None:
        shape = [256, 256]
    good = (lb.state == 1) | (lb.state == 2)
    if isScreen:
        x, y, z = lb.x[good], lb.z[good], lb.y[good]
    else:
        x, y, z = lb.x[good], lb.y[good], lb.z[good]
    goodlen = len(lb.x[good])
    hist2dRGB = np.zeros((shape[1], shape[0], 3), dtype=np.float64)
    hist2d = np.zeros((shape[1], shape[0]), dtype=np.float64)

    if limits is None and goodlen > 0:
        limits = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])

    if goodlen > 0:
        beamLimits = [limits[1], limits[0]] or None
        flux = gb.Jss[good] + gb.Jpp[good]
        hist2d, _, _ = np.histogram2d(y, x, bins=[shape[1], shape[0]], range=beamLimits, weights=flux)
        hist2dRGB = None
    return hist2d, hist2dRGB, limits

Si111 = rmats.crystals_basic.CrystalSi(
    a=5.4307717932001225,
    d=3.1354575567115175,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    name=r"Si111")

Si311 = rmats.crystals_basic.CrystalSi(
    a=5.4307717932001225,
    hkl=[3, 1, 1],
    d=1.6374393054627614,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    name=r"Si311")

pt01 = rmats.elemental.Pt(
    name=r"pt01",
    elements=['Pt'],
    quantities=[1.0])

rh01 = rmats.elemental.Rh(
    name=r"rh01",
    elements=['Rh'],
    quantities=[1.0])

si01 = rmats.elemental.Si(
    name=r"si01",
    elements=['Si'],
    quantities=[1.0])


def build_beamline():
    bl = raycing.BeamLine(
        name=r"BMM",
        description=None)

    bl.TPW = rsources.synchr.Wiggler(
        bl=bl,
        name=r"TPW",
        center=[0.0, 0.0, 0.0],
        nrays=500000,
        eE=3.0,
        eI=0.5,
        eSigmaX=94.86832980505137,
        eSigmaZ=4.47213595499958,
        betaZ=2.0000000000000004,
        xPrimeMax=0.75,
        zPrimeMax=0.1,
        eMin=9040.0,
        eMax=9060.0,
        K=10,
        period=100,
        n=2)

    bl.FE_MASK = rapts.RectangularAperture(
        bl=bl,
        name=r"FE_MASK",
        center=[0.0, 12385.0, 0.0],
        blades={'left': -10, 'right': 10, 'bottom': -1.5, 'top': 1.5},
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.M1_VCM = roes.parametric.ParabolicalMirrorParam(
        p=13000,
        isCylindrical=True,
        bl=bl,
        name=r"M1_VCM",
        center=[0.0, 13000.0, 0.0],
        pitch=0.0035,
        material=rh01,
        limPhysX=[-15.0, 15.0],
        limPhysY=[-550.0, 550.0],
        isParametric=True,
        order=1)

    bl.Diag1 = rscreens.Screen(
        bl=bl,
        name=r"Diag1",
        center=[0.0, 25077, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    bl.DCM = roes.dcm.DCM(
        bragg=[9050],
        limPhysX2=[-50.0, 50.0],
        limPhysY2=[-100.0, 100.0],
        material2=Si111,
        fixedOffset=30,
        bl=bl,
        name=r"DCM",
        center=[0, 26105, r"auto"],
        material=Si111,
        limPhysX=[-50.0, 50.0],
        limPhysY=[-50.0, 50.0],
        order=1)

    bl.PinkBeamStop = rapts.RectangularAperture(
        bl=bl,
        name=r"PinkBeamStop",
        center=[0, 26450, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.Diag2 = rscreens.Screen(
        bl=bl,
        name=r"Diag2",
        center=[0.0, 27050, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    bl.M2_TFM = roes.ToroidMirror(
        bl=bl,
        name=r"M2_TFM",
        center=[0, 28473, r"auto"],
        pitch=-0.0035,
        positionRoll=3.141592653589793,
        material=rh01,
        limPhysX=[-15.0, 15.0],
        limPhysY=[-550.0, 550.0],
        order=1,
        R=7000000.0,
        r=58.5)

    bl.M3_HRM = roes.base.OE(
        bl=bl,
        name=r"M3_HRM",
        center=[0, 30381, r"auto"],
        pitch=r"3mrad",
        positionRoll=3.141592653589793,
        material=si01,
        limPhysX=[-15.0, 15.0],
        limPhysY=[-550.0, 550.0],
        order=1)

    bl.NANO_BPM = rscreens.Screen(
        bl=bl,
        name=r"NANO_BPM",
        center=[0, 31122, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    bl.BeamShutter = rapts.RectangularAperture(
        bl=bl,
        name=r"BeamShutter",
        center=[0, 31555, r"auto"],
        blades={'left': -5, 'right': 5, 'bottom': -3, 'top': 3},
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.XAS_SAMPLE = rscreens.Screen(
        bl=bl,
        name=r"XAS_SAMPLE",
        center=[0, 40300, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[-10.0, 10.0],
        limPhysY=[-10.0, 10.0],
        cLimits=[0.0, 0.0],
        histShape=[1456.0, 1088.0])

    bl.XRD_SAMPLE = rscreens.Screen(
        bl=bl,
        name=r"XRD_SAMPLE",
        center=[0, 44509, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    return bl


def run_process(bl):
    TPW_global = bl.TPW.shine()

    FE_MASK_local = bl.FE_MASK.propagate(
        beam=TPW_global)

    M1_VCM_global, M1_VCM_local = bl.M1_VCM.reflect(
        beam=TPW_global)

    Diag1_local = bl.Diag1.expose(
        beam=M1_VCM_global)

    DCM_global, DCM_local1, DCM_local2 = bl.DCM.double_reflect(
        beam=M1_VCM_global)

    PinkBeamStop_local = bl.PinkBeamStop.propagate(
        beam=DCM_global)

    Diag2_local = bl.Diag2.expose(
        beam=DCM_global)

    M2_TFM_global, M2_TFM_local = bl.M2_TFM.reflect(
        beam=DCM_global)

    M3_HRM_global, M3_HRM_local = bl.M3_HRM.reflect(
        beam=M2_TFM_global)

    NANO_BPM_local = bl.NANO_BPM.expose(
        beam=M3_HRM_global)

    BeamShutter_local = bl.BeamShutter.propagate(
        beam=M3_HRM_global)

    XAS_SAMPLE_local = bl.XAS_SAMPLE.expose(
        beam=M3_HRM_global)

    XRD_SAMPLE_local = bl.XRD_SAMPLE.expose(
        beam=M3_HRM_global)

    outDict = {
        'TPW_global': TPW_global,
        'FE_MASK_local': FE_MASK_local,
        'M1_VCM_global': M1_VCM_global,
        'M1_VCM_local': M1_VCM_local,
        'Diag1_local': Diag1_local,
        'DCM_global': DCM_global,
        'DCM_local1': DCM_local1,
        'DCM_local2': DCM_local2,
        'PinkBeamStop_local': PinkBeamStop_local,
        'Diag2_local': Diag2_local,
        'M2_TFM_global': M2_TFM_global,
        'M2_TFM_local': M2_TFM_local,
        'M3_HRM_global': M3_HRM_global,
        'M3_HRM_local': M3_HRM_local,
        'NANO_BPM_local': NANO_BPM_local,
        'BeamShutter_local': BeamShutter_local,
        'XAS_SAMPLE_local': XAS_SAMPLE_local,
        'XRD_SAMPLE_local': XRD_SAMPLE_local}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"Diag1_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-10, 10]),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-10, 10]),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"01 - Diag 1")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"Diag2_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-10, 10]),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-10, 10]),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"02 - Diag2")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"XAS_SAMPLE_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-10, 10],
            bins=728,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-10, 10],
            bins=544,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=544,
            ppb=1),
        title=r"03 - XAS Sample screen")
    plots.append(plot03)
    return plots


def main():
    BMM = build_beamline()
    E0 = 0.5 * (BMM.TPW.eMin +
                BMM.TPW.eMax)
    BMM.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=BMM)


if __name__ == '__main__':
    main()
