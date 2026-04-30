# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2026-04-09"

Created with xrtQook


None

"""

import numpy as np
# import sys
# sys.path.append(r"C:\GitHub\xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials.elemental as rmatse
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

# limits = [[-0.6, 0.6], [-0.45, 0.45]]
limits = [[-5, 5], [-5, 5]]


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

pt01 = rmatse.Pt(
    name=r"pt01")

rh01 = rmatse.Rh(
    name=r"rh01")

si01 = rmatse.Si(
    name=r"si01")


def build_beamline(ev=9050):
    BeamLine = raycing.BeamLine(
        name=r"BeamLine",
        description=None)

    BeamLine.wiggler01 = rsources.synchr.Wiggler(
        bl=BeamLine,
        name=r"wiggler01",
        center=[0, 0, 0],
        eE=3.0,
        eI=0.5,
        eSigmaX=94.86832980505137,
        eSigmaZ=4.47213595499958,
        xPrimeMax=0.5,
        zPrimeMax=0.5,
        eMin=ev-10,
        eMax=ev+10,
        K=10,
        period=100,
        n=2)

    BeamLine.bentFlatMirror01 = roes.BentFlatMirror(
        bl=BeamLine,
        name=r"bentFlatMirror01",
        # center=[0.0, 10000.0, 0.0],
        center=[0.0, 13000.0, 0.0],
        pitch=r"0.1deg",
        limPhysX=[-10.0, 10.0],
        limPhysY=[-500.0, 500.0],
        order=1,
        R=[10000, 1000000])

    BeamLine.dcM01 = roes.dcm.DCM(
        bragg=[ev],
        limPhysX2=[-50.0, 50.0],
        limPhysY2=[-10.0, 50.0],
        material2=Si111,
        # cryst2roll=0,
        fixedOffset=10,
        bl=BeamLine,
        name=r"dcM01",
        # center=[0, 20000, r"auto"],
        center=[0, 26105, r"auto"],
        # center=[0, r'auto', r"auto"],
        material=Si111,
        limPhysX=[-50.0, 50.0],
        limPhysY=[-50.0, 10.0],
        order=1)

    BeamLine.screen01 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen01",
        center=[0, 25000, r"auto"])

    BeamLine.toroidMirror01 = roes.ToroidMirror(
        bl=BeamLine,
        name=r"toroidMirror01",
        # center=[0, 30000, r"auto"],
        center=[0, 28473, r"auto"],
        # yaw=r"auto",
        pitch=r"0.1deg",
        positionRoll=r"180deg",
        # positionRoll="3.14159"
    )

    BeamLine.screen02 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen02",
        # center=[0, 32000, r"auto"],
        center=[0, 30000, r"auto"],
    )

    BeamLine.oe01 = roes.OE(
        bl=BeamLine,
        name=r"oe01",
        # center=[0, 35000, r"auto"],
        center=[0, 30381, r"auto"],
        pitch=r"5deg",
        positionRoll=r"180deg")

    BeamLine.screen03 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen03",
        # center=[0, 40000, r"auto"],
        center=[0, 30500, r"auto"]
    )

    # BeamLine.rectangularAperture01 = rapts.RectangularAperture(
    #     bl=BeamLine,
    #     name=r"rectangularAperture01",
    #     # center=[0, 36000, r"auto"],
    #     center=[0, 31561, r"auto"]
    # )

    BeamLine.screen04 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen04",
        center=[0, 40300, r"auto"]
    )

    return BeamLine


def run_process(BeamLine):
    wiggler01_global = BeamLine.wiggler01.shine()

    bentFlatMirror01_global, bentFlatMirror01_local = BeamLine.bentFlatMirror01.reflect(
        beam=wiggler01_global)

    dcM01_global, dcM01_local1, dcM01_local2 = BeamLine.dcM01.double_reflect(
        beam=bentFlatMirror01_global)

    screen01_local = BeamLine.screen01.expose(
        beam=dcM01_global)

    toroidMirror01_global, toroidMirror01_local = BeamLine.toroidMirror01.reflect(
        beam=dcM01_global)

    screen02_local = BeamLine.screen02.expose(
        beam=toroidMirror01_global)

    oe01_global, oe01_local = BeamLine.oe01.reflect(
        beam=toroidMirror01_global)

    screen03_local = BeamLine.screen03.expose(
        beam=oe01_global)

    # rectangularAperture01_local = BeamLine.rectangularAperture01.propagate(
        # beam=oe01_global)
    
    screen04_local = BeamLine.screen04.expose(
        beam=oe01_global)

    outDict = {
        'wiggler01_global': wiggler01_global,
        'bentFlatMirror01_global': bentFlatMirror01_global,
        'bentFlatMirror01_local': bentFlatMirror01_local,
        'dcM01_global': dcM01_global,
        'dcM01_local1': dcM01_local1,
        'dcM01_local2': dcM01_local2,
        'screen01_local': screen01_local,
        'toroidMirror01_global': toroidMirror01_global,
        'toroidMirror01_local': toroidMirror01_local,
        'screen02_local': screen02_local,
        'oe01_global': oe01_global,
        'oe01_local': oe01_local,
        # 'rectangularAperture01_local': rectangularAperture01_local,
        'screen03_local': screen03_local,
        'screen04_local': screen04_local}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen03_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot01-screen03_local-energy")
    plots.append(plot01)
    return plots


def main():
    BeamLine = build_beamline()
    E0 = 0.5 * (BeamLine.wiggler01.eMin +
                BeamLine.wiggler01.eMax)
    BeamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=BeamLine)


if __name__ == '__main__':
    main()
