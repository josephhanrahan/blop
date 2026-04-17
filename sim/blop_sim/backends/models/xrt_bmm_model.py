# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2026-04-09"

Created with xrtQook


None

"""

# import numpy as np
# import sys
# sys.path.append(r"C:\GitHub\xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

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
    name=r"pt01")

rh01 = rmats.elemental.Rh(
    name=r"rh01")

si01 = rmats.elemental.Si(
    name=r"si01")


def build_beamline():
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
        eMin=9040,
        eMax=9060,
        K=10,
        period=100,
        n=2)

    BeamLine.bentFlatMirror01 = roes.BentFlatMirror(
        bl=BeamLine,
        name=r"bentFlatMirror01",
        center=[0.0, 10000.0, 0.0],
        pitch=r"0.1deg",
        limPhysX=[-10.0, 10.0],
        limPhysY=[-500.0, 500.0],
        order=1,
        R=[10000, 1000000])

    BeamLine.dcM01 = roes.dcm.DCM(
        bragg=[9050],
        limPhysX2=[-50.0, 50.0],
        limPhysY2=[-10.0, 50.0],
        material2=Si111,
        fixedOffset=10,
        bl=BeamLine,
        name=r"dcM01",
        center=[0, 20000, r"auto"],
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
        center=[0, 30000, r"auto"],
        pitch=r"0.1deg",
        positionRoll=r"180deg")

    BeamLine.screen02 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen02",
        center=[0, 32000, 3])

    BeamLine.oe01 = roes.OE(
        bl=BeamLine,
        name=r"oe01",
        center=[0, 35000, r"auto"],
        pitch=r"5deg",
        positionRoll=r"180deg")

    BeamLine.screen03 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen03",
        center=[0, 40000, r"auto"])

    BeamLine.rectangularAperture01 = rapts.RectangularAperture(
        bl=BeamLine,
        name=r"rectangularAperture01",
        center=[0, 36000, r"auto"])

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

    rectangularAperture01_local = BeamLine.rectangularAperture01.propagate(
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
        'rectangularAperture01_local': rectangularAperture01_local,
        'screen03_local': screen03_local}
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
