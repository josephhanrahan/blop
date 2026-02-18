import itertools
from collections import deque
from collections.abc import Generator, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import scipy as sp  # type: ignore[import-untyped]
from event_model import StreamRange, compose_resource, compose_stream_resource  # type: ignore[import-untyped]
from ophyd import Component as Cpt  # type: ignore[import-untyped]
from ophyd import Device, Kind, Signal  # type: ignore[import-untyped]
from ophyd.sim import NullStatus, new_uid  # type: ignore[import-untyped]
from ophyd.utils import make_dir_tree  # type: ignore[import-untyped]
from . import get_beam_stats
from .handlers import ExternalFileReference

from .srw_model import build_beamline, run_process

TEST = False

class TiledDetector(Device):
    sum = Cpt(Signal, kind=Kind.hinted)
    max = Cpt(Signal, kind=Kind.normal)
    area = Cpt(Signal, kind=Kind.normal)
    cen_x = Cpt(Signal, kind=Kind.hinted)
    cen_y = Cpt(Signal, kind=Kind.hinted)
    wid_x = Cpt(Signal, kind=Kind.hinted)
    wid_y = Cpt(Signal, kind=Kind.hinted)

    image = Cpt(ExternalFileReference, kind=Kind.omitted)
    image_shape = Cpt(Signal, value=(294, 528), kind=Kind.omitted)
    noise = Cpt(Signal, kind=Kind.normal)

    def __init__(self, root_dir: str = "/tmp/blop/sim", verbose: bool = True, noise: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _ = make_dir_tree(datetime.now().year, base_path=root_dir)

        self._root_dir = root_dir
        self._verbose = verbose

        # Used for the emulated cameras only.
        self._img_dir = None

        # Resource/datum docs related variables.
        self._asset_docs_cache: deque[tuple[str, dict[str, Any]]] = deque()
        self._stream_resource_document: dict[str, Any] | None = None
        self._stream_datum_factory: Any | None = None
        self._dataset: h5py.Dataset | None = None

        self.noise.put(noise)
        self.limits = [[-0.6, 0.6], [-0.45, 0.45]]
        if TEST:
            self.mplFig = mpl.figure.Figure()
            self.mplFig.subplots_adjust(left=0.15, bottom=0.15, top=0.92)
            self.mplAx = self.mplFig.add_subplot(111)

            xv = np.random.rand(294, 528)
            self.im = self.mplAx.imshow(
                xv.T,
                aspect="auto",
                origin="lower",
                vmin=0,
                vmax=1e3,
                cmap="jet",
                extent=(self.limits[0][0], self.limits[0][1], self.limits[1][0], self.limits[1][1]),
            )
        self.counter = 0
        # self.beamLine = None
        # self.beamLine = build_beamline(self.parent.crl2_xoff.get(), self.parent.crl2_yoff.get())

    
    def trigger(self):
        super().trigger()

        self.beamLine = build_beamline(self.parent.crl2_xoff.get(), self.parent.crl2_yoff.get())
        # run srw sim and get output here
        raw_image = run_process(self.beamLine)
        # raw_image = self.generate_beam(noise=self.noise.get())

        current_frame = next(self._counter)

        self._dataset.resize((current_frame + 1, *self.image_shape.get()))

        self._dataset[current_frame, :, :] = raw_image

        stream_datum_document = self._stream_datum_factory(
            StreamRange(start=current_frame, stop=current_frame + 1),
        )
        self._asset_docs_cache.append(("stream_datum", stream_datum_document))

        stats = get_beam_stats(raw_image)

        for attr in ["max", "sum", "cen_x", "cen_y", "wid_x", "wid_y"]:
            getattr(self, attr).put(stats[attr])

        super().trigger()
        return NullStatus()
        
    def _generate_file_path(self, date_template="%Y/%m/%d"):
        date = datetime.now()
        assets_dir = date.strftime(date_template)
        data_file = f"{new_uid()}.h5"
        return Path(self._root_dir) / Path(assets_dir) / Path(data_file)

    def stage(self):
        super().stage()

        self._asset_docs_cache.clear()
        full_path = self._generate_file_path()
        image_shape = self.image_shape.get()

        uri = f"file://localhost/{str(full_path).strip('/')}"

        (
            self._stream_resource_document,
            self._stream_datum_factory,
        ) = compose_stream_resource(
            mimetype="application/x-hdf5",
            uri=uri,
            data_key=self.image.name,
            parameters={
                "chunk_shape": (1, *image_shape),
                "dataset": "/entry/image",
            },
        )

        self._data_file = full_path
        self._asset_docs_cache.append(("stream_resource", self._stream_resource_document))

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *image_shape)),
            maxshape=(None, *image_shape),
            chunks=(1, *self.image_shape.get()),
            dtype="float64",
            compression="lzf",
        )

        self._counter = itertools.count()
        
    def unstage(self):
        super().unstage()
        # del self._dataset
        self._h5file_desc.close()
        self._stream_resource_document = None
        self._stream_datum_factory = None

    def describe(self):
        res = super().describe()
        res[self.image.name] = {
            "shape": [1, *self.image_shape.get()],
            "external": "STREAM:",
            "source": "sim",
            "dtype": "array",
            "dtype_numpy": np.dtype(np.float64).str,
        }  # <i8
        return res

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        yield from items
    

class TiledBeamline(Device):
    det = Cpt(TiledDetector)

    crl2_xoff = Cpt(Signal, kind="hinted")
    crl2_yoff = Cpt(Signal, kind="hinted")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    