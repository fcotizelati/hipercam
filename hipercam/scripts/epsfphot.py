"""Experimental ePSF photometry helpers for crowded-field work."""

import argparse
import copy
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable, Table, vstack
from photutils.background import LocalBackground, MMMBackground
from photutils.centroids import centroid_com
from photutils.detection import DAOStarFinder
from photutils.psf import (
    EPSFBuilder,
    ImagePSF,
    IterativePSFPhotometry,
    PSFPhotometry,
    SourceGrouper,
    extract_stars,
)

import hipercam as hcam

__all__ = ["epsfphot"]


def _read_table(path):
    return Table.read(path)


def _read_file_list(path):
    with open(path) as fp:
        return [
            line.strip()
            for line in fp
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _window_data(hcm_file, ccd, window):
    mccd = hcam.MCCD.read(str(hcm_file))
    wind = mccd[str(ccd)][str(window)]
    return np.asarray(wind.data, dtype=float), wind, mccd


def _background_subtract(data, sigma=3.0):
    _, median, _ = sigma_clipped_stats(data, sigma=sigma)
    return data - median, median


def _write_epsf(path, epsf):
    hdr = fits.Header()
    hdr["HIPREPSF"] = True
    hdr["OVERSX"] = int(epsf.oversampling[0])
    hdr["OVERSY"] = int(epsf.oversampling[1])
    hdr["ORIGINX"] = float(epsf.origin[0])
    hdr["ORIGINY"] = float(epsf.origin[1])
    fits.PrimaryHDU(np.asarray(epsf.data, dtype=np.float32), hdr).writeto(
        path, overwrite=True
    )


def _read_epsf(path):
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header
    oversampling = (int(hdr.get("OVERSX", 1)), int(hdr.get("OVERSY", 1)))
    origin = (float(hdr["ORIGINX"]), float(hdr["ORIGINY"])) if "ORIGINX" in hdr else None
    return ImagePSF(data, oversampling=oversampling, origin=origin)


def _fit_shape(value):
    value = int(value)
    if value % 2 == 0:
        value += 1
    return (value, value)


def _make_error(data, read, gain):
    return np.sqrt(read**2 + np.maximum(data, 0.0) / gain)


def _local_background(args):
    if args.local_bkg_inner is None or args.local_bkg_outer is None:
        return None
    return LocalBackground(
        args.local_bkg_inner,
        args.local_bkg_outer,
        bkg_estimator=MMMBackground(),
    )


def _source_grouper(min_separation):
    if min_separation is None or min_separation <= 0:
        return None
    return SourceGrouper(min_separation)


def _build_epsf(args):
    data, _, _ = _window_data(args.hcm, args.ccd, args.window)
    bkgsub, _ = _background_subtract(data, sigma=args.sigma)

    stars_tbl = _read_table(args.stars)
    if "x" not in stars_tbl.colnames or "y" not in stars_tbl.colnames:
        raise hcam.HipercamError("star table must contain 'x' and 'y' columns")

    stars = extract_stars(
        NDData(bkgsub),
        QTable(stars_tbl[["x", "y"]]),
        size=args.stamp_size,
    )
    builder = EPSFBuilder(
        oversampling=args.oversampling,
        shape=None if args.epsf_size is None else (args.epsf_size, args.epsf_size),
        maxiters=args.maxiters,
        smoothing_kernel=None if args.no_smoothing else "quartic",
        recentering_boxsize=(args.recenter_box, args.recenter_box),
        recentering_func=centroid_com,
        progress_bar=args.progress,
    )
    result = builder(stars)
    _write_epsf(args.output, result.epsf)

    summary = Table()
    summary["n_input_stars"] = [len(stars)]
    summary["n_excluded_stars"] = [result.n_excluded_stars]
    summary["converged"] = [result.converged]
    summary["iterations"] = [result.iterations]
    summary["final_center_accuracy"] = [result.final_center_accuracy]
    if args.summary:
        summary.write(args.summary, overwrite=True)
    else:
        print(summary)


def _make_source_list(args):
    data, _, _ = _window_data(args.hcm, args.ccd, args.window)
    bkgsub, _ = _background_subtract(data, sigma=args.sigma)
    epsf = _read_epsf(args.epsf)
    error = _make_error(data, args.read, args.gain)

    finder = DAOStarFinder(threshold=args.threshold, fwhm=args.fwhm)
    photometry = IterativePSFPhotometry(
        epsf,
        _fit_shape(args.fit_size),
        finder,
        grouper=_source_grouper(args.group_separation),
        maxiters=args.maxiters,
        mode=args.mode,
        aperture_radius=args.aperture_radius,
        local_bkg_estimator=_local_background(args),
        progress_bar=args.progress,
    )
    result = photometry(
        bkgsub,
        error=error,
    )
    result = result[result["flags"] == 0]
    result.write(args.output, overwrite=True)

    if args.residual:
        residual = photometry.make_residual_image(
            bkgsub, psf_shape=(args.stamp_size, args.stamp_size)
        )
        fits.PrimaryHDU(np.asarray(residual, dtype=np.float32)).writeto(
            args.residual, overwrite=True
        )


def _forced(args):
    files = _read_file_list(args.flist)
    source_table = _read_table(args.sources)
    if args.id_column not in source_table.colnames:
        source_table[args.id_column] = np.arange(1, len(source_table) + 1)
    for col in (args.x_column, args.y_column):
        if col not in source_table.colnames:
            raise hcam.HipercamError(f"source table must contain '{col}'")

    shift_table = None
    if args.shifts:
        shift_table = _read_table(args.shifts)
        for col in ("file", "dx", "dy"):
            if col not in shift_table.colnames:
                raise hcam.HipercamError("shift table must contain file, dx, dy columns")

    epsf0 = _read_epsf(args.epsf)
    rows = []
    for nframe, fname in enumerate(files, start=1):
        data, _, mccd = _window_data(fname, args.ccd, args.window)
        bkgsub, global_bkg = _background_subtract(data, sigma=args.sigma)
        error = _make_error(data, args.read, args.gain)

        dx = dy = 0.0
        if shift_table is not None:
            match = shift_table[shift_table["file"] == fname]
            if len(match):
                dx = float(match["dx"][0])
                dy = float(match["dy"][0])

        init_params = Table()
        init_params["id"] = source_table[args.id_column]
        init_params["x"] = source_table[args.x_column] + dx
        init_params["y"] = source_table[args.y_column] + dy

        epsf = copy.deepcopy(epsf0)
        if args.fixed_positions:
            epsf.x_0.fixed = True
            epsf.y_0.fixed = True

        photometry = PSFPhotometry(
            epsf,
            _fit_shape(args.fit_size),
            finder=None,
            grouper=_source_grouper(args.group_separation),
            aperture_radius=args.aperture_radius,
            local_bkg_estimator=_local_background(args),
            progress_bar=False,
        )
        result = photometry(bkgsub, error=error, init_params=init_params)
        result["frame"] = nframe
        result["file"] = fname
        result["mjdutc"] = mccd.head.get("MJDUTC", np.nan)
        result["global_bkg"] = global_bkg
        rows.append(result)

    if not rows:
        raise hcam.HipercamError("no frames were processed")
    vstack(rows).write(args.output, overwrite=True)


def _add_common_image_args(parser):
    parser.add_argument("hcm", help="input .hcm image")
    parser.add_argument("ccd", help="CCD label")
    parser.add_argument("window", help="window label")


def _add_photometry_args(parser):
    parser.add_argument("--fit-size", type=int, default=9)
    parser.add_argument("--fwhm", type=float, default=5.0)
    parser.add_argument("--read", type=float, default=3.5)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--group-separation", type=float, default=10.0)
    parser.add_argument("--aperture-radius", type=float, default=5.0)
    parser.add_argument("--local-bkg-inner", type=float)
    parser.add_argument("--local-bkg-outer", type=float)
    parser.add_argument("--progress", action="store_true")


def _parser():
    parser = argparse.ArgumentParser(
        prog="epsfphot",
        description="Experimental HiPERCAM ePSF crowded-field photometry helpers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-epsf")
    _add_common_image_args(build)
    build.add_argument("stars", help="table with x,y columns for isolated PSF stars")
    build.add_argument("output", help="output ePSF FITS file")
    build.add_argument("--summary", help="optional output build-summary table")
    build.add_argument("--stamp-size", type=int, default=25)
    build.add_argument("--epsf-size", type=int)
    build.add_argument("--oversampling", type=int, default=2)
    build.add_argument("--maxiters", type=int, default=10)
    build.add_argument("--recenter-box", type=int, default=7)
    build.add_argument("--sigma", type=float, default=3.0)
    build.add_argument("--no-smoothing", action="store_true")
    build.add_argument("--progress", action="store_true")
    build.set_defaults(func=_build_epsf)

    source = subparsers.add_parser("make-source-list")
    _add_common_image_args(source)
    source.add_argument("epsf", help="input ePSF FITS file")
    source.add_argument("output", help="output source table")
    source.add_argument("--residual", help="optional residual FITS image")
    source.add_argument("--stamp-size", type=int, default=25)
    source.add_argument("--maxiters", type=int, default=5)
    source.add_argument("--mode", choices=("new", "all"), default="all")
    _add_photometry_args(source)
    source.set_defaults(func=_make_source_list)

    forced = subparsers.add_parser("forced")
    forced.add_argument("flist", help="file containing .hcm frames to process")
    forced.add_argument("ccd", help="CCD label")
    forced.add_argument("window", help="window label")
    forced.add_argument("epsf", help="input ePSF FITS file")
    forced.add_argument("sources", help="source table from make-source-list")
    forced.add_argument("output", help="output forced-photometry table")
    forced.add_argument("--shifts", help="optional table with file,dx,dy columns")
    forced.add_argument("--id-column", default="id")
    forced.add_argument("--x-column", default="x_fit")
    forced.add_argument("--y-column", default="y_fit")
    forced.add_argument("--fixed-positions", action="store_true", default=True)
    forced.add_argument("--free-positions", dest="fixed_positions", action="store_false")
    _add_photometry_args(forced)
    forced.set_defaults(func=_forced)

    return parser


def epsfphot(args=None):
    parsed = _parser().parse_args(args)
    parsed.func(parsed)


if __name__ == "__main__":
    epsfphot()
