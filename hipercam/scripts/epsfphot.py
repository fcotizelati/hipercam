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


def _read_mask(path, ccd, window, shape):
    if path is None:
        return None

    mask_path = Path(path)
    try:
        data, _, _ = _window_data(mask_path, ccd, window)
    except Exception as hcam_error:
        try:
            data = np.asarray(fits.getdata(mask_path))
        except Exception as fits_error:
            raise hcam.HipercamError(
                f"could not read mask '{path}' as HiPERCAM or FITS image"
            ) from fits_error
        if data.shape != shape:
            raise hcam.HipercamError(
                f"mask '{path}' has shape {data.shape}, expected {shape}; "
                f"HiPERCAM read error was: {hcam_error}"
            )
    if data.shape != shape:
        raise hcam.HipercamError(
            f"mask '{path}' has shape {data.shape}, expected {shape}"
        )
    return np.asarray(data != 0, dtype=bool)


def _data_mask(data, user_mask=None):
    mask = ~np.isfinite(data)
    if user_mask is not None:
        mask |= user_mask
    return mask if np.any(mask) else None


def _background_subtract(data, sigma=3.0, mask=None):
    _, median, _ = sigma_clipped_stats(data, sigma=sigma, mask=mask)
    return data - median, median


def _xy_table(table, x_column="x", y_column="y", dx=0.0, dy=0.0):
    for col in (x_column, y_column):
        if col not in table.colnames:
            raise hcam.HipercamError(f"table must contain '{col}' column")
    xy = QTable()
    xy["x"] = np.asarray(table[x_column], dtype=float) + dx
    xy["y"] = np.asarray(table[y_column], dtype=float) + dy
    return xy


def _source_mask(shape, x, y, radius):
    mask = np.zeros(shape, dtype=bool)
    if radius is None or radius <= 0:
        return mask

    ny, nx = shape
    radius2 = radius * radius
    for xpos, ypos in zip(np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
        if not np.isfinite(xpos) or not np.isfinite(ypos):
            continue
        xmin = max(0, int(np.floor(xpos - radius)))
        xmax = min(nx, int(np.ceil(xpos + radius)) + 1)
        ymin = max(0, int(np.floor(ypos - radius)))
        ymax = min(ny, int(np.ceil(ypos + radius)) + 1)
        if xmin >= xmax or ymin >= ymax:
            continue
        yy, xx = np.ogrid[ymin:ymax, xmin:xmax]
        mask[ymin:ymax, xmin:xmax] |= (xx - xpos) ** 2 + (yy - ypos) ** 2 <= radius2
    return mask


class SourceMaskedLocalBackground:
    """LocalBackground wrapper that ignores known sources in sky annuli."""

    def __init__(
        self,
        inner_radius,
        outer_radius,
        source_positions,
        source_radius,
        bkg_estimator=None,
    ):
        self.local_background = LocalBackground(
            inner_radius,
            outer_radius,
            bkg_estimator=MMMBackground() if bkg_estimator is None else bkg_estimator,
        )
        self.source_positions = source_positions
        self.source_radius = source_radius
        self._shape = None
        self._source_mask = None

    def __call__(self, data, x, y, mask=None):
        if self._source_mask is None or self._shape != data.shape:
            self._shape = data.shape
            self._source_mask = _source_mask(
                data.shape,
                self.source_positions["x"],
                self.source_positions["y"],
                self.source_radius,
            )
        if mask is None:
            combined_mask = self._source_mask
        else:
            combined_mask = np.asarray(mask, dtype=bool) | self._source_mask
        return self.local_background(data, x, y, mask=combined_mask)


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


def _epsf_arg(args, name, default=None):
    value = getattr(args, f"epsf_{name}", None)
    if value is None:
        value = getattr(args, name, default)
    return value


def _build_epsf_model(data, stars_tbl, args, mask=None):
    stars = extract_stars(
        NDData(data, mask=mask),
        QTable(stars_tbl[["x", "y"]]),
        size=_epsf_arg(args, "stamp_size", 25),
    )
    epsf_size = _epsf_arg(args, "size", None)
    no_smoothing = _epsf_arg(args, "no_smoothing", False)
    builder = EPSFBuilder(
        oversampling=_epsf_arg(args, "oversampling", 2),
        shape=None if epsf_size is None else (epsf_size, epsf_size),
        maxiters=_epsf_arg(args, "maxiters", 10),
        smoothing_kernel=None if no_smoothing else "quartic",
        recentering_boxsize=(
            _epsf_arg(args, "recenter_box", 7),
            _epsf_arg(args, "recenter_box", 7),
        ),
        recentering_func=centroid_com,
        progress_bar=getattr(args, "progress", False),
    )
    result = builder(stars)
    summary = {
        "epsf_n_input_stars": len(stars),
        "epsf_n_excluded_stars": result.n_excluded_stars,
        "epsf_converged": result.converged,
        "epsf_iterations": result.iterations,
        "epsf_final_center_accuracy": result.final_center_accuracy,
    }
    return result.epsf, summary


def _fit_shape(value):
    value = int(value)
    if value % 2 == 0:
        value += 1
    return (value, value)


def _make_error(data, read, gain):
    return np.sqrt(read**2 + np.maximum(data, 0.0) / gain)


def _local_background(args, source_positions=None):
    if args.local_bkg_inner is None or args.local_bkg_outer is None:
        return None
    bkg_mask_radius = getattr(args, "bkg_mask_radius", None)
    if bkg_mask_radius is not None and source_positions is not None:
        return SourceMaskedLocalBackground(
            args.local_bkg_inner,
            args.local_bkg_outer,
            source_positions,
            bkg_mask_radius,
        )
    return LocalBackground(
        args.local_bkg_inner,
        args.local_bkg_outer,
        bkg_estimator=MMMBackground(),
    )


def _source_grouper(min_separation):
    if min_separation is None or min_separation <= 0:
        return None
    return SourceGrouper(min_separation)


def _lookup_shift(shift_table, fname):
    if shift_table is None:
        return 0.0, 0.0
    match = shift_table[shift_table["file"] == fname]
    if len(match):
        return float(match["dx"][0]), float(match["dy"][0])
    return 0.0, 0.0


def _centroid_shift(data, stars_tbl, box_size, mask=None, dx0=0.0, dy0=0.0):
    if stars_tbl is None:
        return dx0, dy0, 0, np.nan, np.nan

    half = max(2, int(box_size) // 2)
    ny, nx = data.shape
    dxs = []
    dys = []

    for xmaster, ymaster in zip(stars_tbl["x"], stars_tbl["y"]):
        xguess = float(xmaster) + dx0
        yguess = float(ymaster) + dy0
        ix = int(round(xguess))
        iy = int(round(yguess))
        xmin = max(0, ix - half)
        xmax = min(nx, ix + half + 1)
        ymin = max(0, iy - half)
        ymax = min(ny, iy + half + 1)
        if xmax - xmin < 3 or ymax - ymin < 3:
            continue

        cutout = np.asarray(data[ymin:ymax, xmin:xmax], dtype=float).copy()
        cutmask = ~np.isfinite(cutout)
        if mask is not None:
            cutmask |= mask[ymin:ymax, xmin:xmax]
        if np.all(cutmask):
            continue
        sky = np.nanmedian(np.where(cutmask, np.nan, cutout))
        cutout[cutmask] = sky
        cutout -= sky
        cutout[cutout < 0] = 0
        if np.sum(cutout) <= 0:
            continue

        xcen, ycen = centroid_com(cutout)
        if not np.isfinite(xcen) or not np.isfinite(ycen):
            continue
        dxs.append((xmin + xcen) - float(xmaster))
        dys.append((ymin + ycen) - float(ymaster))

    if not dxs:
        return dx0, dy0, 0, np.nan, np.nan
    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    return dx, dy, len(dxs), float(np.std(dxs)), float(np.std(dys))


def _auto_shift(data, stars_tbl, box_size, mask=None, dx0=0.0, dy0=0.0, epsf=None):
    if stars_tbl is None:
        return dx0, dy0, 0, np.nan, np.nan

    if epsf is not None:
        init_params = Table()
        init_params["id"] = np.arange(len(stars_tbl))
        init_params["x"] = np.asarray(stars_tbl["x"], dtype=float) + dx0
        init_params["y"] = np.asarray(stars_tbl["y"], dtype=float) + dy0
        try:
            shift_epsf = copy.deepcopy(epsf)
            shift_epsf.x_0.fixed = False
            shift_epsf.y_0.fixed = False
            photometry = PSFPhotometry(
                shift_epsf,
                _fit_shape(box_size),
                finder=None,
                aperture_radius=max(2.0, float(box_size) / 2.0),
                progress_bar=False,
            )
            result = photometry(data, mask=mask, init_params=init_params)
        except Exception:
            result = None

        if result is not None and len(result):
            if "flags" in result.colnames:
                result = result[result["flags"] == 0]
            dxs = []
            dys = []
            if "id" in result.colnames:
                for row in result:
                    try:
                        idx = int(row["id"])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(stars_tbl):
                        dx = float(row["x_fit"]) - float(stars_tbl["x"][idx])
                        dy = float(row["y_fit"]) - float(stars_tbl["y"][idx])
                        if np.isfinite(dx) and np.isfinite(dy):
                            dxs.append(dx)
                            dys.append(dy)
            else:
                for row, xmaster, ymaster in zip(result, stars_tbl["x"], stars_tbl["y"]):
                    dx = float(row["x_fit"]) - float(xmaster)
                    dy = float(row["y_fit"]) - float(ymaster)
                    if np.isfinite(dx) and np.isfinite(dy):
                        dxs.append(dx)
                        dys.append(dy)
            if dxs:
                dx = float(np.median(dxs))
                dy = float(np.median(dys))
                return dx, dy, len(dxs), float(np.std(dxs)), float(np.std(dys))

    return _centroid_shift(data, stars_tbl, box_size, mask, dx0, dy0)


def _restore_fixed_positions(result, init_params):
    if (
        not len(result)
        or "x_fit" not in result.colnames
        or "y_fit" not in result.colnames
    ):
        return

    if "id" in result.colnames and "id" in init_params.colnames:
        forced = {
            str(row["id"]): (float(row["x"]), float(row["y"])) for row in init_params
        }
        xfit = []
        yfit = []
        for row in result:
            x, y = forced.get(str(row["id"]), (np.nan, np.nan))
            xfit.append(x)
            yfit.append(y)
    else:
        nrow = len(result)
        xfit = np.asarray(init_params["x"], dtype=float)[:nrow]
        yfit = np.asarray(init_params["y"], dtype=float)[:nrow]
    result["x_fit"] = xfit
    result["y_fit"] = yfit


def _output_path(directory, prefix, nframe, fname, suffix):
    root = Path(fname).stem
    return Path(directory) / f"{prefix}{nframe:05d}_{root}{suffix}"


def _write_residual(directory, prefix, nframe, fname, photometry, data, psf_shape):
    Path(directory).mkdir(parents=True, exist_ok=True)
    residual = photometry.make_residual_image(data, psf_shape=psf_shape)
    fits.PrimaryHDU(np.asarray(residual, dtype=np.float32)).writeto(
        _output_path(directory, prefix, nframe, fname, ".fits"),
        overwrite=True,
    )


def _build_epsf(args):
    data, _, _ = _window_data(args.hcm, args.ccd, args.window)
    mask = _data_mask(data, _read_mask(args.mask, args.ccd, args.window, data.shape))
    bkgsub, _ = _background_subtract(data, sigma=args.sigma, mask=mask)

    stars_tbl = _read_table(args.stars)
    stars_tbl = _xy_table(stars_tbl)
    epsf, epsf_summary = _build_epsf_model(bkgsub, stars_tbl, args, mask)
    _write_epsf(args.output, epsf)

    summary = Table()
    for key, value in epsf_summary.items():
        summary[key.replace("epsf_", "")] = [value]
    if args.summary:
        summary.write(args.summary, overwrite=True)
    else:
        print(summary)


def _make_source_list(args):
    data, _, _ = _window_data(args.hcm, args.ccd, args.window)
    mask = _data_mask(data, _read_mask(args.mask, args.ccd, args.window, data.shape))
    bkgsub, _ = _background_subtract(data, sigma=args.sigma, mask=mask)
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
        mask=mask,
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

    frame_epsf_stars = None
    if args.frame_epsf_stars:
        frame_epsf_stars = _xy_table(_read_table(args.frame_epsf_stars))

    auto_shift_stars = None
    if args.auto_shift_stars:
        auto_shift_stars = _xy_table(_read_table(args.auto_shift_stars))
    elif args.auto_shifts and frame_epsf_stars is not None:
        auto_shift_stars = frame_epsf_stars
    elif args.auto_shifts:
        raise hcam.HipercamError(
            "--auto-shifts requires --auto-shift-stars or --frame-epsf-stars"
        )

    shift_rows = []
    epsf0 = _read_epsf(args.epsf)
    rows = []
    for nframe, fname in enumerate(files, start=1):
        data, _, mccd = _window_data(fname, args.ccd, args.window)
        mask = _data_mask(
            data, _read_mask(args.mask, args.ccd, args.window, data.shape)
        )
        bkgsub, global_bkg = _background_subtract(data, sigma=args.sigma, mask=mask)
        error = _make_error(data, args.read, args.gain)

        dx, dy = _lookup_shift(shift_table, fname)
        nshifts = 0
        shift_xrms = np.nan
        shift_yrms = np.nan
        if args.auto_shifts:
            dx, dy, nshifts, shift_xrms, shift_yrms = _auto_shift(
                bkgsub,
                auto_shift_stars,
                args.shift_box_size,
                mask=mask,
                dx0=dx,
                dy0=dy,
                epsf=epsf0,
            )
        shift_rows.append((fname, dx, dy, nshifts, shift_xrms, shift_yrms))

        init_params = Table()
        init_params["id"] = source_table[args.id_column]
        init_params["x"] = source_table[args.x_column] + dx
        init_params["y"] = source_table[args.y_column] + dy

        epsf_summary = {}
        if args.rebuild_epsf:
            if frame_epsf_stars is None:
                raise hcam.HipercamError("--rebuild-epsf requires --frame-epsf-stars")
            epsf_stars = _xy_table(frame_epsf_stars, dx=dx, dy=dy)
            epsf, epsf_summary = _build_epsf_model(bkgsub, epsf_stars, args, mask)
            if args.frame_epsf_dir:
                Path(args.frame_epsf_dir).mkdir(parents=True, exist_ok=True)
                _write_epsf(
                    _output_path(
                        args.frame_epsf_dir,
                        args.frame_epsf_prefix,
                        nframe,
                        fname,
                        "_epsf.fits",
                    ),
                    epsf,
                )
        else:
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
            local_bkg_estimator=_local_background(args, init_params),
            progress_bar=False,
        )
        result = photometry(bkgsub, mask=mask, error=error, init_params=init_params)
        if args.fixed_positions:
            _restore_fixed_positions(result, init_params)
        if args.residual_dir:
            _write_residual(
                args.residual_dir,
                args.residual_prefix,
                nframe,
                fname,
                photometry,
                bkgsub,
                (_epsf_arg(args, "stamp_size", args.fit_size),) * 2,
            )
        result["frame"] = nframe
        result["file"] = fname
        result["mjdutc"] = mccd.head.get("MJDUTC", np.nan)
        result["global_bkg"] = global_bkg
        result["shift_dx"] = dx
        result["shift_dy"] = dy
        result["shift_nstars"] = nshifts
        result["shift_xrms"] = shift_xrms
        result["shift_yrms"] = shift_yrms
        for key, value in epsf_summary.items():
            result[key] = value
        rows.append(result)

    if not rows:
        raise hcam.HipercamError("no frames were processed")
    vstack(rows).write(args.output, overwrite=True)
    if args.write_shifts:
        shifts = Table(
            rows=shift_rows,
            names=("file", "dx", "dy", "nstars", "x_rms", "y_rms"),
        )
        shifts.write(args.write_shifts, overwrite=True)


def _parse_id_set(text, ids):
    ids = [str(item) for item in ids]
    if text is None or text.lower() == "all":
        return set(ids)
    if text.lower() == "none":
        return set()
    wanted = {item.strip() for item in text.split(",") if item.strip()}
    missing = wanted.difference(ids)
    if missing:
        raise hcam.HipercamError(
            "variable source IDs not found in source table: " + ", ".join(sorted(missing))
        )
    return wanted


def _psf_values(epsf, yy, xx, x0, y0):
    values = epsf.evaluate(xx, yy, 1.0, x0, y0)
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = 0.0
    return values


def _scene(args):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    files = _read_file_list(args.flist)
    source_table = _read_table(args.sources)
    if args.id_column not in source_table.colnames:
        source_table[args.id_column] = np.arange(1, len(source_table) + 1)
    for col in (args.x_column, args.y_column):
        if col not in source_table.colnames:
            raise hcam.HipercamError(f"source table must contain '{col}'")

    source_ids = [str(item) for item in source_table[args.id_column]]
    variable_ids = _parse_id_set(args.variable_ids, source_ids)
    constant_ids = [sid for sid in source_ids if sid not in variable_ids]
    variable_ids = [sid for sid in source_ids if sid in variable_ids]

    shift_table = None
    if args.shifts:
        shift_table = _read_table(args.shifts)
        for col in ("file", "dx", "dy"):
            if col not in shift_table.colnames:
                raise hcam.HipercamError("shift table must contain file, dx, dy columns")

    frame_epsf_stars = None
    if args.frame_epsf_stars:
        frame_epsf_stars = _xy_table(_read_table(args.frame_epsf_stars))

    auto_shift_stars = None
    if args.auto_shift_stars:
        auto_shift_stars = _xy_table(_read_table(args.auto_shift_stars))
    elif args.auto_shifts and frame_epsf_stars is not None:
        auto_shift_stars = frame_epsf_stars
    elif args.auto_shifts:
        raise hcam.HipercamError(
            "--auto-shifts requires --auto-shift-stars or --frame-epsf-stars"
        )

    epsf0 = _read_epsf(args.epsf)
    column_index = {}
    columns = []
    for sid in constant_ids:
        column_index[("constant", sid, None)] = len(columns)
        columns.append(("constant", sid, None))
    for iframe in range(len(files)):
        for sid in variable_ids:
            column_index[("variable", sid, iframe)] = len(columns)
            columns.append(("variable", sid, iframe))
        if args.fit_background:
            column_index[("background", None, iframe)] = len(columns)
            columns.append(("background", None, iframe))

    row_idx = []
    col_idx = []
    values = []
    rhs_chunks = []
    contexts = []
    shift_rows = []
    row0 = 0

    for iframe, fname in enumerate(files):
        data, _, mccd = _window_data(fname, args.ccd, args.window)
        mask = _data_mask(
            data, _read_mask(args.mask, args.ccd, args.window, data.shape)
        )
        bkgsub, global_bkg = _background_subtract(data, sigma=args.sigma, mask=mask)
        error = _make_error(data, args.read, args.gain)
        dx, dy = _lookup_shift(shift_table, fname)
        nshifts = 0
        shift_xrms = np.nan
        shift_yrms = np.nan
        if args.auto_shifts:
            dx, dy, nshifts, shift_xrms, shift_yrms = _auto_shift(
                bkgsub,
                auto_shift_stars,
                args.shift_box_size,
                mask=mask,
                dx0=dx,
                dy0=dy,
                epsf=epsf0,
            )

        if args.rebuild_epsf:
            if frame_epsf_stars is None:
                raise hcam.HipercamError("--rebuild-epsf requires --frame-epsf-stars")
            epsf_stars = _xy_table(frame_epsf_stars, dx=dx, dy=dy)
            epsf, _ = _build_epsf_model(bkgsub, epsf_stars, args, mask)
            if args.frame_epsf_dir:
                Path(args.frame_epsf_dir).mkdir(parents=True, exist_ok=True)
                _write_epsf(
                    _output_path(
                        args.frame_epsf_dir,
                        args.frame_epsf_prefix,
                        iframe + 1,
                        fname,
                        "_epsf.fits",
                    ),
                    epsf,
                )
        else:
            epsf = copy.deepcopy(epsf0)
        shift_rows.append((fname, dx, dy, nshifts, shift_xrms, shift_yrms))

        xs = np.asarray(source_table[args.x_column], dtype=float) + dx
        ys = np.asarray(source_table[args.y_column], dtype=float) + dy
        pad = int(args.scene_padding)
        xmin = max(0, int(np.floor(np.min(xs) - pad)))
        xmax = min(data.shape[1], int(np.ceil(np.max(xs) + pad)) + 1)
        ymin = max(0, int(np.floor(np.min(ys) - pad)))
        ymax = min(data.shape[0], int(np.ceil(np.max(ys) + pad)) + 1)
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        valid = np.isfinite(bkgsub[ymin:ymax, xmin:xmax])
        if mask is not None:
            valid &= ~mask[ymin:ymax, xmin:xmax]
        err = error[ymin:ymax, xmin:xmax]
        valid &= np.isfinite(err) & (err > 0)
        nvalid = int(np.sum(valid))
        if nvalid == 0:
            raise hcam.HipercamError(f"no valid scene pixels in {fname}")

        rhs_chunks.append((bkgsub[ymin:ymax, xmin:xmax][valid] / err[valid]).ravel())

        for sid, x0, y0 in zip(source_ids, xs, ys):
            if sid in constant_ids:
                col = column_index[("constant", sid, None)]
            else:
                col = column_index[("variable", sid, iframe)]
            template = _psf_values(epsf, yy, xx, x0, y0)
            weighted = np.zeros_like(template, dtype=float)
            weighted[valid] = template[valid] / err[valid]
            nz = np.flatnonzero(np.abs(weighted[valid].ravel()) > 0)
            if len(nz):
                row_idx.extend(row0 + nz)
                col_idx.extend([col] * len(nz))
                values.extend(weighted[valid].ravel()[nz])

        if args.fit_background:
            col = column_index[("background", None, iframe)]
            weighted = np.zeros_like(err, dtype=float)
            weighted[valid] = 1.0 / err[valid]
            nz = np.flatnonzero(weighted[valid].ravel() > 0)
            row_idx.extend(row0 + nz)
            col_idx.extend([col] * len(nz))
            values.extend(weighted[valid].ravel()[nz])

        contexts.append(
            {
                "frame": iframe + 1,
                "file": fname,
                "mjdutc": mccd.head.get("MJDUTC", np.nan),
                "global_bkg": global_bkg,
                "dx": dx,
                "dy": dy,
                "nshifts": nshifts,
                "shift_xrms": shift_xrms,
                "shift_yrms": shift_yrms,
                "bkgsub": bkgsub,
                "epsf": epsf,
                "roi": (xmin, xmax, ymin, ymax),
                "valid": valid,
                "yy": yy,
                "xx": xx,
                "xs": xs,
                "ys": ys,
            }
        )
        row0 += nvalid

    rhs = np.concatenate(rhs_chunks)
    matrix = coo_matrix((values, (row_idx, col_idx)), shape=(len(rhs), len(columns))).tocsr()
    solution = lsqr(
        matrix,
        rhs,
        atol=args.lsqr_tol,
        btol=args.lsqr_tol,
        iter_lim=args.lsqr_iter,
    )[0]

    rows = []
    for iframe, context in enumerate(contexts):
        bkg_fit = (
            solution[column_index[("background", None, iframe)]]
            if args.fit_background
            else 0.0
        )
        model = np.full(context["yy"].shape, bkg_fit, dtype=float)
        for sid, x0, y0 in zip(source_ids, context["xs"], context["ys"]):
            if sid in constant_ids:
                col = column_index[("constant", sid, None)]
                flux_type = "constant"
            else:
                col = column_index[("variable", sid, iframe)]
                flux_type = "variable"
            flux = solution[col]
            model += flux * _psf_values(context["epsf"], context["yy"], context["xx"], x0, y0)
            rows.append(
                (
                    context["frame"],
                    context["file"],
                    context["mjdutc"],
                    sid,
                    flux,
                    flux_type,
                    context["dx"],
                    context["dy"],
                    context["nshifts"],
                    context["shift_xrms"],
                    context["shift_yrms"],
                    context["global_bkg"],
                    bkg_fit,
                )
            )
        if args.residual_dir:
            xmin, xmax, ymin, ymax = context["roi"]
            residual = context["bkgsub"][ymin:ymax, xmin:xmax] - model
            Path(args.residual_dir).mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(np.asarray(residual, dtype=np.float32)).writeto(
                _output_path(
                    args.residual_dir,
                    args.residual_prefix,
                    context["frame"],
                    context["file"],
                    ".fits",
                ),
                overwrite=True,
            )

    output = Table(
        rows=rows,
        names=(
            "frame",
            "file",
            "mjdutc",
            "id",
            "flux_fit",
            "flux_type",
            "shift_dx",
            "shift_dy",
            "shift_nstars",
            "shift_xrms",
            "shift_yrms",
            "global_bkg",
            "scene_bkg_fit",
        ),
    )
    output.write(args.output, overwrite=True)
    if args.write_shifts:
        shifts = Table(
            rows=shift_rows,
            names=("file", "dx", "dy", "nstars", "x_rms", "y_rms"),
        )
        shifts.write(args.write_shifts, overwrite=True)


def _add_common_image_args(parser):
    parser.add_argument("hcm", help="input .hcm image")
    parser.add_argument("ccd", help="CCD label")
    parser.add_argument("window", help="window label")
    parser.add_argument(
        "--mask",
        help="optional HiPERCAM or FITS mask; non-zero pixels are ignored",
    )


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


def _add_forced_frame_args(parser):
    parser.add_argument("--shifts", help="optional table with file,dx,dy columns")
    parser.add_argument(
        "--auto-shifts",
        action="store_true",
        help="measure frame shifts automatically by ePSF-fitting reference stars",
    )
    parser.add_argument(
        "--auto-shift-stars",
        help="table with x,y columns for stars used to measure automatic shifts",
    )
    parser.add_argument("--shift-box-size", type=int, default=15)
    parser.add_argument(
        "--frame-epsf-stars",
        help="table with x,y columns for stars used to rebuild each frame ePSF",
    )
    parser.add_argument(
        "--rebuild-epsf",
        action="store_true",
        help="build a fresh ePSF on each frame using --frame-epsf-stars",
    )
    parser.add_argument("--epsf-stamp-size", type=int, default=25)
    parser.add_argument("--epsf-size", type=int)
    parser.add_argument("--epsf-oversampling", type=int, default=2)
    parser.add_argument("--epsf-maxiters", type=int, default=10)
    parser.add_argument("--epsf-recenter-box", type=int, default=7)
    parser.add_argument("--epsf-no-smoothing", action="store_true")
    parser.add_argument("--residual-dir", help="optional directory for residual FITS files")
    parser.add_argument("--residual-prefix", default="resid_")
    parser.add_argument("--write-shifts", help="optional output table of measured shifts")
    parser.add_argument("--frame-epsf-dir", help="optional directory for per-frame ePSFs")
    parser.add_argument("--frame-epsf-prefix", default="frame_")


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
    _add_forced_frame_args(forced)
    forced.add_argument(
        "--mask",
        help="optional HiPERCAM or FITS mask; non-zero pixels are ignored",
    )
    forced.add_argument("--id-column", default="id")
    forced.add_argument("--x-column", default="x_fit")
    forced.add_argument("--y-column", default="y_fit")
    forced.add_argument("--fixed-positions", action="store_true", default=True)
    forced.add_argument("--free-positions", dest="fixed_positions", action="store_false")
    _add_photometry_args(forced)
    forced.add_argument(
        "--bkg-mask-radius",
        type=float,
        help=(
            "mask known source footprints by this radius only when estimating "
            "local sky backgrounds"
        ),
    )
    forced.set_defaults(func=_forced)

    scene = subparsers.add_parser("scene")
    scene.add_argument("flist", help="file containing .hcm frames to process")
    scene.add_argument("ccd", help="CCD label")
    scene.add_argument("window", help="window label")
    scene.add_argument("epsf", help="input ePSF FITS file")
    scene.add_argument("sources", help="source table from make-source-list")
    scene.add_argument("output", help="output scene-model table")
    _add_forced_frame_args(scene)
    scene.add_argument(
        "--mask",
        help="optional HiPERCAM or FITS mask; non-zero pixels are ignored",
    )
    scene.add_argument("--id-column", default="id")
    scene.add_argument("--x-column", default="x_fit")
    scene.add_argument("--y-column", default="y_fit")
    scene.add_argument("--read", type=float, default=3.5)
    scene.add_argument("--gain", type=float, default=1.0)
    scene.add_argument("--sigma", type=float, default=3.0)
    scene.add_argument("--scene-padding", type=int, default=25)
    scene.add_argument(
        "--variable-ids",
        default="all",
        help=(
            "comma-separated IDs to fit independently in every frame; "
            "use 'all' or 'none'"
        ),
    )
    scene.add_argument("--fit-background", action="store_true", default=True)
    scene.add_argument(
        "--no-fit-background", dest="fit_background", action="store_false"
    )
    scene.add_argument("--lsqr-tol", type=float, default=1e-8)
    scene.add_argument("--lsqr-iter", type=int, default=1000)
    scene.set_defaults(func=_scene)

    return parser


def epsfphot(args=None):
    parsed = _parser().parse_args(args)
    parsed.func(parsed)


if __name__ == "__main__":
    epsfphot()
