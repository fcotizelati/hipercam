"""Experimental ePSF photometry helpers for crowded-field work.

This module provides a script-level workflow around the modern
``photutils.psf`` ePSF API. It is intentionally separate from the normal
HiPERCAM aperture-photometry path: inputs are calibrated HiPERCAM ``.hcm``
frames, a single CCD/window selection, and table files describing master-frame
source or PSF-star positions; outputs are ECSV tables and optional FITS
diagnostic images.

The main assumptions are deliberately conservative. Source positions are
defined on a high-S/N master image, frame-to-frame registration is represented
by a translation, and a bad-pixel mask suppresses unusable pixels without
removing neighbouring stars from the simultaneous PSF fit. Per-frame ePSFs can
be built from reference stars, but the build is guarded by a minimum usable-star
count and falls back to the master ePSF if a frame is not good enough.
"""

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
    """Read an Astropy table from disk.

    Parameters
    ----------
    path : str or path-like
        Path to any table format understood by `astropy.table.Table.read`.
        The command-line interface normally uses ECSV tables because they keep
        column names and metadata explicit, but FITS or other Astropy-supported
        formats are accepted by this helper.

    Returns
    -------
    astropy.table.Table
        The decoded table.

    Notes
    -----
    This thin wrapper exists so that all table reads go through one place. It
    makes later changes, such as adding table validation or default format
    handling, less invasive.
    """
    return Table.read(path)


def _read_file_list(path):
    """Read a HiPERCAM frame-list file.

    Parameters
    ----------
    path : str or path-like
        Text file containing one input frame path per line. Empty lines and
        lines whose first non-blank character is ``#`` are ignored.

    Returns
    -------
    list of str
        Cleaned list of frame paths. The order is preserved and is used as the
        frame order in forced and scene photometry outputs.

    Notes
    -----
    The parser intentionally does not expand wildcards or shell syntax. Keeping
    the list format literal makes reruns reproducible and matches the existing
    HiPERCAM ``source=hf`` file-list convention.
    """
    with open(path) as fp:
        return [
            line.strip()
            for line in fp
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _window_data(hcm_file, ccd, window):
    """Load one CCD/window from a HiPERCAM ``.hcm`` file.

    Parameters
    ----------
    hcm_file : str or path-like
        File readable by `hipercam.MCCD.read`.

    ccd : str or int
        CCD label to select. The label is converted to a string before lookup,
        matching the storage convention inside `hipercam.MCCD`.

    window : str or int
        Window label within the selected CCD, also converted to a string.

    Returns
    -------
    data : numpy.ndarray
        Floating-point view/copy of the selected window data.

    wind : hipercam.Window
        The original HiPERCAM window object. This is returned so callers can
        access geometry or metadata if needed.

    mccd : hipercam.MCCD
        The full multi-CCD object, primarily used here for frame headers such as
        ``MJDUTC``.
    """
    mccd = hcam.MCCD.read(str(hcm_file))
    wind = mccd[str(ccd)][str(window)]
    return np.asarray(wind.data, dtype=float), wind, mccd


def _read_mask(path, ccd, window, shape):
    """Read an optional bad-pixel mask for one science window.

    Parameters
    ----------
    path : str or path-like or None
        Mask file. If `None`, no user mask is returned. Otherwise the helper
        first tries to read the path as a HiPERCAM file and select the same
        ``ccd``/``window`` as the science frame. If that fails, it tries to read
        the path as a plain FITS image.

    ccd, window : str or int
        HiPERCAM CCD/window labels used if the mask is stored as a HiPERCAM
        image.

    shape : tuple of int
        Expected two-dimensional image shape. Shape mismatches are fatal because
        a shifted or mismatched mask would silently corrupt the fit.

    Returns
    -------
    numpy.ndarray or None
        Boolean mask where ``True`` marks pixels to ignore, or `None` when no
        mask path was supplied.

    Raises
    ------
    hipercam.HipercamError
        If the mask cannot be read or if its shape does not match ``shape``.

    Notes
    -----
    This mask is for bad pixels, saturated columns, defects, or similar pixels
    that should not enter the fit. It is not the source-footprint mask used only
    for local sky estimation in ``SourceMaskedLocalBackground``.
    """
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
    """Combine non-finite-pixel masking with an optional user mask.

    Parameters
    ----------
    data : numpy.ndarray
        Science image array.

    user_mask : numpy.ndarray or None, optional
        Boolean mask supplied by `_read_mask`. ``True`` pixels are excluded.

    Returns
    -------
    numpy.ndarray or None
        Combined boolean mask, or `None` if no pixels are masked. Returning
        `None` preserves the convention used by Photutils and avoids allocating
        mask arrays unnecessarily.
    """
    mask = ~np.isfinite(data)
    if user_mask is not None:
        mask |= user_mask
    return mask if np.any(mask) else None


def _background_subtract(data, sigma=3.0, mask=None):
    """Subtract a robust scalar background from an image.

    Parameters
    ----------
    data : numpy.ndarray
        Input image in detector coordinates.

    sigma : float, optional
        Sigma-clipping threshold passed to `astropy.stats.sigma_clipped_stats`.

    mask : numpy.ndarray or None, optional
        Boolean mask identifying pixels to ignore when estimating the scalar
        background.

    Returns
    -------
    bkgsub : numpy.ndarray
        Image with the sigma-clipped median subtracted.

    median : float
        The subtracted scalar background level.

    Notes
    -----
    This is intentionally simple. In crowded fields a scalar background is not a
    replacement for modelling neighbouring stars; neighbours should remain in
    the source list and be fitted by the PSF model.
    """
    _, median, _ = sigma_clipped_stats(data, sigma=sigma, mask=mask)
    return data - median, median


def _xy_table(table, x_column="x", y_column="y", dx=0.0, dy=0.0):
    """Return a normalized ``x``/``y`` position table.

    Parameters
    ----------
    table : astropy.table.Table
        Input table containing source or PSF-star coordinates.

    x_column, y_column : str, optional
        Names of the coordinate columns to read from ``table``.

    dx, dy : float, optional
        Translation to add to the returned coordinates. This is used when a
        master-frame PSF-star list is shifted into an individual frame.

    Returns
    -------
    astropy.table.QTable
        Table with exactly two coordinate columns, ``x`` and ``y``.

    Raises
    ------
    hipercam.HipercamError
        If either coordinate column is missing.
    """
    for col in (x_column, y_column):
        if col not in table.colnames:
            raise hcam.HipercamError(f"table must contain '{col}' column")
    xy = QTable()
    xy["x"] = np.asarray(table[x_column], dtype=float) + dx
    xy["y"] = np.asarray(table[y_column], dtype=float) + dy
    return xy


def _source_mask(shape, x, y, radius):
    """Build a circular footprint mask around known source positions.

    Parameters
    ----------
    shape : tuple of int
        Output mask shape as ``(ny, nx)``.

    x, y : array-like
        Source positions in pixel coordinates.

    radius : float or None
        Mask radius in pixels. A non-positive value disables masking and returns
        an all-False mask.

    Returns
    -------
    numpy.ndarray
        Boolean mask where ``True`` marks pixels lying inside any source
        footprint.

    Notes
    -----
    This helper is used only to protect local background annuli from known
    sources. It must not be confused with the bad-pixel mask, because masking
    neighbouring stars out of the science image would prevent the simultaneous
    PSF fit from deblending them.
    """
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
    """Local background estimator that masks known source footprints.

    Parameters
    ----------
    inner_radius, outer_radius : float
        Inner and outer radii of the local-background annulus, passed to
        `photutils.background.LocalBackground`.

    source_positions : astropy.table.Table or astropy.table.QTable
        Table with ``x`` and ``y`` columns giving all source positions that
        should be ignored while estimating local backgrounds.

    source_radius : float
        Radius of the circular mask placed around every source in
        ``source_positions``.

    bkg_estimator : callable, optional
        Photutils-compatible background estimator. If omitted, an
        `MMMBackground` estimator is used.

    Notes
    -----
    The class is callable because Photutils expects local-background estimators
    to be callables with the signature ``(data, x, y, mask=None)``. It caches the
    source-footprint mask by image shape so repeated calls in one frame do not
    rebuild the same mask.
    """

    def __init__(
        self,
        inner_radius,
        outer_radius,
        source_positions,
        source_radius,
        bkg_estimator=None,
    ):
        """Initialise the wrapped Photutils local-background estimator.

        Parameters
        ----------
        inner_radius, outer_radius : float
            Annulus radii passed directly to `LocalBackground`.

        source_positions : astropy.table.Table
            Source positions used to build the cached footprint mask.

        source_radius : float
            Radius used by `_source_mask` around every known source.

        bkg_estimator : callable, optional
            Background estimator. If absent, `MMMBackground` is used.
        """
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
        """Estimate local background while masking known sources.

        Parameters
        ----------
        data : numpy.ndarray
            Image passed by Photutils.

        x, y : float or array-like
            Positions at which Photutils wants the local background estimated.

        mask : numpy.ndarray or None, optional
            Existing Photutils mask. It is combined with the cached source mask.

        Returns
        -------
        float or numpy.ndarray
            Local background estimate from the wrapped `LocalBackground`
            instance.
        """
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
    """Write an `ImagePSF` model to a compact FITS file.

    Parameters
    ----------
    path : str or path-like
        Output FITS filename. Existing files are overwritten.

    epsf : photutils.psf.ImagePSF
        Empirical PSF model. The image data are written as the primary HDU and
        the oversampling/origin metadata needed by `_read_epsf` are stored in
        header keywords.

    Notes
    -----
    The FITS file is intentionally minimal and local to this script. It is not
    meant to be a general PSF interchange standard; it stores just enough
    information to recreate the Photutils `ImagePSF` object.
    """
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
    """Read an ePSF FITS file written by `_write_epsf`.

    Parameters
    ----------
    path : str or path-like
        FITS file containing an ePSF image and optional ``OVERSX``, ``OVERSY``,
        ``ORIGINX``, and ``ORIGINY`` header keywords.

    Returns
    -------
    photutils.psf.ImagePSF
        ImagePSF model ready for Photutils PSF photometry.
    """
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header
    oversampling = (int(hdr.get("OVERSX", 1)), int(hdr.get("OVERSY", 1)))
    origin = (float(hdr["ORIGINX"]), float(hdr["ORIGINY"])) if "ORIGINX" in hdr else None
    return ImagePSF(data, oversampling=oversampling, origin=origin)


def _epsf_arg(args, name, default=None):
    """Return an ePSF-builder option from a command namespace.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    name : str
        Base option name, for example ``"stamp_size"``. Forced and scene modes
        expose these builder controls with an ``epsf_`` prefix, while
        ``build-epsf`` exposes them without the prefix.

    default : object, optional
        Value returned when neither the prefixed nor unprefixed argument exists.

    Returns
    -------
    object
        The selected argument value.
    """
    value = getattr(args, f"epsf_{name}", None)
    if value is None:
        value = getattr(args, name, default)
    return value


def _build_epsf_model(data, stars_tbl, args, mask=None):
    """Construct an empirical PSF model from selected reference stars.

    Parameters
    ----------
    data : numpy.ndarray
        Background-subtracted image containing the PSF-star positions.

    stars_tbl : astropy.table.Table
        Table with ``x`` and ``y`` columns. The stars should already be vetted
        by the user for isolation, saturation, S/N, and detector defects; this
        function checks only the numerical outcome of the ePSF build.

    args : argparse.Namespace
        Command arguments providing ePSF builder settings. The helper accepts
        both unprefixed names used by ``build-epsf`` and ``epsf_``-prefixed names
        used by frame-processing commands.

    mask : numpy.ndarray or None, optional
        Boolean mask for bad pixels and non-finite pixels.

    Returns
    -------
    epsf : photutils.psf.ImagePSF
        Built empirical PSF model.

    summary : dict
        Diagnostic values including number of input stars, number excluded by
        Photutils, number used, convergence state, iteration count, and final
        centering accuracy.

    Raises
    ------
    hipercam.HipercamError
        If fewer than ``min_stars`` usable stars remain after ePSF building.

    Notes
    -----
    The minimum-star guard is deliberately conservative. It cannot decide
    whether a PSF-star list is scientifically good, but it prevents the pipeline
    from silently rebuilding a per-frame ePSF from too little information.
    """
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
    n_input = len(stars)
    n_excluded = int(result.n_excluded_stars)
    n_used = n_input - n_excluded
    min_stars = int(_epsf_arg(args, "min_stars", 3))
    if n_used < min_stars:
        raise hcam.HipercamError(
            f"ePSF build used {n_used} stars, below minimum {min_stars}"
        )
    summary = {
        "epsf_status": "rebuilt",
        "epsf_n_input_stars": n_input,
        "epsf_n_excluded_stars": n_excluded,
        "epsf_n_used_stars": n_used,
        "epsf_converged": result.converged,
        "epsf_iterations": result.iterations,
        "epsf_final_center_accuracy": result.final_center_accuracy,
        "epsf_warning": "",
    }
    return result.epsf, summary


def _epsf_fallback_summary(error):
    """Create diagnostic metadata for a failed per-frame ePSF rebuild.

    Parameters
    ----------
    error : Exception
        Exception raised while trying to build the frame-specific ePSF.

    Returns
    -------
    dict
        Summary dictionary with the same broad keys as `_build_epsf_model`, but
        marked with ``epsf_status = "fallback_master"`` and zero usable stars.

    Notes
    -----
    Forced and scene photometry use this when ``--rebuild-epsf`` was requested
    but the frame does not meet the ePSF guardrails. The science extraction then
    continues with the master ePSF, and the output table records the fallback so
    the affected frames can be inspected later.
    """
    return {
        "epsf_status": "fallback_master",
        "epsf_n_input_stars": 0,
        "epsf_n_excluded_stars": 0,
        "epsf_n_used_stars": 0,
        "epsf_converged": False,
        "epsf_iterations": 0,
        "epsf_final_center_accuracy": np.nan,
        "epsf_warning": str(error),
    }


def _fit_shape(value):
    """Convert a scalar fit size into a valid two-dimensional PSF fit shape.

    Parameters
    ----------
    value : int-like
        Requested linear size in pixels.

    Returns
    -------
    tuple of int
        ``(ny, nx)`` shape with an odd size in both dimensions.

    Notes
    -----
    Photutils PSF fitting expects a finite pixel stamp around each source. Odd
    sizes are preferable because they provide a central pixel and avoid subtle
    half-pixel asymmetries in small fit boxes.
    """
    value = int(value)
    if value % 2 == 0:
        value += 1
    return (value, value)


def _make_error(data, read, gain):
    """Construct a simple per-pixel uncertainty image.

    Parameters
    ----------
    data : numpy.ndarray
        Image in detector units before scalar background subtraction.

    read : float
        Read noise in electrons, or in units consistent with ``data`` and
        ``gain``.

    gain : float
        Gain in electrons per data unit.

    Returns
    -------
    numpy.ndarray
        Per-pixel 1-sigma uncertainty estimate,
        ``sqrt(read**2 + max(data, 0) / gain)``.

    Notes
    -----
    This is a pragmatic weighting model for the Photutils fits. It does not
    propagate the full HiPERCAM calibration error budget, but it prevents bright
    pixels from being weighted as if they had the same noise as sky pixels.
    """
    return np.sqrt(read**2 + np.maximum(data, 0.0) / gain)


def _local_background(args, source_positions=None):
    """Build the local-background estimator requested by command options.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. The relevant fields are
        ``local_bkg_inner``, ``local_bkg_outer``, and optionally
        ``bkg_mask_radius``.

    source_positions : astropy.table.Table or None, optional
        Positions of known sources. If supplied together with
        ``bkg_mask_radius``, these sources are masked only for local background
        estimation.

    Returns
    -------
    callable or None
        Photutils-compatible local-background estimator, or `None` when no
        local annulus was requested.
    """
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
    """Create a Photutils source grouper for simultaneous local fits.

    Parameters
    ----------
    min_separation : float or None
        Minimum separation in pixels for grouping sources into a simultaneous
        fit. Non-positive values disable grouping.

    Returns
    -------
    photutils.psf.SourceGrouper or None
        Grouper object used by Photutils, or `None` if grouping is disabled.
    """
    if min_separation is None or min_separation <= 0:
        return None
    return SourceGrouper(min_separation)


def _translation_transform(dx=0.0, dy=0.0, method="table", nstars=0):
    """Represent the frame registration used by this script.

    Parameters
    ----------
    dx, dy : float, optional
        Translation from master-frame coordinates to the current frame.

    method : str, optional
        Provenance of the shift. Current values include ``"identity"``,
        ``"table"``, ``"epsf"``, ``"centroid"``, and ``"initial"``.

    nstars : int, optional
        Number of reference stars used to measure the shift.

    Returns
    -------
    dict
        Small transform dictionary consumed by `_apply_transform` and written
        into diagnostic output columns.

    Notes
    -----
    The transform is deliberately limited to translation. This keeps the
    crowded-field photometry tied to the master image while avoiding unstable
    high-order registrations from sparse reference-star lists.
    """
    return {
        "method": method,
        "dx": float(dx),
        "dy": float(dy),
        "nstars": int(nstars),
        "x_rms": np.nan,
        "y_rms": np.nan,
    }


def _lookup_shift(shift_table, fname):
    """Look up a tabulated shift for one frame.

    Parameters
    ----------
    shift_table : astropy.table.Table or None
        Optional table with columns ``file``, ``dx``, and ``dy``.

    fname : str
        Frame filename exactly as it appears in the input file list.

    Returns
    -------
    tuple of float
        ``(dx, dy)`` shift. If the table is absent or the frame is not present,
        a zero shift is returned.
    """
    if shift_table is None:
        return 0.0, 0.0
    match = shift_table[shift_table["file"] == fname]
    if len(match):
        return float(match["dx"][0]), float(match["dy"][0])
    return 0.0, 0.0


def _initial_transform(shift_table, fname):
    """Create the starting transform for a frame.

    Parameters
    ----------
    shift_table : astropy.table.Table or None
        Optional user-supplied shift table.

    fname : str
        Frame filename.

    Returns
    -------
    dict
        Translation transform initialized either from the shift table or from
        the identity transform. Automatic shift measurement, if requested, uses
        this as its initial guess.
    """
    dx, dy = _lookup_shift(shift_table, fname)
    return _translation_transform(dx, dy, method="table" if shift_table else "identity")


def _apply_transform(transform, x, y):
    """Apply a translation transform to coordinates.

    Parameters
    ----------
    transform : dict
        Dictionary created by `_translation_transform`.

    x, y : array-like
        Master-frame coordinates.

    Returns
    -------
    tuple of numpy.ndarray
        Coordinates shifted into the current frame.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x + transform["dx"], y + transform["dy"]


def _transform_table(table, transform):
    """Apply the current frame transform to a coordinate table.

    Parameters
    ----------
    table : astropy.table.Table
        Table with ``x`` and ``y`` columns.

    transform : dict
        Translation transform.

    Returns
    -------
    astropy.table.QTable
        New table with transformed ``x`` and ``y`` coordinates. The input table
        is not modified.
    """
    xy = QTable()
    xy["x"], xy["y"] = _apply_transform(transform, table["x"], table["y"])
    return xy


def _fit_translation_transform(xmaster, ymaster, xframe, yframe):
    """Fit a robust translational registration from reference stars.

    Parameters
    ----------
    xmaster, ymaster : array-like
        Reference-star positions in the master coordinate system.

    xframe, yframe : array-like
        Measured positions of the same reference stars in the current frame.

    Returns
    -------
    dict
        Translation transform whose ``dx`` and ``dy`` are the median measured
        offsets, with RMS residual diagnostics in ``x_rms`` and ``y_rms``.

    Raises
    ------
    hipercam.HipercamError
        If no finite matched reference-star measurements are available.

    Notes
    -----
    The median offset is used rather than an unweighted mean to make the shift
    less sensitive to one poor reference-star measurement or a cosmic-ray hit.
    """
    xmaster = np.asarray(xmaster, dtype=float)
    ymaster = np.asarray(ymaster, dtype=float)
    xframe = np.asarray(xframe, dtype=float)
    yframe = np.asarray(yframe, dtype=float)
    ok = (
        np.isfinite(xmaster)
        & np.isfinite(ymaster)
        & np.isfinite(xframe)
        & np.isfinite(yframe)
    )
    xmaster = xmaster[ok]
    ymaster = ymaster[ok]
    xframe = xframe[ok]
    yframe = yframe[ok]
    if not len(xmaster):
        raise hcam.HipercamError("no valid reference-star positions for shift fit")

    dx = float(np.median(xframe - xmaster))
    dy = float(np.median(yframe - ymaster))
    transform = _translation_transform(dx, dy, method="epsf", nstars=len(xmaster))
    xpred, ypred = _apply_transform(transform, xmaster, ymaster)
    transform["x_rms"] = float(np.sqrt(np.mean((xpred - xframe) ** 2)))
    transform["y_rms"] = float(np.sqrt(np.mean((ypred - yframe) ** 2)))
    return transform


def _transform_summary(transform):
    """Return transform fields in output-table order.

    Parameters
    ----------
    transform : dict
        Translation transform created by this module.

    Returns
    -------
    tuple
        ``(method, dx, dy, nstars, x_rms, y_rms)`` for compact insertion into
        forced, scene, or shift diagnostic tables.
    """
    return (
        transform.get("method", "unknown"),
        float(transform.get("dx", 0.0)),
        float(transform.get("dy", 0.0)),
        int(transform.get("nstars", 0)),
        float(transform.get("x_rms", np.nan)),
        float(transform.get("y_rms", np.nan)),
    )


def _centroid_reference_positions(data, stars_tbl, box_size, mask=None, base_transform=None):
    """Measure reference-star positions with center-of-mass centroids.

    Parameters
    ----------
    data : numpy.ndarray
        Background-subtracted image.

    stars_tbl : astropy.table.Table
        Table with master-frame ``x`` and ``y`` positions.

    box_size : int
        Linear size of the search box around each predicted star position.

    mask : numpy.ndarray or None, optional
        Bad-pixel mask.

    base_transform : dict or None, optional
        Starting translation used to predict where each reference star should
        fall in the current frame.

    Returns
    -------
    indices : list of int
        Indices of successfully measured reference stars.

    xframe, yframe : list of float
        Centroid positions in the current frame.

    Notes
    -----
    This is a fallback for automatic shift measurement when ePSF fitting of the
    reference stars fails. It is intentionally simple and therefore should be
    judged by the returned RMS diagnostics.
    """
    if stars_tbl is None:
        return [], [], []

    if base_transform is None:
        base_transform = _translation_transform()

    xguess_all, yguess_all = _apply_transform(
        base_transform, stars_tbl["x"], stars_tbl["y"]
    )
    xframe = []
    yframe = []
    indices = []
    half = max(2, int(box_size) // 2)
    ny, nx = data.shape
    for idx, (xguess, yguess) in enumerate(zip(xguess_all, yguess_all)):
        ix = int(round(float(xguess)))
        iy = int(round(float(yguess)))
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
        xframe.append(xmin + xcen)
        yframe.append(ymin + ycen)
        indices.append(idx)

    return indices, xframe, yframe


def _measure_reference_positions(
    data, stars_tbl, box_size, mask=None, epsf=None, base_transform=None
):
    """Measure frame positions of reference stars for shift estimation.

    Parameters
    ----------
    data : numpy.ndarray
        Background-subtracted science image.

    stars_tbl : astropy.table.Table or None
        Master-frame reference-star table with ``x`` and ``y`` columns.

    box_size : int
        Fit/search-box size in pixels.

    mask : numpy.ndarray or None, optional
        Bad-pixel mask.

    epsf : photutils.psf.ImagePSF or None, optional
        ePSF model used to fit reference-star positions. If this fit fails or
        produces no good positions, the function falls back to centroiding.

    base_transform : dict or None, optional
        Initial master-to-frame translation.

    Returns
    -------
    indices : list of int
        Reference-star indices measured successfully.

    xframe, yframe : list of float
        Measured current-frame positions.

    method : str
        ``"epsf"`` when ePSF fitting succeeded, ``"centroid"`` when the
        fallback centroid path was used, or ``"none"`` if no star table was
        supplied.

    Notes
    -----
    Exceptions raised by the Photutils reference-star fit are intentionally
    swallowed here. Failed shift-star fitting should not crash the entire
    reduction; it should fall back to the simpler centroid method and leave the
    quality assessment to the shift RMS diagnostics.
    """
    if stars_tbl is None:
        return [], [], [], "none"

    if base_transform is None:
        base_transform = _translation_transform()

    if epsf is not None:
        xguess, yguess = _apply_transform(base_transform, stars_tbl["x"], stars_tbl["y"])
        init_params = Table()
        init_params["id"] = np.arange(len(stars_tbl))
        init_params["x"] = xguess
        init_params["y"] = yguess
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
            indices = []
            xframe = []
            yframe = []
            for row in result:
                try:
                    idx = int(row["id"])
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(stars_tbl):
                    xfit = float(row["x_fit"])
                    yfit = float(row["y_fit"])
                    if np.isfinite(xfit) and np.isfinite(yfit):
                        indices.append(idx)
                        xframe.append(xfit)
                        yframe.append(yfit)
            if indices:
                return indices, xframe, yframe, "epsf"

    indices, xframe, yframe = _centroid_reference_positions(
        data, stars_tbl, box_size, mask=mask, base_transform=base_transform
    )
    return indices, xframe, yframe, "centroid"


def _auto_transform(
    data,
    stars_tbl,
    box_size,
    mask=None,
    epsf=None,
    base_transform=None,
):
    """Measure the automatic master-to-frame translation for one image.

    Parameters
    ----------
    data : numpy.ndarray
        Background-subtracted image.

    stars_tbl : astropy.table.Table or None
        Reference-star table in master-frame coordinates.

    box_size : int
        Pixel size used for reference-star fitting or centroiding.

    mask : numpy.ndarray or None, optional
        Bad-pixel mask.

    epsf : photutils.psf.ImagePSF or None, optional
        ePSF used for reference-star position fitting.

    base_transform : dict or None, optional
        Starting transform from a user shift table or identity.

    Returns
    -------
    dict
        Translation transform. If no reference star can be measured, the input
        transform is returned with method ``"initial"`` and zero measured stars.
    """
    if base_transform is None:
        base_transform = _translation_transform()
    if stars_tbl is None:
        return base_transform

    indices, xframe, yframe, method = _measure_reference_positions(
        data,
        stars_tbl,
        box_size,
        mask=mask,
        epsf=epsf,
        base_transform=base_transform,
    )
    if not indices:
        transform = copy.deepcopy(base_transform)
        transform["method"] = "initial"
        transform["nstars"] = 0
        return transform

    xmaster = np.asarray(stars_tbl["x"], dtype=float)[indices]
    ymaster = np.asarray(stars_tbl["y"], dtype=float)[indices]
    transform = _fit_translation_transform(xmaster, ymaster, xframe, yframe)
    transform["method"] = method
    return transform


def _restore_fixed_positions(result, init_params):
    """Restore forced coordinates in Photutils fixed-position output tables.

    Parameters
    ----------
    result : astropy.table.Table
        Photutils result table from `PSFPhotometry`.

    init_params : astropy.table.Table
        Initial source table passed to Photutils. Its ``x`` and ``y`` columns
        are the actual forced coordinates used for the fit.

    Notes
    -----
    Some Photutils fixed-position configurations can leave ``x_fit`` and
    ``y_fit`` reflecting model defaults rather than the supplied forced
    positions. The flux fit is still performed at the requested coordinates, but
    the output table would be misleading. This helper overwrites those columns
    with the forced coordinates for auditability.
    """
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
    """Construct a deterministic per-frame diagnostic filename.

    Parameters
    ----------
    directory : str or path-like
        Destination directory.

    prefix : str
        Filename prefix, for example ``"resid_"`` or ``"frame_"``.

    nframe : int
        One-based frame number in the input file list.

    fname : str
        Input frame filename. Its stem is included in the diagnostic filename.

    suffix : str
        Filename suffix, including extension.

    Returns
    -------
    pathlib.Path
        Path of the form ``directory/prefixNNNNN_inputstem_suffix``.
    """
    root = Path(fname).stem
    return Path(directory) / f"{prefix}{nframe:05d}_{root}{suffix}"


def _write_residual(directory, prefix, nframe, fname, photometry, data, psf_shape):
    """Write a Photutils residual image for one forced-photometry frame.

    Parameters
    ----------
    directory, prefix, nframe, fname
        Components used by `_output_path` to name the output FITS file.

    photometry : photutils.psf.PSFPhotometry
        Photutils object after a fit has been run.

    data : numpy.ndarray
        Image passed to the fit, normally scalar-background-subtracted data.

    psf_shape : tuple of int
        Shape passed to ``make_residual_image``.

    Notes
    -----
    Residual images are one of the most important diagnostics for crowded-field
    work. Structured residuals near the target usually indicate a bad PSF,
    missing neighbour, or incorrect geometry.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    residual = photometry.make_residual_image(data, psf_shape=psf_shape)
    fits.PrimaryHDU(np.asarray(residual, dtype=np.float32)).writeto(
        _output_path(directory, prefix, nframe, fname, ".fits"),
        overwrite=True,
    )


def _build_epsf(args):
    """Command implementation for ``epsfphot build-epsf``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing the input HiPERCAM image, CCD/window labels,
        PSF-star table, output ePSF path, and ePSF-builder controls.

    Side Effects
    ------------
    Writes an ePSF FITS file to ``args.output`` and either writes or prints a
    one-row build-summary table.

    Notes
    -----
    The input frame is treated as the image on which the PSF-star coordinates
    are defined. In the recommended workflow this is usually the high-S/N master
    image rather than an individual science frame.
    """
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
    """Command implementation for ``epsfphot make-source-list``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing a master image, ePSF file, output source
        table path, detection/fitting controls, and optional residual path.

    Side Effects
    ------------
    Writes an ECSV/FITS-style source table to ``args.output``. If requested,
    writes a residual FITS image for the master-frame source detection pass.

    Notes
    -----
    This command uses `IterativePSFPhotometry` to detect and fit sources on the
    master image. For a faint known target that is below the automatic detection
    threshold, the user should add a row manually to the resulting table using
    the target's master-frame coordinates.
    """
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
    """Command implementation for frame-by-frame forced ePSF photometry.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``epsfphot forced`` arguments. The key inputs are a file list,
        CCD/window labels, a master ePSF, a master-frame source table, optional
        shift information, and optional frame-ePSF/rebuild controls.

    Side Effects
    ------------
    Writes a stacked output table to ``args.output``. Optional side effects
    include residual FITS images, per-frame ePSF FITS files, and a measured
    shift table.

    Notes
    -----
    The source table coordinates are assumed to live in the master-frame
    coordinate system. Each science frame receives a translational correction
    from either a user shift table or ePSF/centroid measurements of reference
    stars. Unless ``--free-positions`` is supplied, positions are fixed and the
    fit primarily solves for source fluxes, which is usually the safer choice
    for faint crowded targets.
    """
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

        transform = _initial_transform(shift_table, fname)
        if args.auto_shifts:
            transform = _auto_transform(
                bkgsub,
                auto_shift_stars,
                args.shift_box_size,
                mask=mask,
                epsf=epsf0,
                base_transform=transform,
            )
        (
            shift_method,
            dx,
            dy,
            nshifts,
            shift_xrms,
            shift_yrms,
        ) = _transform_summary(transform)
        shift_rows.append(
            (
                fname,
                shift_method,
                dx,
                dy,
                nshifts,
                shift_xrms,
                shift_yrms,
            )
        )

        init_params = Table()
        init_params["id"] = source_table[args.id_column]
        init_params["x"], init_params["y"] = _apply_transform(
            transform, source_table[args.x_column], source_table[args.y_column]
        )

        epsf_summary = {"epsf_status": "master"}
        if args.rebuild_epsf:
            if frame_epsf_stars is None:
                raise hcam.HipercamError("--rebuild-epsf requires --frame-epsf-stars")
            epsf_stars = _transform_table(frame_epsf_stars, transform)
            try:
                epsf, epsf_summary = _build_epsf_model(bkgsub, epsf_stars, args, mask)
            except hcam.HipercamError as err:
                epsf = copy.deepcopy(epsf0)
                epsf_summary = _epsf_fallback_summary(err)
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
        result["shift_method"] = shift_method
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
            names=(
                "file",
                "method",
                "dx",
                "dy",
                "nstars",
                "x_rms",
                "y_rms",
            ),
        )
        shifts.write(args.write_shifts, overwrite=True)


def _parse_id_set(text, ids):
    """Parse a comma-separated source-ID selection.

    Parameters
    ----------
    text : str or None
        Selection string. ``"all"`` selects every ID, ``"none"`` selects no ID,
        and comma-separated values select explicit IDs.

    ids : iterable
        Valid source IDs. Values are compared as strings so integer table IDs
        and command-line text match naturally.

    Returns
    -------
    set of str
        Selected source IDs.

    Raises
    ------
    hipercam.HipercamError
        If the selection names an ID not present in ``ids``.
    """
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
    """Evaluate a unit-flux ePSF template on a pixel grid.

    Parameters
    ----------
    epsf : photutils.psf.ImagePSF
        Empirical PSF model.

    yy, xx : numpy.ndarray
        Pixel-coordinate grids as returned by `numpy.mgrid`.

    x0, y0 : float
        Source position at which to center the template.

    Returns
    -------
    numpy.ndarray
        Unit-flux PSF values on the supplied grid. Non-finite values are set to
        zero so sparse scene matrices do not inherit NaNs from the PSF model.
    """
    values = epsf.evaluate(xx, yy, 1.0, x0, y0)
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = 0.0
    return values


def _scene_positions(context, source_ids, position_offsets):
    """Compute current-frame source positions for the scene model.

    Parameters
    ----------
    context : dict
        Per-frame context assembled by `_scene`. It contains the master
        coordinates, the frame transform, and diagnostic metadata.

    source_ids : list of str
        Source IDs in the same order as the source table.

    position_offsets : dict
        Optional nonlinear master-frame position offsets keyed by source ID.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)`` current-frame positions after applying selected master-frame
        offsets and the frame translation.
    """
    base_x = np.asarray(context["base_x"], dtype=float).copy()
    base_y = np.asarray(context["base_y"], dtype=float).copy()
    for nsource, sid in enumerate(source_ids):
        if sid in position_offsets:
            dx, dy = position_offsets[sid]
            base_x[nsource] += dx
            base_y[nsource] += dy
    return _apply_transform(context["transform"], base_x, base_y)


def _scene_model(context, source_ids, column_index, constant_ids, solution, position_offsets):
    """Render the scene model for one frame.

    Parameters
    ----------
    context : dict
        Per-frame image, ePSF, coordinate, and ROI information.

    source_ids : list of str
        Source IDs in table order.

    column_index : dict
        Mapping from logical model parameters to columns in the linear
        solution vector.

    constant_ids : list of str
        Source IDs whose flux is shared across all frames.

    solution : numpy.ndarray
        Current vector of flux and optional background parameters.

    position_offsets : dict
        Optional nonlinear master-frame position offsets.

    Returns
    -------
    model : numpy.ndarray
        Model image over the frame ROI.

    xs, ys : numpy.ndarray
        Current-frame source positions used for the model.

    bkg_fit : float
        Scalar background term fitted for this frame, or zero if background
        fitting is disabled.
    """
    iframe = context["frame"] - 1
    bkg_fit = (
        solution[column_index[("background", None, iframe)]]
        if ("background", None, iframe) in column_index
        else 0.0
    )
    model = np.full(context["yy"].shape, bkg_fit, dtype=float)
    xs, ys = _scene_positions(context, source_ids, position_offsets)
    for sid, x0, y0 in zip(source_ids, xs, ys):
        if sid in constant_ids:
            col = column_index[("constant", sid, None)]
        else:
            col = column_index[("variable", sid, iframe)]
        model += solution[col] * _psf_values(
            context["epsf"], context["yy"], context["xx"], x0, y0
        )
    return model, xs, ys, bkg_fit


def _refine_scene_nonlinear(
    args,
    solution,
    contexts,
    source_ids,
    column_index,
    constant_ids,
):
    """Optionally refine selected global source positions with least squares.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed scene arguments. The relevant controls are
        ``refine_position_ids``, ``nl_max_nfev``, and ``nl_loss``.

    solution : numpy.ndarray
        Initial linear least-squares scene solution.

    contexts : list of dict
        Per-frame scene contexts built by `_scene`.

    source_ids : list of str
        Source IDs in table order.

    column_index : dict
        Mapping from scene parameter labels to solution-vector indices.

    constant_ids : list of str
        Sources whose flux is global across the sequence.

    Returns
    -------
    fluxes : numpy.ndarray
        Refined flux/background parameter vector. If no position refinement was
        requested, this is the input ``solution``.

    offsets : dict
        Master-frame ``(dx, dy)`` offsets keyed by refined source ID.

    info : dict
        Diagnostic information from `scipy.optimize.least_squares`, including
        success state, cost, and number of function evaluations.

    Notes
    -----
    This is a constrained refinement stage, not a full scene-modelling engine.
    It adjusts selected master-frame positions and the already-defined
    flux/background parameters while keeping the ePSF choice and frame
    translations fixed.
    """
    from scipy.optimize import least_squares

    refine_ids = _parse_id_set(args.refine_position_ids, source_ids)
    refine_ids = [sid for sid in source_ids if sid in refine_ids]
    if not refine_ids:
        return solution, {}, {"cost": np.nan, "nfev": 0, "success": True}

    nflux = len(solution)
    p0 = np.concatenate([solution, np.zeros(2 * len(refine_ids), dtype=float)])

    def unpack(params):
        """Split the nonlinear parameter vector into fluxes and offsets."""
        offsets = {}
        start = nflux
        for nsource, sid in enumerate(refine_ids):
            offsets[sid] = (
                params[start + 2 * nsource],
                params[start + 2 * nsource + 1],
            )
        return params[:nflux], offsets

    def residuals(params):
        """Return weighted residuals for the nonlinear scene refinement."""
        fluxes, offsets = unpack(params)
        chunks = []
        for context in contexts:
            model, _, _, _ = _scene_model(
                context,
                source_ids,
                column_index,
                constant_ids,
                fluxes,
                offsets,
            )
            valid = context["valid"]
            resid = (model[valid] - context["data"][valid]) / context["err"][valid]
            chunks.append(resid.ravel())
        return np.concatenate(chunks)

    result = least_squares(
        residuals,
        p0,
        max_nfev=args.nl_max_nfev,
        loss=args.nl_loss,
        x_scale="jac",
    )
    fluxes, offsets = unpack(result.x)
    info = {"cost": result.cost, "nfev": result.nfev, "success": result.success}
    return fluxes, offsets, info


def _scene(args):
    """Command implementation for multi-frame scene ePSF photometry.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``epsfphot scene`` arguments. Required inputs are a file list,
        CCD/window labels, an ePSF file, a source table, and an output table.

    Side Effects
    ------------
    Writes a long-format scene-photometry table. Optional outputs include a
    shift table, residual FITS images for each frame ROI, and per-frame ePSF
    FITS files.

    Notes
    -----
    The scene model builds one weighted sparse linear system across all frames.
    Sources named by ``--variable-ids`` get independent fluxes in every frame;
    all other sources have one shared flux across the full sequence. This is a
    useful stabilizer for faint targets blended with neighbours that are
    expected to be constant.
    """
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
        transform = _initial_transform(shift_table, fname)
        if args.auto_shifts:
            transform = _auto_transform(
                bkgsub,
                auto_shift_stars,
                args.shift_box_size,
                mask=mask,
                epsf=epsf0,
                base_transform=transform,
            )
        (
            shift_method,
            dx,
            dy,
            nshifts,
            shift_xrms,
            shift_yrms,
        ) = _transform_summary(transform)

        epsf_summary = {"epsf_status": "master"}
        if args.rebuild_epsf:
            if frame_epsf_stars is None:
                raise hcam.HipercamError("--rebuild-epsf requires --frame-epsf-stars")
            epsf_stars = _transform_table(frame_epsf_stars, transform)
            try:
                epsf, epsf_summary = _build_epsf_model(bkgsub, epsf_stars, args, mask)
            except hcam.HipercamError as err:
                epsf = copy.deepcopy(epsf0)
                epsf_summary = _epsf_fallback_summary(err)
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
        shift_rows.append(
            (
                fname,
                shift_method,
                dx,
                dy,
                nshifts,
                shift_xrms,
                shift_yrms,
            )
        )

        base_x = np.asarray(source_table[args.x_column], dtype=float)
        base_y = np.asarray(source_table[args.y_column], dtype=float)
        xs, ys = _apply_transform(transform, base_x, base_y)
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

        data_roi = bkgsub[ymin:ymax, xmin:xmax]
        rhs_chunks.append((data_roi[valid] / err[valid]).ravel())

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
                "shift_method": shift_method,
                "nshifts": nshifts,
                "shift_xrms": shift_xrms,
                "shift_yrms": shift_yrms,
                "epsf_summary": epsf_summary,
                "bkgsub": bkgsub,
                "data": data_roi,
                "err": err,
                "epsf": epsf,
                "transform": transform,
                "roi": (xmin, xmax, ymin, ymax),
                "valid": valid,
                "yy": yy,
                "xx": xx,
                "base_x": base_x,
                "base_y": base_y,
                "xs": xs,
                "ys": ys,
            }
        )
        row0 += nvalid

    rhs = np.concatenate(rhs_chunks)
    matrix = coo_matrix(
        (values, (row_idx, col_idx)), shape=(len(rhs), len(columns))
    ).tocsr()
    solution = lsqr(
        matrix,
        rhs,
        atol=args.lsqr_tol,
        btol=args.lsqr_tol,
        iter_lim=args.lsqr_iter,
    )[0]

    position_offsets = {}
    nonlinear_info = {"cost": np.nan, "nfev": 0, "success": False}
    if args.nonlinear_scene:
        solution, position_offsets, nonlinear_info = _refine_scene_nonlinear(
            args,
            solution,
            contexts,
            source_ids,
            column_index,
            constant_ids,
        )

    rows = []
    for iframe, context in enumerate(contexts):
        model, xs, ys, bkg_fit = _scene_model(
            context,
            source_ids,
            column_index,
            constant_ids,
            solution,
            position_offsets,
        )
        for nsource, (sid, x0, y0) in enumerate(zip(source_ids, xs, ys)):
            if sid in constant_ids:
                col = column_index[("constant", sid, None)]
                flux_type = "constant"
            else:
                col = column_index[("variable", sid, iframe)]
                flux_type = "variable"
            flux = solution[col]
            pos_dx, pos_dy = position_offsets.get(sid, (0.0, 0.0))
            master_x = context["base_x"][nsource] + pos_dx
            master_y = context["base_y"][nsource] + pos_dy
            rows.append(
                (
                    context["frame"],
                    context["file"],
                    context["mjdutc"],
                    sid,
                    x0,
                    y0,
                    master_x,
                    master_y,
                    pos_dx,
                    pos_dy,
                    flux,
                    flux_type,
                    context["dx"],
                    context["dy"],
                    context["shift_method"],
                    context["nshifts"],
                    context["shift_xrms"],
                    context["shift_yrms"],
                    context["epsf_summary"].get("epsf_status", "unknown"),
                    context["epsf_summary"].get("epsf_n_input_stars", 0),
                    context["epsf_summary"].get("epsf_n_used_stars", 0),
                    context["epsf_summary"].get("epsf_warning", ""),
                    context["global_bkg"],
                    bkg_fit,
                    bool(args.nonlinear_scene),
                    bool(nonlinear_info["success"]),
                    float(nonlinear_info["cost"]),
                    int(nonlinear_info["nfev"]),
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
            "x_fit",
            "y_fit",
            "master_x_fit",
            "master_y_fit",
            "master_dx_fit",
            "master_dy_fit",
            "flux_fit",
            "flux_type",
            "shift_dx",
            "shift_dy",
            "shift_method",
            "shift_nstars",
            "shift_xrms",
            "shift_yrms",
            "epsf_status",
            "epsf_n_input_stars",
            "epsf_n_used_stars",
            "epsf_warning",
            "global_bkg",
            "scene_bkg_fit",
            "nonlinear_scene",
            "nonlinear_success",
            "nonlinear_cost",
            "nonlinear_nfev",
        ),
    )
    output.write(args.output, overwrite=True)
    if args.write_shifts:
        shifts = Table(
            rows=shift_rows,
            names=(
                "file",
                "method",
                "dx",
                "dy",
                "nstars",
                "x_rms",
                "y_rms",
            ),
        )
        shifts.write(args.write_shifts, overwrite=True)


def _add_common_image_args(parser):
    """Add image-selection arguments shared by image-based subcommands.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Subparser to mutate.

    Notes
    -----
    These options identify one HiPERCAM window and an optional bad-pixel mask.
    They are used by commands that operate on a single image, such as
    ``build-epsf`` and ``make-source-list``.
    """
    parser.add_argument("hcm", help="input .hcm image")
    parser.add_argument("ccd", help="CCD label")
    parser.add_argument("window", help="window label")
    parser.add_argument(
        "--mask",
        help="optional HiPERCAM or FITS mask; non-zero pixels are ignored",
    )


def _add_photometry_args(parser):
    """Add common Photutils detection and PSF-fitting arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Subparser to mutate.

    Notes
    -----
    These controls apply mainly to source detection and forced Photutils fits:
    fit stamp size, approximate FWHM, detector noise model, detection threshold,
    source grouping scale, aperture radius for initial flux estimates, optional
    local-background annuli, and progress-bar display.
    """
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
    """Add frame-sequence arguments shared by forced and scene modes.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Subparser to mutate.

    Notes
    -----
    These options describe how master-frame coordinates are mapped to each
    science frame, whether a fresh ePSF should be attempted per frame, and which
    diagnostic products should be written. The registration model is a
    translation only; higher-order transforms are intentionally not exposed.
    """
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
    parser.add_argument("--epsf-min-stars", type=int, default=3)
    parser.add_argument("--epsf-no-smoothing", action="store_true")
    parser.add_argument("--residual-dir", help="optional directory for residual FITS files")
    parser.add_argument("--residual-prefix", default="resid_")
    parser.add_argument("--write-shifts", help="optional output table of measured shifts")
    parser.add_argument("--frame-epsf-dir", help="optional directory for per-frame ePSFs")
    parser.add_argument("--frame-epsf-prefix", default="frame_")


def _parser():
    """Build the ``epsfphot`` command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``build-epsf``, ``make-source-list``, ``forced``, and
        ``scene`` subcommands registered.

    Notes
    -----
    Keeping parser construction in one function allows tests and the ``reduce``
    handoff to call `epsfphot` with an explicit argument list, without going
    through a shell command.
    """
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
    build.add_argument("--min-stars", type=int, default=3)
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
    scene.add_argument(
        "--nonlinear-scene",
        action="store_true",
        help="jointly refine selected master-frame source positions and fluxes",
    )
    scene.add_argument(
        "--refine-position-ids",
        default="none",
        help="comma-separated source IDs with global positions refined; use 'all' or 'none'",
    )
    scene.add_argument("--nl-max-nfev", type=int, default=100)
    scene.add_argument(
        "--nl-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="linear",
    )
    scene.set_defaults(func=_scene)

    return parser


def epsfphot(args=None):
    """Entry point for the experimental ePSF photometry command.

    Parameters
    ----------
    args : list of str or None, optional
        Command-line arguments excluding the program name. If `None`, arguments
        are read from ``sys.argv`` through `argparse`.

    Side Effects
    ------------
    Dispatches to one of the subcommand implementations and writes the
    requested ECSV/FITS products.

    Notes
    -----
    This function is intentionally small so it can be reused by
    ``hipercam.scripts.reduce`` as an internal handoff target. All scientific
    behaviour lives in the subcommand implementation functions above.
    """
    parsed = _parser().parse_args(args)
    parsed.func(parsed)


if __name__ == "__main__":
    epsfphot()
