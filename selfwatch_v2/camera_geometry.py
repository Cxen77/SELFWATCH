"""CPU-only calibration and conservative floor evidence for a fixed camera rig."""
from dataclasses import dataclass
from itertools import combinations, product
import math

import cv2
import numpy as np


_PIXEL_PROBES = np.array(list(product((-1, 0, 1), repeat=2)))


def _points(values, minimum):
    points = np.asarray(values, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum
            or not np.isfinite(points).all()):
        raise ValueError("Expected finite XY points")
    return points


def _hull(points):
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) <= 1e-8:
        raise ValueError("Calibration points must span an area")
    return hull


def _distance(polygon, xy):
    return cv2.pointPolygonTest(polygon, tuple(map(float, xy)), True)


def _positive(*values):
    return all(math.isfinite(value) and value > 0 for value in values)


@dataclass(frozen=True)
class ErrorStats:
    count: int
    median: float
    p95: float
    maximum: float

    @classmethod
    def measure(cls, residuals):
        return cls(len(residuals), float(np.median(residuals)),
                   float(np.percentile(residuals, 95)), float(np.max(residuals)))


@dataclass(frozen=True)
class FloorContact:
    xy_m: tuple[float, float]
    radius_m: float

    def __post_init__(self):
        if (len(self.xy_m) != 2 or not all(map(math.isfinite, self.xy_m))
                or not math.isfinite(self.radius_m) or self.radius_m < 0):
            raise ValueError("Invalid floor contact")


class FloorMap:
    """Project visible floor contacts with a matching optional pinhole model.

    Error radii are operational estimates whose coverage depends on the
    distribution of held-out targets. Keep the footprint on one flat surface.
    """
    def __init__(self, size, floor_from_image, footprint_m, residual_m,
                 *, pinhole=None, distortion=None):
        self.width, self.height = size
        self.matrix = np.array(floor_from_image, dtype=float, copy=True)
        if (self.width < 2 or self.height < 2 or self.matrix.shape != (3, 3)
                or not np.isfinite(self.matrix).all() or np.linalg.cond(self.matrix) > 1e10
                or not math.isfinite(residual_m) or residual_m < 0):
            raise ValueError("Invalid floor calibration")
        self.matrix /= np.max(np.abs(self.matrix))
        self.footprint = _hull(_points(footprint_m, 3))
        self.residual_m = residual_m
        self.pinhole = None if pinhole is None else np.array(pinhole, dtype=float, copy=True)
        self.distortion = None if distortion is None else np.array(distortion, dtype=float, copy=True).ravel()
        if (self.pinhole is None) != (self.distortion is None):
            raise ValueError("Pinhole and distortion must be supplied together")
        if self.pinhole is not None and (
            self.pinhole.shape != (3, 3) or not np.isfinite(self.pinhole).all()
            or min(self.pinhole[0, 0], self.pinhole[1, 1]) <= 0
            or self.distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(self.distortion).all()
        ):
            raise ValueError("Invalid pinhole model")
        image_from_floor = np.linalg.inv(self.matrix)
        scale = np.column_stack((self.footprint, np.ones(len(self.footprint)))) @ image_from_floor[2]
        if not (np.all(scale > 1e-9) or np.all(scale < -1e-9)):
            raise ValueError("Floor footprint crosses the camera horizon")
        self._side = 1 if scale[0] > 0 else -1

    def _undistort(self, pixels):
        if self.pinhole is None:
            return pixels
        return cv2.undistortPoints(pixels.reshape(-1, 1, 2), self.pinhole,
                                   self.distortion, P=self.pinhole).reshape(-1, 2)

    def _inside_image(self, pixels):
        return np.isfinite(pixels).all() and ((0 <= pixels) & (
            pixels <= [self.width - 1, self.height - 1])).all()

    def _project(self, pixels):
        undistorted = self._undistort(pixels)
        homogeneous = np.column_stack((undistorted, np.ones(len(pixels)))) @ self.matrix.T
        if not np.isfinite(homogeneous).all() or np.any(self._side * homogeneous[:, 2] <= 1e-9):
            return None
        return homogeneous[:, :2] / homogeneous[:, 2, None]

    @classmethod
    def fit(cls, size, fit_px, fit_m, check_px, check_m, *, max_error_m,
            pinhole=None, distortion=None):
        """Fit on surveyed targets and require distinct held-out targets."""
        fit_px, fit_m = _points(fit_px, 4), _points(fit_m, 4)
        check_px, check_m = _points(check_px, 3), _points(check_m, 3)
        if (len(fit_px) != len(fit_m) or len(check_px) != len(check_m)
                or not _positive(max_error_m)):
            raise ValueError("Invalid surveyed target pairs or tolerance")
        for fit, check in ((fit_px, check_px), (fit_m, check_m)):
            combined = np.vstack((fit, check))
            if len(np.unique(combined, axis=0)) != len(combined):
                raise ValueError("Validation targets must be distinct from fit targets")
        hull = _hull(fit_m)
        if any(_distance(hull, point) < 0 for point in check_m):
            raise ValueError("Validation targets must be within the surveyed footprint")
        base = cls(size, np.eye(3), hull, 0, pinhole=pinhole, distortion=distortion)
        if not all(base._inside_image(points) for points in (fit_px, check_px)):
            raise ValueError("Survey pixels fall outside the calibrated frame")
        corrected = base._undistort(fit_px)
        _hull(corrected)
        matrix, _ = cv2.findHomography(corrected, fit_m)
        if matrix is None:
            raise ValueError("Survey targets do not determine a homography")
        floor = cls(size, matrix, hull, 0, pinhole=pinhole, distortion=distortion)
        projected = floor._project(check_px)
        if projected is None:
            raise ValueError("Could not project held-out targets")
        stats = ErrorStats.measure(np.linalg.norm(projected - check_m, axis=1))
        if stats.maximum > max_error_m:
            raise ValueError(f"Held-out floor error {stats.maximum:.3f} m exceeds tolerance")
        floor.residual_m = stats.maximum
        return floor, stats

    def contact(self, pixel_xy, pixel_error, size):
        if tuple(size) != (self.width, self.height) or not _positive(pixel_error):
            return None
        xy = np.asarray(pixel_xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return None
        probes = xy + pixel_error * _PIXEL_PROBES
        if not self._inside_image(probes):
            return None
        projected = self._project(probes)
        if projected is None:
            return None
        center = projected[4]
        radius = self.residual_m + float(np.max(np.linalg.norm(projected - center, axis=1)))
        if _distance(self.footprint, center) <= radius:
            return None
        return FloorContact(tuple(map(float, center)), radius)


@dataclass(frozen=True)
class AlignedTime:
    time_s: float
    uncertainty_s: float

    def __post_init__(self):
        if not math.isfinite(self.time_s) or not _positive(self.uncertainty_s):
            raise ValueError("Invalid aligned time")


@dataclass(frozen=True)
class CameraClock:
    """Affine capture-time alignment in one stream epoch."""
    raw_origin_s: float
    rig_origin_s: float
    rate: float
    valid_from_s: float
    valid_to_s: float
    uncertainty_s: float

    def __post_init__(self):
        if (not all(math.isfinite(value) for value in (
                self.raw_origin_s, self.rig_origin_s, self.valid_from_s, self.valid_to_s))
                or self.valid_from_s >= self.valid_to_s
                or not _positive(self.rate, self.uncertainty_s)):
            raise ValueError("Invalid clock model")

    @classmethod
    def fit(cls, fit_events, check_events, *, quantization_s, max_error_s):
        """Fit (camera seconds, rig seconds) pairs from shared physical events."""
        fit, check = _points(fit_events, 3), _points(check_events, 2)
        if not _positive(quantization_s, max_error_s):
            raise ValueError("Positive timing tolerances required")
        events = np.vstack((fit, check))
        events = events[np.argsort(events[:, 0])]
        if np.any(np.diff(events, axis=0) <= 0):
            raise ValueError("Sync events must be distinct and time-ordered")
        lower, upper = float(np.min(fit[:, 0])), float(np.max(fit[:, 0]))
        if np.min(check[:, 0]) < lower or np.max(check[:, 0]) > upper:
            raise ValueError("Held-out sync events must be within the measured interval")
        raw_origin, rig_origin = map(float, fit.mean(axis=0))
        raw_delta, rig_delta = (fit - [raw_origin, rig_origin]).T
        rate = float((raw_delta @ rig_delta) / (raw_delta @ raw_delta))
        residuals = np.abs(rig_origin + (check[:, 0] - raw_origin) * rate - check[:, 1])
        stats = ErrorStats.measure(residuals)
        bound = stats.maximum + quantization_s
        if bound > max_error_s:
            raise ValueError(f"Held-out clock uncertainty {bound:.3f} s exceeds tolerance")
        return cls(raw_origin, rig_origin, rate, lower, upper, bound), stats

    def align(self, raw_s):
        if not self.valid_from_s <= raw_s <= self.valid_to_s:
            return None
        aligned = self.rig_origin_s + self.rate * (raw_s - self.raw_origin_s)
        return AlignedTime(aligned, self.uncertainty_s) if math.isfinite(aligned) else None


@dataclass(frozen=True)
class FloorObservation:
    camera_id: str
    stream_epoch: str
    floor_id: str
    time: AlignedTime
    contact: FloorContact


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    stream_epoch: str
    floor_id: str
    floor: FloorMap
    clock: CameraClock

    def observe(self, frame, foot_px, pixel_error):
        """Accept a visible ground contact on a frame from this calibrated epoch."""
        if (frame.camera_id, frame.stream_epoch) != (self.camera_id, self.stream_epoch):
            return None
        time = self.clock.align(frame.captured_at_s)
        if time is None:
            return None
        contact = self.floor.contact(foot_px, pixel_error, (frame.width, frame.height))
        if contact is None:
            return None
        return FloorObservation(self.camera_id, self.stream_epoch, self.floor_id, time, contact)


def simultaneous(a, b, tolerance_s):
    return (a is not None and b is not None and a.floor_id == b.floor_id
            and a.camera_id != b.camera_id and _positive(tolerance_s)
            and abs(a.time.time_s - b.time.time_s)
            + a.time.uncertainty_s + b.time.uncertainty_s <= tolerance_s)


class Zone:
    def __init__(self, floor_id, kind, boundary_m):
        self.boundary = _points(boundary_m, 3).astype(np.float32)
        if kind not in {"aisle", "shelf", "checkout"} or cv2.contourArea(self.boundary) <= 1e-8:
            raise ValueError("Invalid zone kind or boundary")
        self.floor_id = floor_id
        self.kind = kind

    def contains(self, observation):
        return (observation is not None and observation.floor_id == self.floor_id
                and _distance(self.boundary, observation.contact.xy_m)
                > observation.contact.radius_m)


class ExitGate:
    """Directed finite segment. Outbound moves from the right to the left side."""
    def __init__(self, floor_id, start_m, end_m, *, hysteresis_m, max_gap_s, max_speed_m_s):
        self.start, end = _points([start_m, end_m], 2)
        delta = end - self.start
        self.length = float(np.linalg.norm(delta))
        if not _positive(self.length, hysteresis_m, max_gap_s, max_speed_m_s):
            raise ValueError("Invalid gate geometry or tolerances")
        self.tangent = delta / self.length
        self.normal = np.array([-self.tangent[1], self.tangent[0]])
        self.floor_id = floor_id
        self.hysteresis_m = hysteresis_m
        self.max_gap_s = max_gap_s
        self.max_speed_m_s = max_speed_m_s

    def crossed_outbound(self, before, after):
        """Check consecutive observed contacts belonging to one track."""
        if before is None or after is None:
            return False
        if (before.floor_id != self.floor_id or after.floor_id != self.floor_id
                or (before.camera_id == after.camera_id and before.stream_epoch != after.stream_epoch)):
            return False
        elapsed = after.time.time_s - before.time.time_s
        timing_error = after.time.uncertainty_s + before.time.uncertainty_s
        if elapsed <= timing_error or elapsed + timing_error > self.max_gap_s:
            return False
        start, end = (np.asarray(obs.contact.xy_m) - self.start for obs in (before, after))
        u, v = before.contact.radius_m, after.contact.radius_m
        if np.linalg.norm(end - start) + u + v > self.max_speed_m_s * (elapsed - timing_error):
            return False
        if (start @ self.normal >= -u - self.hysteresis_m
                or end @ self.normal <= v + self.hysteresis_m):
            return False
        # The uncertainty disks must both fit within the strip of the endpoints.
        return all(radius < point @ self.tangent < self.length - radius
                   for point, radius in ((start, u), (end, v)))


class CameraRig:
    """One floor revision and one commissioned epoch per available camera."""
    def __init__(self, floor_id, cameras):
        self.floor_id = floor_id
        self.cameras = {camera.camera_id: camera for camera in cameras}
        if (not floor_id or not self.cameras or len(self.cameras) != len(cameras)
                or any(camera.floor_id != floor_id for camera in cameras)):
            raise ValueError("Expected unique cameras on the same floor revision")
        self.overlaps = {}
        for a, b in combinations(sorted(self.cameras), 2):
            area, polygon = cv2.intersectConvexConvex(self.cameras[a].floor.footprint,
                                                    self.cameras[b].floor.footprint)
            if area > 1e-8:
                self.overlaps[a, b] = polygon.reshape(-1, 2)

    def observe(self, frame, foot_px, pixel_error):
        camera = self.cameras.get(frame.camera_id)
        return camera.observe(frame, foot_px, pixel_error) if camera else None

    def shares_coverage(self, a, b, tolerance_s):
        """Check geometry and time support for a simultaneous pair."""
        if not simultaneous(a, b, tolerance_s) or a.floor_id != self.floor_id:
            return False
        overlap = self.overlaps.get(tuple(sorted((a.camera_id, b.camera_id))))
        return overlap is not None and all(
            self.cameras[obs.camera_id].stream_epoch == obs.stream_epoch
            and _distance(overlap, obs.contact.xy_m) > obs.contact.radius_m
            for obs in (a, b)
        )
