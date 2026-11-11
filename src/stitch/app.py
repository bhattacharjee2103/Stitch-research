"""Mac desktop capture, reference registration and live rescan interface."""
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import yaml

from .database import PlaceDatabase
from .depth import DEFAULT_DEPTH_MODEL, MonocularDepthEstimator, colorize_depth
from .features import LocalEngine
from .pipeline import Pipeline, live_config, resize_for_scan
from .ui_dialogs import confirm_quad, select_box
from .multiview import coverage, load_primary_view, match_views, place_views, relate_view
from .pose3d import camera_matrix, crop_camera_matrix, oriented_box_from_roi, project_points

ROOT = Path(__file__).resolve().parents[2]


def camera_devices():
    """Return camera names reported by macOS AVFoundation via ffmpeg, if installed."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return []
    try:
        result = subprocess.run([ffmpeg, '-hide_banner', '-f', 'avfoundation',
                                 '-list_devices', 'true', '-i', ''],
                                capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return []
    devices = []
    for line in (result.stdout + result.stderr).splitlines():
        if 'AVFoundation audio devices' in line:
            break
        match = re.search(r'\[(\d+)\]\s+(.+?)\s*$', line)
        if match and 'AVFoundation' not in match.group(2):
            devices.append((match.group(1), match.group(2)))
    return devices


class StitchApp:
    def __init__(self, root):
        self.root = root
        root.title('Stitch · Reference Register & Rescan')
        root.geometry('1280x850')
        root.minsize(1000, 700)
        self.database_path = ROOT / 'data' / 'places.json'
        self.config = yaml.safe_load((ROOT / 'configs' / 'default.yaml').read_text())
        self.capture = None
        self.capture_thread = None
        self.capture_running = False
        self.camera_opened_at = None
        self.camera_warning_shown = False
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.reference_crop = None
        self.reference_box = None
        self.reference_frame_size = None
        self.reference_source = ''
        self.reference_file_path = ''
        self.reference_captured_at = None
        self.reference_is_photo = False
        self.reference_sift_points = np.empty((0, 2), dtype=np.float32)
        self.reference_sift_count = None
        self.registration_preview = None
        self.sift_engine = None
        self.reference_depth_map = None
        self.reference_anchor_3d = None
        self.reference_camera_matrix = None
        self.reference_fov_x_deg = None
        self.depth_estimator = None
        self.depth_model_id_loaded = None
        self.depth_results = Queue()
        self.depth_job_generation = 0
        self.overlay_path = tk.StringVar()
        self.source = tk.StringVar(value='0')
        self.status = tk.StringVar(value='Connect the Mac camera or choose a camera source.')
        self.selection_status = tk.StringVar(value='No reference region selected')
        self.scan_status = tk.StringVar(value='Scanning is idle.')
        self.place_id = tk.StringVar()
        self.place_name = tk.StringVar()
        self.latitude = tk.StringVar()
        self.longitude = tk.StringVar()
        self.scan_latitude = tk.StringVar()
        self.scan_longitude = tk.StringVar()
        self.method = tk.StringVar(value='classical')
        self.depth_model_id = tk.StringVar(value=str(self.config.get('depth_model_id', DEFAULT_DEPTH_MODEL)))
        self.camera_fov_x_deg = tk.StringVar(value=str(self.config.get('camera_fov_x_deg', 70.0)))
        self.capture_sweep_deg = tk.StringVar(value=str(self.config.get('capture_angle_sweep_deg', 30)))
        self.capture_yaw_deg = tk.StringVar(value='0')
        self.view_tolerance_deg = tk.StringVar(value=str(self.config.get('view_tolerance_deg', 15)))
        self.capture_yaw_source = 'primary'
        self.reference_view_relation = None
        self.relation_engine = None
        self.overlay_yaw_deg = tk.StringVar(value='0')
        self.overlay_pitch_deg = tk.StringVar(value='0')
        self.overlay_roll_deg = tk.StringVar(value='0')
        self.capture_angle_guidance = tk.StringVar()
        self.depth_placement_status = tk.StringVar(value='No depth map or 3D placement selected.')
        self.scan_enabled = False
        self.pipeline = None
        self.scan_busy = False
        self.scan_generation = 0
        self.scan_result_lock = threading.Lock()
        self.scan_result = None
        self.last_rendered = None
        self.photo_reference = None
        self.camera_names = {}

        self._build_ui()
        self.refresh_cameras()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(30, self._tick)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill='x')
        ttk.Label(top, text='Camera source').pack(side='left')
        self.camera_box = ttk.Combobox(top, textvariable=self.source, width=32)
        self.camera_box.pack(side='left', padx=(8, 6))
        ttk.Button(top, text='Refresh cameras', command=self.refresh_cameras).pack(side='left')
        ttk.Button(top, text='Connect', command=self.connect_camera).pack(side='left', padx=6)
        ttk.Button(top, text='Disconnect', command=self.disconnect_camera).pack(side='left')

        body = ttk.Panedwindow(self.root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=12, pady=(0, 8))
        video_box = ttk.Frame(body, padding=6)
        controls = ttk.Frame(body, padding=8)
        body.add(video_box, weight=3)
        body.add(controls, weight=2)
        ttk.Label(video_box, text='Live camera view').pack(anchor='w')
        self.video_label = ttk.Label(video_box, text='Camera preview appears here', anchor='center')
        self.video_label.pack(fill='both', expand=True, pady=(8, 0))

        tabs = ttk.Notebook(controls)
        tabs.pack(fill='both', expand=True)
        register_tab = ttk.Frame(tabs, padding=12)
        scan_tab = ttk.Frame(tabs, padding=12)
        placement_tab = ttk.Frame(tabs, padding=12)
        tabs.add(register_tab, text='1 · Register')
        tabs.add(scan_tab, text='2 · Rescan')
        tabs.add(placement_tab, text='3 · 3D placement')
        self._build_register_tab(register_tab)
        self._build_scan_tab(scan_tab)
        self._build_placement_tab(placement_tab)
        ttk.Label(self.root, textvariable=self.status, relief='sunken', anchor='w', padding=6).pack(fill='x')

    def _field(self, parent, label, variable, row, width=32):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=5)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky='ew', pady=5)
        return entry

    def _update_capture_angle_guidance(self):
        try:
            sweep = float(self.capture_sweep_deg.get())
            tolerance = float(self.view_tolerance_deg.get())
            fov = float(self.camera_fov_x_deg.get())
            if not (math.isfinite(sweep) and 1 <= sweep <= 120 and math.isfinite(tolerance)
                    and 1 <= tolerance <= 60 and math.isfinite(fov) and 1.0 < fov < 179.0):
                raise ValueError
        except (TypeError, ValueError):
            self.capture_angle_guidance.set(
                'Enter a sweep of 1–120°, a per-view tolerance of 1–60° and a camera FOV of 1–179°.')
            return
        place_id = self.place_id.get().strip()
        captured = []
        if place_id and self.database_path.exists():
            try:
                captured = [v['yaw_deg'] for v in place_views(
                    PlaceDatabase(str(self.database_path), allow_empty=True), place_id)]
            except Exception:
                captured = []
        report = coverage(captured, sweep, tolerance)
        fmt = lambda values: ', '.join('0°' if round(v) == 0 else f'{v:+.0f}°' for v in values) or 'none'
        text = (f'Capture plan for a {sweep:g}° rescan range with ±{tolerance:g}° per view: '
                f'views at {fmt(report["plan"])}.')
        if not captured:
            text += (' Start with the first view (0°) and draw the floating box once; '
                     'later views reuse that box and their angles are measured automatically.')
        else:
            text += f' Registered for {place_id}: {fmt(report["captured"])}.'
            if report['complete']:
                text += ' The whole range is covered.'
            else:
                gaps = ', '.join(f'{a:+.0f}° to {b:+.0f}°' for a, b in report['missing'])
                text += f' Not yet covered: {gaps}. Next suggested view: {fmt(report["pending"][:1])}.'
        try:
            current = float(self.capture_yaw_deg.get())
            if math.isfinite(current) and captured:
                text += f' This view: {current:+.1f}° ({self.capture_yaw_source}).'
        except (TypeError, ValueError):
            pass
        text += f' Use one camera throughout (FOV {fov:g}°).'
        self.capture_angle_guidance.set(text)

    def _build_register_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text='Place identity and capture location', font=('', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))
        self._field(parent, 'Place ID', self.place_id, 1)
        self._field(parent, 'Place name', self.place_name, 2)
        self._field(parent, 'Latitude', self.latitude, 3)
        self._field(parent, 'Longitude', self.longitude, 4)
        ttk.Label(parent, text='Enter the registered place latitude and longitude.').grid(
            row=5, column=0, columnspan=2, sticky='w', pady=(0, 10))
        ttk.Separator(parent).grid(row=6, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Label(parent, text='Select a textured scene region to recognize', font=('', 12, 'bold')).grid(
            row=7, column=0, columnspan=2, sticky='w', pady=(5, 8))
        ttk.Button(parent, text='Capture live frame and draw box',
                   command=self.select_live_reference).grid(row=8, column=0, columnspan=2, sticky='ew', pady=4)
        ttk.Button(parent, text='Load photo and draw box',
                   command=self.select_photo_reference).grid(row=9, column=0, columnspan=2, sticky='ew', pady=4)
        ttk.Label(parent, textvariable=self.selection_status, wraplength=350).grid(
            row=10, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Button(parent, text='Choose overlay image…', command=self.choose_overlay).grid(
            row=11, column=0, sticky='ew', pady=5)
        ttk.Label(parent, textvariable=self.overlay_path, wraplength=250).grid(
            row=11, column=1, sticky='w', padx=5)
        ttk.Button(parent, text='Register this reference and overlay',
                   command=self.register_reference).grid(row=12, column=0, columnspan=2,
                                                           sticky='ew', pady=(14, 4))
        ttk.Label(parent, text='A later capture can use the same Place ID to add another view.',
                  wraplength=350).grid(row=13, column=0, columnspan=2, sticky='w', pady=6)
        ttk.Button(parent, text='Clear selected region', command=self.clear_reference).grid(
            row=14, column=0, columnspan=2, sticky='ew', pady=4)

    def _build_scan_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text='Recognize and render registered content', font=('', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))
        ttk.Label(parent, text='Matching method').grid(row=1, column=0, sticky='w', pady=5)
        ttk.Combobox(parent, textvariable=self.method, state='readonly',
                     values=('classical', 'local', 'full', 'depth3d')).grid(row=1, column=1, sticky='ew', pady=5)
        self._field(parent, 'Scan latitude (optional)', self.scan_latitude, 2)
        self._field(parent, 'Scan longitude (optional)', self.scan_longitude, 3)
        ttk.Button(parent, text='Start rescan', command=self.start_scan).grid(
            row=4, column=0, columnspan=2, sticky='ew', pady=(12, 4))
        ttk.Button(parent, text='Stop rescan', command=self.stop_scan).grid(
            row=5, column=0, columnspan=2, sticky='ew', pady=4)
        ttk.Label(parent, textvariable=self.scan_status, wraplength=350).grid(
            row=6, column=0, columnspan=2, sticky='w', pady=10)
        ttk.Label(parent, text='An overlay appears after image matching and geometric verification.').grid(
            row=7, column=0, columnspan=2, sticky='w', pady=5)

    def _build_placement_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text='Depth-backed placement', font=('', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))
        ttk.Label(parent, text=(
            'First select the reference scene in Register. Then create a metric depth map '
            'and draw a second box where the overlay should float. For extra views of the '
            'same Place ID the box is transferred from the first view automatically.'),
            wraplength=360).grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        self._field(parent, 'Outdoor depth model', self.depth_model_id, 2)
        self._field(parent, 'Horizontal camera FOV (°)', self.camera_fov_x_deg, 3)
        self._field(parent, 'Rescan range (° total)', self.capture_sweep_deg, 4)
        self._field(parent, 'Per-view tolerance (°)', self.view_tolerance_deg, 5)
        self._field(parent, 'View angle from first view (°)', self.capture_yaw_deg, 6)
        ttk.Label(parent, textvariable=self.capture_angle_guidance, wraplength=360).grid(
            row=7, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Label(parent, text='3D overlay plane rotation (degrees)', font=('', 10, 'bold')).grid(
            row=8, column=0, columnspan=2, sticky='w', pady=(10, 2))
        self._field(parent, 'Yaw', self.overlay_yaw_deg, 9)
        self._field(parent, 'Pitch', self.overlay_pitch_deg, 10)
        self._field(parent, 'Roll', self.overlay_roll_deg, 11)
        ttk.Button(parent, text='Generate depth map and select 3D overlay box',
                   command=self.create_3d_placement).grid(
            row=12, column=0, columnspan=2, sticky='ew', pady=(12, 4))
        ttk.Label(parent, textvariable=self.depth_placement_status, wraplength=360).grid(
            row=13, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Label(parent, text=(
            'The view angle is measured automatically for additional views. Camera FOV must be '
            'calibrated for reliable metric placement.'),
            wraplength=360).grid(row=14, column=0, columnspan=2, sticky='w', pady=(8, 4))
        for variable in (self.capture_sweep_deg, self.capture_yaw_deg, self.camera_fov_x_deg,
                         self.view_tolerance_deg, self.place_id):
            variable.trace_add('write', lambda *_: self._update_capture_angle_guidance())
        self._update_capture_angle_guidance()

    def refresh_cameras(self):
        devices = camera_devices()
        self.camera_names = {index: name for index, name in devices}
        values = [f'{index} — {name}' for index, name in devices]
        values.extend(str(i) for i in range(6))
        self.camera_box['values'] = list(dict.fromkeys(values))
        if devices and self.source.get() == '0':
            self.source.set(f'{devices[0][0]} — {devices[0][1]}')
        if not devices:
            self.status.set('No AVFoundation camera listed yet. Check Mac camera permissions, then Refresh cameras; you can also try camera ID 0–5 or paste a stream URL.')
        else:
            self.status.set('Camera devices found. Choose the Mac camera, then Connect.')

    def _source_value(self):
        value = self.source.get().strip()
        if ' — ' in value:
            value = value.split(' — ', 1)[0]
        return int(value) if value.isdigit() else value

    def connect_camera(self):
        self.disconnect_camera()
        source = self._source_value()
        backend = cv2.CAP_AVFOUNDATION if isinstance(source, int) and hasattr(cv2, 'CAP_AVFOUNDATION') else cv2.CAP_ANY
        capture = cv2.VideoCapture(source, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            self.status.set('Could not open that camera. Check macOS Camera privacy permission, then try another ID or stream URL.')
            return
        self.capture = capture
        self.capture_running = True
        self.camera_opened_at = time.monotonic()
        self.camera_warning_shown = False
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.status.set('Camera connected; wait for the live preview.')

    def _capture_loop(self):
        while self.capture_running and self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(.03)

    def disconnect_camera(self):
        self.capture_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.stop_scan()

    def _current_frame(self):
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def select_live_reference(self):
        frame = self._current_frame()
        if frame is None:
            messagebox.showinfo('Camera not ready', 'Connect the Mac camera and wait for its preview first.')
            return
        self._select_box(frame, 'live camera')

    def select_photo_reference(self):
        path = filedialog.askopenfilename(title='Choose a reference photo',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.heic *.tif *.tiff'), ('All files', '*')])
        if not path:
            return
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg') as converted:
                    conversion = subprocess.run(['sips', '-s', 'format', 'jpeg', path,
                                                 '--out', converted.name],
                                                capture_output=True, timeout=15)
                    if conversion.returncode == 0:
                        frame = cv2.imread(converted.name, cv2.IMREAD_COLOR)
            except (OSError, subprocess.TimeoutExpired):
                frame = None
            if frame is None:
                messagebox.showerror('Cannot read photo', 'Could not decode this photo. Export it as JPEG from Photos and try again.')
                return
        self._select_box(frame, 'photo', path)

    def _select_box(self, frame, source_description, source_path=''):
        try:
            fov = self._number(self.camera_fov_x_deg.get(), 'Horizontal camera FOV', 1.01, 178.99)
        except ValueError as exc:
            messagebox.showerror('Invalid camera FOV', str(exc))
            return
        self.status.set('Draw a rectangle around a textured scene region, then click Use box or press Enter.')
        box = select_box(self.root, frame, 'Stitch · draw the reference box',
                         'Drag a box around a textured scene region that should be recognised later. '
                         'Click "Use box" or press Enter to accept; Esc cancels.', min_size=12)
        if box is None:
            self.status.set('Selection cancelled or too small; select a larger surface region.')
            return
        x, y, w, h = box
        crop = frame[int(y):int(y+h), int(x):int(x+w)].copy()
        if min(crop.shape[:2]) < 2:
            self.status.set('The selected rectangle was empty.')
            return
        self.reference_crop = crop
        self.depth_job_generation += 1
        self.reference_depth_map = None
        self.reference_anchor_3d = None
        self.depth_placement_status.set('No depth map or 3D placement selected.')
        self.reference_box = [int(x), int(y), int(w), int(h)]
        self.reference_frame_size = [int(frame.shape[1]), int(frame.shape[0])]
        self.reference_source = source_description
        self.reference_file_path = source_path
        self.reference_captured_at = datetime.now(timezone.utc).isoformat()
        self.reference_is_photo = source_description == 'photo'
        self.config['camera_fov_x_deg'] = fov
        self.reference_fov_x_deg = fov
        frame_matrix = camera_matrix(frame.shape, fov)
        self.reference_camera_matrix = crop_camera_matrix(frame_matrix, x, y)
        try:
            if self.sift_engine is None:
                self.sift_engine = LocalEngine(
                    'sift', self.config['max_keypoints'], self.config['sift_ratio'])
            crop_height, crop_width = crop.shape[:2]
            scale = min(1.0, self.config['max_side'] / max(crop_height, crop_width))
            analysis_crop = crop
            if scale < 1.0:
                analysis_crop = cv2.resize(
                    crop,
                    (max(1, round(crop_width * scale)), max(1, round(crop_height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            features = self.sift_engine.extract(analysis_crop)
            point_scale = np.array(
                [analysis_crop.shape[1] / crop_width,
                 analysis_crop.shape[0] / crop_height], dtype=np.float32)
            self.reference_sift_points = features.points / point_scale
            self.reference_sift_count = len(features.points)
        except Exception as exc:
            self.reference_sift_points = np.empty((0, 2), dtype=np.float32)
            self.reference_sift_count = None
            self.status.set(f'Reference selected, but SIFT preview failed: {exc}')

        preview = frame.copy()
        for point_x, point_y in self.reference_sift_points:
            center = (int(round(x + point_x)), int(round(y + point_y)))
            cv2.circle(preview, center, 3, (40, 230, 80), 1, cv2.LINE_AA)
        cv2.rectangle(preview, (int(x), int(y)), (int(x + w), int(y + h)),
                      (40, 230, 80), 2, cv2.LINE_AA)
        if self.reference_sift_count is None:
            details = 'SIFT analysis unavailable; scan will analyse this crop.'
        else:
            required = int(self.config['min_inliers'])
            details = f'{self.reference_sift_count} SIFT features; scan needs at least {required} inliers.'
            if self.reference_sift_count < required:
                details += ' Try a larger, more textured region for more reliable matching.'
            cv2.putText(preview, f'SIFT features: {self.reference_sift_count}',
                        (int(x), max(22, int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (40, 230, 80), 2, cv2.LINE_AA)
        self.registration_preview = preview
        self.selection_status.set(
            f'Selected {w} × {h}px from {source_description}. {details}')
        self._show_frame(preview)

    def clear_reference(self):
        self.depth_job_generation += 1
        self.reference_crop = None
        self.reference_box = None
        self.reference_frame_size = None
        self.reference_source = ''
        self.reference_file_path = ''
        self.reference_captured_at = None
        self.reference_is_photo = False
        self.reference_sift_points = np.empty((0, 2), dtype=np.float32)
        self.reference_sift_count = None
        self.reference_depth_map = None
        self.reference_anchor_3d = None
        self.reference_camera_matrix = None
        self.reference_fov_x_deg = None
        self.registration_preview = None
        self.selection_status.set('No reference region selected')
        self.depth_placement_status.set('No depth map or 3D placement selected.')
        self.status.set('Reference selection cleared; the live camera preview is shown.')

    def choose_overlay(self):
        path = filedialog.askopenfilename(title='Choose overlay image',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp'), ('All files', '*')])
        if path:
            self.overlay_path.set(path)

    def create_3d_placement(self):
        if self.reference_crop is None:
            messagebox.showerror('Reference required', 'Capture a frame and draw the scene reference box first.')
            return
        try:
            sweep = self._number(self.capture_sweep_deg.get(), 'Rescan range', 1, 120)
            self._number(self.view_tolerance_deg.get(), 'Per-view tolerance', 1, 60)
            yaw = self._number(self.overlay_yaw_deg.get(), 'Overlay yaw', -180, 180)
            pitch = self._number(self.overlay_pitch_deg.get(), 'Overlay pitch', -180, 180)
            roll = self._number(self.overlay_roll_deg.get(), 'Overlay roll', -180, 180)
            capture_yaw = self._number(self.capture_yaw_deg.get(), 'View angle', -180, 180)
            fov = self._number(self.camera_fov_x_deg.get(), 'Horizontal camera FOV', 1.01, 178.99)
            self.config['camera_fov_x_deg'] = fov
            model_id = self.depth_model_id.get().strip()
            if not model_id:
                raise ValueError('Enter a depth model ID')
            frame_width, frame_height = self.reference_frame_size
            box_x, box_y, _, _ = self.reference_box
            self.reference_camera_matrix = crop_camera_matrix(
                camera_matrix((frame_height, frame_width), fov), box_x, box_y)
            self.reference_fov_x_deg = fov
            self.depth_job_generation += 1
            job = dict(generation=self.depth_job_generation, crop=self.reference_crop.copy(),
                       model_id=model_id, angles=(yaw, pitch, roll), capture_yaw=capture_yaw,
                       sweep=sweep, fov=fov, place_id=self.place_id.get().strip(),
                       matrix=self.reference_camera_matrix.copy())
            self.depth_placement_status.set(
                'Loading the metric depth model and estimating the selected reference…')
            threading.Thread(target=self._estimate_depth_background, args=(job,), daemon=True).start()
        except ValueError as exc:
            messagebox.showerror('Invalid 3D placement settings', str(exc))

    def _relate_to_primary(self, job, depth_map):
        """Relate this view to the place's first 3D view; ``(None, None)`` if it is the first."""
        if not job['place_id'] or not self.database_path.exists():
            return None, None
        primary = load_primary_view(PlaceDatabase(str(self.database_path), allow_empty=True), job['place_id'])
        if primary is None:
            return None, None
        if primary['fov'] is not None and not math.isclose(float(primary['fov']), job['fov'], abs_tol=1e-3):
            raise ValueError(f"The first view used a {float(primary['fov']):g}° FOV; use the same camera "
                             'and FOV for every view of a place')
        method = self.config.get('depth3d_features', 'sift')
        if self.relation_engine is None or self.relation_engine.method != method:
            self.relation_engine = LocalEngine(method, live_config(self.config, 'depth3d')['max_keypoints'],
                                               self.config['sift_ratio'], self.config['device'])
        primary_points, new_points = match_views(self.relation_engine, primary['image'], job['crop'],
                                                 self.config['max_side'])
        relation = relate_view(primary_points, new_points, primary['depth'], primary['matrix'],
                               job['matrix'], primary['anchor'], new_depth=depth_map,
                               min_inliers=self.config['min_inliers'],
                               min_ratio=self.config['min_inlier_ratio'],
                               reprojection_px=self.config['ransac_px'],
                               max_depth_m=float(self.config.get('depth_max_m', 80.0)),
                               min_support=float(self.config.get('pose_min_support', 0.01)))
        return primary, relation

    def _estimate_depth_background(self, job):
        result = dict(job, depth_map=None, estimator=None, primary=None, relation=None,
                      error=None, relation_error=None)
        try:
            estimator = self.depth_estimator
            if estimator is None or self.depth_model_id_loaded != job['model_id']:
                estimator = MonocularDepthEstimator(job['model_id'], self.config.get('device', 'cpu'))
            result['estimator'] = estimator
            result['depth_map'] = estimator.estimate(job['crop'])
        except Exception as exc:
            result['error'] = str(exc)
            self.depth_results.put(result)
            return
        try:
            result['primary'], result['relation'] = self._relate_to_primary(job, result['depth_map'])
        except Exception as exc:
            result['relation_error'] = str(exc)
        self.depth_results.put(result)

    def _finish_depth_placement(self, result):
        if result['generation'] != self.depth_job_generation:
            return
        if result['error']:
            self.depth_placement_status.set(f"Depth estimation failed: {result['error']}")
            messagebox.showerror('Depth estimation failed', result['error'])
            return
        depth_map = result['depth_map']
        self.depth_estimator = result['estimator']
        self.depth_model_id_loaded = result['model_id']
        depth_colors = colorize_depth(depth_map)
        preview = cv2.addWeighted(self.reference_crop, 0.48, depth_colors, 0.52, 0)
        relation, corners_3d, capture_yaw, source = result['relation'], None, result['capture_yaw'], 'user-entered'
        if result['relation_error']:
            messagebox.showwarning('Could not relate to the first view', result['relation_error'])
        elif result['primary'] is None:
            capture_yaw, source = 0.0, 'primary'
        elif relation is not None and relation.accepted:
            capture_yaw, source = relation.yaw_deg, 'measured'
            quad = project_points(relation.anchor_3d, np.zeros(3), np.zeros(3), self.reference_camera_matrix)
            use = confirm_quad(
                self.root, preview, quad, 'Stitch · overlay box transferred from the first view',
                f'This view is {relation.yaw_deg:+.1f}° around the anchor from the first view '
                f'(height change {relation.pitch_deg:+.1f}°), measured from {relation.pose.inliers} '
                f'depth-backed matches; depth scale correction ×{relation.depth_scale:.2f}.\n\n'
                '"Use this box" keeps the same physical overlay position as the first view (recommended). '
                '"Draw my own box" lets you draw it again for this view only.')
            if use:
                corners_3d = relation.anchor_3d
        else:
            inliers = relation.pose.inliers if relation is not None else 0
            messagebox.showwarning(
                'Could not relate to the first view',
                f'Only {inliers} reliable depth-backed matches with the first view of this place. '
                'Move closer to an already registered angle or include more of the same textured '
                'scene. You can draw the box manually, but it may not line up exactly with other views.')
        if corners_3d is None:
            box = select_box(self.root, preview, 'Stitch · draw the floating 3D overlay box',
                             'Colours show depth (near to far). Drag a box where the overlay should '
                             'float in 3D. Click "Use box" or press Enter; Esc cancels.')
            if box is None:
                self.depth_placement_status.set('3D placement cancelled; no depth box selected.')
                return
            x, y, width, height = box
            try:
                corners_3d, _ = oriented_box_from_roi(
                    (x, y, width, height), depth_map, self.reference_camera_matrix, result['angles'])
            except ValueError as exc:
                self.depth_placement_status.set(f'3D placement failed: {exc}')
                messagebox.showerror('3D placement failed', str(exc))
                return
        self.reference_depth_map = depth_map
        self.reference_anchor_3d = np.asarray(corners_3d, dtype=np.float32)
        self.reference_view_relation = relation if source == 'measured' else None
        self.capture_yaw_source = source
        self.capture_yaw_deg.set(f'{capture_yaw:.1f}')
        self.capture_sweep_deg.set(f"{result['sweep']:g}")
        distance = float(np.linalg.norm(self.reference_anchor_3d.mean(axis=0)))
        origin = ('transferred from the first view' if source == 'measured' and relation is not None
                  and np.allclose(self.reference_anchor_3d, relation.anchor_3d) else 'drawn in this view')
        self.depth_placement_status.set(
            f'3D box {origin}, about {distance:.2f} m from this camera; view angle {capture_yaw:+.1f}° '
            f'({source}). Register this view, then select depth3d when rescanning.')

    @staticmethod
    def _number(value, label, low, high):
        try:
            result = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{label} must be a number')
        if not math.isfinite(result) or not low <= result <= high:
            raise ValueError(f'{label} must be between {low} and {high}')
        return result

    def register_reference(self):
        place_id = self.place_id.get().strip()
        overlay = self.overlay_path.get().strip()
        if self.reference_crop is None:
            messagebox.showerror('Reference required', 'Capture a live frame or load a photo and draw a box first.')
            return
        if not place_id or not overlay:
            messagebox.showerror('Missing information', 'Enter a Place ID and choose an overlay image.')
            return
        try:
            lat = self._number(self.latitude.get(), 'Latitude', -90, 90)
            lon = self._number(self.longitude.get(), 'Longitude', -180, 180)
            capture_yaw = self._number(self.capture_yaw_deg.get(), 'View angle', -180, 180)
            capture_sweep = self._number(self.capture_sweep_deg.get(), 'Rescan range', 1, 120)
            tolerance = self._number(self.view_tolerance_deg.get(), 'Per-view tolerance', 1, 60)
            fov = self._number(self.camera_fov_x_deg.get(), 'Horizontal camera FOV', 1.01, 178.99)
            if abs(capture_yaw) > capture_sweep / 2.0 + tolerance and not messagebox.askyesno(
                    'View outside the rescan range',
                    f'This view is at {capture_yaw:+.1f}°, outside the planned ±{capture_sweep / 2:g}° range. '
                    'Register it anyway?'):
                return
            self.config['camera_fov_x_deg'] = fov
            if (self.reference_anchor_3d is not None and self.reference_fov_x_deg is not None
                    and not math.isclose(fov, self.reference_fov_x_deg, abs_tol=1e-6)):
                raise ValueError('Camera FOV changed after 3D placement; regenerate the depth placement')
            if self.reference_frame_size is not None and self.reference_box is not None:
                frame_width, frame_height = self.reference_frame_size
                box_x, box_y, _, _ = self.reference_box
                self.reference_camera_matrix = crop_camera_matrix(
                    camera_matrix((frame_height, frame_width), fov), box_x, box_y)
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            if self.database_path.exists():
                db = PlaceDatabase(str(self.database_path), allow_empty=True)
            else:
                self.database_path.write_text(json.dumps({'schema_version': 1, 'places': []}, indent=2) + '\n')
                db = PlaceDatabase(str(self.database_path), allow_empty=True)
            with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                if not cv2.imwrite(tmp.name, self.reference_crop):
                    raise OSError('Could not save selected reference crop')
                capture_metadata = dict(
                    captured_at_utc=self.reference_captured_at,
                    source=self.reference_source,
                    camera_source=str(self._source_value()) if not self.reference_is_photo else None,
                    source_photo=self.reference_file_path or None,
                    source_frame_box_xywh=self.reference_box,
                    source_frame_size_px=self.reference_frame_size,
                    selected_reference_size_px=[int(self.reference_crop.shape[1]), int(self.reference_crop.shape[0])],
                    sift_keypoints_at_scan_size=self.reference_sift_count,
                    reference_camera_matrix=self.reference_camera_matrix.tolist(),
                    camera_fov_x_deg=fov,
                    capture_yaw_offset_deg=capture_yaw,
                    capture_yaw_source=self.capture_yaw_source,
                    capture_yaw_sweep_deg=capture_sweep,
                    view_tolerance_deg=tolerance,
                    view_relation=(None if self.reference_view_relation is None else dict(
                        inliers=int(self.reference_view_relation.pose.inliers),
                        median_reprojection_px=float(self.reference_view_relation.pose.median_error_px),
                        pitch_deg=float(self.reference_view_relation.pitch_deg),
                        depth_scale=float(self.reference_view_relation.depth_scale))),
                    location_source='user-entered',
                    coordinate_order='latitude, longitude')
                place = db.register(place_id, self.place_name.get().strip(), tmp.name, overlay,
                                    lat, lon, anchor=[[0,0],[1,0],[1,1],[0,1]],
                                    capture_metadata=capture_metadata,
                                    depth_map=self.reference_depth_map,
                                    anchor_3d_camera=self.reference_anchor_3d)
            self.status.set(f'Registered {place["name"]} ({place_id}) at {lat:.6f}, {lon:.6f}.')
            self.scan_status.set('Reference saved. Start rescan to search for it with the live camera.')
            self._update_capture_angle_guidance()
            if self.reference_anchor_3d is not None:
                self.method.set('depth3d')
                self.scan_status.set('3D reference saved. Start rescan with depth3d pose matching.')
            messagebox.showinfo('Reference registered',
                f'{place["name"]} is saved in {self.database_path}\nA later capture can use the same Place ID for another view.')
        except Exception as exc:
            messagebox.showerror('Registration failed', str(exc))

    def start_scan(self):
        if self._current_frame() is None:
            messagebox.showerror('Camera not connected', 'Connect the Mac camera before starting a rescan.')
            return
        if self.scan_enabled:
            return
        if not self.database_path.exists():
            messagebox.showerror('No registered places', 'Register a reference first.')
            return
        lat_text, lon_text = self.scan_latitude.get().strip(), self.scan_longitude.get().strip()
        if bool(lat_text) != bool(lon_text):
            messagebox.showerror('Incomplete location', 'Enter both scan latitude and longitude, or leave both empty.')
            return
        try:
            self.registration_preview = None
            location = (self._number(lat_text, 'Scan latitude', -90, 90),
                        self._number(lon_text, 'Scan longitude', -180, 180)) if lat_text else (None, None)
            self.config['camera_fov_x_deg'] = self._number(
                self.camera_fov_x_deg.get(), 'Horizontal camera FOV', 1.01, 178.99)
            db = PlaceDatabase(str(self.database_path))
            self.scan_generation += 1
            generation = self.scan_generation
            self.scan_busy = True
            self.scan_status.set('Loading registered references…')
            threading.Thread(target=self._load_pipeline,
                             args=(generation, db, self.method.get(), location), daemon=True).start()
        except Exception as exc:
            messagebox.showerror('Cannot start scan', str(exc))

    def _load_pipeline(self, generation, db, method, location):
        try:
            pipeline = Pipeline(db, live_config(self.config, method), method)
            with self.scan_result_lock:
                self.scan_result = (generation, 'ready', pipeline, location)
        except Exception as exc:
            with self.scan_result_lock:
                self.scan_result = (generation, 'error', str(exc), None)

    def stop_scan(self):
        self.scan_enabled = False
        self.scan_busy = False
        self.pipeline = None
        self.last_rendered = None
        self.scan_generation += 1
        if hasattr(self, 'scan_status'):
            self.scan_status.set('Scanning is idle.')

    def _launch_query(self, frame, generation, location):
        try:
            frame = resize_for_scan(frame, self.config)
            prediction = self.pipeline.stabilise(self.pipeline.query(frame, *location))
            rendered = self.pipeline.render(frame, prediction)
            with self.scan_result_lock:
                self.scan_result = (generation, 'frame', rendered, prediction)
        except Exception as exc:
            with self.scan_result_lock:
                self.scan_result = (generation, 'query-error', str(exc), None)

    def _tick(self):
        try:
            while True:
                self._finish_depth_placement(self.depth_results.get_nowait())
        except Empty:
            pass
        frame = self._current_frame()
        if (self.capture_running and frame is None and self.camera_opened_at is not None
                and time.monotonic() - self.camera_opened_at > 5
                and not self.camera_warning_shown):
            self.camera_warning_shown = True
            self.status.set('Camera opened but no frames arrived. Check System Settings → Privacy & Security → Camera for the launching app, then verify the selected camera source.')
        if frame is not None:
            display = (self.registration_preview
                       if self.registration_preview is not None and not self.scan_enabled
                       else frame)
            with self.scan_result_lock:
                result, self.scan_result = self.scan_result, None
            if result:
                generation, kind, data, extra = result
                if generation == self.scan_generation:
                    if kind == 'ready':
                        self.pipeline, self._scan_location = data, extra
                        self.scan_enabled, self.scan_busy = True, False
                        self.scan_status.set('Scanning live frames…')
                    elif kind == 'error':
                        self.scan_busy = False
                        self.scan_status.set(f'Could not load scan pipeline: {data}')
                    elif kind == 'frame':
                        self.scan_busy = False
                        self.last_rendered = data
                        display = data
                        prediction = extra
                        if prediction.place_id:
                            text = (f'Matched {prediction.place_id} · {prediction.verification.inliers} '
                                    f'inliers · {prediction.elapsed_ms:.0f} ms')
                            if prediction.view_yaw_deg is not None:
                                text += f' · viewing {prediction.view_yaw_deg:+.0f}° from the first view'
                                text += (' (inside the captured range)' if prediction.in_captured_range
                                         else ' (outside the captured range: register a view from here)')
                            if prediction.held:
                                text += ' · holding last stable pose'
                            self.scan_status.set(text)
                        else:
                            self.scan_status.set(f'No verified surface match · {prediction.elapsed_ms:.0f} ms')
                    elif kind == 'query-error':
                        self.scan_busy = False
                        self.scan_status.set(f'Rescan error: {data}')
            if self.scan_enabled and not self.scan_busy and self.pipeline is not None:
                self.scan_busy = True
                threading.Thread(target=self._launch_query,
                                 args=(frame.copy(), self.scan_generation, self._scan_location),
                                 daemon=True).start()
            elif self.scan_enabled and self.last_rendered is not None:
                display = self.last_rendered
            self._show_frame(display)
        self.root.after(30, self._tick)

    def _show_frame(self, frame):
        if frame is None:
            return
        height, width = frame.shape[:2]
        scale = min(1.0, 900 / width, 650 / height)
        if scale < 1:
            frame = cv2.resize(frame, (round(width*scale), round(height*scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=photo, text='')
        self.photo_reference = photo

    def close(self):
        self.stop_scan()
        self.capture_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.root.destroy()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Stitch reference register and rescan app')
    parser.add_argument('--camera', help='Camera to select: index (0-5), part of its name '
                        '(e.g. "FaceTime" for the Mac camera) or a stream URL')
    parser.add_argument('--connect', action='store_true', help='Connect the selected camera on start')
    args = parser.parse_args(argv)
    root = tk.Tk()
    app = StitchApp(root)
    if args.camera:
        match = next((f'{index} — {name}' for index, name in camera_devices()
                      if args.camera.lower() in name.lower()), None)
        app.source.set(match or args.camera)
    if args.connect:
        root.after(300, app.connect_camera)
    root.mainloop()


if __name__ == '__main__':
    main()
