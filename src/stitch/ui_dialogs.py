"""Native Tkinter dialogs for drawing and confirming boxes on an image.

OpenCV HighGUI windows (``cv2.selectROI``/``cv2.imshow``) do not cooperate
with a Tkinter main loop on macOS: they open behind the app, miss mouse and
keyboard input and block the interface. These dialogs stay inside Tk.
"""
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


def display_scale(image_shape, max_width, max_height):
    """Scale that fits an image inside the given display area (never enlarges)."""
    height, width = image_shape[:2]
    return min(1.0, max_width / float(width), max_height / float(height))


def to_image_box(start, end, scale, image_shape):
    """Convert two canvas corners to an ``(x, y, w, h)`` box in image pixels."""
    height, width = image_shape[:2]
    (x0, y0), (x1, y1) = start, end
    left, right = sorted((x0 / scale, x1 / scale))
    top, bottom = sorted((y0 / scale, y1 / scale))
    left, right = max(0.0, left), min(float(width), right)
    top, bottom = max(0.0, top), min(float(height), bottom)
    return (int(round(left)), int(round(top)),
            max(0, int(round(right - left))), max(0, int(round(bottom - top))))


class _ImageDialog:
    def __init__(self, root, image_bgr, title, instruction):
        self.root, self.image = root, image_bgr
        self.top = tk.Toplevel(root)
        self.top.title(title)
        self.top.transient(root)
        self.top.resizable(False, False)
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        self.scale = display_scale(image_bgr.shape, screen_w * 0.8, screen_h * 0.7)
        height, width = image_bgr.shape[:2]
        size = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        shown = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA) if self.scale < 1 else image_bgr
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
        ttk.Label(self.top, text=instruction, wraplength=size[0], padding=(10, 8)).pack(anchor='w')
        self.canvas = tk.Canvas(self.top, width=size[0], height=size[1], highlightthickness=0, cursor='crosshair')
        self.canvas.pack(padx=10)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.buttons = ttk.Frame(self.top, padding=10)
        self.buttons.pack(fill='x')
        self.result = None

    def run(self):
        self.top.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - self.top.winfo_width()) // 2)
        y = self.root.winfo_rooty() + 30
        self.top.geometry(f'+{max(0, x)}+{max(0, y)}')
        self.top.lift()
        self.top.focus_force()
        self.top.grab_set()
        self.top.protocol('WM_DELETE_WINDOW', self.cancel)
        self.root.wait_window(self.top)
        return self.result

    def close(self, result):
        self.result = result
        self.top.grab_release()
        self.top.destroy()

    def cancel(self, _event=None):
        self.close(None)


class BoxSelector(_ImageDialog):
    """Drag a rectangle; Enter or "Use box" accepts, Esc or "Cancel" aborts."""

    def __init__(self, root, image_bgr, title, instruction, min_size=8):
        super().__init__(root, image_bgr, title, instruction)
        self.min_size, self.start, self.end, self.rect = min_size, None, None, None
        self.size_text = tk.StringVar(value='Drag on the image to draw a box.')
        ttk.Label(self.buttons, textvariable=self.size_text).pack(side='left')
        ttk.Button(self.buttons, text='Cancel (Esc)', command=self.cancel).pack(side='right')
        self.use_button = ttk.Button(self.buttons, text='Use box (Enter)', command=self.accept, state='disabled')
        self.use_button.pack(side='right', padx=6)
        self.canvas.bind('<ButtonPress-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.canvas.bind('<ButtonRelease-1>', self._drag)
        for key in ('<Return>', '<KP_Enter>'):
            self.top.bind(key, self.accept)
        self.top.bind('<Escape>', self.cancel)

    def _press(self, event):
        self.start = self.end = (event.x, event.y)
        if self.rect is not None:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                 outline='#28e650', width=2)

    def _drag(self, event):
        if self.start is None:
            return
        width, height = int(self.canvas['width']), int(self.canvas['height'])
        self.end = (min(max(event.x, 0), width - 1), min(max(event.y, 0), height - 1))
        self.canvas.coords(self.rect, *self.start, *self.end)
        box = to_image_box(self.start, self.end, self.scale, self.image.shape)
        valid = box[2] >= self.min_size and box[3] >= self.min_size
        self.size_text.set(f'Box {box[2]} × {box[3]} px' + ('' if valid else ' (too small)'))
        self.use_button.configure(state='normal' if valid else 'disabled')

    def box(self):
        if self.start is None or self.end is None:
            return None
        box = to_image_box(self.start, self.end, self.scale, self.image.shape)
        return box if box[2] >= self.min_size and box[3] >= self.min_size else None

    def accept(self, _event=None):
        box = self.box()
        if box is not None:
            self.close(box)


class QuadConfirm(_ImageDialog):
    """Show a projected quadrilateral; returns True (use it) or False (draw instead)."""

    def __init__(self, root, image_bgr, quad, title, message,
                 yes_text='Use this box (Enter)', no_text='Draw my own box'):
        super().__init__(root, image_bgr, title, message)
        if quad is not None:
            points = (np.asarray(quad, dtype=np.float64).reshape(-1, 2) * self.scale).ravel().tolist()
            self.canvas.create_polygon(*points, outline='white', fill='', width=4)
            self.canvas.create_polygon(*points, outline='#ff3030', fill='', width=2)
        ttk.Button(self.buttons, text=no_text, command=lambda: self.close(False)).pack(side='right')
        ttk.Button(self.buttons, text=yes_text, command=lambda: self.close(True)).pack(side='right', padx=6)
        for key in ('<Return>', '<KP_Enter>'):
            self.top.bind(key, lambda _e: self.close(True))
        self.top.bind('<Escape>', lambda _e: self.close(False))

    def cancel(self, _event=None):
        self.close(False)


def select_box(root, image_bgr, title, instruction, min_size=8):
    """Modal box selection; returns ``(x, y, w, h)`` in image pixels or ``None``."""
    return BoxSelector(root, image_bgr, title, instruction, min_size).run()


def confirm_quad(root, image_bgr, quad, title, message):
    """Modal confirmation of a projected box; returns True or False."""
    return bool(QuadConfirm(root, image_bgr, quad, title, message).run())
