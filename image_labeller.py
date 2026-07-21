import json
import os
import sys
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False


class StackReader:
    """Lazy frame reader: only one frame is decoded/held in memory at a time."""

    @staticmethod
    def _coerce_stack(raw):
        if hasattr(raw, "keys") and hasattr(raw, "__getitem__") and not isinstance(raw, np.ndarray):
            for key in ("data", "stack", "images", "frames", "array", "arr"):
                if key in raw:
                    return StackReader._coerce_stack(raw[key])
            keys = list(raw.keys())
            if len(keys) == 1:
                return StackReader._coerce_stack(raw[keys[0]])
            raise ValueError(f"Cannot find array in keys: {keys}")

        if hasattr(raw, "data") and not isinstance(raw, np.ndarray):
            return StackReader._coerce_stack(raw.data)

        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
            return StackReader._coerce_stack(raw.item())

        if isinstance(raw, (list, tuple)):
            if not raw:
                raise ValueError("Empty stack")
            return np.asarray(raw, dtype=object)

        if not isinstance(raw, np.ndarray):
            raise ValueError(f"Expected array stack, got {type(raw).__name__}")

        while raw.ndim > 3:
            raw = raw[0]
        return raw

    @staticmethod
    def _load_memmap(path: str):
        meta_path = os.path.splitext(path)[0] + ".meta.json"
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        shape = tuple(int(v) for v in meta["shape"])
        dtype = np.dtype(meta["dtype"])
        arr = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        if arr.ndim == 2:
            return arr, "npy", 1, arr.shape, arr.dtype
        if arr.ndim == 3:
            return arr, "npy", arr.shape[0], arr.shape[1:], arr.dtype
        raise ValueError(f"Unsupported memmap shape {shape} in {path}")

    @staticmethod
    def _load_npy(path: str):
        memmap_stack = StackReader._load_memmap(path)
        if memmap_stack is not None:
            return memmap_stack

        arr = None
        last_error = None

        for loader in (
            lambda: np.load(path, mmap_mode="r"),
            lambda: np.load(path, allow_pickle=True),
            lambda: np.load(path, allow_pickle=True, encoding="latin1"),
        ):
            try:
                arr = loader()
                break
            except Exception as exc:
                last_error = exc

        if arr is None:
            import pickle
            try:
                with open(path, "rb") as f:
                    arr = pickle.load(f)
            except Exception as exc:
                last_error = exc
                msg = str(last_error)
                raise ValueError(f"Could not load {path}: {msg}") from last_error

        arr = StackReader._coerce_stack(arr)

        if arr.dtype == object:
            sample = np.asarray(arr[0])
            return arr, "npy_obj", len(arr), sample.shape, sample.dtype
        if arr.ndim == 2:
            return arr, "npy", 1, arr.shape, arr.dtype
        if arr.ndim == 3:
            return arr, "npy", arr.shape[0], arr.shape[1:], arr.dtype
        raise ValueError(f"Unsupported array shape {arr.shape} in {path}")

    def __init__(self, path: str):
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tif", ".tiff"):
            import tifffile
            self._mode = "tif"
            self._tif = tifffile.TiffFile(path)
            self._pages = self._tif.pages
            self._n = len(self._pages)
            sample = self._pages[0].asarray()
            self._frame_shape = sample.shape
            self._dtype = sample.dtype
        elif ext == ".npy":
            self._data, self._mode, self._n, self._frame_shape, self._dtype = (
                self._load_npy(path)
            )
        else:
            import hyperspy.api as hs
            self._mode = "lazy"
            sig = hs.load(path, lazy=True)
            if isinstance(sig, list):
                sig = sig[0]
            arr = sig.data
            while arr.ndim > 3:
                arr = arr[0]
            self._data = arr
            self._n = arr.shape[0] if arr.ndim == 3 else 1
            self._frame_shape = arr.shape[1:] if arr.ndim == 3 else arr.shape
            self._dtype = arr.dtype

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> np.ndarray:
        if not 0 <= i < self._n:
            raise IndexError(i)
        if self._mode == "tif":
            return self._pages[i].asarray()
        if self._mode == "npy_obj":
            return np.asarray(self._data[i])
        if self._mode == "npy":
            return np.asarray(self._data[i] if self._data.ndim == 3 else self._data)
        arr = self._data
        s = arr if arr.ndim == 2 else arr[i]
        if hasattr(s, "compute"):
            s = s.compute()
        return np.asarray(s)

    def close(self):
        if self._mode == "tif":
            self._tif.close()
        elif self._mode == "npy" and isinstance(self._data, np.memmap):
            self._data._mmap.close()


def normalize(frame: np.ndarray) -> np.ndarray:
    mn = float(frame.min())
    mx = float(frame.max())
    if mx <= mn:
        return np.zeros(frame.shape, dtype=np.uint8)
    scale = 255.0 / (mx - mn)
    return ((frame.astype(np.float32, copy=False) - mn) * scale).astype(np.uint8)


class ColorButton(tk.Frame):
    def __init__(self, parent, color, command):
        self.default_bg = parent.cget("bg")
        super().__init__(parent, bg=self.default_bg, padx=5, pady=5)
        self.inner = tk.Label(self, bg=color, width=14, height=3, cursor="hand2")
        self.inner.pack()
        self.inner.bind("<Button-1>", lambda e: command())

    def set_selected(self, selected: bool):
        self.config(bg="black" if selected else self.default_bg)


class LabelChart(tk.Canvas):
    W, H = 180, 110
    BAR_W = 40

    def __init__(self, parent):
        super().__init__(
            parent, width=self.W, height=self.H, bg="#ffffff",
            highlightthickness=1, highlightbackground="#888",
        )
        self._green = 0
        self._red = 0
        self._total = 1

    def update_counts(self, labels: np.ndarray):
        self._green = int((labels == 1).sum())
        self._red = int(len(labels) - self._green)
        self._total = max(1, len(labels))
        self._draw()

    def _draw(self):
        self.delete("all")
        W, H = self.W, self.H
        y0 = H - 22
        chart_h = y0 - 22
        cx1 = W // 2 - self.BAR_W - 8
        cx2 = W // 2 + 8
        g_h = int((self._green / self._total) * chart_h)
        r_h = int((self._red / self._total) * chart_h)
        self.create_rectangle(cx1, y0 - g_h, cx1 + self.BAR_W, y0,
                              fill="#4caf50", outline="")
        self.create_rectangle(cx2, y0 - r_h, cx2 + self.BAR_W, y0,
                              fill="#f44336", outline="")
        self.create_line(cx1 - 4, y0, cx2 + self.BAR_W + 4, y0, fill="#333")
        self.create_text(cx1 + self.BAR_W // 2, y0 - g_h - 9,
                         text=str(self._green), fill="#222",
                         font=("Helvetica", 10, "bold"))
        self.create_text(cx2 + self.BAR_W // 2, y0 - r_h - 9,
                         text=str(self._red), fill="#222",
                         font=("Helvetica", 10, "bold"))
        self.create_text(cx1 + self.BAR_W // 2, y0 + 10, text="green",
                         fill="#333", font=("Helvetica", 9))
        self.create_text(cx2 + self.BAR_W // 2, y0 + 10, text="red",
                         fill="#333", font=("Helvetica", 9))


class Labeller:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeller")
        self.root.geometry("1280x900")
        self.stack: StackReader | None = None
        self.labels: np.ndarray | None = None
        self.frame_idx = 0
        self.path: str | None = None
        self._src_pil: Image.Image | None = None
        self._src_idx = -1
        self._last_render_size = (0, 0)

        self.drop_label = tk.Label(
            root,
            text="Drag & drop an image stack here\n(.emd, .tif, .tiff, .npy)",
            bg="#f0f0f0", relief="ridge", fg="#555", font=("Helvetica", 14),
        )
        self.drop_label.pack(padx=20, pady=20, fill="both", expand=True)

        if HAS_DND:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.drop_label.config(
                text="tkinterdnd2 not installed.\nPass file path as CLI arg, or run:\n  pip install tkinterdnd2"
            )

        self.root.protocol("WM_DELETE_WINDOW", self.done)

    def on_drop(self, event):
        path = event.data.strip().strip("{}").strip('"')
        self.load(path)

    def load(self, path: str):
        self.path = path
        try:
            self.stack = StackReader(path)
        except Exception as e:
            self.drop_label.config(text=f"Failed to load:\n{e}")
            return
        n = len(self.stack)
        self.labels = self._load_existing_labels(path, n)
        self.frame_idx = 0
        self._src_pil = None
        self._src_idx = -1
        self._last_render_size = (0, 0)
        self.build_viewer()

    @staticmethod
    def _load_existing_labels(path: str, n: int) -> np.ndarray:
        label_path = os.path.splitext(path)[0] + "_labels.npy"
        if os.path.exists(label_path):
            try:
                saved = np.load(label_path)
                if saved.shape == (n,):
                    print(f"Loaded existing labels from {label_path}")
                    return saved.astype(np.uint8)
                print(f"Ignoring {label_path}: shape {saved.shape} != ({n},)")
            except Exception as e:
                print(f"Failed to read {label_path}: {e}")
        return np.zeros(n, dtype=np.uint8)

    def build_viewer(self):
        self.drop_label.destroy()

        controls = tk.Frame(self.root)
        controls.pack(side="bottom", fill="x", pady=(0, 12))

        title_row = tk.Frame(controls)
        title_row.pack(pady=(6, 4))
        tk.Label(title_row, text="Frame ", font=("Helvetica", 12)).pack(side="left")
        self.frame_entry = tk.Entry(title_row, width=6, font=("Helvetica", 12),
                                    justify="center")
        self.frame_entry.pack(side="left")
        self.frame_entry.bind("<Return>", self._on_frame_entry)
        self.frame_entry.bind("<FocusOut>", self._on_frame_entry)
        self.frame_entry.bind("<Escape>", lambda e: (self._sync_frame_entry(),
                                                     self.root.focus_set()))
        self.total_label = tk.Label(title_row, text="", font=("Helvetica", 12))
        self.total_label.pack(side="left")

        btn_row = tk.Frame(controls)
        btn_row.pack(pady=4)

        self.green_btn = ColorButton(btn_row, "#4caf50", lambda: self.set_label(1))
        self.green_btn.pack(side="left", padx=15)

        self.red_btn = ColorButton(btn_row, "#f44336", lambda: self.set_label(0))
        self.red_btn.pack(side="left", padx=15)

        tk.Button(controls, text="Done", command=self.done, width=12,
                  takefocus=0).pack(pady=(10, 0))

        img_row = tk.Frame(self.root)
        img_row.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.prev_btn = tk.Button(img_row, text="\u25c0", command=self.prev_frame,
                                  width=3, takefocus=0)
        self.prev_btn.pack(side="left", fill="y", padx=(0, 5))

        self.canvas = tk.Canvas(img_row, bg="black", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._canvas_img_id: int | None = None

        self.next_btn = tk.Button(img_row, text="\u25b6", command=self.next_frame,
                                  width=3, takefocus=0)
        self.next_btn.pack(side="left", fill="y", padx=(5, 0))

        self.chart = LabelChart(img_row)
        self.chart.place(relx=0.0, rely=1.0, x=10, y=-10, anchor="sw")
        self.chart.update_counts(self.labels)

        self.canvas.bind("<Configure>", lambda e: self._render_image())

        self.root.bind("<Left>", lambda e: self._shortcut(self.prev_frame))
        self.root.bind("<Right>", lambda e: self._shortcut(self.next_frame))
        self.root.bind("<Up>", lambda e: self._shortcut(self.toggle_label))
        self.root.focus_set()

        self.update_display()

    def _shortcut(self, action):
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        action()

    def _sync_frame_entry(self):
        if self.stack is None:
            return
        self.frame_entry.delete(0, "end")
        self.frame_entry.insert(0, str(self.frame_idx + 1))
        self.total_label.config(text=f" / {len(self.stack)}")

    def _on_frame_entry(self, event=None):
        if self.stack is None:
            return
        try:
            target = int(self.frame_entry.get()) - 1
        except ValueError:
            self._sync_frame_entry()
            self.root.focus_set()
            return
        n = len(self.stack)
        target = max(0, min(n - 1, target))
        if target != self.frame_idx:
            self.frame_idx = target
            self.update_display()
        else:
            self._sync_frame_entry()
        self.root.focus_set()

    def update_display(self):
        if self._src_idx != self.frame_idx:
            frame = self.stack[self.frame_idx]
            self._src_pil = Image.fromarray(normalize(frame), mode="L")
            self._src_idx = self.frame_idx
            self._last_render_size = (0, 0)
            del frame
        self._sync_frame_entry()
        self._render_image()
        self.update_highlight()

    def _render_image(self):
        if self._src_pil is None:
            return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2:
            return
        sw, sh = self._src_pil.size
        scale = min(w / sw, h / sh)
        tw = max(1, int(sw * scale))
        th = max(1, int(sh * scale))
        if (tw, th) != self._last_render_size or self._canvas_img_id is None:
            resample = Image.NEAREST if scale >= 1 else Image.BILINEAR
            resized = self._src_pil.resize((tw, th), resample)
            self.tk_img = ImageTk.PhotoImage(resized)
            self._last_render_size = (tw, th)
        self.canvas.delete("img")
        self._canvas_img_id = self.canvas.create_image(
            w // 2, h // 2, image=self.tk_img, tags="img"
        )

    def update_highlight(self):
        is_green = bool(self.labels[self.frame_idx] == 1)
        self.green_btn.set_selected(is_green)
        self.red_btn.set_selected(not is_green)

    def set_label(self, value: int):
        prev = int(self.labels[self.frame_idx])
        if prev == value:
            return
        self.labels[self.frame_idx] = value
        self.update_highlight()
        self.chart.update_counts(self.labels)

    def toggle_label(self):
        self.set_label(1 - int(self.labels[self.frame_idx]))

    def prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self.update_display()

    def next_frame(self):
        if self.stack is not None and self.frame_idx < len(self.stack) - 1:
            self.frame_idx += 1
            self.update_display()

    def done(self):
        if self.path is not None and self.labels is not None:
            out = os.path.splitext(self.path)[0] + "_labels.npy"
            np.save(out, self.labels)
            print(f"Saved labels to {out}")
            print(self.labels)
        if self.stack is not None:
            self.stack.close()
        self.root.destroy()


def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = Labeller(root)
    if len(sys.argv) > 1:
        app.load(sys.argv[1])
    root.mainloop()


if __name__ == "__main__":
    main()
