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
        arr = self._data
        s = arr if arr.ndim == 2 else arr[i]
        if hasattr(s, "compute"):
            s = s.compute()
        return np.asarray(s)

    def close(self):
        if self._mode == "tif":
            self._tif.close()


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


class Labeller:
    MAX_IMG = 512

    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeller")
        self.root.geometry("760x720")
        self.stack: StackReader | None = None
        self.labels: np.ndarray | None = None
        self.frame_idx = 0
        self.path: str | None = None

        self.drop_label = tk.Label(
            root,
            text="Drag & drop an image stack here\n(.emd, .tif, .tiff)",
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

        container = tk.Frame(self.root)
        container.pack(padx=10, pady=10)

        img_row = tk.Frame(container)
        img_row.pack()

        self.prev_btn = tk.Button(img_row, text="\u25c0", command=self.prev_frame,
                                  width=3, height=15)
        self.prev_btn.pack(side="left", padx=5)

        self.canvas = tk.Label(img_row, bg="black", width=self.MAX_IMG, height=self.MAX_IMG)
        self.canvas.pack(side="left")

        self.next_btn = tk.Button(img_row, text="\u25b6", command=self.next_frame,
                                  width=3, height=15)
        self.next_btn.pack(side="left", padx=5)

        self.title_label = tk.Label(container, text="", font=("Helvetica", 12))
        self.title_label.pack(pady=6)

        btn_row = tk.Frame(container)
        btn_row.pack(pady=4)

        self.green_btn = ColorButton(btn_row, "#4caf50", lambda: self.set_label(1))
        self.green_btn.pack(side="left", padx=15)

        self.red_btn = ColorButton(btn_row, "#f44336", lambda: self.set_label(0))
        self.red_btn.pack(side="left", padx=15)

        tk.Button(container, text="Done", command=self.done, width=12).pack(pady=12)

        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<space>", lambda e: self.toggle_label())
        self.root.focus_set()

        self.update_display()

    def update_display(self):
        frame = self.stack[self.frame_idx]
        pil = Image.fromarray(normalize(frame), mode="L")
        del frame
        pil.thumbnail((self.MAX_IMG, self.MAX_IMG))
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(image=self.tk_img, width=pil.width, height=pil.height)
        self.title_label.config(text=f"Frame {self.frame_idx + 1} / {len(self.stack)}")
        self.update_highlight()

    def update_highlight(self):
        is_green = bool(self.labels[self.frame_idx] == 1)
        self.green_btn.set_selected(is_green)
        self.red_btn.set_selected(not is_green)

    def set_label(self, value: int):
        self.labels[self.frame_idx] = value
        self.update_highlight()

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
