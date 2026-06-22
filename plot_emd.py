import sys
import hyperspy.api as hs
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_emd(path: str) -> None:
    signal = hs.load(path)
    if isinstance(signal, list):
        signal = signal[0]
    data = signal.data
    while data.ndim > 3:
        data = data[0]

    if data.ndim == 2:
        plt.imshow(data, cmap="gray")
        plt.axis("off")
        plt.title(path)
        plt.show()
        return

    n = data.shape[0]
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    img = ax.imshow(data[0], cmap="gray")
    ax.axis("off")
    ax.set_title(f"frame 0 / {n - 1}")

    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, "frame", 0, n - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        img.set_data(data[i])
        ax.set_title(f"frame {i} / {n - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    plot_emd(sys.argv[1])
