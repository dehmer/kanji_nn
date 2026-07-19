import matplotlib.pyplot as plt
from kanji_nn.plot import multi_channel_plot

def plot_mcp(stroke, channels=["P_norm", "dP/dt", "dP", "ds", "c_speed", "at"], show=False, save=True):
    figure = multi_channel_plot(stroke, channels, figsize=(18, 8))

    if save:
        filename = f"data/dataset/{stroke.dataset}/mcp/{stroke.literal}-{stroke.stroke_index}"
        plt.savefig(filename)

    if show:
        plt.show()

    plt.close(figure)
