import matplotlib.pyplot as plt

def plot_stick_man(glyph):
    fig, ax = plt.subplots(figsize=(10, 10))

    image = glyph["image:binary"]
    ax.imshow(image)

    for xy in glyph["stick_man"]:
        ax.plot(xy[:, 0], xy[:, 1], color="red", linewidth=2)

    plt.show()

    return glyph
