import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Here is a utility function used to display the transformed dataset.
# The color of each point refers to the actual digit (of course,
# this information was not used by the dimensionality reduction algorithm).
# For general classification problem (not MNIST digit recognition), colors
# contain the class labels
def scatter2D(x, colors):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    labels = np.unique(colors)
    for i in labels:
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def scatter3D(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a 3D scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = Axes3D(f)
    sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-25, 25)
    ax.axis('tight')

    return f, ax, sc


# Display 25 images in a 5x5 grid
def show_imgs(imgs):
    cnt = 0
    r, c = 5, 5
    fi, ax = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            ax[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
            ax[i, j].axis('off')
            cnt += 1
    plt.show()