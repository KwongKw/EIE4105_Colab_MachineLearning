# That's an impressive list of imports.
import numpy as np

# We import sklearn.
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# Random state.
RS = 20150101

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,rc={"lines.linewidth": 2.5})

# Now we load the classic handwritten digits datasets. It contains 1797 images with \(8*8=64\) pixels each.
digits = load_digits()
digits.data.shape
print(digits.data.shape)
print(digits['DESCR'])

# Display the digits
import os
try:
    os.stat('images')
except:
    os.mkdir('images')
nrows, ncols = 2, 5
plt.figure(figsize=(6,3))
plt.gray()
for i in range(ncols * nrows):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i,...])
    plt.xticks([]); plt.yticks([])
    plt.title(digits.target[i])
plt.savefig('images/digits-generated.png', dpi=150)

import matplotlib.image as mpimg
img = mpimg.imread('images/digits-generated.png')
plt.imshow(img)
#plt.show()

# Now let's run the t-SNE algorithm on the dataset. It just takes one line with scikit-learn.
# We first reorder the data points according to the handwritten numbers.
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])

digits_proj = TSNE(random_state=RS).fit_transform(X)

# Here is a utility function used to display the transformed dataset.
# The color of each point refers to the actual digit (of course, this information was not used by the dimensionality reduction algorithm).
def scatter(x, colors):
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
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(digits_proj, y)
plt.savefig('images/digits_tsne-generated.png', dpi=120)
img = mpimg.imread('images/digits_tsne-generated.png')
plt.imshow(img)
plt.show()
