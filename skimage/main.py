from skimage import data
import plotly
import plotly.express as px
import numpy as np
from skimage.filters import rank

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk, ball
from skimage import data, color
from skimage import exposure

import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters




def show_cell3d():
    img = data.cells3d()[20:]

    # omit some slices that are partially empty
    img = img[5:26]

    upper_limit = 1.5 * np.percentile(img, q=99)
    img = np.clip(img, 0, upper_limit)
    fig = px.imshow(
        img,
        facet_col=1,
        animation_frame=0,
        binary_string=True,
        binary_format="jpg",
    )
    fig.layout.annotations[0]["text"] = "Cell membranes"
    fig.layout.annotations[1]["text"] = "Nuclei"
    plotly.io.show(fig)


def show_sample_images():
    fig, axs = plt.subplots(nrows=3, ncols=3)
    for ax in axs.flat:
        ax.axis("off")
    axs[0, 0].imshow(data.astronaut())
    axs[0, 1].imshow(data.binary_blobs(), cmap=plt.cm.gray)
    axs[0, 2].imshow(data.brick(), cmap=plt.cm.gray)
    axs[1, 0].imshow(data.colorwheel())
    axs[1, 1].imshow(data.camera(), cmap=plt.cm.gray)
    axs[1, 2].imshow(data.cat())
    axs[2, 0].imshow(data.checkerboard(), cmap=plt.cm.gray)
    axs[2, 1].imshow(data.clock(), cmap=plt.cm.gray)
    further_img = np.full((300, 300), 255)
    for xpos in [100, 150, 200]:
        further_img[150 - 10 : 150 + 10, xpos - 10 : xpos + 10] = 0
    axs[2, 2].imshow(further_img, cmap=plt.cm.gray)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)


def plot_img_and_hist(image: np.ndarray, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.axis("off")
    ax_hist: plt.axes._axes.Axes
    ax_cdf: plt.axes._axes.Axes
    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype="stepfilled", color="black")
    ax_hist.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_hist.set_xlabel("Pixel intensity")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, "r")
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def show_contrast_adjustment():
    # Load an example image
    img = data.moon()
    img = (img / 255).astype(float)

    # Gamma
    gamma_corrected = exposure.adjust_gamma(img, 2)

    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img, 1)

    # Display results
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(8, 12))
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:2, 0])
    ax_img.set_title("Low contrast image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:2, 1])
    ax_img.set_title("Gamma correction")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:2, 2])
    ax_img.set_title("Logarithmic correction")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    p2, p98 = np.percentile(img, (0, 98))
    contrast_stretched = exposure.rescale_intensity(img, in_range=(p2, p98))
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(contrast_stretched, axes[2:4, 0])
    ax_img.set_title("Contrast stretching")

    equilized = exposure.equalize_hist(img)
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(equilized, axes[2:4, 1])
    ax_img.set_title("Histogram equalization")

    adaptive_equ = exposure.equalize_adapthist(img, clip_limit=0.03)
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(adaptive_equ, axes[2:4, 2])
    ax_img.set_title("Adaptive equalization")



    #Local Equalization
    footprint = disk(30)
    img_eq = rank.equalize(img, footprint=footprint)
    img_eq = img_eq / 255  # rank.equalize returns an uint8 image
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[4:6, 0])
    ax_img.set_title("Local Equalization")


    brain_img = data.brain() #10 layers of a brain scan
    global_eq = exposure.equalize_hist((brain_img / 255).astype(float))
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(global_eq[0], axes[4:6, 1])
    ax_img.set_title("Brain scan Global Equalization")

    local_eq = rank.equalize(brain_img / 255, footprint=ball(5))
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(local_eq[0], axes[4:6, 2])
    ax_img.set_title("Brain Local Equalization")

    # prevent overlap of y-axis labels
    fig.tight_layout()
    fig.savefig("contrast_adjustment.png")

def hue_adjustment():

    reference = data.astronaut()
    image = data.coffee()
    matched = exposure.match_histograms(image, reference, channel_axis=-1)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
    axes[0].imshow(reference)
    axes[1].imshow(image)
    axes[2].imshow(matched)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("hue_adjustment.png")

def colorize(image, hue, saturation=1):
    """Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)



@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

def as_gray(image_filter, image, *args, **kwargs):
    gray_image = color.rgb2gray(image)
    return image_filter(gray_image, *args, **kwargs)

@adapt_rgb(as_gray)
def sobel_gray(image):
    return filters.sobel(image)

def show_sobel_masks():
    image = data.astronaut()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
    axes[0].imshow(1-sobel_hsv(image))
    axes[1].imshow(1-sobel_gray(image), cmap=plt.cm.gray)
    axes[2].imshow(1-sobel_each(image))
    fig.savefig("sobel_masks.png")

def color_deconvolution():
    # Example of color deconvolution
    from skimage import data
    from skimage.color import rgb2hed, hed2rgb, separate_stains

    image = data.immunohistochemistry()
    ihc = rgb2hed(image)
    null = np.zeros_like(ihc[:, :, 0])
    h, e, d = [hed2rgb(np.roll(np.stack((ihc[:, :, i], null, null), axis=-1), i)) for i in range(3)]



    ax = plt.subplot(2, 2, 1)
    ax.imshow(ihc)
    ax.set_title("IHC")

    ax = plt.subplot(2, 2, 2)
    ax.imshow(h)
    ax.set_title("Hematoxylin")
    ax = plt.subplot(2, 2, 3)
    ax.imshow(e)
    ax.set_title("Eosin")           

    ax = plt.subplot(2, 2, 4)  
    ax.imshow(d)
    ax.set_title("DAB")      




    plt.tight_layout()
    plt.savefig("color_deconvolution.png")

def find_countours():
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage import measure 
    from matplotlib.markers import MarkerStyle

    # works with marching squares algorithm

    # Construct some test data
    x, y = np.ogrid[-np.pi : np.pi : 100j, -np.pi : np.pi : 100j]
    r = np.sin(np.exp(np.sin(x) ** 3 + np.cos(y) ** 2))

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(r, 0.7)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(r, cmap=plt.cm.gray)

    for i in range(1, 10):
        contours = measure.find_contours(r, i/10)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, marker= MarkerStyle.filled_markers[i])

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("find_countours.png")




# show_contrast_adjustment()
# show_sobel_masks()
# color_deconvolution()
find_countours()

