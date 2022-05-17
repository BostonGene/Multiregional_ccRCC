import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from functions.utils import item_series, to_common_samples


def axis_net(x, y, title='', x_len=4, y_len=4, title_y=1, gridspec_kw=None):
    """
    Return an axis iterative for subplots arranged in a net
    :param x: int, number of subplots in a row
    :param y: int, number of subplots in a column
    :param title: str, plot title
    :param x_len: float, width of a subplot in inches
    :param y_len: float, height of a subplot in inches
    :param gridspec_kw: is used to specify axis ner with different rows/cols sizes.
            A dict: height_ratios -> list + width_ratios -> list
    :param title_y: absolute y position for suptitle
    :return: axs.flat, numpy.flatiter object which consists of axes (for further plots)
    """
    if x == y == 1:
        fig, ax = plt.subplots(figsize=(x * x_len, y * y_len))
        af = ax
    else:
        fig, axs = plt.subplots(y, x, figsize=(x * x_len, y * y_len), gridspec_kw=gridspec_kw)
        af = axs.flat

    fig.suptitle(title, y=title_y)
    return af


def lin_colors(factors_vector, cmap='default', sort=True, min_v=0, max_v=1, linspace=True):
    """
    Return dictionary of unique features of "factors_vector" as keys and color hexes as entries
    :param factors_vector: pd.Series
    :param cmap: matplotlib.colors.LinearSegmentedColormap, which colormap to base the returned dictionary on
        default - matplotlib.cmap.hsv with min_v=0, max_v=.8, lighten_color=.9
    :param sort: bool, whether to sort the unique features
    :param min_v: float, for continuous palette - minimum number to choose colors from
    :param max_v: float, for continuous palette - maximum number to choose colors from
    :param linspace: bool, whether to spread the colors from "min_v" to "max_v"
        linspace=False can be used only in discrete cmaps
    :return: dict
    """

    unique_factors = factors_vector.dropna().unique()
    if sort:
        unique_factors = np.sort(unique_factors)

    if cmap == 'default':
        cmap = matplotlib.cm.rainbow
        max_v = .92

    if linspace:
        cmap_colors = cmap(np.linspace(min_v, max_v, len(unique_factors)))
    else:
        cmap_colors = np.array(cmap.colors[:len(unique_factors)])

    return dict(list(zip(unique_factors, [matplotlib.colors.to_hex(x) for x in cmap_colors])))


def patch_plot(patches, ax=None, order='sort', w=0.25, h=0, legend_right=True,
               show_ticks=False):
    cur_patches = pd.Series(patches)

    if order == 'sort':
        order = list(np.sort(cur_patches.index))

    data = pd.Series([1] * len(order), index=order[::-1])
    if ax is None:
        if h == 0:
            h = 0.3 * len(patches)
        _, ax = plt.subplots(figsize=(w, h))

    data.plot(kind='barh', color=[cur_patches[x] for x in data.index], width=1, ax=ax)
    ax.set_xticks([])
    if legend_right:
        ax.yaxis.tick_right()

    sns.despine(offset={'left': -2}, ax=ax)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not show_ticks:
        ax.tick_params(length=0)

    return ax


def simple_palette_scatter(
    x: pd.Series,
    y: pd.Series,
    grouping: pd.Series,
    palette = None,
    order = None,
    legend: bool = 'out',
    patch_size: int = 10,
    centroid_complement_color: bool = True,
    ax = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot a scatter for 2 vectors, coloring by grouping.
    Only samples with common indexes are plotted.

    See Also
    --------------------
    plotting.simple_scatter

    :param x: pd.Series, numerical values
    :param y: pd.Series, numerical values
    :param grouping: pd.Series, which group each sample belongs to
    :param palette: dict, palette for plotting. Keys are unique values from groups, entries are color hexes
    :param order: list, order to plot the entries in. Contains ordered unique values from grouping
    :param legend: bool, whether to plot legend
    :param patch_size: float, size of legend
    :param centroid_complement_color: bool, whether to plot centroids in complement color
    :param ax: plt.axes to plot on
    :param kwargs:
    :return: matplotlib axis
    """

    if palette is None:
        palette = lin_colors(grouping)

    if order is None:
        order = np.sort(list(palette.keys()))

    c_grouping, c_x, c_y = to_common_samples([grouping, x, y])

    patch_location = 2
    if 'loc' in kwargs:
        patch_location = kwargs.pop('loc')

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(kwargs.get('figsize', (4, 4))))

    kwargs['marker'] = kwargs.get('marker', 'o')
    kwargs['edgecolor'] = kwargs.get('edgecolor', 'black')
    kwargs['linewidth'] = kwargs.get('linewidth', 0)

    for label in order:
        samps = c_grouping[c_grouping == label].index
        simple_scatter(c_x[samps], c_y[samps], color=palette[label], ax=ax, **kwargs)
        handles = [mpatches.Patch(color=palette[label], label=label) for label in order]

    
    if legend:
        ax.legend(
            bbox_to_anchor=(1, 1) if legend == 'out' else None,
            handles=handles,
            loc=patch_location,
            prop={'size': patch_size},
            borderaxespad=0.1,
        )

    return ax



def simple_scatter(x, y, ax=None, title='', color='b', figsize=(5, 5), s=20, **kwargs):
    """
    Plot a scatter for 2 vectors. Only samples with common indexes are plotted.
    If color is a pd.Series - it will be used to color the dots
    :param x: pd.Series, numerical values
    :param y: pd.Series, numerical values
    :param ax: matplotlib axis, axis to plot on
    :param title: str, plot title
    :param color: str, color to use for points
    :param figsize: (float, float), figure size in inches
    :param s: float, size of points
    :param alpha: float, alpha of points
    :param marker: str, marker to use for points
    :param linewidth: float, width of marker borders
    :param edgecolor: str, color of marker borders
    :return: matplotlib axis
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    try:
        c_x, c_y, c_color = to_common_samples([x, y, color])
    except Exception:
        c_x, c_y = to_common_samples([x, y])
        c_color = color

    ax.set_title(title)

    ax.scatter(c_x, c_y, color=c_color, s=s, **kwargs)

    if hasattr(x, 'name'):
        ax.set_xlabel(x.name)
    if hasattr(y, 'name'):
        ax.set_ylabel(y.name)

    return ax



def axis_matras(ys, title='', x_len=8, title_y=1, sharex=True):
    """
    Return an axis iterative for subplots stacked vertically
    :param ys: list, list of lengths by 'y'
    :param title: str, title for plot
    :param x_len: int, length by 'x'
    :param sharex: boolean, images will be shared if True
    :param title_y: absolute y position for suptitle
    :return: axs.flat, numpy.flatiter object which consists of axes (for further plots)
    """
    fig, axs = plt.subplots(len(ys), 1, figsize=(x_len, np.sum(ys)), gridspec_kw={'height_ratios': ys}, sharex=sharex)
    fig.suptitle(title, y=title_y)

    for ax in axs:
        ax.tick_params(axis='x', which='minor', length=0)

    return axs.flat


def bot_bar_plot(data, palette=None, lrot=0, figsize=(5, 5), title='', ax=None, order=None, stars=False, percent=False,
    pvalue=False, p_digits=5, legend=True, xl=True, offset=-0.1, linewidth=0, align='center', bar_width=0.9,
    edgecolor=None, hide_grid=True, draw_horizontal=False, plot_all_borders=True):
    """
    Plot a stacked bar plot based on contingency table
    :param data: pd.DataFrame, contingency table for plotting. Each element of index corresponds to a bar.
    :param palette: dict, palette for plotting. Keys are unique values from groups, entries are color hexes
    :param lrot: float, rotation angle of bar labels in degrees
    :param figsize: (float, float), figure size in inches
    :param title: str, plot title
    :param ax: matplotlib axis, axis to plot on
    :param order: list, what order to plot the stacks of each bar in. Contains column labels of "data"
    :param stars: bool, whether to use the star notation for p value instead of numerical value
    :param percent: bool, whether to normalize each bar to 1
    :param pvalue: bool, whether to add the p value (chi2 contingency test) to the plot title.
    :param p_digits: int, number of digits to round the p value to
    :param legend: bool, whether to plot the legend
    :param xl: bool, whether to plot bar labels (on x axis for horizontal plot, on y axis for vertical plot)
    :param plot_all_borders: bool, whether to plot top and right border
    :return: matplotlib axis
    """
    from matplotlib.ticker import FuncFormatter

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if pvalue:
        from scipy.stats import chi2_contingency

        chi2_test_data = chi2_contingency(data)
        p = chi2_test_data[1]
        if title is not False:
            title += '\n' + str(p)

    if percent:
        c_data = data.apply(lambda x: x * 1.0 / x.sum(), axis=1)
        if title:
            title = '% ' + title
        ax.set_ylim(0, 1)
    else:
        c_data = data

    c_data.columns = [str(x) for x in c_data.columns]

    if order is None:
        order = c_data.columns
    else:
        order = [str(x) for x in order]

    if palette is None:
        c_palette = lin_colors(pd.Series(order))

        if len(order) == 1:
            c_palette = {order[0]: blue_color}
    else:
        c_palette = {str(k): v for k, v in palette.items()}

    if edgecolor is not None:
        edgecolor = [edgecolor] * len(c_data)


    c_data[order].plot(kind='bar', stacked=True, position=offset, width=bar_width,
                       color=pd.Series(order).map(c_palette).values, ax=ax, linewidth=linewidth,
                       align=align, edgecolor=edgecolor)

    if legend:
        ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
    else:
        ax.legend_.remove()
        
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    return ax
