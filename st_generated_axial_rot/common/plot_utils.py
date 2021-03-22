"""This module provides utilities that will be broadly used by code that performs plotting."""

from typing import List, Dict, Any, Tuple, Union
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
from biokinepy.np_utils import find_runs
from matplotlib import rcParams
from matplotlib import font_manager
import mplcursors
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from spm1d.stats._spm import _SPM0Dinference


def init_graphing(backend) -> None:
    """Specify the default font and backend for Matplotlib."""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    matplotlib.use(backend)


def make_interactive(multiple: bool = True) -> None:
    """Enable interaction with the current Matplotlib figure."""
    mplcursors.cursor(multiple=multiple)


def update_xticks(ax: matplotlib.axes.Axes, font_size: float = 12, font_weight: str = 'bold') -> None:
    """Stylize x-ticks."""
    ax.xaxis.set_tick_params(width=2)
    fp = font_manager.FontProperties(weight=font_weight, size=font_size)
    for label in ax.get_xticklabels():
        label.set_fontproperties(fp)


def update_yticks(ax: matplotlib.axes.Axes, fontsize: float = 12, fontweight: str = 'bold') -> None:
    """Stylize y-ticks."""
    ax.yaxis.set_tick_params(width=2, pad=0.1)
    fp = font_manager.FontProperties(weight=fontweight, size=fontsize)
    for label in ax.get_yticklabels():
        label.set_fontproperties(fp)


def update_spines(ax: matplotlib.axes.Axes, line_width: float = 2) -> None:
    """Stylizes spines for a single axes."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def update_spines_add_right(ax: matplotlib.axes.Axes, line_width: float = 2):
    """Stylizes spines for the right y-axis in an axes with two y-axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_linewidth(line_width)


def update_spines_right(ax: matplotlib.axes.Axes, line_width: float = 2):
    """Stylizes spines for the right y-axis in an axes with one y-axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_linewidth(line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def update_ylabel(ax: matplotlib.axes.Axes, ylabel_text: str, font_size: float = 12, font_weight: str = 'bold',
                  labelpad: float = 2) -> None:
    """Set and stylize y-label."""
    ax.set_ylabel(ylabel_text, fontdict=dict(fontsize=font_size, fontweight=font_weight), labelpad=labelpad)


def update_xlabel(ax: matplotlib.axes.Axes, xlabel_text: str, font_size: float = 12, font_weight: str = 'bold') -> None:
    """Set and stylize y-label."""
    ax.set_xlabel(xlabel_text, fontdict=dict(fontsize=font_size, fontweight=font_weight), labelpad=2)


def update_title(ax: matplotlib.axes.Axes, title_text: str, font_size: float = 12, font_weight: str = 'bold',
                 pad: float = -8) -> None:
    """Set and stylize title."""
    ax.set_title(title_text, fontdict=dict(fontsize=font_size, fontweight=font_weight), pad=pad)


def mean_sd_plot(ax: matplotlib.axes.Axes, x: np.ndarray, mean: np.ndarray, sd: np.ndarray,
                 shaded_kwargs: Dict[str, Any], line_kwargs: Dict[str, Any]) -> List[matplotlib.lines.Line2D]:
    """Create mean +- sd plot."""
    ax.fill_between(x, mean - sd, mean + sd, **shaded_kwargs)
    return ax.plot(x, mean, **line_kwargs)


def quantile_plot(ax: matplotlib.axes.Axes, x: np.ndarray, median: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  shaded_kwargs: Dict[str, Any], line_kwargs: Dict[str, Any]) -> List[matplotlib.lines.Line2D]:
    """Create mean +- sd plot."""
    ax.fill_between(x, lower, upper, **shaded_kwargs)
    return ax.plot(x, median, **line_kwargs)


def spm_plot(ax: matplotlib.axes.Axes, x: np.ndarray, spm_test: _SPM0Dinference, shaded_kwargs: Dict[str, Any],
             line_kwargs: Dict[str, Any]) -> List[matplotlib.lines.Line2D]:
    """Create SPM plot."""
    ax.axhline(spm_test.zstar, ls='--', color='grey')
    ax.axhline(-spm_test.zstar, ls='--', color='grey')
    ax.fill_between(x, spm_test.zstar, spm_test.z, where=(spm_test.z > spm_test.zstar), **shaded_kwargs)
    ax.fill_between(x, -spm_test.zstar, spm_test.z, where=(spm_test.z < -spm_test.zstar), **shaded_kwargs)
    return ax.plot(x, spm_test.z, **line_kwargs)


def spm_plot_alpha(ax: matplotlib.axes.Axes, x: np.ndarray, spm_test: _SPM0Dinference, shaded_kwargs: Dict[str, Any],
                   line_kwargs: Dict[str, Any]) -> Tuple[List[matplotlib.lines.Line2D], List[matplotlib.lines.Line2D]]:
    """Create SPM plot."""
    ax.axhline(spm_test.zstar, ls='--', color='grey')
    ax.axhline(-spm_test.zstar, ls='--', color='grey')
    alpha_line = ax.axhline(spm_test.zstar, ls='--', color=line_kwargs['color'], alpha=0.5)
    ax.axhline(-spm_test.zstar, ls='--', color=line_kwargs['color'], alpha=0.5)
    ax.fill_between(x, spm_test.zstar, spm_test.z, where=(spm_test.z > spm_test.zstar), **shaded_kwargs)
    ax.fill_between(x, -spm_test.zstar, spm_test.z, where=(spm_test.z < -spm_test.zstar), **shaded_kwargs)
    return ax.plot(x, spm_test.z, **line_kwargs), alpha_line


def style_figure(fig: matplotlib.figure.Figure, title: str, bottom: float = 0.2) -> None:
    """Stylize the supplied matplotlib Figure instance."""
    fig.tight_layout()
    if bottom is not None:
        fig.subplots_adjust(bottom=bottom)
    fig.suptitle(title)
    fig.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left')


def style_axes(ax: matplotlib.axes.Axes, x_label: Union[str, None], y_label: Union[str, None]):
    """Stylize the supplied matplotlib Axes instance."""
    update_spines(ax)
    update_xticks(ax, font_size=8)
    update_yticks(ax, fontsize=8)
    if x_label:
        update_xlabel(ax, x_label, font_size=10)
    if y_label:
        update_ylabel(ax, y_label, font_size=10)


def style_axes_right(ax: matplotlib.axes.Axes, x_label: Union[str, None], y_label: Union[str, None]):
    """Stylize the supplied matplotlib Axes instance."""
    update_spines_right(ax)
    update_xticks(ax, font_size=8)
    update_yticks(ax, fontsize=8)
    if x_label:
        update_xlabel(ax, x_label, font_size=10)
    if y_label:
        update_ylabel(ax, y_label, font_size=10)


def style_axes_add_right(ax: matplotlib.axes.Axes, y_label: Union[str, None]):
    """Stylize the supplied matplotlib Axes instance."""
    update_spines_add_right(ax)
    update_yticks(ax, fontsize=8)
    if y_label:
        update_ylabel(ax, y_label, font_size=10)


def sig_string(x: np.ndarray, sig: np.ndarray) -> str:
    """Create a string describing signifiance based on the domain x and boolean vector of signifiance (sig)."""
    run_values, run_starts, run_lengths = find_runs(sig)
    run_starts_sign = run_starts[run_values]
    run_lengths_sign = run_lengths[run_values]
    sec = []
    for run, length in zip(run_starts_sign, run_lengths_sign):
        sec.append('{:.2f} - {:.2f}'.format(x[run], x[run + length - 1]))
    return '\n'.join(sec)


def extract_sig(spm_test: _SPM0Dinference, x: np.ndarray) -> str:
    """Create a string describing where spm_test is significant in the domain of x."""
    sig = np.logical_or(spm_test.z < -spm_test.zstar, spm_test.z > spm_test.zstar)
    return sig_string(x, sig)


def extract_non_sig(spm_test: _SPM0Dinference, x: np.ndarray) -> str:
    """Create a string describing where spm_test is NOT significant in the domain of x."""
    non_sig = np.logical_and(spm_test.z > -spm_test.zstar, spm_test.z < spm_test.zstar)
    return sig_string(x, non_sig)


def extract_sig_gt(spm_test: _SPM0Dinference, x: np.ndarray) -> str:
    """Create a string describing where spm_test is significant in the domain of x but only when z > z*."""
    sig = spm_test.z > spm_test.zstar
    return sig_string(x, sig)


def extract_sig_lt(spm_test: _SPM0Dinference, x: np.ndarray) -> str:
    """Create a string describing where spm_test is significant in the domain of x but only when z < -z*."""
    sig = spm_test.z < -spm_test.zstar
    return sig_string(x, sig)


def sig_filter(spm_test: _SPM0Dinference, x: np.ndarray) -> np.ndarray:
    """Return a vector which has NaNs everywhere apart where spm_test is significant in the domain of x."""
    x_ret = np.copy(x)
    x_ret[np.logical_and(spm_test.z > -spm_test.zstar, spm_test.z < spm_test.zstar)] = np.nan
    return x_ret


def non_sig_filter(spm_test: _SPM0Dinference, x: np.ndarray) -> np.ndarray:
    """Return a vector which has NaNs where spm_test is significant in the domain of x."""
    x_ret = np.copy(x)
    x_ret[np.logical_or(spm_test.z < -spm_test.zstar, spm_test.z > spm_test.zstar)] = np.nan
    return x_ret


def sig_filter_gt(spm_test: _SPM0Dinference, x: np.ndarray) -> np.ndarray:
    """Return a vector which has NaNs everywhere apart where spm_test is significant in the domain of x and z > z*."""
    x_ret = np.copy(x)
    x_ret[spm_test.z < spm_test.zstar] = np.nan
    return x_ret


def sig_filter_lt(spm_test: _SPM0Dinference, x: np.ndarray) -> np.ndarray:
    """Return a vector which has NaNs everywhere apart where spm_test is significant in the domain of x and z < -z*."""
    x_ret = np.copy(x)
    x_ret[spm_test.z > -spm_test.zstar] = np.nan
    return x_ret


def output_spm_p(spm_test: _SPM0Dinference):
    """Output SPM cluster p-values."""
    ret_str = ''
    for idx, cluster in enumerate(spm_test.clusters):
        ret_str += 'Cluster {}: {:.5f}, {:.5f} '.format(idx + 1, cluster.P, cluster.threshold)
    if hasattr(spm_test, 'p_set'):
        ret_str += '\nSet-Level: {:.5f}'.format(spm_test.p_set)
    return ret_str


def retrieve_bp_stats(bp: Dict[str, List]) -> Tuple[float, float, float, float, float]:
    """Return min, 25%, median, 75%, and max given a boxplot."""
    def get_bp_limits(bp):
        min_max = np.zeros((len(bp['fliers']), 3))
        for idx, (flier, lWhisker, uWhisker) in enumerate(
                zip(bp['fliers'], bp['whiskers'][0::2], bp['whiskers'][1::2])):
            x_data = np.append(flier.get_xdata(), np.append(lWhisker.get_xdata(), uWhisker.get_xdata()))
            unique_x = np.unique(x_data.round(decimals=4))
            assert (len(unique_x) == 1)
            y_data = np.append(flier.get_ydata(), np.append(lWhisker.get_ydata(), uWhisker.get_ydata()))
            min_max[idx, 0] = unique_x
            min_max[idx, 1] = min(y_data)
            min_max[idx, 2] = max(y_data)
        return min_max

    bp_lims = get_bp_limits(bp)
    return (bp_lims[0, 1], bp['boxes'][0].get_ydata()[0], bp['medians'][0].get_ydata()[0],
            bp['boxes'][0].get_ydata()[2], bp_lims[0, 2])


# from https://stackoverflow.com/a/31548752/2577053
class HandlerTupleVertical(HandlerTuple):
    def __init__(self, **kwargs):
        HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # How many lines are there.
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()

        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        height_y = (height / numlines)

        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)

            legline = handler.create_artists(legend, handle,
                                             xdescent,
                                             (2*i + 1)*height_y,
                                             width,
                                             2*height,
                                             fontsize, trans)
            leglines.extend(legline)

        return leglines
