# Non-uniform Savitzky-Golay filter

Repository that extends the Savitzky-Golay filter (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) implemented by [scipy.signal.savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) to a dataset with non-uniform spacing.

This is a slightly improved version of the code exposed in https://dsp.stackexchange.com/a/64313.

The borders are interpolated like [scipy.signal.savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) would do.
