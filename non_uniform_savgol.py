import numpy as np
import warnings


def check_input(x: np.ndarray, y: np.ndarray, window_size: int, fit_polynom_degree: int = 2) -> None:
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window_size:
        raise ValueError('The data size must be larger than the window size')

    if type(window_size) is not int:
        raise TypeError('"window" must be an integer')

    if window_size % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(fit_polynom_degree) is not int:
        raise TypeError('"polynom" must be an integer')

    if fit_polynom_degree >= window_size:
        raise ValueError('"polynom" must be less than "window"')


def non_uniform_savgol(x: np.ndarray, y: np.ndarray, window_size: int, fit_polynom_degree: int = 2) -> np.array:
    """
    Applies a Savitzky-Golay filter (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) to y with non-uniform spacing
    as defined in x.

    This is a slightly improved version of the code exposed in https://dsp.stackexchange.com/a/64313.

    The borders are interpolated like scipy.signal.savgol_filter would do.

    Parameters
    ----------
    x : np.ndarray
        Floats representing the x values of the data
    y : np.ndarray
        Floats representing the y values. Must have same length as x
    window : int
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array
        The smoothed y values
    """

    check_input(x, y, window_size, fit_polynom_degree)

    half_window_size: int = window_size // 2
    fit_polynom_degree += 1

    # Initialize variables
    A = np.empty((window_size, fit_polynom_degree))     # Matrix
    tA = np.empty((fit_polynom_degree, window_size))    # Transposed matrix
    t = np.empty(window_size)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window_size, len(x) - half_window_size, 1):
        # Center a window of x values on x[i]
        for j in range(0, window_size, 1):
            t[j] = x[i + j - half_window_size] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window_size, 1):
            r = 1.0
            for k in range(0, fit_polynom_degree, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        try:
            tAA = np.linalg.inv(tAA)
        except np.linalg.LinAlgError:
            warnings.warn("""
            Line not smoothed. Check if there are repeated x values in the input data
            """)
            return y

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window_size, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window_size]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window_size:
            first_coeffs = np.zeros(fit_polynom_degree)
            for j in range(0, window_size, 1):
                for k in range(fit_polynom_degree):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window_size - 1:
            last_coeffs = np.zeros(fit_polynom_degree)
            for j in range(0, window_size, 1):
                for k in range(fit_polynom_degree):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window_size + j]

    # Interpolate the result at the left border
    for i in range(0, half_window_size, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, fit_polynom_degree, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window_size]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window_size, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, fit_polynom_degree, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window_size - 1]

    return y_smoothed
