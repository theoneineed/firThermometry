'''
This module stores the functions used for the FIR based optical thermometry workflow.
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import pandas as pd




def getBaselineCountOfManifold(arr2D, lambdaStart, lambdaEnd):
    """
    Deprecated, use getBaselineCountSpectrum instead: Retrieves the lowest photon count for each temperature within a specified wavelength range.
 
    This function extracts the minimum photon count across all wavelengths within 
    the [lambdaStart, lambdaEnd] range for each temperature, serving as a baseline 
    for photon count data in manifold regions.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
        wavelength (first element) across varying temperatures.
        lambdaStart (float): The starting wavelength for search (in nm).
        lambdaEnd (float): The ending wavelength for search (in nm).
 
    Returns:
        numpy.ndarray: A 1D array containing the lowest photon count for each temperature 
            within the [lambdaStart, lambdaEnd] range.
    """

    warnings.warn(
        "getBaselineCountOfManifold() is deprecated as it uses the minimum of the manifold which is not the best approach. "
        "Use getBaselineCountSpectrum() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return [min(arr2D[lambdaStart - 250 : lambdaEnd - 250 + 1, i]) for i in range(1, len(arr2D[0]))]





def getBaselineCountSpectrum(arr2D, startLambdaFlatRegion, endLambdaFlatRegion):
    """
    Computes the average photon count for each temperature within a flat spectral region.
 
    This function averages photon counts across the flat (featureless) region of the spectrum
    to establish a baseline for subsequent background subtraction in spectral analysis.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
            wavelength (first element) across varying temperatures.
        startLambdaFlatRegion (float): The starting wavelength of the flat region to use 
            as baseline (in nm).
        endLambdaFlatRegion (float): The ending wavelength of the flat region to use 
            as baseline (in nm).
 
    Returns:
        numpy.ndarray: A 1D array containing the average photon count for each temperature 
            within the flat region defined by the given wavelength range.
    """

    try:
        startIdx = np.where(arr2D[:, 0] == startLambdaFlatRegion)[0][0]
        endIdx   = np.where(arr2D[:, 0] == endLambdaFlatRegion)[0][0]

    except IndexError:
        raise ValueError(
            f"One or both wavelengths ({startLambdaFlatRegion}, "
            f"{endLambdaFlatRegion}) were not found in the wavelength axis."
        )

    return [np.mean(arr2D[startIdx:endIdx + 1, i]) for i in range(1, arr2D.shape[1])]




def getBaselineFittedCountOfManifold(arr2D, lambdaStart, lambdaEnd):
    """
    Deprecated, use getBaselineFittedCountSpectrum instead: Computes baseline-subtracted integrated photon counts within a spectral manifold with peaks.
 
    Integrates photon counts across the specified wavelength range for each temperature 
    and subtracts the minimum count value (baseline) determined internally from the range. 
    Designed for analyzing spectral regions containing emission peaks or features.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
            wavelength (first element) across varying temperatures.
        lambdaStart (float): The starting wavelength for integration (in nm).
        lambdaEnd (float): The ending wavelength for integration (in nm).
 
    Returns:
        numpy.ndarray: A 1D array containing the integrated photon counts minus the baseline 
            (minimum count in the range) for each temperature within the [lambdaStart, lambdaEnd] range.
    """

    warnings.warn(
        "getBaselineFittedCountOfManifold() is deprecated as it uses the minimum of the manifold which is not the best approach. "
        "Use getBaselineFittedCountSpectrum() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return getTotalCount(arr2D, lambdaStart, lambdaEnd) - getBaselineCountOfManifold(arr2D, lambdaStart, lambdaEnd)




def getBaselineFittedCountSpectrum(arr2D, lambdaStart, lambdaEnd, baseCountArr):
    """
    Computes baseline-subtracted integrated photon counts within a specified wavelength range.
 
    Integrates photon counts across the specified wavelength range for each temperature 
    and subtracts the corresponding baseline value. Used for correcting spectral data 
    for background contributions.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
            wavelength (first element) across varying temperatures.
        lambdaStart (float): The starting wavelength for integration (in nm).
        lambdaEnd (float): The ending wavelength for integration (in nm).
        baseCountArr (numpy.ndarray): The baseline count array obtained from getBaselineCountSpectrum or any other function
            (1D array with one value per temperature).
 
    Returns:
        numpy.ndarray: A 1D array containing the integrated photon counts minus the baseline 
            for each temperature within the [lambdaStart, lambdaEnd] range.
    """

    tempArr = getTotalCount(arr2D, lambdaStart, lambdaEnd)
    assert len(tempArr) == len(baseCountArr)
    return tempArr - baseCountArr













def getErrorsOfFit(yData, yFit):
    """
    Computes goodness-of-fit metrics comparing observed and fitted data.
 
    Calculates multiple error metrics to quantify how well the fitted model describes 
    the observed data, including both absolute and relative measures.
 
    Args:
        yData (numpy.ndarray): 1D array of observed data values.
        yFit (numpy.ndarray): 1D array of fitted/predicted data values (same length as yData).
 
    Returns:
        tuple: A tuple containing four metrics:
            - MSE (float): Mean Squared Error.
            - RMSE (float): Root Mean Squared Error.
            - MAE (float): Mean Absolute Error.
            - R_squared (float): Coefficient of determination (R²), ranging from -∞ to 1.
    """

    import numpy as np
    assert len(yData) == len(yFit)
    yMean   = np.mean(yData)

    MSE     = np.sum((yData - yFit)**2)/len(yData)
    RMSE    = np.sqrt(MSE)
    MAE     = np.sum(np.abs(yData - yFit))/len(yData)
    R2      = 1 - np.sum((yData - yFit)**2)/np.sum((yData - yMean)**2)

    return MSE, RMSE, MAE, R2




def getFIR(arr2D, lambdaStart1, lambdaEnd1, lambdaStart2, lambdaEnd2, baseCountArr):
    """
    Computes the Fluorescence Intensity Ratio (FIR) based on two wavelength intervals.
 
    Calculates the FIR as the ratio of baseline-subtracted integrated photon counts 
    in two non-overlapping wavelength ranges. The FIR is commonly used as a temperature-dependent 
    parameter in fluorescence thermometry.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
            wavelength (first element) across varying temperatures.
        lambdaStart1 (float): The starting wavelength for the first range (in nm).
        lambdaEnd1 (float): The ending wavelength for the first range (in nm).
        lambdaStart2 (float): The starting wavelength for the second range (in nm).
        lambdaEnd2 (float): The ending wavelength for the second range (in nm).
        baseCountArr (numpy.ndarray): The baseline count array obtained from getBaselineCountSpectrum or any other functions
            (1D array with one value per temperature).
 
    Returns:
        numpy.ndarray: A 1D array containing the FIR values (ratio of integrated counts) 
            for each temperature, calculated as:
            FIR = (integral in [lambdaStart1, lambdaEnd1] - baseline) / (integral in [lambdaStart2, lambdaEnd2] - baseline)
    """

    firArr  = getBaselineFittedCountSpectrum(arr2D, lambdaStart1, lambdaEnd1, baseCountArr)/getBaselineFittedCountSpectrum(arr2D, lambdaStart2, lambdaEnd2, baseCountArr)
    
    return firArr





def getSensitivities(tempArr, slope, intercept):
    """
    Computes relative and absolute temperature sensitivities for the FIR-based thermometry model.
 
    Evaluates the sensitivity of the FIR (Fluorescence Intensity Ratio) to temperature changes 
    at each temperature point, based on a linear model: ln(FIR) = slope * (1/T) + intercept.
    Relative sensitivity indicates fractional change in FIR per Kelvin; absolute sensitivity 
    indicates absolute change in FIR per Kelvin.
 
    Args:
        tempArr (numpy.ndarray): 1D array of temperature values (in Kelvin).
        slope (float or numpy.ndarray): The slope of the linear fit in the ln(FIR) vs. 1/T model 
            (from getSlopeAndIntercept), or a 1D array of slopes for multiple fits.
        intercept (float or numpy.ndarray): The y-intercept of the linear fit (from getSlopeAndIntercept), 
            or a 1D array of intercepts. slope and intercept must have the same dimensions.

        slope and intercept must be of same dimensions.
 
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            A tuple containing:
            - relSens2D (numpy.ndarray): Relative sensitivity (dimensionless) evaluated at each temperature.
              Shape: (n_fits,) if slope/intercept are scalars, or (n_fits, n_temperatures) if arrays.
            - absSens2D (numpy.ndarray): Absolute sensitivity (in FIR units per Kelvin) evaluated at each temperature.
              Same shape as relSens2D.
    """

    if isinstance(slope, np.ndarray) and isinstance(intercept, np.ndarray):
        if not len(slope)== len(intercept):
            raise ValueError("slope and intercept arrays must have the same length.")
        else:
            slopeArr = slope
            interceptArr = intercept
    else:
        if isinstance(slope, np.ndarray) or isinstance(intercept, np.ndarray):
            raise ValueError("slope and intercept must either both be scalars or both be NumPy arrays.")
        else:
            slopeArr = np.array([slope])
            interceptArr = np.array([intercept])
    
    relSens2D = np.empty((len(slopeArr), len(tempArr)))
    absSens2D = np.empty((len(slopeArr), len(tempArr)))

    for idx, (currSlope, currIntercept) in enumerate(zip(slopeArr, interceptArr)):
        relSens2D[idx, :] = - currSlope/tempArr ** 2 #because the slope is -deltaE/k and Sr is deltaE/(kT^2)
        absSens2D[idx, :] = (np.exp(currSlope/tempArr) * (- currSlope/tempArr ** 2)) * np.exp(currIntercept)

    return relSens2D, absSens2D






def getSlopeAndIntercept(inverseTempArr, logOfFir):
    """
    Fits a linear model to inverse temperature and log(FIR) data.
 
    Performs least-squares linear regression on ln(FIR) vs. 1/T data to extract 
    the calibration parameters for FIR-based thermometry.
 
    Args:
        inverseTempArr (numpy.ndarray): 1D array of inverse temperature values 
            (in K^{-1}, e.g., 1/T).
        logOfFir (numpy.ndarray): 1D array of natural logarithm of FIR values 
            (must be same length as inverseTempArr).
 
    Returns:
        tuple: A tuple containing:
            - slope (float): The slope of the best-fit line (dimensionless).
            - intercept (float): The y-intercept of the best-fit line (dimensionless).
    """

    import numpy as np
    slope, intercept    = np.polyfit(inverseTempArr, logOfFir, 1)
    return slope, intercept






# this assumes that the separation is 1 nm
def getTotalCount(arr2D, lambdaStart, lambdaEnd):
    """
    Integrates photon counts within a specified wavelength range for each temperature.
 
    Sums all photon counts across wavelengths in the [lambdaStart, lambdaEnd] range 
    for each temperature.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
            wavelength (first element) across varying temperatures.
        lambdaStart (float): The starting wavelength for integration (in nm).
        lambdaEnd (float): The ending wavelength for integration (in nm).
 
    Returns:
        numpy.ndarray: A 1D array containing the total integrated photon count for each 
            temperature within the [lambdaStart, lambdaEnd] range.

    """
    wavelengths_all = arr2D[:, 0] 
    if lambdaStart > lambdaEnd: 
        raise ValueError( f"lambdaStart ({lambdaStart}) must be <= lambdaEnd ({lambdaEnd})." ) 

    start_matches = np.where(np.isclose(wavelengths_all, lambdaStart))[0] 
    end_matches = np.where(np.isclose(wavelengths_all, lambdaEnd))[0] 

    if len(start_matches) == 0 or len(end_matches) == 0: 
        raise ValueError( f"One or both wavelengths ({lambdaStart}, {lambdaEnd}) " "were not found in the wavelength axis." ) 

    startIdx = start_matches[0] 
    endIdx = end_matches[0] 

    wavelengths = wavelengths_all[startIdx:endIdx + 1] 
    spectra = arr2D[startIdx:endIdx + 1, 1:] 

    return np.trapezoid(spectra, x=wavelengths, axis=0)





def graphCountsVsWavelength(
    arr2D,
    tempArr,
    usePlotly=False,
    fromTempIdx=0,
    toTempIdx=-1,
    saveGraph=False,
    figSize=(20, 5),
    title="Emission spectra (wavelength in nm, temperature in Celsius)"
):
    """
    Plots photon counts against wavelength for multiple temperatures.
 
    Generates a 2D line plot showing emission spectra for a subset of temperatures. 
    Each line represents the photon count spectrum for a single temperature. 
    
    Use plotly for interactive experience.

    Args:
        arr2D (numpy.ndarray):
            2D array where the first column contains wavelengths and
            the remaining columns contain photon counts for each
            temperature.

        tempArr (numpy.ndarray):
            1D array of temperature values in Celsius corresponding
            to the columns in arr2D (excluding the wavelength column).

        usePlotly (bool, optional):
            If True, uses Plotly for interactive visualization.
            Otherwise uses Matplotlib. Defaults to False.

        fromTempIdx (int, optional):
            Starting temperature index (inclusive). Defaults to 0.

        toTempIdx (int, optional):
            Ending temperature index (exclusive). Use -1 to include
            all remaining temperatures. Defaults to -1.

        saveGraph (bool, optional):
            If True, saves the figure to disk. Defaults to False.

        figSize (tuple, optional):
            Matplotlib figure dimensions (width, height) in inches.
            Defaults to (20, 5).

        title (str, optional):
            Plot title.

    Returns:
        None
    """

    if toTempIdx == -1:
        toTempIdx = len(tempArr)

    wavelengths = arr2D[:, 0]

    if not usePlotly:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figSize, dpi=200)

        for temp_idx in range(fromTempIdx, toTempIdx):
            ax.plot(
                wavelengths,
                arr2D[:, temp_idx + 1],
                label=tempArr[temp_idx]
            )

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Photon counts per second")
        ax.set_title(title)
        ax.legend()

        if saveGraph:
            fig.savefig(
                "spectralLineMPL.png",
                transparent=True,
                bbox_inches="tight"
            )

        plt.show()

    else:
        import plotly.graph_objects as go

        fig = go.Figure()

        for temp_idx in range(fromTempIdx, toTempIdx):
            fig.add_scatter(
                x=wavelengths,
                y=arr2D[:, temp_idx + 1],
                name=str(tempArr[temp_idx])
            )

        fig.update_layout(
            title=title,
            xaxis_title="Wavelength (nm)",
            yaxis_title="Photon counts per second"
        )

        if saveGraph:
            fig.write_image("spectralLinePLT.png")

        fig.show()





def graphCountsVsWavelength3D(arr2D, tempArr, fromTempIdx = 0, toTempIdx = -1, saveGraph = False, figSize = (25, 25), saveFileName = "spectralLineMPL_3D.png"):
    """
    Generates a 3D surface plot of photon counts vs. wavelength and temperature.
 
    Creates a three-dimensional visualization with wavelength and temperature as the 
    independent axes and photon count as the dependent axis. Useful for visualizing 
    spectral evolution across a temperature range.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
            wavelength (first element) across varying temperatures.
        tempArr (numpy.ndarray): 1D array of temperature values corresponding to the columns 
            in arr2D (excluding the first wavelength column).
        fromTempIdx (int, optional): Starting index of temperature to plot. Defaults to 0.
        toTempIdx (int, optional): Ending index of temperature to plot (-1 for last). Defaults to -1.
        saveGraph (bool, optional): If True, saves the figure to disk. Defaults to False.
        figSize (tuple, optional): Figure dimensions (width, height) in inches. Defaults to (25, 25).
        saveFileName (str, optional): Filename for saved figure. Defaults to "spectralLineMPL_3D.png".
 
    Returns:
        None. Displays the plot and optionally saves it.
    """

    if arr2D.shape[1] != len(tempArr) + 1:
        raise ValueError(
            "arr2D must contain one wavelength column plus "
            "one column for each temperature."
        )
    
    if toTempIdx == -1:
        toTempIdx = len(tempArr)

        
    lowerLambda = arr2D[1,  0]
    upperLambda = arr2D[-1, 0]

    lowerTemp   = tempArr[fromTempIdx] + 273.15
    upperTemp   = tempArr[toTempIdx] + 273.15


    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 35, 'font.family':'serif', 'font.serif':'Times New Roman'})

    fig     = plt.figure(figsize=figSize, dpi=200)
    ax      = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=325, roll=0)

    ax.set_box_aspect((2.5, 2.3, 0.8)) # Example: Stretch x and y by a factor of 2

    for temp_idx in range(fromTempIdx, toTempIdx):
        temperature     = tempArr[temp_idx] + 273.15
        counts          = 1e-5 * arr2D[:, temp_idx + 1]

        ax.plot(temperature, arr2D[:, 0], counts, 'b-', alpha=0.8)

    
    ax.grid(False)

    # Manually plot grid lines at the 'floor_z' level
    tempFloor       = np.linspace(int(lowerTemp//10)*10, int(upperTemp//10 + 1)*10, int(upperTemp - lowerTemp)//10 + 1)
    lambdaFloor     = np.linspace(lowerLambda, upperLambda, 50)
    zFloor          = 0

    for y_val in lambdaFloor:
        ax.plot(tempFloor, np.full_like(tempFloor, y_val), np.full_like(tempFloor, zFloor), color='gray', linewidth=0.2)

    for x_val in tempFloor:
        ax.plot(np.full_like(lambdaFloor, x_val), lambdaFloor, np.full_like(lambdaFloor, zFloor), color='gray', linewidth=0.2)
        

    # Hide the planes
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)

    # Hide the spines
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    ax.zaxis.line.set_visible(False)

    # Hide the tick labels (already included)
    ax.tick_params(axis='x', which='major',  labelrotation=15)
    ax.tick_params(axis='y', which='major',  labelrotation=335)
    # ax.tick_params(axis='z', which='major', length=0, labelsize=0)

    #make labels
    ax.set_xlabel('Temperature: T [K]',                 labelpad=55)
    ax.set_ylabel(r'Wavelength: $\lambda$ [nm]',        labelpad=65)
    ax.set_zlabel(r'Photon Counts [10$^5$ a.u.]')
    
    
    if saveGraph:
        fig.savefig(saveFileName, transparent=True)





def graphFirAnalysis(arrOfFirArr, arrOfTempArrKelvins, figsize=(12,6), arrOfLabels=None, saveGraph = False):
    """
    Displays four FIR analysis visualization for single or multiple datasets.
 
    Generates four related plots for FIR-based thermometry analysis. Automatically handles 
    both single and multiple FIR/temperature array pairs.
 
    The four plots show:
        1. FIR vs. Temperature
        2. Log(FIR) vs. Inverse Temperature (with linear fit)
        3. Relative and Absolute Sensitivity vs. Temperature
        4. Temperature Uncertainty vs. Temperature
 
    Args:
        arrOfFirArr (numpy.ndarray): 1D array of FIR values, or a 2D array where each row 
            is a 1D FIR array for a separate process/sample.
        arrOfTempArrKelvins (numpy.ndarray): 1D array of temperature values (in Kelvin), 
            or a 2D array matching the structure of arrOfFirArr (one temperature array per row).
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (12, 6).
        arrOfLabels (numpy.ndarray, optional): Labels for each FIR/temperature pair (e.g., process names). 
            Should have length equal to number of rows in arrOfFirArr. Defaults to None (uses integer indices 0, 1, 2, ...).
        saveGraph (bool, optional): If True, saves the figure to disk. Defaults to False.
 
    Returns:
        None. Displays the plot and optionally saves it.
 
    Raises:
        ValueError: If dimensions of arrOfFirArr and arrOfTempArrKelvins do not agree.
    """
 
    
    import numpy as np
 
    if isinstance(arrOfFirArr, np.ndarray):
        if isinstance(arrOfTempArrKelvins, np.ndarray):
            if isinstance(arrOfFirArr[0], np.ndarray) and isinstance(arrOfTempArrKelvins[0], np.ndarray):
                if arrOfLabels is None or (len(arrOfFirArr) != len(arrOfLabels)):
                    arrOfLabels = np.array([str(idx) for idx in range(len(arrOfFirArr))])

                if len(arrOfFirArr) != len(arrOfTempArrKelvins):
                    raise ValueError(
                        "The number of FIR arrays must equal "
                        "the number of temperature arrays."
                    )

                graphFirAnalysisGrouped(arrOfFirArr, arrOfTempArrKelvins, figsize, arrOfLabels, saveGraph)
            else:
                if not (isinstance(arrOfFirArr[0], np.ndarray) or isinstance(arrOfTempArrKelvins[0], np.ndarray)):
                    graphFirAnalysisSingle(arrOfFirArr, arrOfTempArrKelvins, figsize, saveGraph)

                else:
                    raise ValueError("FIR and temperature array should be of same type, and of same length, same dimension too.")
        else:
            raise ValueError(f"{type(arrOfTempArrKelvins)} is not a np.ndarray") 
    else:
        raise ValueError(f"{type(arrOfFirArr)} is not a np.ndarray") 


def graphFirAnalysisGrouped(arrOfFirArr, arrOfTempArrKelvins, figsize=(12,6), arrOfLabels=None, saveGraph = False):
    """
    Displays four graphs for FIR analysis: R v T, lnR v invT, sensitivities v invT, temp uncertainty v T. 
    This is for the case when you have multiple FIR array, with same number of temperature array is given (meaning, for different processes shown together.)

    The four plots show:
    1. FIR vs. Temperature for all arrays
    2. Log(FIR) vs. Inverse Temperature for all arrays
    3. Relative and Absolute Sensitivity vs. Temperature for all arrays
    4. Temperature uncertainty vs Temperature for all arrays

    Args:
        arrOfFirArr (numpy.ndarray): an array of 1D array of FIR values.
        arrOfTempArrKelvins (numpy.ndarray): an array of 1D array of temperature values in Kelvin.
        figsize (tuple, optional): The size of the figure (width, height) in inches. Defaults to (12,8).
        arrOfLabels(numpy.ndarray): The array of labels you want to display if multiple FIRs, tempArrays given. There should be a label for each process (each individual FIR array). Defaults to integers 0,1,2... if any issue.
        saveGraph (bool, optional): If True, saves graph. Defaults to False.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    plt.rcParams.update({'font.size': 25, 'font.family':'serif', 'font.serif':'Times New Roman'})
    pointSize = 35
    
    fig1, ax1 = plt.subplots(figsize = figsize) # R v T
    fig2, ax2 = plt.subplots(figsize = figsize) # lnR v invT
    fig3, ax3 = plt.subplots(figsize = figsize) #relSens v invT
    ax4 = ax3.twinx() #ax4, which is twin of ax3 is assigned for the absSens v invT
    fig5, ax5 = plt.subplots(figsize = figsize) #temperature uncertainty v temperature


    legend_handles =   []
    legend_handles2 =  []
    legend_labels =    []

    if len(arrOfFirArr) != len(arrOfLabels):
        arrOfLabels = np.array([str(idx) for idx in range(len(arrOfFirArr))])

    for idx, (firArr, tempArrKelvins) in enumerate(zip(arrOfFirArr, arrOfTempArrKelvins)):

        if len(firArr) != len(tempArrKelvins):
            raise ValueError(
                f"FIR and temperature arrays for process "
                f"{arrOfLabels[idx]} must have the same length."
            )

        if np.any(firArr <= 0):
            raise ValueError(
                f"FIR values for process {arrOfLabels[idx]} "
                "must be positive for log transformation."
            )

        if np.any(tempArrKelvins <= 0):
            raise ValueError(
                f"Temperatures for process {arrOfLabels[idx]} "
                "must be greater than zero Kelvin."
            )

        slope, intercept    = getSlopeAndIntercept(1/tempArrKelvins, np.log(firArr))
        relSens, absSens    = getSensitivities(tempArrKelvins, np.array([slope]), np.array([intercept]))
        relSens, absSens    = relSens[0], absSens[0]
        thermalUncertainty  = (0.5*0.01)/relSens #because 0.5 or 0.03 are percentage
        thermalUncertaintyM = (0.03*0.01)/relSens 

        # color1 = 'tab:blue'
        # color2 = 'tab:red'

        # color1 = plt.cm.tab20(2*idx)
        # color2 = plt.cm.tab20(2*idx+1)

        color = plt.cm.tab10(idx)

        print(f"\nCurrently, working with: {arrOfLabels[idx]}:\n")


        print(" relSens : ", relSens, "\n\n  absSens : ", absSens, "\n\n uncertainty in general condition : ", thermalUncertainty, "\n\n uncertainty in max SNR : ", thermalUncertaintyM, "\n\n temperature : ", tempArrKelvins, "\n\n for slope = ", slope, " and intercept = ", intercept, "\n")
        
        x                   = np.linspace(np.min(1/tempArrKelvins), np.max(1/tempArrKelvins), 30)
        y                   = slope * x + intercept

        equationTxt = f' {arrOfLabels[idx]}: ln(R) =  {intercept:.4f} {slope:+.4f}/T '

        # Plot 1: FIR vs temperature
        ax1.plot(tempArrKelvins, firArr, label=f"{arrOfLabels[idx]}")
        ax1.scatter(tempArrKelvins, firArr, s = pointSize)

        # Plot 2: Log of FIR vs inverse temperature
        # ax2.plot(1/tempArrKelvins, np.log(firArr))
        ax2.plot(x, y, label=equationTxt)
        ax2.scatter(1/tempArrKelvins, np.log(firArr), s = pointSize)

        #since the linestyles can differentiate the plots for relSens and absSens, no need for different color
        
        # Plot 3: Relative sensitivity vs temperature, same figure with absSens
        rel_line, = ax3.plot(tempArrKelvins, relSens, color=color, linestyle="-")
        ax3.scatter(tempArrKelvins, relSens, s = pointSize, marker='s', color=color)

        # Plot 4: Absolute sensitivity vs temperature
        # fig4, ax4 = plt.subplots(figsize = figsize)
        abs_line, = ax4.plot(tempArrKelvins, absSens, color=color, linestyle='--')
        ax4.scatter(tempArrKelvins, absSens, s = pointSize, color=color)

        # Plot 5: temperature uncertainty vs temperature
        thermUnc5, = ax5.plot(tempArrKelvins,      thermalUncertainty, color=color, linestyle="-")
        ax5.scatter(tempArrKelvins,   thermalUncertainty, marker='s', color=color, s = pointSize)
        
        thermUnc03, = ax5.plot(tempArrKelvins,      thermalUncertaintyM, color=color, linestyle='--')
        ax5.scatter(tempArrKelvins,   thermalUncertaintyM, color=color, s = pointSize)        
        
        legend_handles.append((rel_line, abs_line))
        legend_handles2.append((thermUnc5, thermUnc03))
        legend_labels.append(f"{arrOfLabels[idx]}")
        
    
    ## adding the bells and whistles around the plots
    
    ax1.set_xlabel("Temperature: T [K]")
    ax1.set_ylabel("R")
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop={'size': 20},framealpha=0)



    ax2.set_ylabel("Log(R)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel(r"Inverse Temperature: 1/T [K$^{-1}$]")
    ax2.legend(prop={'size': 20}, framealpha=0)



    # ax3.set_ylabel(r"Relative Sensitivity: Sr [K$^{-1}$]", color=color1)
    ax3.set_ylabel(r"Relative Sensitivity: Sr [K$^{-1}$]")
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # ax3.tick_params(axis='y', labelcolor=color1)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Temperature: T [K]")
    # ax3.legend(framealpha=0)
    ax3.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)}, prop={'size': 20},framealpha=0)


    # ax4.set_ylabel(r"Absolute Sensitivity : Sa [K$^{-1}$]", color=color2)
    ax4.set_ylabel(r"Absolute Sensitivity : Sa [K$^{-1}$]")
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # ax4.tick_params(axis='y', labelcolor=color2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Temperature: T [K]")
    # ax4.legend(framealpha=0)



    ax5.set_ylabel(r'Temperature Uncertainty: $\delta$T [K] ')
    ax5.yaxis.set_label_position("left")
    ax5.xaxis.set_label_position("bottom")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel("Temperature: T [K]")
    ax5.legend(legend_handles2, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)}, prop={'size': 20},framealpha=0)

    # ax5.legend(framealpha=0)


    if saveGraph:
        fig1.savefig("images/pdf/FirVTemp.pdf",             dpi=300, transparent=True, bbox_inches='tight')
        fig2.savefig("images/pdf/logFirVInvTemp.pdf",       dpi=300, transparent=True, bbox_inches='tight')
        fig3.savefig("images/pdf/sensitivitiesVTemp.pdf",   dpi=300, transparent=True, bbox_inches='tight')
        fig5.savefig("images/pdf/tempUncertainty.pdf",      dpi=300, transparent=True, bbox_inches='tight')

    # Show all plots
    plt.show()



def graphFirAnalysisSingle(arrOfFirArr, arrOfTempArrKelvins, figsize=(12,6), saveGraph = False):
    """
    Displays four graphs for FIR analysis: R v T, lnR v invT, sensitivities v invT, temp uncertainty v T. 
    This is for the case when only a single FIR array, with single temperature array is given (meaning, for a single process.)

    The four plots show:
    1. FIR vs. Temperature
    2. Log(FIR) vs. Inverse Temperature
    3. Relative and Absolute Sensitivity vs. Temperature
    4. Temperature uncertainty vs Temperature

    Args:
        arrOfFirArr (numpy.ndarray): 1D array of FIR values.
        arrOfTempArrKelvins (numpy.ndarray): 1D array of temperature values in Kelvin.
        figsize (tuple, optional): The size of the figure (width, height) in inches. Defaults to (12,8).
        saveGraph (bool, optional): If True, saves graph. Defaults to False.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 25, 'font.family':'serif', 'font.serif':'Times New Roman'})
    pointSize = 35

    if len(arrOfFirArr) != len(arrOfTempArrKelvins):
        raise ValueError("FIR and temperature arrays must have the same length.")

    if np.any(arrOfFirArr <= 0):
        raise ValueError("FIR values must be positive for log transformation.")

    if np.any(arrOfTempArrKelvins <= 0):
        raise ValueError("Temperature values must be greater than zero Kelvin.")



    
    fig1, ax1 = plt.subplots(figsize = figsize) # R v T
    fig2, ax2 = plt.subplots(figsize = figsize) # lnR v invT
    fig3, ax3 = plt.subplots(figsize = figsize) #relSens v invT
    ax4 = ax3.twinx() #ax4, which is twin of ax3 is assigned for the absSens v invT
    fig5, ax5 = plt.subplots(figsize = figsize) #temperature uncertainty v temperature

    ############################## assigning the variable name to work with what is already below
    firArr = arrOfFirArr
    tempArrKelvins = arrOfTempArrKelvins
    #####################


    slope, intercept    = getSlopeAndIntercept(1/tempArrKelvins, np.log(firArr))
    relSens, absSens    = getSensitivities(tempArrKelvins, np.array([slope]), np.array([intercept]))
    relSens, absSens    = relSens[0], absSens[0]
    thermalUncertainty  = (0.5*0.01)/relSens #because 0.5 and 0.03 are percentages
    thermalUncertaintyM = (0.03*0.01)/relSens 

    color = 'tab:blue'


    print(" relSens : ", relSens, "\n\n  absSens : ", absSens, "\n\n uncertainty in general condition : ", thermalUncertainty, "\n\n uncertainty in max SNR : ", thermalUncertaintyM, "\n\n temperature : ", tempArrKelvins, "\n\n for slope = ", slope, " and intercept = ", intercept, "\n")
    
    x                   = np.linspace(np.min(1/tempArrKelvins), np.max(1/tempArrKelvins), 30)
    y                   = slope * x + intercept

    equationTxt = f' ln(R) =  {intercept:.4f} {slope:+.4f}/T '

    # Plot 1: FIR vs temperature
    ax1.plot(tempArrKelvins, firArr)
    ax1.scatter(tempArrKelvins, firArr, s = pointSize)

    # Plot 2: Log of FIR vs inverse temperature
    # ax2.plot(1/tempArrKelvins, np.log(firArr))
    ax2.plot(x, y, label=equationTxt)
    ax2.scatter(1/tempArrKelvins, np.log(firArr), s = pointSize)

    #since the linestyles can differentiate the plots for relSens and absSens, no need for different color
    
    # Plot 3: Relative sensitivity vs temperature, same figure with absSens
    ax3.plot(tempArrKelvins, relSens, color=color, linestyle='-')
    ax3.scatter(tempArrKelvins, relSens, s = pointSize, marker='s', color=color)

    # Plot 4: Absolute sensitivity vs temperature
    ax4.plot(tempArrKelvins, absSens, color=color, linestyle='--')
    ax4.scatter(tempArrKelvins, absSens, s = pointSize, color=color)

    # Plot 5: temperature uncertainty vs temperature
    ax5.plot(tempArrKelvins,      thermalUncertainty, color=color, linestyle='-')
    ax5.scatter(tempArrKelvins,   thermalUncertainty, color=color, marker='s', s = pointSize)
    ax5.plot(tempArrKelvins,      thermalUncertaintyM, color=color, linestyle='--')
    ax5.scatter(tempArrKelvins,   thermalUncertaintyM, color=color, s = pointSize)        
    
    
    
    ## adding the bells and whistles around the plots
    
    ax1.set_xlabel("Temperature: T [K]")
    ax1.set_ylabel("R")
    ax1.grid(True, alpha=0.3)



    ax2.set_ylabel("Log(R)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel(r"Inverse Temperature: 1/T [K$^{-1}$]")
    ax2.legend(prop={'size': 20}, framealpha=0)



    # ax3.set_ylabel(r"Relative Sensitivity: Sr [K$^{-1}$]", color=color1)
    ax3.set_ylabel(r"Relative Sensitivity: Sr [K$^{-1}$]")
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Temperature: T [K]")
    ax3.legend(prop={'size': 20},framealpha=0)


    ax4.set_ylabel(r"Absolute Sensitivity : Sa [K$^{-1}$]")
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Temperature: T [K]")
    ax4.legend(prop={'size': 20},framealpha=0)



    ax5.set_ylabel(r'Temperature Uncertainty: $\delta$T [K] ')
    ax5.yaxis.set_label_position("left")
    ax5.xaxis.set_label_position("bottom")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel("Temperature: T [K]")

    ax5.legend(prop={'size': 20},framealpha=0)


    if saveGraph:
        fig1.savefig("images/pdf/FirVTemp.pdf",             dpi=300, transparent=True, bbox_inches='tight')
        fig2.savefig("images/pdf/logFirVInvTemp.pdf",       dpi=300, transparent=True, bbox_inches='tight')
        fig3.savefig("images/pdf/sensitivitiesVTemp.pdf",   dpi=300, transparent=True, bbox_inches='tight')
        fig5.savefig("images/pdf/tempUncertainty.pdf",      dpi=300, transparent=True, bbox_inches='tight')

    # Show all plots
    plt.show()












def plotIntegratedIntensities(arr2D
                              , temperatureArrKelvin
                              , startLambdaManifoldArr
                              , endLambdaManifoldArr
                              , labelsArr
                              , fromTempIdx = None
                              , toTempIdx   = None
                              , saveFigure  = False
                              , saveFileName = "images/integratedIntensities.pdf"
                              , figSize=(12, 12)
                              , dpi = 300
                              , fontSize = 28
                              ):

    """
    Plots integrated spectral intensities across multiple wavelength regions as a function of temperature.
 
    For each wavelength manifold (region), computes baseline-subtracted integrated intensities 
    and plots them vs. temperature on a common figure, allowing comparison of multiple spectral features.
 
    Args:
        arr2D (numpy.ndarray): 2D array where each row contains photon counts for a specific 
            wavelength (first element) across varying temperatures.
        temperatureArrKelvin (numpy.ndarray): 1D array of temperature values (in Kelvin) 
            corresponding to the columns in arr2D.
        startLambdaManifoldArr (numpy.ndarray): 1D array of starting wavelengths (in nm) 
            for each manifold region to integrate.
        endLambdaManifoldArr (numpy.ndarray): 1D array of ending wavelengths (in nm) 
            for each manifold region to integrate. Must have same length as startLambdaManifoldArr.
        labelsArr (numpy.ndarray or list): Labels for each wavelength region (e.g., "peak 1", "peak 2"). 
            Must have same length as startLambdaManifoldArr.
        fromTempIdx (int, optional): Starting index of temperature range to plot. 
            If None, starts from 0. Defaults to None.
        toTempIdx (int, optional): Ending index of temperature range to plot. 
            If None, uses last temperature. Defaults to None.
        saveFigure (bool, optional): If True, saves the figure to the specified path. Defaults to False.
        saveFileName (str, optional): Path and filename for saved figure. 
            Defaults to "images/integratedIntensities.pdf".
        figSize (tuple, optional): Figure dimensions (width, height) in inches. Defaults to (12, 12).
        dpi (int, optional): Resolution in dots per inch for saved figure. Defaults to 300.
        fontSize (int, optional): Font size for plot labels and legend. Defaults to 28.
 
    Returns:
        None. Displays the plot and optionally saves it.
    """
    
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': fontSize, 'font.family':'serif', 'font.serif':'Times New Roman'})

    figIntensities, axIntensities = plt.subplots(figsize = figSize, dpi=dpi)

    if not (len(startLambdaManifoldArr) == len(endLambdaManifoldArr) == len(labelsArr)):
        raise ValueError("startLambdaManifoldArr, endLambdaManifoldArr, and labelsArr must have the same length")

    if fromTempIdx is None:
        fromTempIdx = 0

    if toTempIdx is None:
        toTempIdx = len(temperatureArrKelvin) - 1

    for idx in range(len(labelsArr)):
        tempCount = getTotalCount(arr2D, startLambdaManifoldArr[idx], endLambdaManifoldArr[idx])
        assert len(tempCount) == len(temperatureArrKelvin)

        axIntensities.plot(temperatureArrKelvin[fromTempIdx:toTempIdx+1], tempCount[fromTempIdx:toTempIdx+1], label = labelsArr[idx], linewidth=2)
        axIntensities.scatter(temperatureArrKelvin[fromTempIdx:toTempIdx+1], tempCount[fromTempIdx:toTempIdx+1])
        
    # axIntensities.set_xlabel(r"Temperature: T [$^{\circ}$C]")
    axIntensities.set_xlabel(r"Temperature: T [K]")
    axIntensities.set_ylabel("Intensity [a.u]")

    axIntensities.set_yscale('log')

    axIntensities.legend(framealpha=0, fontsize=24)

    if saveFigure:
    # if True:
        figIntensities.savefig(saveFileName, transparent=True)



def plotXRD(xrdData, saveFigure, saveName = "images/xrdPattern.pdf"):
    """
    Plots X-ray diffraction (XRD) data and optionally saves the figure.
 
    Generates a publication-quality plot of diffraction intensity vs. diffraction angle (2θ). 
    Customizes appearance with appropriate axis labels and formatting for scientific publications.
 
    Args:
        xrdData (numpy.ndarray or list of lists): The XRD data. Each row should contain 
            two values: the diffraction angle (2θ, in degrees) and the corresponding intensity 
            (arbitrary units). Data should have shape (n_points, 2).
        saveFigure (bool): If True, the plot is saved to the file path specified by saveName.
        saveName (str, optional): The file path (including filename and extension) where the 
            plot will be saved. Defaults to "images/xrdPattern.pdf". Common formats: 
            .png, .pdf, .jpg.
 
    Returns:
        None. Displays the plot and optionally saves it.
    """

    plt.rcParams.update({
        "font.size": 28,
        "font.family": "serif",
        "font.serif": "Times New Roman",
    })

    angles = xrdData["angle"]
    intensity = xrdData["intensity"]

    plt.figure(figsize=(12, 10))
    plt.plot(angles, intensity, "k")

    plt.xlim(10, angles.max() + 0.5)
    plt.xticks(np.arange(10, 90, 10))

    plt.xlabel(r"Diffraction angle: 2$\theta$ [degrees]")
    plt.ylabel("Intensity : I [a.u]")
    plt.tight_layout()

    if saveFigure:
        plt.savefig(saveName, transparent=True, bbox_inches="tight", pad_inches=0.02)

    plt.show()




def readEmissionTextFile(filename):
    """
    Reads emission spectral data from a formatted text file.
 
    Parses a text file containing wavelength and temperature-dependent photon counts 
    to extract spectral and temperature information. The file is expected to follow a 
    specific format: emission data begins at line 25, with columns for wavelength, 
    followed by photon counts at different temperatures.
 
    Args:
        filename (str): The path to the text file containing the emission data.
 
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - arr2D (numpy.ndarray): A 2D array where each row corresponds to a wavelength 
              entry with the first column as wavelength (in nm) and subsequent columns as 
              photon counts for each temperature.
            - tempArr (numpy.ndarray): A 1D array of temperature values (in Kelvin or Celsius, 
              as recorded in the file) corresponding to the columns in arr2D (after the wavelength column).
 
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If a line in the file cannot be properly parsed into numerical values,
            or if the file format does not match the expected structure.
 
    Note:
        - The file must contain emission data starting from line 25 onwards.
        - The first column is wavelength; subsequent columns are photon counts per temperature.
        - The function returns an empty array if the file is empty or if an error occurs 
          during reading.
    """
    
    try:
        with open(filename, 'r', encoding='latin-1') as file:
            content = file.readlines()
    except Exception as e:
        print(f"Error occured while reading file: {e}")


    for i, line in enumerate(content):
        if line.startswith("Labels"):
            labelsIndex = i
            break

    print(labelsIndex)

    tempArr     = content[labelsIndex + 15].rstrip(",\n")
    tempArr     = tempArr.split(",")
    tempArr     = [float(i) for i in tempArr[1:]]

    arr2D       = []

    for i in range(labelsIndex + 22, len(content)):
        lineI   = content[i].rstrip(",\n")
        lineI   = lineI.split(",")
        arr2D.append([float(val) for val in lineI])
    
    assert len(arr2D[0]) - 1 == len(tempArr)
    
    print("num of wavelengths : ", len(arr2D), " number of temperatures available : ", len(tempArr))
    return np.array(arr2D), np.array(tempArr)




def readXrdFile(xrdFilename, encoding=None):
    """
    Reads X-ray diffraction (XRD) data from a CSV, XLS, or XLSX file.
 
    Loads XRD diffraction pattern data consisting of two columns: diffraction angle (2θ) 
    and intensity. Automatically detects delimiters in CSV files and handles Excel formats.
 
    Args:
        xrdFilename (str or pathlib.Path): Path to the XRD data file. Supported file 
            formats are .csv, .xls, and .xlsx.
        encoding (str, optional): Text encoding to use when reading CSV files (e.g., 'utf-8', 
            'latin-1'). Ignored for Excel files, which handle encoding automatically. 
            If None, pandas uses its default encoding. Defaults to None.
 
    Returns:
        pandas.DataFrame: A DataFrame with two columns:
            - ``angle``: Diffraction angle (2θ, in degrees).
            - ``intensity``: Measured diffraction intensity (arbitrary units).
            Each row represents one diffraction peak or data point.
 
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file extension is not .csv, .xls, or .xlsx.
        pandas.ParserError: If the file cannot be parsed (e.g., corrupted or unsupported format).
 
    Note:
        - The input file should not contain a header row; data columns are named automatically.
        - CSV files may be comma-, tab-, or otherwise delimiter-separated; the function 
          detects the delimiter automatically.
        - Excel files (.xls and .xlsx) are read directly without delimiter detection.
    """

    path = Path(xrdFilename)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(
            path,
            sep=None,          # automatically detect comma/tab/etc.
            engine="python",
            header=None,
            names=["angle", "intensity"],
            encoding=encoding,
        )
    elif path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(
            path,
            header=None,
            names=["angle", "intensity"],
        )
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return df.astype(float)
    