import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
from scipy import interpolate
import batman
import configparser
import corner
import emcee
import george
from tabulate import tabulate
from tqdm import tqdm
import pdb
from itertools import chain

config = configparser.RawConfigParser()

def create_param_file(newfile=''):
    """
    Creates a blank parameter file for the user to fill.

    :param newfile: Name of the file that will be created

    :return: Writes file to pwd
    """

    config.add_section('What to Fit')
    config.set('What to Fit', 'Planet Radius                      ', 'False')
    config.set('What to Fit', 'Limb Darkening Coefficients        ', 'False')
    config.set('What to Fit', 'Time of Mid-Transit                ', 'False')
    config.set('What to Fit', 'Period                             ', 'False')
    config.set('What to Fit', 'Scaled Semi-Major Axis             ', 'False')
    config.set('What to Fit', 'Inclination                        ', 'False')
    config.set('What to Fit', 'Eccentricity                       ', 'False')
    config.set('What to Fit', 'Argument of Periastron Longitude   ', 'False')

    config.add_section('Multi-Transits:What to Trust')
    config.set('Multi-Transits:What to Trust', 'rp  ', '[2, 2, 0, 0]')
    config.set('Multi-Transits:What to Trust', 'u   ', '[1, 0, 0, 0]')
    config.set('Multi-Transits:What to Trust', 't0  ', '[1, 1, 1, 1]')
    config.set('Multi-Transits:What to Trust', 'per ', '[2, 2, 0, 0]')
    config.set('Multi-Transits:What to Trust', 'a   ', '[1, 0, 0, 0]')
    config.set('Multi-Transits:What to Trust', 'inc ', '[1, 0, 0, 0]')
    config.set('Multi-Transits:What to Trust', 'ecc ', '[1, 0, 0, 0]')
    config.set('Multi-Transits:What to Trust', 'w   ', '[1, 0, 0, 0]')

    config.add_section('Planet Name')
    config.set('Planet Name', 'rp                      ', '[0.0,        error]')
    config.set('Planet Name', 'u                       ', '[[0.0, 0.0], [error, error]]')
    config.set('Planet Name', 't0                      ', '[2450000.0,  error]')
    config.set('Planet Name', 'per                     ', '[1.0,        error]')
    config.set('Planet Name', 'a                       ', '[0.0,        error]')
    config.set('Planet Name', 'inc                     ', '[0.5*sp.pi,  error]')
    config.set('Planet Name', 'ecc                     ', '[0.0,        error]')
    config.set('Planet Name', 'w                       ', '[0.0,        error]')
    config.set('Planet Name', 'limb_dark               ', 'quadratic')

    with open(newfile, 'wt') as configfile:
        config.write(configfile)


def create_results_file(results, filename):
    """
    Saves a file with a latex ready table of the results

    :param results: results from the fit
    :param filename: the name of the file to be saved
    """

    f = open(filename, 'w')
    f.write(tabulate(results, tablefmt="latex", floatfmt=".5f"))
    f.close()


def separate_limbdark(variables):
    """
    Separates the Limb Darkening variables from the input structure to a form usable by the MCMC code.

    :param variables: List of variables provided from BatSignal class, containing a nested array for the limb darkening
        variables.

    :return: List of variables without the nested array.
    """

    length = 0
    for i, v in enumerate(variables):
        try:
            length = len(v)
        except TypeError:
            variables[i] = [v]

    new = list(chain.from_iterable(variables))

    return new, length


def together_limbdark(variables, length):
    """
    Returns the Limb Darkening variables to the original nested structure.

    :param variables: List of variables provided from BatSignal class, with the limb darkening variables separated.
    :param length: Number of limb darkening coefficients

    :return: List of variables with the limb darkening coefficients in a nested array.
    """

    ld = []
    for _ in range(length):
        ld.append(variables.pop(1))

    assert isinstance(ld, list)
    variables.insert(1, ld)

    return variables


def run_batman(variables, date_real, names):

    date = np.linspace(date_real[0], date_real[-1], len(date_real))

    batinputs = batman.TransitParams()

    [setattr(batinputs, name, val) for name, val in zip(names, variables) if name not in ["amp, scale"]]

    bats = batman.TransitModel(batinputs, date)
    model = bats.light_curve(batinputs)

    return model


def lnprior(variables, sigmas):
    """
    Verifies that the new guesses for the parameters are within a reasonable (3sigma) range of the original guesses.
    Then determines a likelihood value related to how distant the new guesses are from the originals.

    :param variables: guesses of the user for the parameters.
    :param sigmas: values of error from the user guesses in the input file.

    :return: Likelihood value based off of the distance between the new guesses and the original input.
    """

    value = list()

    for key in variables.keys():
        if key == 'usr':
            pass
        else:
            for i, s in enumerate(sigmas):
                val = 0.0
                if s[0] == 'amp' or s[0] == 'scale':
                    if sp.log(s[1][0]) < variables[key][i] < sp.log(s[1][1]):
                        val = val - (variables[key][i] - variables['usr'][i]) ** 2 / 2 * 0.3 * sp.log(s[1][1])
                    else:
                        val = -sp.inf
                    value.append(val)
                elif s[0] == 't0' and not isinstance(variables['usr'][4], float):
                    num = int(key[7:])
                    if variables['usr'][i][num] - 3 * s[1] < variables[key][i] < variables['usr'][i][num] + 3 * s[1]:
                        val = val - (variables[key][i] - variables['usr'][i][num]) ** 2 / (2 * (s[1] ** 2))
                    else:
                        val = -sp.inf
                    value.append(val)
                else:
                    if isinstance(s[1], float):
                        if variables['usr'][i] - 3 * s[1] < variables[key][i] < variables['usr'][i] + 3 * s[1]:
                            val = val - (variables[key][i] - variables['usr'][i]) ** 2 / (2 * (s[1] ** 2))
                        else:
                            val = -sp.inf
                        value.append(val)
                    else:
                        for n in range(len(s[1])):
                            if val != -sp.inf and variables['usr'][i][n] - 3 * s[1][n] < variables[key][i][n] < \
                                            variables['usr'][i][n] + 3 * s[1][n]:
                                val = val - (variables[key][i][n] - variables['usr'][i][n]) ** 2 / (2 * (s[1][n] ** 2))
                            else:
                                val = -sp.inf
                            value.append(val)

    val = 0.0
    for i in value:
        if not sp.isfinite(i):
            val = -sp.inf
        else:
            val += i

    return val


def lnlike(variables, datadict, names):
    """
    Changes the inputs for batman to the new guesses. Computes a new model for the light curve and uses Gaussian
    Process Regression to determine the likelihood of the model as a whole.

    :param variables: dictionary of variables to use in the fit of each transit.
    :param datadict: dictionary containting lightcurve data
    :param names: ordered list of names of the parameters

    :return: A number representing the Gaussian Process liklihood of the model being correct
    """

    val = 0.0
    for i in range(len(datadict.keys())):

        key = 'transit' + str(i)
        date = datadict[key][3]
        date_real = datadict[key][0]
        flux = datadict[key][1]
        error = datadict[key][2]

        model = run_batman(variables[key], date_real, names)

        tck = interpolate.splrep(date, model, s=0)
        model_real_times = interpolate.splev(date_real, tck, der=0)

        kernel = variables[key][0] * george.kernels.Matern52Kernel(variables[key][1])
        gp = george.GP(kernel)
        gp.compute(date_real, error)

        val += gp.lnlikelihood(flux - model_real_times)

    return val


def lnprob(theta, datadict, parameters, sigmas, multi):
    """
    Returns the combined likelihoods from lnprior and lnlike. If this number is smaller than the previous model, this
    model is saved as the most probable fit of the data.

    :param theta: New values for all variables being fit
    :param datadict: Dictionary of lightcurve data
    :param parameters: list of user input parameters
    :param sigmas: list of parameter names coupled with the user input error on each
    :param multi: Dictionary for each transit listing the location in theta of a given parameter (or nan if not
                  being fit)

    :return: A value representing the liklihood of the model being thr true fit to the data. This value should be
        minimized over time.
    """

    vardict = {x: parameters[:] for x in datadict.keys()}
    usr = parameters[:]
    for key in vardict.keys():
        vardict[key][0] = theta[0]
        vardict[key][1] = theta[1]
        for i in range(len(parameters) - 4):
            if multi[key][i] is not np.nan and i == 1:
                for n in range(parameters[-1]):
                    vardict[key][i+2][n] = theta[multi[key][i]+n]
            elif multi[key][i] is not np.nan:
                vardict[key][i+2] = theta[multi[key][i]]
            else:
                pass

    vardict['usr'] = usr
    names = [n for n, v in sigmas]
    names.append('limb_dark')

    ln_prior = lnprior(vardict, sigmas)

    if sp.isfinite(ln_prior):
        ln_like = lnlike(vardict, datadict, names)
        return ln_prior + ln_like
    else:
        return -sp.inf


def normalize_flux(flux):
    """
    Normalizes the flux by the median of the baseline
    :param flux: Light collected from star, y axis

    :return: Flux divided by the median of the transit's baseline
    """

    # Compute the mean value of flux
    mean_flux = np.mean(flux)

    # Determinate the points above the mean of the flux
    flux_upper = flux[flux >= mean_flux]

    # Compute the median of the points above the mean of the flux
    median_flux = np.median(flux_upper)

    # Normalize the flux by the median of the no_transit flux
    norm_flux = flux / median_flux

    return norm_flux


def compute_tzero(date, flux):
    """
    Computes the time of mid transit for secondary transits from the value given for the primary and the period of the
    orbit.
    :param date: Time of observations, x axis
    :param flux: Flux for each observation, y axis

    :return: Time of mid transit for a secondary transit
    """

    _, redate = np.histogram(date, bins=int(len(date)*0.05))
    reflux = np.interp(redate, date, flux)
    idx = np.argmin(reflux)

    dlist = date.tolist()
    low = min(date, key=lambda x: np.abs(x - redate[idx - 1]))
    low = dlist.index(low)
    high = min(date, key=lambda x: np.abs(x - redate[idx + 1]))
    high = dlist.index(high)
    zoom = date[low:high]

    _, redate = np.histogram(zoom, bins=int(len(zoom)*0.15))
    reflux = np.interp(redate, date, flux)
    idx = np.argmin(reflux)

    t0_new = redate[idx]

    return t0_new


def noparams(func):
    """
    Wrapper to check if the user named the parameters needed as input to the function they ran. If they didn't, it
    requests them in the command line and calls the function again.

    :param func: The function being called: Enable, Disable

    :return: The function again, however this time with the parameters named
    """

    def func_wrapper(self, parameters=''):
        if parameters == '':
            parameters = input("Which parameter(s) would you like to update: rp, u1-u4, t0, per, a, inc, ecc, w, "
                               "or all? Please write them in list format:  ")

        ret = func(self, parameters)

        return ret

    return func_wrapper


def nofiles(func):
    """
    Wrapper to check if the user named the input files when calling the instance. If they didn't, it requests them in
    the command line and calls the instance again.

    :param func: The function being called (in this case, BatSignal's init function)

    :return: The function again, however this time with the input files named
    """

    def func_wrapper(self, input_param_file='', light_curve_file='', planet=''):
        if input_param_file == '':
            check = input('Do you have an input file .cfg file? [y/n] ')

            if check == 'y':
                input_param_file = input('What is the name of the file containing your input parameters? ')
            elif check == 'n':
                fname = input('We will create one for you, then. What would you like it to be called? (please use '
                              'the extention ".cfg". ')
                create_param_file(newfile=fname)

        if '.cfg' not in input_param_file:
            raise ValueError("The parameter file should have the extention .cfg. To generate a \
                  a blank file, call create_param_file().")

        if light_curve_file == '':
            light_curve_file = input("What is the name of the file containing your light curve? ")

        ret = func(self, input_param_file, light_curve_file, planet)

        return ret

    return func_wrapper


class BatSignal:
    """
    Fits a model created by BATMAN to reduced exoplanet light curves. Uses MCMC and Gaussian Process Regression to
    determine the most likely model fit to the data.
    """

    @nofiles
    def __init__(self, input_param_file, light_curve_file, planet=''):
        """
        Init function to set up variables in self.
        :param input_param_file: File such as that created in create_param_file(). Contains user guesses for planet
            details and errors on guesses.
        :param light_curve_file: File contacting the Julian date of observations, flux, and optional errors on the flux
        :param planet: The name of the planet. Only required if there are multiple planets in the input file.
        """

        self.input_param_file = input_param_file
        self.light_curve_file = light_curve_file

        # Check if the parameter file is in the correct format
        if input_param_file[-4:] != '.cfg':
            raise TypeError('The parameter file should be of format ".cfg". Run create_param_file() for an example.')

        # Create a dictionary containing the Julian date, flux, and error information of each light curve. The last
        # column has the Julian date in equal time intervals.
        self._datadict = {}
        if isinstance(light_curve_file, (list, tuple)):
            for i in range(len(light_curve_file)):
                n = 'transit' + str(i)
                self._datadict[n] = sp.loadtxt(light_curve_file[i], unpack=True)
                col = np.linspace(self._datadict[n][0][0], self._datadict[n][0][-1], len(self._datadict[n][0]))
                col = col.reshape(1, np.shape(col)[0])
                self._datadict[n] = np.concatenate([self._datadict[n], col])
        elif isinstance(light_curve_file, str):
            self._datadict['transit0'] = sp.loadtxt(light_curve_file, unpack=True)
            col = np.linspace(self._datadict['transit0'][0][0], self._datadict['transit0'][0][-1],
                              len(self._datadict['transit0'][0]))
            col = col.reshape(1, np.shape(col)[0])
            self._datadict['transit0'] = np.concatenate([self._datadict['transit0'], col])
        else:
            raise TypeError('The light curve file should be either a string or list')
        for key in self._datadict.keys():
            if len(self._datadict[key]) != 4:
                raise ValueError("The input file should contain a column with the Julian date,"
                                 "a column of data, and an error column only.")

        # Finds the name of the planet and the cooresponding parameters from the parameter file
        config.readfp(open(self.input_param_file))
        if planet == '':
            section = config.sections()[2]
        else:
            section = planet
        self.planet = section

        # List of all parameter names from the parameter file
        self.all_param_names = sp.array(config.items(section))[:, 0].tolist()
        self.all_param_names = [str(self.all_param_names[i]) for i in range(len(self.all_param_names))]
        self.all_param_names.insert(0, 'amp')
        self.all_param_names.insert(1, 'scale')

        # Organizes the input from the parameter file
        usr = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
        self.limb_dark_law = usr.pop(8)
        self._usr_in = [i[0] for i in usr]
        self._usr_err = [i[1] for i in usr]

        # Relax is a multiplication factor for sigma in each value, it can be modified by the user with the
        # update_relax method
        self._usr_err, self._ldlength = separate_limbdark(self._usr_err)
        self.relax = np.ones(len(self._usr_err)).tolist()

        # Determines the time of mid transit for all transits and sets the input parameter for batman as the value for
        # the first transit, with all values in a separate list.
        self.t0s = {}
        if isinstance(self._usr_in[2], float):
            if self._datadict['transit0'][0][0] < self._usr_in[2] < self._datadict['transit0'][0][-1]:
                if len(self._datadict.keys()) == 1:
                    self.t0s[self._datadict.keys()[0]] = self._usr_in[2]
                else:
                    self.t0s['transit0'] = self._usr_in[2]
                    for key in sorted(self._datadict.keys()):
                        if key != 'transit0':
                            self.t0s[key] = compute_tzero(self._datadict[key][0], self._datadict[key][1])
            else:
                for key in sorted(self._datadict.keys()):
                    self.t0s[key] = compute_tzero(self._datadict[key][0], self._datadict[key][1])
        else:
            count = 0
            for key in sorted(self._datadict.keys()):
                self.t0s[key] = self._usr_in[2][count]
                count += 1

        # Determines which parameters the user would like to fit
        section = config.sections()[0]
        self._usr_change = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
        change_arr = np.where(self._usr_change)[0]

        # Determines the number of limb-darkening coefficients for the law requested if quadratic wasn't used
        self.names_change = [self.all_param_names[i+2] for i in change_arr]
        self.names_change.insert(0, 'amp')
        self.names_change.insert(1, 'scale')

        # Creates an array of the varibales to be fit
        self.variables = [self._usr_in[i] for i in change_arr]

        # Updates the sigma array to contain only values for the parameters to be fit, along with the addition of the
        # amplitude and scale values for the gp kernel
        self._usr_err.insert(0, [1., 5000.])
        self._usr_err.insert(1, [1., 100.])

        # Determines the number of walkers to use and number of variables to fit
        self.variables.insert(0, 0.06)
        self.variables.insert(1, 0.8)
        self.init = dict(zip(self.names_change, self.variables))

        # Creates a dictionary for which transit parameters should be fit independanly or simultaneously in the case
        # of multiple transits
        section = config.sections()[1]
        multi = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
        self._multi = dict(zip(self.all_param_names[2:], multi))

        # Reformats the multi dictionary and creates the list of variables to be fit for
        lcs = sorted(self._datadict.keys())
        self._multi_lc = dict.fromkeys(lcs)
        pos = dict(zip(self.names_change, np.ones(len(self.names_change))))
        self.initnames = ['amp', 'scale']
        self.initial = [0.06, 0.8]
        count = 2
        for n in self.all_param_names[2:-1]:
            if n not in pos.keys():
                pos[n] = 0
        for lc in lcs:
            self._multi_lc[lc] = []
            for i, n in enumerate(self.all_param_names[2:-1]):
                if self._multi[n][lcs.index(lc)] == 0:
                    self._multi_lc[lc].append(np.nan)
                elif self._multi[n][lcs.index(lc)] == 1 and pos[n] != 0:
                        self._multi_lc[lc].append(count)
                        if n == 'u':
                            for l in range(self._ldlength):
                                self.initnames.append(n)
                            count += self._ldlength-1
                        else:
                            self.initnames.append(n)
                        if n == 't0':
                            self.initial.append(self.t0s[lc])
                        else:
                            self.initial.append(self.init[n])
                        count += 1
                elif pos[n] != 0:
                    group = 'pos' + str(self._multi[n][lcs.index(lc)])
                    try:
                        self._multi_lc[lc].append(locals()[group][n])
                    except KeyError:
                        self._multi_lc[lc].append(count)
                        try:
                            locals()[group][n] = count
                            if n == 'u':
                                for l in range(self._ldlength):
                                    self.initnames.append(n)
                                count += self._ldlength - 1
                            else:
                                self.initnames.append(n)
                            if n == 't0':
                                self.initial.append(self.t0s[lc])
                            else:
                                self.initial.append(self.init[n])
                            count += 1
                        except KeyError:
                            locals()[group] = dict()
                            locals()[group][n] = count
                            if n == 'u':
                                for l in range(self._ldlength):
                                    self.initnames.append(n)
                                count += self._ldlength - 1
                            else:
                                self.initnames.append(n)
                            if n == 't0':
                                self.initial.append(self.t0s[lc])
                            else:
                                self.initial.append(self.init[n])
                            count += 1
                else:
                    self._multi_lc[lc].append(np.nan)

        self.initial, _ = separate_limbdark(self.initial)

        # Normalizes the flux for all transits
        for key in sorted(self._datadict.keys()):
            self._datadict[key][1] = normalize_flux(self._datadict[key][1])

        # Sets up empty variables for future use
        self.sigmas = None
        self._sampler = None
        self._output = None
        self._bat_model = None
        self.results = list()

    def update_relax(self, **kwargs):
        """
        Allows the user to update the relaxation factor for sigma of each parameter listed in keywords.

        :keyword: "rp" - ratio of the radii
        :keyword: "u"  - limb darkening coefficients
        :keyword: "t0" - time of mid transit
        :keyword: "per"- period of orbit
        :keyword: "a"  - scaled semi-major axis
        :keyword: "inc"- inclination
        :keyword: "ecc"- eccentricity
        :keyword: "w"  - argument of periastron longitude
        """

        for key, value in kwargs.items():
            idx = self.names_change.index(key)
            self.relax[idx] = value

        return self

    def bat(self, *args, **kwargs):
        """
        Runs fit for light curve model using MCMC and Gaussian Process Regression.

        :keyword: err_nlvl      - new value for error in amplitude for gaussian process regression
        :keyword: err_scale     - new value for error in scale for gaussian process regression

        :return: input_param_file - Name of file containing user's guesses for planet details
        :return: light_curve_file - Name of file containing light curve data
        :return: date             - Interpolated dates so that they have equal time intervals
        :return: date_real        - Dates of observations imported from light curve file
        :return: flux             - Flux values imported from light curve file
        :return: error            - Error on flux values imported from light curve file
        :return: names            - Names of the parameters BATMAN uses to create its models
        :return: variables        - Values of the parameters that are being fit for
        :return: relax            - Relaxation on the sigma parameter for each variable being fit for
        :return: model            - Model light curve of transit as determined by BatSignal
        :return: results          - Best fit parameters used to create the model light curve
        """

        # Organizes any keyword aguments included when the method was called
        for key, value in kwargs.items():
            if key == "err_nlvl":
                self._usr_err[0] = value
            elif key == "err_scale":
                self._usr_err[1] = value
            elif key == 'uniform':
                uniform = value
            elif key == 'gaussian' or key == 'normal':
                gaussian = value
            else:
                pass

        # Creates a dictionary of sigma values including any updates to the relaxation parameter.
        sigmas = []
        for i, r in enumerate(self.relax):
            sigmas.append(self._usr_err[i+2]*r)
        sigmas = together_limbdark(sigmas, self._ldlength)
        self.sigmas = zip(self.all_param_names[2:-1], sigmas)
        self.sigmas.insert(0, (self.names_change[0], self._usr_err[0]))
        self.sigmas.insert(1, (self.names_change[1], self._usr_err[1]))

        initial = zip(self.initnames, self.initial)
        nwalkers = len(self.initnames) * 10

        self.variables, _ = separate_limbdark(self.variables)
        ndim = len(self.variables)

        pos = np.zeros([nwalkers, len(self.initnames)])
        scales = dict.fromkeys(['amp', 'scale', 'rp', 'per', 'a', 't0', 'ecc', 'w'], 1e-2)
        scales['u'] = 1e-3
        scales['inc'] = 5.
        for i in range(nwalkers):
            for j in enumerate(self.initnames):
                if j[1] == 'amp' or j[1] == 'scale' or j[1] == 't0':
                    pos[i, j[0]] = initial[j[0]][1] + scales[j[1]] * sp.random.randn(1)
                else:
                    pos[i, j[0]] = sp.random.normal(initial[j[0]][1], scales[j[1]])

            if 'noprior' in locals():
                scales = dict.fromkeys(['amp', 'scale', 'rp', 'per', 'a', 'ecc', 'w'], 1.)
                scales['t0'] = 0.1
                scales['u'] = 1e-2
                scales['inc'] = 10.
                if isinstance(noprior, (list, tuple)):
                    for v in noprior:
                        idx = [i for i, x in enumerate(self.initnames) if x == v]
                        for m in idx:
                            pos[i, m] = initial[m][1] + scales[v] * sp.random.randn(1)
                else:
                    idx = [i for i, x in enumerate(self.initnames) if x == noprior]
                    for m in idx:
                        pos[i, m] = initial[m][1] + scales[noprior] * sp.random.randn(1)

            if 'uniform' in locals():
                if isinstance(uniform, (list, tuple)):
                    for v in uniform:
                        idx = [i for i, x in enumerate(self.initnames) if x == v]
                        for m in idx:
                            pos[i, m] = initial[m][1] + scales[v] * sp.random.randn(1)
                else:
                    idx = [i for i, x in enumerate(self.initnames) if x == uniform]
                    for m in idx:
                        pos[i, m] = initial[m][1] + scales[uniform] * sp.random.randn(1)

            if 'gaussian' in locals():
                if isinstance(gaussian, (list, tuple)):
                    for v in gaussian:
                        idx = [i for i, x in enumerate(self.initnames) if x == v]
                        for m in idx:
                            pos[i, m] = sp.random.normal(initial[m][1], scales[v])
                else:
                    idx = [i for i, x in enumerate(self.initnames) if x == gaussian]
                    for m in idx:
                        pos[i, m] = sp.random.normal(initial[m][1], scales[gaussian])

        dimensions = len(self.initnames)

        # Set-up Recovery File
        filename = self.planet + "_backend.h5"
        backend = emcee.backends.HDFBackend(filename)

        if "recover" in args:
            pass
        else:
            backend.reset(nwalkers, dimensions)

        variables = [x for x in self._usr_in]
        variables.insert(0, 0.06)
        variables.insert(1, 0.8)
        variables.append(self.limb_dark_law)
        variables.append(self._ldlength)

        sampler = emcee.EnsembleSampler(nwalkers, dimensions, lnprob, args=(self._datadict, variables, self.sigmas,
                                                                            self._multi_lc), backend=backend)

        max_n = 3000
        # convergence things
        index = 0
        autocorr = np.empty(max_n)
        np.warnings.filterwarnings('ignore')

        old_tau = np.inf
        for position, prob, state in tqdm(sampler.sample(pos, iterations=max_n, store=True)):
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.10)
            if converged:
                break
            old_tau = tau

        samples = sampler.get_chain(discard=100, thin=15, flat=True)

        self._output = list(map(lambda b: (b[1], b[1] - b[0], b[2] - b[1]),
                                zip(*np.percentile(samples, [16, 50, 84], axis=0))))

        self._usr_in = together_limbdark(self._usr_in, self._ldlength)
        self.results = np.zeros(len(self.initnames)).tolist()
        for i in range(len(self.initnames)):
            self.results[i] = [self.initnames[i], self._output[i][0], self._output[i][1], self._output[i][2]]

        if int(0.1 * len(samples)) <= 1000:
            sample_size = int(0.1 * len(samples))
        else:
            sample_size = 1000

        ms = dict.fromkeys(self._datadict.keys(), list())
        vardict = {x: variables[:-1] for x in self._datadict.keys()}
        for s in tqdm(samples[np.random.randint(len(samples), size=sample_size)]):

            amp, scale = s[:2]

            for key in vardict.keys():
                for i in range(len(variables) - 4):
                    if self._multi_lc[key][i] is not np.nan and i == 1:
                        for n in range(variables[-1]):
                            vardict[key][i + 2][n] = s[self._multi_lc[key][i] + n]
                    elif self._multi_lc[key][i] is not np.nan:
                        vardict[key][i + 2] = s[self._multi_lc[key][i]]
                    else:
                        pass

                kernel = amp * george.kernels.Matern52Kernel(scale)
                gp = george.GP(kernel)
                gp.compute(self._datadict[key][0], self._datadict[key][2])
                model = run_batman(vardict[key], self._datadict[key][0], self.all_param_names)
                m = gp.sample_conditional(self._datadict[key][1] - model, self._datadict[key][3]) + model
                ms[key].append([m])

        for key in vardict.keys():
            arr = list()
            arr.append([x[0] for x in ms[key]])
            ms[key] = np.array(arr[0]).transpose(1,0)

            avg = list()
            stdv = list()
            for m in ms[key]:
                avg.append(np.mean(m))
                stdv.append(np.std(m))
            avg = np.array(avg)
            stdv = np.array(stdv)

            avg = avg.reshape(1, np.shape(avg)[0])
            self._datadict[key] = np.concatenate([self._datadict[key], avg])
            stdv = stdv.reshape(1, np.shape(stdv)[0])
            self._datadict[key] = np.concatenate([self._datadict[key], stdv])

        create_results_file(self.results, (self.planet + '.out'))
        os.remove(filename)

        return self

    def create_test_data(self, **kwargs):
        """
        Creates synthetic data set for use in example run and error analysis for BatSignal. The data is based off of
        parameter values in the input parameter file from which a light curve is produced using BATMAN. White noise is
        added to the data from a cauchy distribution, and red noise is added from either a sine wave or the real part of
        the inverse fourier transform of the white noise.

        :argument: fourier - Red noise based off of the real part of the inverse fourier transform of the white noise
        :argument: sine    - Red noise based off of a sine wave

        :return: Synthetic light curve saved in a file called "testcurve.txt" in pwd
        """

        inputs.rp = self._usr_in[0]
        inputs.limb_dark = "quadratic"
        inputs.u = self._usr_in[1]
        inputs.t0 = np.median(self.date)
        inputs.per = self._usr_in[3]
        inputs.a = self._usr_in[4]
        inputs.inc = self._usr_in[5]
        inputs.ecc = self._usr_in[6]
        inputs.w = self._usr_in[7]

        bats = batman.TransitModel(inputs, self.date)
        model = bats.light_curve(inputs)

        sigma = 0.0008
        n = len(self.date)
        if n % 2 != 0:
            m = n + 1
        else:
            m = n

        for i in range(2):
            noise = sigma * sp.random.standard_normal(m)
            red = np.fft.fft(noise)

            numu = m/2 + 1
            k = np.linspace(1, numu, numu)

            red = (red[:numu]/k).tolist()
            conj = (red[1:-1].conjugate()).tolist()
            red = red + conj
            red = np.real(np.fft.ifft(red))

            noise_model = model + noise + red

            sp.savetxt('whitetest' + str(i) + '.txt', zip(self.date, noise_model, noise+red))

        return self

    @noparams
    def enable(self, parameters):
        """
        Lets the user enable fitting for parameters of their choice without manually opening the parameter file

        :param parameters: Parameters that the user would like to enable fitting for.
        """

        file = open(self.input_param_file, 'r')
        config.readfp(file)
        section = config.sections()[0]
        u = eval(sp.array(config.items(section)[1][1]).tolist())
        file.close()

        if 'all' in parameters:
            ld = len(u)
            parameters = ['rp', 'u1', 't0', 'per', 'a', 'inc', 'ecc', 'w']
            if ld == 2:
                parameters.insert(2, 'u2')
            elif ld == 4:
                parameters.insert(2, 'u2')
                parameters.insert(3, 'u3')
                parameters.insert(4, 'u4')
            else:
                pass

        file = open(self.input_param_file, 'w')

        if 'rp' in parameters:
            config.set(section, 'Planet Radius', value=True)
        if 'u1' in parameters:
            u[0] = True
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 'u2' in parameters:
            u[1] = True
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 'u3' in parameters:
            u[2] = True
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 'u4' in parameters:
            u[3] = True
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 't0' in parameters:
            config.set(section, 'Time of Mid-Transit', value=True)
        if 'per' in parameters:
            config.set(section, 'Period', value=True)
        if 'a' in parameters:
            config.set(section, 'Scaled Semi-Major Axis', value=True)
        if 'inc' in parameters:
            config.set(section, 'Inclination', value=True)
        if 'ecc' in parameters:
            config.set(section, 'Eccentricity', value=True)
        if 'w' in parameters:
            config.set(section, 'Argument of Periastron Longitude', value=True)

        config.write(file)
        file.close()

        self.__init__(self.input_param_file, self.light_curve_file, self.planet)

    @noparams
    def disable(self, parameters):
        """
        Lets the user disable fitting for parameters of their choice without manually opening the parameter file

        :param parameters: Parameters that the user would like to disable fitting for.
        """

        file = open(self.input_param_file, 'r')
        config.readfp(file)
        section = config.sections()[0]
        u = eval(sp.array(config.items(section)[1][1]).tolist())
        file.close()

        if 'all' in parameters:
            ld = len(u)
            parameters = ['rp', 'u1', 't0', 'per', 'a', 'inc', 'ecc', 'w']
            if ld == 2:
                parameters.insert(2, 'u2')
            elif ld == 4:
                parameters.insert(2, 'u2')
                parameters.insert(3, 'u3')
                parameters.insert(4, 'u4')
            else:
                pass

        file = open(self.input_param_file, 'w')

        if 'rp' in parameters:
            config.set(section, 'Planet Radius', value=False)
        if 'u1' in parameters:
            u[0] = False
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 'u2' in parameters:
            u[1] = False
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 'u3' in parameters:
            u[2] = False
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 'u4' in parameters:
            u[3] = False
            config.set(section, 'Limb Darkening Coefficients', value=u)
        if 't0' in parameters:
            config.set(section, 'Time of Mid-Transit', value=False)
        if 'per' in parameters:
            config.set(section, 'Period', value=False)
        if 'a' in parameters:
            config.set(section, 'Scaled Semi-Major Axis', value=False)
        if 'inc' in parameters:
            config.set(section, 'Inclination', value=False)
        if 'ecc' in parameters:
            config.set(section, 'Eccentricity', value=False)
        if 'w' in parameters:
            config.set(section, 'Argument of Periastron Longitude', value=False)

        config.write(file)
        file.close()

        self.__init__(self.input_param_file, self.light_curve_file, self.planet)

    def plot_model(self):

        fig, ax = plt.subplots(len(self._datadict.keys()), 2)

        for i in range(len(self._datadict.keys())):
            key = 'transit' + str(i)
            shift = self.t0s[key] - self.t0s['transit0']
            low = self._datadict[key][4] - self._datadict[key][5]
            high = self._datadict[key][4] - self._datadict[key][5]
            res = self._datadict[key][1] - self._datadict[key][4]

            _ = ax[i][0].plot(self._datadict[key][0] + shift, self._datadict[key][1], 'o', c='#4bd8ce', alpha=0.5)
            _ = ax[i][0].plot(self._datadict[key][0] + shift, self._datadict[key][4], c='#5d1591')
            _ = ax[i][0].fill_between(self._datadict[key][0] + shift, high, low, alpha=0.4, edgecolor='#7619b8',
                                      facecolor='#ae64e3')
            _ = ax[i][1].plot(self._datadict[key][0] + shift, res, 'o', c='#5d1591')
            ax[i][1].yaxis.set_major_locator(plt.MaxNLocator(5))
            yticks = ax[i][0].yaxis.get_major_ticks()
            yticks[-1].label1.set_visible(False)
            ax[i][1].yaxis.set_major_locator(plt.MaxNLocator(5))
            yticks = ax[i][1].yaxis.get_major_ticks()
            yticks[-1].label1.set_visible(False)

        plt.savefig(self.planet + "_model.png")
        plt.show()

    def plot_walkers(self):
        """
        Plots a graphical representation of the walkers as they explored the parameter space for the MCMC

        :return: Saves figure as "chain.png" in current working directory.
        """

        samples = self._samples
        n = int(len(self.names_change) / 2)
        remainder = len(self.names_change) % 2
        count = 2

        if remainder == 0:
            f, ax = plt.subplots(n + 1, 2)
        else:
            f, ax = plt.subplots(n + 2, 2)

        _ = ax[0][0].plot(np.swapaxes(samples[:, 0], 0, 1))
        ax[0][0].set_title('amp')
        _ = ax[0][1].plot(np.swapaxes(samples[:, 1], 0, 1))
        ax[0][1].set_title('scale')

        for i in range(n):
            _ = ax[i + 1][0].plot(np.swapaxes(samples[:, count], 0, 1))
            ax[i + 1][0].set_title(self.names_change[count - 2])
            count += 1
            _ = ax[i + 1][1].plot(np.swapaxes(samples[:, count], 0, 1))
            ax[i + 1][1].set_title(self.names_change[count - 2])
            count += 1

        if remainder == 1:
            _ = ax[n + 1][0].plot(np.swapaxes(samples[:, count], 0, 1))
            ax[n + 1][0].set_title(self.names_change[count - 2])
        else:
            pass

        plt.savefig("chain.png")
        plt.show()

    def plot_corner(self):
        """
        Plots corner plot depicting the parameter space explored by the models

        :return: Saves figure as corner.png in current working directory
        """

        variables = self.variables[:]
        names = self.names_change[:]
        samples = self._samples

        variables.insert(0, -5)
        variables.insert(1, -2)

        names.insert(0, 'amp')
        names.insert(1, 'scale')

        fig = corner.corner(samples, truths=variables, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig.set_size_inches(10, 10)
        fig.savefig("corner.png")
        plt.show()
