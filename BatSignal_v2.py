import configparser
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate
from tabulate import tabulate
import numpy as np
import time
import corner
import batman
import emcee
import george
import pdb

config = configparser.RawConfigParser()
inputs = batman.TransitParams()

def create_param_file(newfile=''):
    """
    Creates a blank parameter file for the user to fill.

    :param newfile: Name of the file that will be created

    :return: Writes file to pwd
    """

    # if newfile == '':
    #    newfile = input("What would you like your file name to be? ")

    config.add_section('What to Fit')
    config.set('What to Fit', 'Planet Radius                      ', 'False')
    config.set('What to Fit', 'Limb Darkening Coefficients        ', '[False, False]')
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
    config.set('Planet Name', 't0                      ', '[2400000.0,  error]')
    config.set('Planet Name', 'per                     ', '[1.0,        error]')
    config.set('Planet Name', 'a                       ', '[0.0,        error]')
    config.set('Planet Name', 'inc                     ', '[0.5*sp.pi,  error]')
    config.set('Planet Name', 'ecc                     ', '[0.0,        error]')
    config.set('Planet Name', 'w                       ', '[0.0,        error]')

    config.add_section('References')
    config.set('References', 'r1', '')
    config.set('References', 'r2', '')


    with open(newfile, 'wt') as configfile:
        # config.write(bytes(configfile, 'UTF-8'))
        config.write(configfile)


def create_results_file(results, planet, newfile):

    f = open(newfile, 'w')
    f.write(tabulate(results, tablefmt="latex", floatfmt=".5f"))
    f.close()


def separate_limbdark(variables):
    """
    Separates the Limb Darkening variables from the input structure to a form usable by the MCMC code.

    :param variables: List of variables provided from BatSignal class, containing a nested array for the limb darkening
        variables.

    :return: List of variables without the nested array.
    """

    ld = variables.pop(1)
    try:
        length = len(ld)
        for i in range(length):
            variables.insert(i + 1, ld[i])
    except TypeError:
        length = 1
        variables.insert(1, ld)


    return variables, length


def together_limbdark(variables, length):
    """
    Returns the Limb Darkening variables to the original nested structure.

    :param variables: List of variables provided from BatSignal class, with the limb darkening variables separated.
    :param length: Number of limb darkening coefficients

    :return: List of variables with the limb darkening coefficients in a nested array.
    """
    ld = []
    for i in range(length):
        ld.append(variables.pop(1))

    variables.insert(1, ld)

    return variables


def run_batman(inputs, variables, names, date, change, usr_in):
    if 'u' in names:
        inputs.u = []
    for n in variables:
        if n[0] == 'rp':
            inputs.rp = n[1]
        elif n[0][0] == 'u':
            inputs.u.append(n[1])
        elif n[0][0] == 't0':
            inputs.t0 = n[1]
        elif n[0][0] == 'per':
            inputs.per = n[1]
        elif n[0][0] == 'a':
            inputs.a = n[1]
        elif n[0][0] == 'inc':
            inputs.inc = n[1]
        elif n[0][0] == 'ecc':
            inputs.ecc = n[1]
        elif n[0][0] == 'w':
            inputs.w = n[1]
            pass
        else:
            pass

        if inputs.limb_dark == 'nonlinear':
            if len(inputs.u) != 4:
                for i in enumerate(change[1]):
                    if not i[1]:
                        inputs.u.insert(i[0], usr_in[i[0] + 1])
        elif inputs.limb_dark == 'quadratic' or inputs.limb_dark == 'squareroot' or inputs.limb_dark == 'logarithmic' or \
                        inputs.limb_dark == 'exponential' or inputs.limb_dark == 'power2':
            if len(inputs.u) != 2:
                for i in enumerate(change[1]):
                    if not i[1]:
                        inputs.u.insert(i[0], usr_in[i[0] + 1])

    bats = batman.TransitModel(inputs, date)
    model = bats.light_curve(inputs)

    return model


def lnprior(variables, sigma, error_nlvl, error_scale):
    """
    Verifies that the new guesses for the parameters are within a reasonable (3sigma) range of the original guesses.
    Then determines a likelihood value related to how distant the new guesses are from the originals.

    :param theta: theta[0] and theta[1] are variables for scaling the kernel for the gaussian process part. 
                  theta[2:] are the new guesses for the parameters being fitting for each chain.
    :param variables: guesses of the user for the parameters.
    :param sigma: values of error from the user guesses in the input file.
    :param error_nlvl: error on the amplitude parameter for the gaussian process regression
    :param error_scale: error on the scale parameter for the gaussian process regression

    :return: Likelihood value based off of the distance between the new guesses and the original input.
    """

    value = list()
    for i in variables.keys():

        val = 0.0
        if i == 'amp':
            if sp.log(error_nlvl[0]) < variables[i][1] < sp.log(error_nlvl[1]):
                val = val - (variables[i][1] - 5) ** 2 / 2 * 0.3 * sp.log(error_nlvl[1])
            else:
                val = -sp.inf
            value.append(val)

        elif i == 'scale':
            if val != -sp.inf and sp.log(error_scale[0]) < variables[i][1] < sp.log(error_scale[1]):
                val = val - (variables[i][1] - 0.1) ** 2 / 2 * 0.3 * sp.log(error_scale[1])
            else:
                val = -sp.inf
            value.append(val)

        else:
            if len(variables[i]) == 2:
                if val != -sp.inf and variables[i][0] - 3 * sigma[i] < variables[i][1] < variables[i][0] + 3 * sigma[i]:
                    val = val - (variables[i][1] - variables[i][0]) ** 2 / (2 * (sigma[i] ** 2))
                else:
                    val = -sp.inf
                value.append(val)
            else:
                for n in range(int(len(variables[i]) / 2)):
                    if val != -sp.inf and variables[i][0 + n * 2] - 3 * sigma[i] < variables[i][1 + n * 2] < \
                                    variables[i][0 + n * 2] + 3 * sigma[i]:
                        val = val - (variables[i][1 + n * 2] - variables[i][0 + n * 2]) ** 2 / (2 * (sigma[i] ** 2))
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


def lnlike(variables, date_real, date, flux, error, ins, change, usr, dict, multi):
    """
    Changes the inputs for batman to the new guesses. Computes a new model for the light curve and uses Gaussian
    Process Regression to determine the likelihood of the model as a whole.

    :param theta: theta[0] and theta[1] are variables for scaling the kernel for the gaussian process part.
                  theta[2:] are the new guesses for the parameters being fitting for each chain
    :param names: names of the variables the user is changing.
    :param date_real: Evenly spaced time intervals for x axis
    :param date: Time of observations, x axis
    :param flux: Light collected from star, y axis
    :param error: Errors on the observations
    :param ins: Input parameters to be read by BATMAN
    :param change: The list of parameters the user is fitting for
    :param usr: The user input values

    :return: A number representing the Gaussian Process liklihood of the model being correct
    """

    val = 0.0
    for i in range(len(dict.keys()) + 1):

        if i == 0:
            pass
        else:
            key = 'data' + str(i)
            date = dict[key][3]
            date_real = dict[key][0]
            flux = dict[key][1]
            error = dict[key][2]

        names = []
        v = []
        count = dict.fromkeys(multi.keys(), 0)

        for k in variables.keys():
            if k == 'amp' or k == 'scale':
                pass
            else:
                if multi[k][i] == 0:
                    pass
                elif multi[k][i] == 1:
                    names = np.append(names, k)
                    if k == 'u':
                        if ins.limb_dark == 'nonlinear':
                            names = np.append(names, k)
                            names = np.append(names, k)
                            names = np.append(names, k)
                            v.append(variables[k][1 + count[k] * 8])
                            v.append(variables[k][3 + count[k] * 8])
                            v.append(variables[k][5 + count[k] * 8])
                            v.append(variables[k][7 + count[k] * 8])
                        elif ins.limb_dark == 'quadratic' or ins.limb_dark == 'squareroot' or ins.limb_dark == \
                                'logarithmic' or ins.limb_dark == 'exponential' or ins.limb_dark == 'power2':
                            names = np.append(names, k)
                            v.append(variables[k][1 + count[k] * 4])
                            v.append(variables[k][3 + count[k] * 4])
                        elif ins.limb_dark == 'linear':
                            v.append(variables[k][1 + count[k] * 2])
                        else:
                            pass

                    else:
                        v.append(variables[k][1 + count[k] * 2])
                    count[k] += 1
                elif multi[k][i] == 2:
                    names = np.append(names, k)
                    if k == 'u':
                        if ins.limb_dark == 'nonlinear':
                            names = np.append(names, k)
                            names = np.append(names, k)
                            names = np.append(names, k)
                            v.append(variables[k][1 + count[k] * 8])
                            v.append(variables[k][3 + count[k] * 8])
                            v.append(variables[k][5 + count[k] * 8])
                            v.append(variables[k][7 + count[k] * 8])
                        elif ins.limb_dark == 'quadratic' or ins.limb_dark == 'squareroot' or ins.limb_dark == \
                                'logarithmic' or ins.limb_dark == 'exponential' or ins.limb_dark == 'power2':
                            names = np.append(names, k)
                            v.append(variables[k][1 + count[k] * 4])
                            v.append(variables[k][3 + count[k] * 4])
                        elif ins.limbdark == 'linear':
                            v.append(variables[k][1 + count[k] * 2])
                        else:
                            pass
                    else:
                        v.append(variables[k][1 + count[k] * 2])
                else:
                    raise ValueError('Please use only the following values for the multitransit section: 0 - You do not'
                                     'want to fit the parameter for that specific transit, 1 - You would like to fit the'
                                     'parameter independantly from the other transits, or 2 - You would like to force the'
                                     'parameter to be the same as another transit as the parameter is fit.')

        model = run_batman(ins, zip(names, v), names, date, change, usr)

        tck = interpolate.splrep(date, model, s=0)
        model_real_times = interpolate.splev(date_real, tck, der=0)

        kernel = variables['amp'][1] * george.kernels.Matern52Kernel(variables['scale'][1])
        gp = george.GP(kernel)
        gp.compute(date_real, error)

        val += gp.lnlikelihood(flux - model_real_times)

    return val


def lnprob(theta, ins, date_real, date, flux, error, variables, usr_err_dict, change, usr_in, error_nlvl,
           error_scale, datadict, multi):
    """
    Returns the combined likelihoods from lnprior and lnlike. If this number is smaller than the previous model, this
    model is saved as the most probable fit of the data.

    :param theta: theta[0] and theta[1] are variables for scaling the kernel for the gaussian process part.
        theta[2:] are the new guesses for the parameters being fitting for each chain
    :param ins: Input parameters to be read by BATMAN
    :param names: names of the variables the user is changing.
    :param date_real: Evenly spaced time intervals for x axis
    :param date: Time of observations, x axis
    :param flux: Light collected from star, y axis
    :param error: Errors on the observations
    :param variables: List of variables that the user is fitting for
    :param usr_err: Error on the variables suggested by the user
    :param change: The list of parameters the user is fitting for
    :param usr_in: The user input values
    :param error_nlvl: error for nlvl parameter for the gaussian process regression
    :param error_scale: error for scale parameter for the gaussian procdess regression

    :return: A value representing the liklihood of the model being thr true fit to the data. This value should be
        minimized over time.
    """

    v = zip(*variables)
    variables = {}
    for i in range(len(v[0])):
        try:
            variables[v[0][i]].append(v[1][i])
            variables[v[0][i]].append(theta[i])
        except KeyError:
            variables[v[0][i]] = [v[1][i], theta[i]]

    if datadict is None:
        datadict = {}

    ln_prior = lnprior(variables, usr_err_dict, error_nlvl, error_scale)
    if sp.isfinite(ln_prior):
        ln_like = lnlike(variables, date_real, date, flux, error, ins, change, usr_in, datadict, multi)
        return ln_prior + ln_like
    else:
        return -sp.inf


def show_chain(sampler, names):
    """
    Plots a graphical representation of the walkers as they explored the parameter space for the MCMC

    :param sampler: Walker information from the emcee code as provided by BatSignal.
    :param names: Names of the varibles being fit for

    :return: Saves figure as "chain.png" in current working directory.
    """

    n = int(len(names) / 2)
    remainder = len(names) % 2
    count = 2

    if remainder == 0:
        f, ax = plt.subplots(n + 1, 2)
    else:
        f, ax = plt.subplots(n + 2, 2)

    _ = ax[0][0].plot(sampler.chain[:, :, 0])
    ax[0][0].set_title('amp')
    _ = ax[0][1].plot(sampler.chain[:, :, 1])
    ax[0][1].set_title('scale')

    for i in range(n):
        _ = ax[i + 1][0].plot(sampler.chain[:, :, count])
        ax[i + 1][0].set_title(names[count - 2])
        count += 1
        _ = ax[i + 1][1].plot(sampler.chain[:, :, count])
        ax[i + 1][1].set_title(names[count - 2])
        count += 1

    if remainder == 1:
        _ = ax[n + 1][0].plot(sampler.chain[:, :, count])
        ax[n + 1][0].set_title(names[count - 2])
    else:
        pass

    plt.savefig("chain.png")
    plt.show()


def show_corner(samples, variables, names):
    """
    Plots corner plot depicting the parameter space explored by the models

    :param samples: Sampler from emcee
    :param variables: List of variables that the user is fitting for
    :param names:names of the variables the user is changing.

    :return: Saves figure as corner.png in current working directory
    """

    variables.insert(0, -5)
    variables.insert(1, -2)

    names.insert(0, 'amp')
    names.insert(1, 'scale')

    fig = corner.corner(samples, truths=variables, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.set_size_inches(10, 10)
    fig.savefig("corner.png")
    plt.show()


def normalize_flux(per, a, t0, date, flux):
    """
    Normalizes the flux by the median of the baseline
    :param per: Period
    :param a: Scaled semi-major axis
    :param t0: Time of mid transit
    :param date: Time of observations, x axis
    :param flux: Light collected from star, y axis

    :return: Flux divided by the median of the transit's baseline
    """

    # Compute the mean value of flux
    mean_flux = np.mean(flux)

    # Determinate the points above the mean of the flux
    flux_upper = flux[flux >= mean_flux]
    date_upper = date[flux >= mean_flux]

    # Compute the median of the points above the mean of the flux
    median_flux = np.median(flux_upper)

    # Normalize the flux by the median of the no_transit flux
    norm_flux = flux / median_flux

    return norm_flux


def compute_tzero(date, per, t0):
    """
    Computes the time of mid transit for secondary transits from the value given for the primary and the period of the
    orbit.
    :param date: Time of observations, x axis
    :param per: Period
    :param t0: Original time of mid transit

    :return: Time of mid transit for a secondary transit
    """

    medtime = np.median(date)
    number_transit_fraction = ((medtime - t0) / per) + 0.5
    number_transit = int(number_transit_fraction)
    t0_new = t0 + (per * number_transit)

    return t0_new


def multiply_one_one(a, b):
    """
    Multiplication of a list of numbers by numbers in a list of equal length

    :param a: array of numbers
    :param b: array of numbers of equal length as a

    :return: array of multiplication results
    """
    c = []
    for i in range(len(a)):
        c.append(a[i] * b[i])
    return c


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
            input_param_file = input('What is the name of the file containing your input parameters? ')

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

        if isinstance(light_curve_file, (list, tuple)):
            self._dict = {}
            self.out = {}
            data = sp.loadtxt(light_curve_file[0], unpack=True)
            for i in range(len(light_curve_file) - 1):
                n = 'data' + str(i + 1)
                self._dict[n] = sp.loadtxt(light_curve_file[i + 1], unpack=True)
                col = np.linspace(self._dict[n][0][0], self._dict[n][0][-1], len(self._dict[n][0]))
                col = col.reshape(1, np.shape(col)[0])
                self._dict[n] = np.concatenate([self._dict[n], col])
        elif isinstance(light_curve_file, str):
            self._dict = None
            self.out = None
            data = sp.loadtxt(light_curve_file, unpack=True)
        else:
            raise TypeError('The light curve file should be either a string or list')

        if len(data) == 2:
            error = []
        elif len(data) == 3:
            error = data[2]
        else:
            raise ValueError("The input file should contain a column with the Julian date, \
                 a column of data, and an optional error column only.")

        if input_param_file[-4:] != '.cfg':
            raise TypeError('The parameter file should be of format ".cfg". Run create_param_file() for an example.')

        self.date_real = data[0]
        self.date = np.asarray(self.date_real)
        self.date = np.linspace(self.date_real[0], self.date_real[-1], len(self.date))

        self.input_param_file = input_param_file
        self.light_curve_file = light_curve_file
        self._median = sp.median(self.date_real)
        self.flux = data[1]
        self.error = error
        self.model = list()
        self.results = list()
        self.variables = list()
        self._steps_burn_in = 600 #600
        self._steps_full = 1800 #1800
        self._error_nlvl = [1., 5000.]
        self._error_scale = [1., 100.]


        # Reads in the user's guesses for the planet system parameters
        config.readfp(open(self.input_param_file))
        if planet == '':
            section = config.sections()[2]
        else:
            section = planet

        self.planet = section

        self.names = sp.array(config.items(section))[:, 0].tolist()
        self.names = [str(self.names[i]) for i in range(len(self.names))]

        self._usr = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
        self._usr_in, self._ldlength = separate_limbdark([i[0] for i in self._usr])
        self._usr_err, _ = separate_limbdark([i[1] for i in self._usr])

        # Relax is a factor for sigma in each value, it can be modified by the user with update_relax method
        self.relax = np.ones(len(self._usr_err))
        self._usr_err_relax = multiply_one_one(self._usr_err, self.relax)

        # Determines which parameters the user would like to fit
        section = config.sections()[0]
        self._usr_change = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
        c = self._usr_change[:]
        c, _ = separate_limbdark(c)
        self._change = sp.where(c)[0]

        # Separates the limb darkening arrays into individual parameters
        self._usr_in = together_limbdark(self._usr_in, self._ldlength)

        section = config.sections()[1]
        multi = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
        self._multi = dict(zip(self.names, multi))

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
        :keyword: "third_light" - background light level added from a third body
        """

        for key, value in kwargs.items():

            lngth = len(self.relax)
            if lngth == 11:
                count = 3
            elif lngth == 9:
                count = 1
            else:
                count = 0

            if key == "rp":
                self.relax[0] = value
            elif key == "u":
                for i in range(count + 1):
                    self.relax[i + 1] = value
            elif key == "t0":
                self.relax[count + 2] = value
            elif key == "per":
                self.relax[count + 3] = value
            elif key == "a":
                self.relax[count + 4] = value
            elif key == "inc":
                self.relax[count + 5] = value
            elif key == "ecc":
                self.relax[count + 6] = value
            elif key == "w":
                self.relax[count + 7] = value
        print(self.relax)

    def bat(self, *args, **kwargs):
        """
        Runs fit for light curve model using MCMC and Gaussian Process Regression.

        :argument: quadratic    - quadratic mode for limb darkening (default)
        :argument: nonlinear    - nonlinear mode for limb darkening
        :argument: linear       - linear mode for limb darkening
        :argument: uniform      - uniform mode for limb darkening
        :argument: squareroot   - square root mode for limb darkening
        :argument: logarithmic  - logarithmic mode for limb darkening
        :argument: exponential  - exponential mode for limb darkening
        :argument: corner       - saves corner plot in pwd
        :argument: chain        - saves walker plot in pwd
        :argument: model        - saves plot of data and model in pwd

        :keyword: steps_burn_in - new value for number of MCMCs during burn-in phase
        :keyword: steps_full    - new value for number of MCMCs during full run
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

        # Prepares the inputs for the initial model
        inputs = batman.TransitParams()
        inputs.rp = self._usr_in[0]
        inputs.limb_dark = "quadratic"
        inputs.u = self._usr_in[1]
        inputs.t0 = self._usr_in[2]
        inputs.per = self._usr_in[3]
        inputs.a = self._usr_in[4]
        inputs.inc = self._usr_in[5]
        inputs.ecc = self._usr_in[6]
        inputs.w = self._usr_in[7]

        # Computes the t0 for the current transit
        if self._dict is not None:
            self.t0s = [compute_tzero(self.date, inputs.per, inputs.t0)]
            for k in sorted(self._dict.keys()):
                day = self._dict[k][0]
                self.t0s.append(compute_tzero(day, inputs.per, inputs.t0))

        inputs.t0 = compute_tzero(self.date, inputs.per, inputs.t0)
        self._usr_in[2] = compute_tzero(self.date, self._usr_in[3], self._usr_in[2])

        # Normalize flux
        self.flux = normalize_flux(inputs.per, inputs.a, inputs.t0, self.date, self.flux)

        if 'nonlinear' in args:
            inputs.limb_dark = "nonlinear"
            name1 = [self.names[i] for i in self._change if i <= 1]
            name2 = ['u' for i in self._change if 1 < i <= 4]
            name3 = [self.names[i - 3] for i in self._change if i > 4]
            self.names = name1 + name2 + name3
        elif 'linear' in args:
            inputs.limb_dark = "linear"
            self.names = [self.names[i] for i in self._change]
        elif 'uniform' in args:
            inputs.limb_dark = "uniform"
            self.names = [self.names[i] for i in self._change]
        else:
            if 'squareroot' in args:
                inputs.limb_dark = "squareroot"
            if 'logarithmic' in args:
                inputs.limb_dark = "logarithmic"
            if 'exponential' in args:
                inputs.limb_dark = "exponential"
            if 'power2' in args:
                inputs.limb_dark = "power2"
            name1 = [self.names[i] for i in self._change if i <= 1]
            name2 = [self.names[i - 1] for i in self._change if i > 1]
            self.names = name1 + name2

        self._usr_in, _ = separate_limbdark(self._usr_in)
        self.variables = [self._usr_in[i] for i in self._change]
        self._usr_err = [self._usr_err[i] for i in self._change]

        for key, value in kwargs.items():
            if key == "steps_burn_in":
                self._steps_burn_in = value
            elif key == "steps_full":
                self._steps_full = value
            elif key == "err_nlvl":
                self._error_nlvl = value
            elif key == "err_scale":
                self._error_scale = value
            elif key == 'uniform':
                uniform = value
            elif key == 'gaussian' or key == 'normal':
                gaussian = value
            else:
                pass


        # Initializes the model and uses the user's guesses to model the transit
        bats = batman.TransitModel(inputs, self.date)
        self.model = bats.light_curve(inputs)

        # Determines the number of walkers to use and number of variables to fit
        ndim = len(self.variables)

        self._usr_err_relax = multiply_one_one(self._usr_err, self.relax)
        self._usr_err_dict = dict(zip(self.names, self._usr_err_relax))

        # Initial values for nlvl and scale
        initial = np.array([0.06, 0.8])


        # Add values of the variables
        if self._dict is not None:
            initnames = ('amp', 'scale')
            for i in range(len(self._dict.keys())+1):
                nums = None
                for n in self.names:
                    trust = self._multi[n][i]
                    if n == 't0':
                        variable = self.t0s[i]
                    if n == 'u':
                        if nums != None:
                            pass
                        else:
                            nums = [x for x, y in enumerate(self.names) if y == 'u']
                            variable = [self.variables[x] for x in nums]
                    else:
                        variable = self.variables[self.names.index(n)]
                    if trust == 1:
                        initial = np.append(initial, variable)
                        initnames = np.append(initnames, n)
                    elif trust == 2:
                        if n == 'u':
                            check = [x for x,y in enumerate(initnames) if y =='u']
                            if len(check) >= len(variable):
                                pass
                            else:
                                for v in variable:
                                    initial = np.append(initial, v)
                                    initnames = np.append(initnames, n)
                        if n in initnames:
                            pass
                        else:
                            initial = np.append(initial, variable)
                            initnames = np.append(initnames, n)
                    else:
                        pass
        else:
            for i in range(ndim):
                initial = np.append(initial, self.variables[i])
            initnames = ('amp', 'scale')
            initnames = np.append(initnames, [n for n in self.names])

        initial = zip(initnames, initial)

        nwalkers = len(initnames)*10

        pos = np.zeros([nwalkers, len(initnames)])
        scales = dict.fromkeys(['amp', 'scale', 'rp', 'per', 'a', 't0', 'ecc', 'w'], 1e-2)
        scales['u'] = 1e-3
        scales['inc'] = 5.
        for i in range(nwalkers):
            for j in enumerate(initnames):
                if j[1] == 'amp' or j[1] == 'scale' or j[1] == 't0':
                    pos[i, j[0]] = initial[j[0]][1] + scales[j[1]] * sp.random.randn(1)
                else:
                    pos[i, j[0]] = sp.random.normal(initial[j[0]][1], scales[j[1]])

            if 'noprior' in locals():
                scales = dict.fromkeys(['amp', 'scale', 'rp', 'per', 'a', 't0', 'ecc', 'w'], 1.)
                scales['u'] = 1e-2
                scales['inc'] = 10.
                if isinstance(noprior, (list, tuple)):
                    for v in noprior:
                        idx = [i for i, x in enumerate(initnames) if x == v]
                        for m in idx:
                            pos[i, m] = initial[m][1] + scales[v] * sp.random.randn(1)
                else:
                    idx = [i for i, x in enumerate(initnames) if x == noprior]
                    for m in idx:
                        pos[i, m] = initial[m][1] + scales[uniform] * sp.random.randn(1)


            if 'uniform' in locals():
                if isinstance(uniform, (list, tuple)):
                    for v in uniform:
                        idx = [i for i, x in enumerate(initnames) if x == v]
                        for m in idx:
                            pos[i, m] = initial[m][1] + scales[v] * sp.random.randn(1)
                else:
                    idx = [i for i, x in enumerate(initnames) if x == uniform]
                    for m in idx:
                        pos[i, m] = initial[m][1] + scales[uniform] * sp.random.randn(1)

            if 'gaussian' in locals():
                if isinstance(gaussian, (list, tuple)):
                    for v in gaussian:
                        idx = [i for i, x in enumerate(initnames) if x == v]
                        for m in idx:
                            pos[i, m] = sp.random.normal(initial[m][1], scales[v])
                else:
                    idx = [i for i, x in enumerate(initnames) if x == gaussian]
                    for m in idx:
                        pos[i, m] = sp.random.normal(initial[m][1], scales[gaussian])

        dimensions = len(initnames)

        # Run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, dimensions, lnprob, args=(inputs, self.date_real,
                                                                            self.date, self.flux, self.error,
                                                                            initial, self._usr_err_dict,
                                                                            self._usr_change, self._usr_in,
                                                                            self._error_nlvl, self._error_scale,
                                                                            self._dict, self._multi))

        time0 = time.time()
        position, prob, state = sampler.run_mcmc(pos, self._steps_burn_in)
        sampler.reset()
        time1 = time.time()
        print(time1 - time0)

        time0 = time.time()
        position, prob, state = sampler.run_mcmc(position, self._steps_full)
        time1 = time.time()
        print(time1 - time0)
        samples = sampler.flatchain

        if "chain" in args:
            show_chain(sampler, self.names)

        if "corner" in args:
            show_corner(samples, self.variables[:], self.names[:])

        self._output = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                zip(*sp.percentile(samples, [16, 50, 84], axis=0))))

        self._usr_in = together_limbdark(self._usr_in, self._ldlength)
        self.results = np.zeros(len(initnames)).tolist()
        for i in range(len(initnames)):
            self.results[i] = [initnames[i], self._output[i][0], self._output[i][1], self._output[i][2]]

        if self._dict is not None:
            for idx in range(len(self._dict.keys()) + 1):

                if idx == 0:
                    date = self.date
                    date_real = self.date_real
                    flux = self.flux
                    error = self.error
                else:
                    key = 'data' + str(idx)
                    date = self._dict[key][3]
                    date_real = self._dict[key][0]
                    flux = self._dict[key][1]
                    error = self._dict[key][2]


                ms = list()
                for n in samples[np.random.randint(len(samples), size=int(0.1 * len(samples)))]:
                    # nlvl, scale = np.exp(n[:2])

                    variables = {}
                    for i in range(len(initnames)):
                        try:
                            variables[initnames[i]].append(n[i])
                        except KeyError:
                            variables[initnames[i]] = [n[i]]

                    names = []
                    v = []
                    count = dict.fromkeys(self._multi.keys(), 0)
                    for k in variables.keys():
                        if k == 'amp' or k == 'scale':
                            pass
                        else:
                            if self._multi[k][idx] == 0:
                                pass
                            elif self._multi[k][idx] == 1:
                                names = np.append(names, k)
                                if k == 'u':
                                    if inputs.limb_dark == 'nonlinear':
                                        names = np.append(names, k)
                                        names = np.append(names, k)
                                        names = np.append(names, k)
                                        v.append(variables[k][0 + count[k] * 4])
                                        v.append(variables[k][1 + count[k] * 4])
                                        v.append(variables[k][2 + count[k] * 4])
                                        v.append(variables[k][3 + count[k] * 4])
                                    elif inputs.limb_dark == 'quadratic' or inputs.limb_dark == 'squareroot' or inputs.limb_dark == \
                                            'logarithmic' or inputs.limb_dark == 'exponential' or inputs.limb_dark == 'power2':
                                        names = np.append(names, k)
                                        v.append(variables[k][0 + count[k] * 2])
                                        v.append(variables[k][1 + count[k] * 2])
                                    elif inputs.limb_dark == 'linear':
                                        v.append(variables[k][0 + count[k]])
                                    else:
                                        pass

                                else:
                                    v.append(variables[k][0 + count[k]])
                                count[k] += 1
                            elif self._multi[k][idx] == 2:
                                names = np.append(names, k)
                                if k == 'u':
                                    if inputs.limb_dark == 'nonlinear':
                                        v.append(variables[k][0 + count[k] * 4])
                                        v.append(variables[k][1 + count[k] * 4])
                                        v.append(variables[k][2 + count[k] * 4])
                                        v.append(variables[k][3 + count[k] * 4])
                                        names = np.append(names, 'u')
                                        names = np.append(names, 'u')
                                        names = np.append(names, 'u')
                                    elif inputs.limb_dark == 'quadratic' or inputs.limb_dark == 'squareroot' or inputs.limb_dark == \
                                            'logarithmic' or inputs.limb_dark == 'exponential' or inputs.limb_dark == 'power2':
                                        v.append(variables[k][0 + count[k] * 2])
                                        v.append(variables[k][1 + count[k] * 2])
                                        names = np.append(names, 'u')
                                    elif inputs.limbdark == 'linear':
                                        v.append(variables[k][0 + count[k]])
                                    else:
                                        pass
                                else:
                                    v.append(variables[k][0 + count[k]])
                            else:
                                raise ValueError(
                                    'Please use only the following values for the multitransit section: 0 - You do not'
                                    'want to fit the parameter for that specific transit, 1 - You would like to fit the'
                                    'parameter independantly from the other transits, or 2 - You would like to force the'
                                    'parameter to be the same as another transit as the parameter is fit.')

                    amp, scale = n[:2]
                    kernel = amp * george.kernels.Matern52Kernel(scale)
                    gp = george.GP(kernel)
                    gp.compute(date, error)
                    model = run_batman(inputs, zip(names, v), names, date, self._usr_change, self._usr_in)
                    m = gp.sample_conditional(flux - model, date) + model
                    ms.append(m)

                ms = np.array(ms).transpose(1, 0)
                avg = list()
                stdv = list()
                for m in ms:
                    avg.append(np.mean(m))
                    stdv.append(np.std(m))
                avg = np.array(avg)
                stdv = np.array(stdv)

                if idx == 0:
                    self.model = avg
                    self.stdv = stdv

                else:
                    avg = avg.reshape(1, np.shape(avg)[0])
                    self._dict[key] = np.concatenate([self._dict[key], avg])
                    stdv = stdv.reshape(1, np.shape(stdv)[0])
                    self._dict[key] = np.concatenate([self._dict[key], stdv])

        else:
            ms = list()
            for n in samples[np.random.randint(len(samples), size=int(0.1 * len(samples)))]:
                amp, scale = n[:2]
                kernel = amp * george.kernels.Matern52Kernel(scale)
                gp = george.GP(kernel)
                gp.compute(self.date, self.error)
                model = run_batman(inputs, zip(self.names, n[2:]), self.names, self.date, self._usr_change,
                                   self._usr_in)
                m = gp.sample_conditional(self.flux - model, self.date) + model
                ms.append(m)

            ms = np.array(ms).transpose(1, 0)
            avg = list()
            stdv = list()
            for m in ms:
                avg.append(np.mean(m))
                stdv.append(np.std(m))
            avg = np.array(avg)
            stdv = np.array(stdv)

            self.model = avg
            self.stdv = stdv

        create_results_file(self.results, self.planet, (self.planet + '.out'))

        return self

    def create_test_data(self, *args, **kwargs):
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

        idx = (np.abs(self.date - inputs.t0)).argmin()
        diff = model[0] - model[idx]

        if kwargs:
            for k in kwargs:
                if k == 'sigma':
                    percent = [kwargs[k]]
        else:
            # percent = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
            percent = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05]

#        count = 0
#        for i in percent:
#            count += 1
#            sigma = i * diff
#            noise = sigma * sp.random.standard_normal(len(self.date))

#            if "fourier" in args:
#                for n in range(len(noise)):
#                    if -0.018 >= noise[n] or noise[n] >= 0.018:
#                        noise[n] = i
#                red = 8 * np.fft.fft(noise) ** 2
#                model += noise + red.real
#                sp.savetxt('SynthCurve' + str(i)[2:] + '.txt', zip(self.date, model, noise + red.real))
#            elif "sine" in args:
#                red = np.sin(2 * np.pi * self.date * 3 / max(self.date))
#                model += noise + red
#                sp.savetxt('SynthCurve' + str(i)[2:] + '.txt', zip(self.date, model, noise + red))
#            else:
#                model += noise
#                sp.savetxt('SynthCurve' + str(i)[2:] + '.txt', zip(self.date, model, noise))

        count = 0
        res = self.flux - self.model
        for i,p in enumerate(percent):
            count +=1
            model += res*p
            sp.savetxt('ResCurve' + str(i) + '.txt', zip(self.date, model, res*p))

        self.model = model

        return self

    @noparams
    def Enable(self, parameters):
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
    def Disable(self, parameters):
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

        if self._dict is None:
            low = self.model - self.stdv
            high = self.model + self.stdv
            res = self.flux - self.model

            fig, (plt1, plt2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            fig.subplots_adjust(hspace=0)

            plt1.plot(self.date, self.model, c='#5d1591')
            plt1.fill_between(self.date, high, low, alpha=0.3, edgecolor='#7619b8', facecolor='#ae64e3')
            plt1.plot(self.date, self.flux, 'o', c='#4bd8ce')

            plt2.plot(self.date, res, 'o', c='#5d1591')
            plt2.yaxis.set_major_locator(plt.MaxNLocator(3))
            yticks = plt2.yaxis.get_major_ticks()
            yticks[-1].label1.set_visible(False)

            fig.suptitle('BatSignal Output')
            plt2.set_xlabel('Julian Days')
            plt1.set_ylabel('Relative Flux')

            plt.savefig(self.planet + "_model.png")
            plt.show()

        else:
            n = len(self._dict.keys()) + 1
            fig, ax = plt.subplots(n, 2)

            for i in range(n):
                if i == 0:
                    low = self.model - self.stdv
                    high = self.model + self.stdv
                    res = self.flux - self.model

                    _ = ax[0][0].plot(self.date, self.model, c='#5d1591')
                    _ = ax[0][0].fill_between(self.date, high, low, alpha=0.3, edgecolor='#7619b8', facecolor='#ae64e3')
                    _ = ax[0][0].plot(self.date, self.flux, 'o', c='#4bd8ce')
                    _ = ax[0][1].plot(self.date, res, 'o', c='#5d1591')
                    ax[0][0].yaxis.set_major_locator(plt.MaxNLocator(5))
                    yticks = ax[0][0].yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
                    ax[0][1].yaxis.set_major_locator(plt.MaxNLocator(5))
                    yticks = ax[0][1].yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
                else:
                    key = 'data' + str(i)
                    shift = self.t0s[i] - self.t0s[0]
                    low = self._dict[key][4] - self._dict[key][5]
                    high = self._dict[key][4] - self._dict[key][5]
                    res = self._dict[key][1] - self._dict[key][4]


                    _ = ax[i][0].plot(self._dict[key][0] + shift, self._dict[key][4], c='#5d1591')
                    _ = ax[i][0].fill_between(self._dict[key][0] + shift, high, low, alpha=0.3, edgecolor='#7619b8', facecolor='#ae64e3')
                    _ = ax[i][0].plot(self._dict[key][0] + shift, self._dict[key][1], 'o', c='#4bd8ce')
                    _ = ax[i][1].plot(self._dict[key][0] + shift, self._dict[key][1] - self._dict[key][4], 'o', c='#5d1591')
                    ax[i][1].yaxis.set_major_locator(plt.MaxNLocator(5))
                    yticks = ax[i][0].yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
                    ax[i][1].yaxis.set_major_locator(plt.MaxNLocator(5))
                    yticks = ax[i][1].yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)

            plt.savefig(self.planet + "_model.png")
            plt.show()


