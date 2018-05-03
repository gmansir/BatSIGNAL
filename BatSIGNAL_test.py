import pytest
from astropy.utils.data import get_pkg_data_filename
import tempfile
from BatSignal_prev2 import *


@pytest.fixture
def initialize():
    """Creates a BatSIGNAL instance"""
    inputfile = get_pkg_data_filename('BatSIGNAL_test.cfg')
    datafile = get_pkg_data_filename('WASP-4b_20121031.txt')
    a = BatSignal(inputfile, datafile)
    assert type(config) is configparser.RawConfigParser
    assert type(inputs) is batman.transitmodel.TransitParams
    return a


#def test_createinput():
#    tmpdir = tempfile.mkdtemp()
#    create_param_file(tmpdir+'/tmp.cfg')
#    config.read_file(open(tmpdir+'/tmp.cfg'))
#    assert config.sections()[0] == 'What to Fit'


def test_initial(initialize):
    """Verifies that Batman and ConfigParser are installed correctly"""

    data = sp.loadtxt(initialize.light_curve_file, unpack=True)
    assert isinstance(initialize.date_real[0], float)
    assert [initialize.date_real[i] == data[0][i] for i in range(len(initialize.date_real))]
    assert isinstance(initialize.flux[0], float)


def test_updaterelax(initialize):
    """"Tests that the function to update the relaxation factors on the error is working properly"""

    initialize.update_relax(rp = 5)
    assert initialize.relax[0] == 5


def test_bat(initialize):
    """Runs BatSIGNAL's main function and verifies that the results are resonable"""

    out = initialize.bat()

    names = ['amp', 'scale', 'rp', 'u', 'u', 't0', 'per', 'a', 'inc', 'ecc', 'w']
    count = 0

    for n in range(len(names)):
        assert out.results[n][0] == names[n]
        assert out.results[n][1] < (out.results[n][1] + out.results[n][2])
        assert out.results[n][1] > (out.results[n][1] - out.results[n][3])
        if names[n] == 'amp' or names[n] =='scale':
            pass
        elif names[n] == 'u':
            assert out.results[n][1] < out._usr[n-2-count][0][count] + out._usr[n-2-count][1][count] * 5
            assert out.results[n][1] > out._usr[n-2-count][0][count] - out._usr[n-2-count][1][count] * 5
            if n == 3:
                count += 1
        elif names[n] != 't0':
            assert out.results[n][1] < out._usr[n-2-count][0] + out._usr[n-2-count][1] * 5
            assert out.results[n][1] > out._usr[n-2-count][0] - out._usr[n-2-count][1] * 5

def test_disable(initialize):
    """Checks that the Disable function properly updates the input file"""

    initialize.Disable('all')
    config.read_file(open(initialize.input_param_file))
    section = config.sections()[0]
    check = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
    ld = check.pop(1)
    for i in range(len(ld)):
        check.insert(i + 1, ld[i])
    for i in check:
        assert i == False

def test_enable(initialize):
    """Checks that the Enable function properly updates the input file"""

    initialize.Enable('all')
    config.read_file(open(initialize.input_param_file))
    section = config.sections()[0]
    check = [eval(i) for i in (sp.array(config.items(section))[:, 1]).tolist()]
    ld = check.pop(1)
    for i in range(len(ld)):
        check.insert(i + 1, ld[i])
    for i in check:
        assert i == True


i = initialize()
test_initial(i)
test_updaterelax(i)
test_bat(i)
test_disable(i)
test_enable(i)
print("Tests Completed")


