from __future__ import division

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import random

# Earth mover's distance between two numpy arrays.
# Requires the python-emd module.
def my_emd(a, b):
    import emd
    pos = range(len(a))
    return emd.emd((pos, list(a)), (pos, list(b)), lambda x,y: abs(x-y)+0.0)

# Kullback-Leibler divergence
def kl_div(p, q, reg=0):
    # Kludge to prevent div-by-zero messages.  Doesn't affect the output.
    pp = np.where(p > 0, p, 1)
    qq = np.where(p > 0, q, 1)
    return np.sum(np.where(p > 0, pp * np.log(pp / qq), 0))

def normalize_dist(p):
    assert np.min(p) >= 0
    return p / np.sum(p)

def parse_timeseries(timeseries):
    (t, x, y, z) = timeseries[:,:4].T

    out = dict()

    mag = np.sqrt(x*x+y*y+z*z)

    # Average magnitude of acceleration
    avg_mag = np.average(mag)
    out['avg_mag'] = avg_mag

    # Histogram of acceleration magnitude
    out['hg_mag'] = normalize_dist(np.histogram(mag, bins=50, range=(7,13))[0])

    # Histogram of log deviation of acceleration magnitude from its average
    mag_dev = np.abs(mag-avg_mag)
    out['hg_log_mag_dev'] = normalize_dist(
            np.histogram(np.log(mag_dev), bins=50, range=(-5,5))[0])

    # Histogram of orientation
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x*x+y*y))
    out['hg_orientation'] = normalize_dist(np.histogram2d(lon, lat, bins=5, \
            range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]])[0])

    # Histogram of log of jerk magnitude
    dt = t[1:] - t[:-1]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    dr2 = np.sqrt(dx*dx + dy*dy + dz*dz)
    jerk_good = (dt > 0) & (dt < 1000) & (dr2 > 0)
    jerk = dr2[jerk_good] / dt[jerk_good]
    out['hg_jerk'] = normalize_dist(np.histogram(np.log(jerk), bins=50, range=(-15,4))[0])

    return out

data_stores = {}

def parse_from_file(fn, k, take_slice=None):
    if take_slice is None:
        take_slice = slice(None, None)
    # Data file are opened on demand.  If using multiprocessing, it is important that each
    # process opens its own copy, and that the parent process doesn't open one at all.
    if not fn in data_stores:
        data_stores[fn] = pd.HDFStore(fn, 'r')
    return parse_timeseries(data_stores[fn]['/'+str(k)].as_matrix()[take_slice,:])

# Compare a histogram (by Kullback-Leibler divergence) to the histograms of the training data.
def compare_histograms(hg_key, train_wisdom, test):
    hw = [ w[hg_key] for w in train_wisdom ]
    t = test[hg_key]

    n = 1e-6 # prevent infinities
    return np.array([ kl_div(t, w*(1-n)+t*n) for w in hw ])

# How close the test data is to the training data for each device.
def get_dist_vector(qw):
    dm = compare_histograms('hg_mag', train_wisdom, qw)
    dj = compare_histograms('hg_jerk', train_wisdom, qw)
    #do = compare_histograms('hg_orientation', train_wisdom, qw)
    dm /= np.average(dm)
    dj /= np.average(dj)
    dv = dm + dj
    dv /= np.average(dv)
    return dv

# Is the proposed device likely to be the real device?
def answer_question((qid, sid, proposed_dev)):
    qw = parse_from_file('test_by_seq.h5', sid)
    dev_idx = idx_for_dev[proposed_dev]
    dv = get_dist_vector(qw)
    # threshold tuned to give 50% yes answers
    return dv[dev_idx] < 0.5

def simulated_question():
    true_dev = random.choice(devices)
    proposed_dev = random.choice([true_dev, random.choice(devices)])
    qw = parse_from_file('train_by_dev.h5', true_dev, slice(None, 300))
    dv = get_dist_vector(qw)
    answer = dv[idx_for_dev[proposed_dev]] < 0.5
    win = 1 if (answer == (true_dev == proposed_dev)) else 0
    #print true_dev, proposed_dev, answer, win
    return win

######################################################################

# If 1, then simulate the classification using a portion of the training data.
# If 0, then answer the contest questions.
simulate = 1

# Number of CPUs to use.
num_cpus = 4

if num_cpus > 1:
    par = Parallel(n_jobs=num_cpus, verbose=5)
else:
    # If only one CPU, then just run in the parent process to facilitate debugging.
    par = lambda seq: [ f(*args, **kwargs) for (f, args, kwargs) in seq ]

# Only read the data if it's not already loaded.
if not 'questions' in locals() or questions is 'None':
    print 'Reading questions.'
    questions = np.loadtxt(open('questions.csv', 'r'), delimiter=',', skiprows=1, dtype=int)
    # Get list of devices.  Assumes each device is asked about at least once.
    devices = np.unique(questions[:,2])
    idx_for_dev = { dev: idx for (idx, dev) in enumerate(devices) }

if not 'train_wisdom' in locals() or train_wisdom is None:
    print 'Training.'

    if simulate:
        # Train on all but the first 300 samples (which will be used for testing).
        train_slice = slice(300, None)
    else:
        train_slice = slice(None, None)

    train_wisdom = par(delayed(parse_from_file)
            ('train_by_dev.h5', k, train_slice) for k in devices)

if simulate:
    print 'Getting data for simulated test.'
    # Use first 300 samples of training data for test.
    test_slice = slice(None, 300)
    test_wisdom = par(delayed(parse_from_file)
            ('train_by_dev.h5', k, test_slice) for k in devices)

    print 'Running test simulation.'
    dist_mat = np.array(par(delayed(get_dist_vector)(qw) for qw in test_wisdom ))
    answers = dist_mat < 0.5

    # Probability of accepting a correct suggestion:
    avg_on_diag = np.average(np.diag(answers))
    # Probability of accepting a wrong suggestion:
    avg_off_diag = np.average([ v for ((i,j),v) in np.ndenumerate(answers) if i != j ])

    print '-- Simulation results:'
    print 'P(y|y) =', avg_on_diag
    print 'P(n|n) =', 1-avg_off_diag
    print 'P(y) =', (avg_on_diag + avg_off_diag) / 2.0
    print 'success rate =', (avg_on_diag + (1-avg_off_diag)) / 2.0

    #scores = par(delayed(simulated_question)() for i in range(100))
    #print np.average(scores)
else:
    print 'Evaluating contest questions.'

    answers = par(delayed(answer_question)(q) for q in questions)

    # How often we said 'yes'.
    print 'P(y) =', np.average(answers)

    with open('answers.csv', 'w') as fh:
        fh.write('QuestionId,IsTrue\n')
        for (answer, (qid, sid, proposed_dev)) in zip(answers, questions):
            fh.write('%d,%d\n' % (qid, answer))
