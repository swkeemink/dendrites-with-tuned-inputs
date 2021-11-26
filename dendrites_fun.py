'''Dendrite Functions¶

This python file contains some helpful functions for dendrite modelling in Brian2.'''

# imports
import holoviews as hv
from brian2 import *
from numpy import exp,sin,cos,pi
import numpy as np
import random
hv.extension('matplotlib')

# morphology functions
# morphology functions
def plotmorph(ang, N, pos, w, group='A'):
    '''
    Plot the morphology for a 1-depth dendritic tree (the dendrites don't have sub dendrites)

    Inputs
    --------------
    - ang: orientation of the inputs
    - N:   number of dendrites
    - pos: the positions of the inputs (see output of ConnectStim)
    - w: matrix of weights
    - group: the group name for the plot

    (for now assumes dendrite length = 100um)
    '''
    # add connections and color for tuning
    points = np.zeros((len(ang), 4))
    points[:, 0] = pos[0, :] - 1
    points[:, 1] = pos[1, :]
    points[:, 2] = ang[:] / pi
    points[:, 3] = w/w.max()

    # plot the incoming connections
    copts = hv.opts.Points(s=hv.dim('size')*800)
    ins = hv.Points(points, vdims=['angle', 'size'], group=group).opts(copts)
    curves = ins

    # plot dendrite curves (from end of dendrites to soma, the final kink is just visual)
    curves *= hv.Curve(zip([0, 0, (N - 1) / 2.], [100, 0, -20]), extents=(-1, -0.1, N, 101)).opts(color='k')
    for i in range(1, N):
        curves *= hv.Curve(zip([i, i, (N - 1) / 2.], [100, 0, -20])).opts(color='k')

    # calculate alphas
    alphas = w/w.max()

    # add a soma representation
    curves *= hv.Scatter(zip([(N - 1) / 2.], [-20]))

    return curves

# input functions
def vm(k, a1, a2):
    '''
    Simple von mises function
    returns exp(k*cos(a1-a2))/exp(k)
    '''
    return exp(k * cos(a1 - a2)) / exp(k)


def setupstimulation(N, nA, stim, p=0, kn=0.1, k=1, A=10):
    '''
    stimulation,r,ang = setupstimulation(N,nA,c,s,p=0,kn=0.1,kc=1,ks=0.5,A=10)
    Sets up the the stimulation spike traces.

    Note: when we talk about preferred orientations in this
    script we usually mean the preferred orientation of an input
    connection. If we specifically talk of the output preferred
    orientation we mean the resulting output preferred orientation.

    Inputs
    -------------------
    - N : total number of inputs
    - nA: number of different orientations (should be at least an order of magnitude smaller than N)
    - p : output preferred orientation
    - stim : input orientation presented to neurons
    - kn: width of preferred orientation distribution
    - k: width of individual input orientation tunings
    - A:  basic firing rate drive

    Returns
    --------------------
    - stimulation: Brian stimulation object
    - r          : firing rates
    - ang        : preferred orientations
    '''
    # setup different preferred orientations
    ang = np.arange(-pi, pi, 2 * pi / nA)

    # setup number of inputs for each preferred orientations
    # if kn = 0, then all directions are represented equally
    # if kn > 0 there is a bias towards the output preferred orientation
    dist = vm(kn, ang, p)  # basic distribution
    dist = (N * dist / sum(dist)).round().astype(int)  # distribution changed to have something around N neurons
    N = sum(dist)  # recalculate N (the distribution doesn't always sum up perfectly to N)

    # set the corresponding firing rates
    r = np.zeros(N)  # where firing rates will be stored
    angs = np.zeros(N)  # where orientations will be stored
    count = 0  # counter
    for i, dp in enumerate(ang):  # for all preferred orientations (dp = dendrite preferred)
        for j in range(dist[i]):  # for as many of these preferred inputs there are (could be done in one line too)
            r[count] = A * vm(k, dp, stim)
            angs[count] = dp
            count += 1  # change counter

    # setup stimulation
    stimulation = PoissonGroup(N, r * Hz)
    return stimulation, r, angs


def ConnectStim(neuron, stimulation, S, pos='random'):
    '''
    Connect the trains in the stimulation object to the structures in the neuron object
    through synapses

    Inputs
    --------------------
    neuron : spatialneuron object
        neuron object from brian
    stimulation : stimulation object
        stimulation object from brian
    S : synapse object
        Synapse object from brian
    pos : string
        'nonrandom' -- places input synapses regularly
        'random' -- places input synapses randomly

    Outputs
    ---------------------
    Synapses:  (already connected to neuron)
    '''
    # find relevant numbers
    nStim = len(stimulation)  # number of inputs
    nDen = len(neuron.morphology.children)  # number of dendrites
    lDen = neuron.morphology.children['den2'].distance[-1]  # length of a dendrite

    # setup basic synapses
    positions = np.zeros((2, nStim))

    # connect them as appropriate
    if pos == 'nonrandom':
        # get roughly equal number of inputs per dendrites
        num, div = nStim, nDen
        perDen = [num // div + (1 if x < num % div else 0) for x in range(div)]  # number of dendrites per den
        # from https://stackoverflow.com/questions/20348717/algo-for-dividing-a-number-into-almost-equal-whole-numbers#20348992
        random.shuffle(perDen)

        # loop over stimuli
        ids = np.arange(nStim)
        np.random.shuffle(ids)
        count = 0
        for iDen in range(nDen):
            counter = 1  # counter for current dendrite
            places = np.linspace(0, 1, perDen[iDen]) * lDen
            for j in range(perDen[iDen]):
                # get place
                ID = ids[count]
                place =places[j]

                # connect synapse
                S.connect(i=ID, j=neuron.morphology['den' + str(int(iDen + 1))][
                    place])  # connect neuron 0 to left dendrite at end

                # store positions
                positions[:, ID] = [iDen + 1, place / um]
                counter += 1

                count += 1
    elif pos == 'random':
        for i in range(nStim):  # for all inputs
            # draw a random dendrite
            iDen = np.random.randint(1, nDen + 1)

            # get random position on dendrite
            place = np.random.rand() * lDen  # (0.5+np.random.rand()/4)*lDen

            # connect synapse
            S.connect(i=i,
                      j=neuron.morphology['den' + str(int(iDen))][place])  # connect neuron 0 to left dendrite at end

            # store positions
            positions[:, i] = [iDen, place / um]
    else:
        raise ValueError('Wrong pos argument. Should be random or nonrandom')

    # return positions of all trains
    return positions


def ConnectStimSimple(neuron, angid, stimulation, S, pos='random'):
    '''
    Connect the trains in the stimulation object to the structures in the neuron object
    through synapses

    Same as ConnectStim, but each orientation gets its own dendrite.

    Inputs
    --------------------
    neuron: spatialneuron object
    angid: which dendrite this input should go on
    stimulation: stimulation object
    S: brian Synapse object
    pos: string or list
    Outputs
    ---------------------
    Synapses:  (already connected to neuron)
    '''
    # find relevant numbers and predefine arrays
    nStim = len(stimulation)  # number of inputs
    nDen = len(neuron.morphology.children)  # number of dendrites
    lDen = neuron.morphology.children['den2'].distance[-1]  # length of a dendrite
    positions = np.zeros((2, nStim))

    # start counter, to see how many connections there are on each dendrite
    counter = np.zeros(nDen)
    # connect them as appropriate
    for ii in range(nStim):  # for all inputs
        # pick dendrite
        iDen = angid[ii]
        nDen = float(np.sum(angid == iDen))  # total number of connections on this dendrite
        iDen = int(round(iDen))
        counter[iDen - 1] += 1
        if pos == 'random':
            # get random position on dendrite
            place = (0.5 + np.random.rand() / 4) * lDen
        elif pos == 'nonrandom':
            # get dendrite place
            place = (counter[iDen - 1] / (nDen + 1)) * lDen
        else:
            place = pos[1, ii] * um

        # connect synapse
        S.connect(i=ii, j=morpho['den' + str(int(iDen))][place])  # connect neuron 0 to left dendrite at end

        # store positions
        positions[:, ii] = [iDen, place / um]

    # return positions of all trains
    return positions

