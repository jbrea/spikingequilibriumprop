#!/bin/bash
parallel echo -e 'nice -n19 julia spikingequilibriumprop/experiments/runsims.jl\
   n_ofsamples=\"10^7\" learningratefactor=\"1\" id={3}\
   beta=2 stepsforward=100 stepsbackward=1\
   ns=\"[2,30,30,2]\"\
   backwardupdate=\"SpikingEquilibriumProp.NetSim.updatenetasync!\"\
   backwardconnectiontype=\"SpikingEquilibriumProp.StaticDenseConnection\"' \
   ::: {1..10}
#parallel echo -e 'nice -n19 julia spikingequilibriumprop/experiments/runsims.jl\
#    n_ofsamples=\"10^7\" learningratefactor=.25 beta=1 stepsforward=100\
#	noiselevel=.01 id={}' ::: {1..10}
#parallel echo -e 'nice -n19 julia spikingequilibriumprop/experiments/runsims.jl\
#	n_ofsamples=\"10^6\" learningratefactor=2.5 beta=1 stepsforward=1000\
#	stepsbackward=20 tau_trace=10 noiselevel=.01 id={}' ::: {1..10}
