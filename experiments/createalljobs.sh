#!/bin/bash
parallel echo -e "julia noisedependence.jl {}" ::: {1..24}
parallel echo -e "julia forwardphasedependence.jl {}" ::: {1..8}
parallel echo -e "julia learningratedependence.jl {1} {2}" ::: {1..8} ::: C L
parallel echo -e "julia spikingexample.jl {1} {2} Y" ::: 50 150 250 ::: {1..5}
