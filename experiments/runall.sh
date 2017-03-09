#!/bin/bash
re='^[0-9]+$'
if ! [[ $1 =~ $re ]] ; then
   echo -e "Please enter a number of concurrent jobs,
e.g. 'runall.sh 24' runs 24 jobs in parallel." >&2; exit 1
fi
./createalljobs.sh | parallel -j$1 --progress --group --result stdout
