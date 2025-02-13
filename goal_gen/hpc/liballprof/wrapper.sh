#!/bin/bash

#COMPRESS="bzip2 -c"
#SUFFIX="bz2"
COMPRESS="gzip -c"
SUFFIX="gz"

HOST=$(hostname -s)
VERBOSE=false
if [ -f $HOME/.wrapper_verbose ]; then
  VERBOSE=true
fi

#echo "[$HOST] clearing /tmp ..."
rm -f /tmp/pmpi-trace-rank-*txt

if $VERBOSE; then
  echo "[$HOST] htor profiling wrapper: executing $@ ..."
fi


# execute the command ...
$@

if [ x"$HTOR_PMPI_FILE_PREFIX" == "x" ]; then
  HTOR_PMPI_FILE_PREFIX="/tmp/pmpi-trace-rank-"
fi;

for i in $(ls -1 $HTOR_PMPI_FILE_PREFIX*txt 2>/dev/null); do
  if test -f $i; then
    TMP=$(mktemp)
    if $VERBOSE; then
      echo "[$HOST] moving $i to $TMP to have exclusive access ..."
    fi
    # one process wins the move -- and mv should be atomic in any
    # reasonable FS :)
    mv $i $TMP 2> /dev/null
    # if I won ... compress it ...
    if test -s $TMP; then
      if $VERBOSE; then
        echo "[$HOST] compressing $i ($TMP) ..."
      fi
      cat $TMP | $COMPRESS > $(basename $i).$SUFFIX;
    fi;
    rm $TMP
  fi;
done;

