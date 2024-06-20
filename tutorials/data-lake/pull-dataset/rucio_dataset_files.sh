#/bin/bash
#
# G. Guerrieri & E. Garcia (CERN) - Jun 2024
#
# This script runs only on VEGA
# 
# Usage - on a terminal run
# > ./rucio_dataset_files.sh <SCOPE:DataSet> <output_file> <output_symlink_dir>

set -e

ds=$1
name=$2
location=$3

pw=`pwd -P`

if [[ -f "${name}" ]]; then rm ${name}.txt; fi
touch ${name}.txt

if [ -d "${location}" ]; then echo -e "Directory exists. Exiting\n${pw}/${location}" ; exit 1 ; fi
mkdir $location

for file in `rucio list-file-replicas --rse VEGA-DCACHE $ds | awk '{ print $12 }' | sed 's|https://dcache.sling.si:2880|/dcache/sling.si|g'`
do
  if [[ $file == "|" ]]; then continue; fi
  fileReduced=`basename $file`
  echo linking $fileReduced "..."
  link=$location/${ds/:/.}.$fileReduced
  ln -s $file $link
  echo ${pw}/$link >> ${name}.txt
done

chmod -R 777 $3
