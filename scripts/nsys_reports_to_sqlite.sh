#!/bin/bash
# Converts all nsys reports in the given directory to sqlite
# Usage: bash nsys_reports_to_sqlite.sh <directory>
REPORT_DIR=$1
if [ -z "$REPORT_DIR" ]; then
    echo "Usage: bash nsys_reports_to_sqlite.sh <directory>"
    exit 1
fi
# Get the absolute path of the directory
REPORT_DIR=$(realpath $REPORT_DIR)

# The sqlite files will be saved in the same directory as the nsys reports
echo "Number of nsys reports: $(ls ${REPORT_DIR}/*.nsys-rep | wc -l)"
num_files=$(ls ${REPORT_DIR}/*.nsys-rep | wc -l)
count=0
for file in ${REPORT_DIR}/*.nsys-rep; do
    echo "Converting ${file} to ${file}.sqlite"
    # Extracts the filename without the extension
    nsys_report_file=$file
    file=$(basename -- "$file")
    file="${file%.*}"
    # Converts the report to sqlite
    nsys export --type=sqlite --force-overwrite=true --output=${REPORT_DIR}/${file}.sqlite ${nsys_report_file} &
done
# Wait for all the processes to finish
wait
exit 0