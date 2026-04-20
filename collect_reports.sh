#!/bin/bash

shopt -s nullglob

for d in static/docs/*/report-12/final-report.md
do
  name=$(basename "$(dirname "$(dirname "$d")")")
  echo $name
  cp -u "$d" "static/docs/reports/$name.md"
done
