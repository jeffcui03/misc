#!/usr/bin/bash

date
$targetDir=<mydir>
for i in $(seq 32); do
    cp --sparse=never hh $target
done

echo "Done!"
