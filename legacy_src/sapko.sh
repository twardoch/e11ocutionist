#!/usr/bin/env bash
# this_file: work2/sapko.sh
#
# Script to process Sapkowski's Witcher text through the Malmo pipeline
# This script runs the full processing chain from chunking to 11labs format

set -euo pipefail

python ../malmo_orator.py "Andrzej Sapkowski - Wiedźmin Geralt z Rivii 0.1 - Rozdroże kruków_step2_entited.xml" "Andrzej Sapkowski - Wiedźmin Geralt z Rivii 0.1 - Rozdroże kruków_step3_orated.xml" --all_steps --verbose --backup

python ../malmo_tonedown.py "Andrzej Sapkowski - Wiedźmin Geralt z Rivii 0.1 - Rozdroże kruków_step3_orated.xml" "Andrzej Sapkowski - Wiedźmin Geralt z Rivii 0.1 - Rozdroże kruków_step4_toneddown.xml" --verbose

python ../malmo_11labs.py "Andrzej Sapkowski - Wiedźmin Geralt z Rivii 0.1 - Rozdroże kruków_step4_toneddown.xml" "Andrzej Sapkowski - Wiedźmin Geralt z Rivii 0.1 - Rozdroże kruków_step5_11.txt" --verbose
