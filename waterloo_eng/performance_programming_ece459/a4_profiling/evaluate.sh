#!/bin/bash

verbose=false
while getopts "v" opt; do
  case $opt in
    v)
	verbose=true 
      ;;
    \?)
      ;;
  esac
done

echo "Evaulating Correctness:"

# Path to base code
path="/home/urban/urban/uw/4b/ece459/ece459-1231-a4-upistek"

REGEX_IDEA='(?<=Idea Generator: ).*'
REGEX_STUDENT_IDEA='(?<=Student Idea: ).*'
REGEX_STUDENT_PKG='(?<=Student Package: ).*'
REGEX_DOWN='(?<=Package Downloader: ).*'

args=(
	"4 2 20 2 2"
	"80 2 4000 6 6"
	"200 8 10000 8 8"
	"20 1 1000 1 1"
	"4 1 400 2 2"
	"400 8 4000 8 16"
	"50 20 20000 20 20"
	"80 8 8000 8 16"
	"4 2 100 2 2"
	"100 16 40000 20 16"
	"100 50 40000 20 50"
	"40 10 10000 20 32"
)
len=${#args[@]}	

# for a in "${args[@]}"
for ((i=0; i<len; i++))
do
	echo ""
	echo "Testing with args: ${args[$i]}"

	output_o="$(cargo run --release -- ${args[$i]})"

	value_idea_o=$(echo "$output_o" | grep -oP "$REGEX_IDEA")
	value_student_idea_o=$(echo "$output_o" | grep -oP "$REGEX_STUDENT_IDEA")
	value_student_pkg_o=$(echo "$output_o" | grep -oP "$REGEX_STUDENT_PKG")
	value_down_o=$(echo "$output_o" | grep -oP "$REGEX_DOWN")

	output_b="$(cd $path && cargo run --release -- ${args[$i]})"

	value_idea_b=$(echo "$output_b" | grep -oP "$REGEX_IDEA")
	value_student_idea_b=$(echo "$output_b" | grep -oP "$REGEX_STUDENT_IDEA")
	value_student_pkg_b=$(echo "$output_b" | grep -oP "$REGEX_STUDENT_PKG")
	value_down_b=$(echo "$output_b" | grep -oP "$REGEX_DOWN")

	# Check if all variables ending in "b" have the same value as those ending in "o"
	all_equal=true
	if [ "$value_idea_o" != "$value_idea_b" ]; then
		all_equal=false
	fi
	if [ "$value_student_idea_b" != "$value_student_idea_o" ]; then
		all_equal=false
	fi
	if [ "$value_student_pkg_b" != "$value_student_pkg_o" ]; then
		all_equal=false
	fi
	if [ "$value_down_b" != "$value_down_o" ]; then
		all_equal=false
	fi
	
	# Print out sets of checksums
	if $verbose; then
		echo ""
		echo "cargo run --release -- ${args[$i]}"
		echo "[Optimized Output]"
		python3 cli.py -p -v "$output_o"
		echo "[Base Output]"
		python3 cli.py -p -v "$output_b"
	fi 

	if $all_equal; then
		echo "=> All checksums are equal"
	else
		echo "=> FAILED: All checksums are not equal"
	fi

done
