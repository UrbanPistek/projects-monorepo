#!/bin/bash

echo "Evaulating Performance:"

# Path to base code
path="/home/urban/urban/uw/4b/ece459/ece459-1231-a4-upistek"

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

for ((i=0; i<len; i++))
do
	echo ""
	echo "Testing with args: ${args[$i]}"

	target=$(printf 'target/release/lab4 %s' "${args[$i]}")
	output_o="$(hyperfine --warmup 10 --min-runs 25 -i "$target" --shell=none --export-json benchmark.json 2> /dev/null)"
	echo "Optimized:"
	python3 cli.py -j

	target=$(printf '%s/target/release/lab4 %s' $path "${args[$i]}")
	output_b="$(hyperfine --warmup 10 --min-runs 25 -i "$target" --shell=none --export-json benchmark.json 2> /dev/null)"
	echo "Base:"
	python3 cli.py -j
done

