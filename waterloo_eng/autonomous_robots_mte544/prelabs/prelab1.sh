#!/bin/bash

usage() {
    echo "usage: -l <line>"
}

line=""
while getopts "l:" flag; do
    case "${flag}" in
        l)
            line=${OPTARG}
            ;;
        ?)
            echo "Invalid argument"
            exit 1
            ;;
    esac
done

[[ "$line" == "" ]] && {
    usage
    exit 1
}

printf "\nSearching for line: \n ${line} \n"

search=$(grep ${line} ~/.bashrc)

[[ "$search" == "" ]] && {
	printf "\nError: Line was not found... \n"
    exit 1
}

printf "Found: \n ${search} \n"

