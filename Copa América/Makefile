SHELL := /bin/bash
PWD := $(shell pwd)

SIMULATIONS = 10000000

INPUT_DATA = "./data/matches.csv"
OUTPUT_DATA = "./output/results.csv"
INTERMEDIATE_DATA = "./output/raw-data.json"

default: run

build:
	pip3 install numpy
	pip3 install psutil
	pip3 install pandas
	pip3 install pathlib
	pip3 install argparse

run:
	@python3 -W ignore ./scripts/simulation.py --input $(INPUT_DATA) --output $(INTERMEDIATE_DATA) --simulations $(SIMULATIONS)
	@python3 -W ignore ./scripts/prepare.py --input $(INTERMEDIATE_DATA) --output $(OUTPUT_DATA)

prepare:
	@python3 -W ignore ./scripts/prepare.py --input $(INTERMEDIATE_DATA) --output $(OUTPUT_DATA)
