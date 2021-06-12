SHELL := /bin/bash
PWD := $(shell pwd)

default: run

build:
	pip3 install psutil
	pip3 install numpy
	pip3 install pandas

run:
	@python3 -W ignore ./Simulation.py
