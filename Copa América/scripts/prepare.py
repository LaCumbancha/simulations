# Imports
import csv
import json
import argparse

# Default values
DEFAULT_CSV_OUTPUT = "./output/raw-data.json"
DEFAULT_JSON_INPUT = "./output/predictions.csv"

# Loading arguments
parser = argparse.ArgumentParser(description='I/O data')
parser.add_argument('--input', dest='input', type=str, help='Input data in JSON format')
parser.add_argument('--output', dest='output', type=str, help='Output data in CSV format')
args = parser.parse_args()

if args.input is None:
    args.input = DEFAULT_JSON_INPUT

if args.output is None:
    args.output = DEFAULT_CSV_OUTPUT
  
# Reading JSON file
with open(args.input, "r") as json_file:
    predictions = json.load(json_file)

predictions_csv = []
for country, results in predictions.items():
    groups_stage = results['GROUPS']
    quarterfinals = results['QF']
    fourth_place = results['4TH']
    third_place = results['3RD']
    second_place = results['2ND']
    champion = results['1ST']
    simulations = groups_stage + quarterfinals + fourth_place + third_place + second_place + champion
    predictions_csv += [[country, groups_stage/simulations, quarterfinals/simulations, fourth_place/simulations, third_place/simulations, second_place/simulations, champion/simulations]]

# Writing CSV file
with open(args.output, 'w', encoding='UTF8') as csv_file:
    writer = csv.writer(csv_file)
    for prediction in predictions_csv:
        writer.writerow(prediction)
