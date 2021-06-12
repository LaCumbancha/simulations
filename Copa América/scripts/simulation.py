# Imports
import os
import json
import enum
import math
import psutil
import pathlib
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import date, datetime

logger = logging.getLogger(name="Copa América 2021 Simulator")
logging.basicConfig(format='%(name)s | %(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Default values
DEFAULT_SIMULATIONS = 10000000
DEFAULT_INPUT_DATA = './data/matches.csv'
DEFAULT_OUTPUT_DATA = './output/raw-data.json'

# Loading arguments
parser = argparse.ArgumentParser(description='I/O data')
parser.add_argument('--input', dest='input', type=str, help='Input data in JSON format')
parser.add_argument('--output', dest='output', type=str, help='Output data in CSV format')
parser.add_argument('--simulations', dest='simulations', type=int, help='Number of simulations')
args = parser.parse_args()

if args.input is None:
    logger.warn(f'Missing input data. Setting default: {DEFAULT_INPUT_DATA}')
    args.input = DEFAULT_JSON_INPUT

if args.output is None:
    logger.warn(f'Missing output data. Setting default: {DEFAULT_OUTPUT_DATA}')
    args.output = DEFAULT_CSV_OUTPUT

if args.simulations is None:
    logger.warn(f'Missing simulations number. Setting default: {DEFAULT_SIMULATIONS}')
    args.simulations = DEFAULT_SIMULATIONS

# Load data
matches = pd.read_csv(args.input)

# Load teams
CONMEBOL_TEAMS = {'Argentina': 'ARG', 'Brazil': 'BRA', 'Uruguay': 'URU', 'Bolivia': 'BOL', 'Paraguay': 'PAR', 'Chile': 'CHI', 'Peru': 'PER', 'Colombia': 'COL', 'Ecuador': 'ECU', 'Venezuela': 'VEN'}
CONMEBOL_TEAMS_NAMES = list(CONMEBOL_TEAMS.keys())

# Conmebol matches
conmebol_matches = matches.loc[matches['home_team'].isin(CONMEBOL_TEAMS_NAMES) & matches['away_team'].isin(CONMEBOL_TEAMS_NAMES)]

# Map dates
conmebol_matches['date'] = conmebol_matches['date'].apply(lambda match_date: datetime.strptime(match_date, "%Y-%m-%d").date())
conmebol_matches = conmebol_matches.sort_values(by=['date'])

# Map neutral
conmebol_matches['neutral'] = conmebol_matches.apply(lambda match: match['home_team'] != match['country'], axis=1)

# Map names
conmebol_matches['home_team'] = conmebol_matches['home_team'].apply(lambda team: CONMEBOL_TEAMS[team])
conmebol_matches['away_team'] = conmebol_matches['away_team'].apply(lambda team: CONMEBOL_TEAMS[team])

# Simulation rules

TIER_1 = ['Friendly']
TIER_2 = ['Copa Lipton', 'Copa Newton', 'Copa Premio Honor Argentino', 'Copa Premio Honor Uruguayo', 'Copa Roca', 'Copa Chevallier Boutell', 'Copa Rio Branco', 'Copa Oswaldo Cruz', 
'Pan American Championship', 'Copa del Pacífico', 'Copa Bernardo O\'Higgins', 'Atlantic Cup', 'Copa Paz del Chaco', 'Copa Carlos Dittborn', 'Copa Juan Pinto Durán', 'Copa Artigas', 
'Brazil Independence Cup', 'Copa Ramón Castilla', 'Copa Félix Bogado', 'Gold Cup']
TIER_3 = ['FIFA World Cup qualification']
TIER_4 = ['Copa América', 'Confederations Cup', 'Mundialito']
TIER_5 = ['FIFA World Cup']

def tournament_value(tournament):
    if tournament in TIER_5:
        tournament_points = 5
    elif tournament in TIER_4:
        tournament_points = 4
    elif tournament in TIER_3:
        tournament_points = 3
    elif tournament in TIER_2:
        tournament_points = 2
    else:
        tournament_points = 1
    return math.sqrt(tournament_points)


BUCKETS = 10
CURRENT_YEAR = date.today().year
BUCKET_LAPSE = math.ceil((CURRENT_YEAR - conmebol_matches.iloc[0]['date'].year) / BUCKETS)

def date_value(match_date):
    return math.sqrt(BUCKETS - math.floor((CURRENT_YEAR - match_date.year) / BUCKET_LAPSE))


def result_value(home_score, away_score, neutral):
    home, tie, away = 0, 0, 0
    if home_score > away_score:
        home = 1
    elif home_score == away_score:
        tie = 1
    else:
        away = 1
        
    if not neutral:
        if tie != 0:
            tie += 0.25
        if away != 0:
            away += 0.25
        
    if home_score >= 3 + away_score:
        home += 0.5
    
    if away_score >= 3 + home_score:
        away += 0.5
        
    return home, tie, away


def total_points(match, result):
    date_points = date_value(match['date'])
    home, tie, away = result_value(match['home_score'], match['away_score'], match['neutral'])    
    tournament_points = tournament_value(match['tournament'])
    
    if result == 'team1':
        result_points = home if match['home_team'] < match['away_team'] else away
    elif result == 'team2':
        result_points = away if match['home_team'] < match['away_team'] else home
    else:
        result_points = tie

    return result_points * date_points * tournament_points


def match_name(home_team, away_team):
    if home_team < away_team:
        return f'{home_team}-{away_team}'
    else:
        return f'{away_team}-{home_team}'
    
def teams(match):
    return match.split('-')

# Team handicap

PRESENT_MATCHES = 8
PRESENT_MATCHES_MAX_POINTS = PRESENT_MATCHES * 3

def match_points(home_score, away_score, neutral):
    if home_score > away_score:
        home_points = 2.5
        away_points = 0
    elif home_score == away_score:
        home_points = 1
        away_points = 1
    else:
        home_points = 0
        away_points = 3
        
    if not neutral and away_points != 0:
        away_points += 1
        
    if home_score >= 3 + away_score:
        home_points += 1
    
    if away_score >= 3 + home_score:
        away_points += 1
        
    return home_points, away_points

handicaps = {}
for TEAM in CONMEBOL_TEAMS_NAMES:
    team_points = 0
    team_matches = matches.loc[(matches['home_team'] == TEAM) | (matches['away_team'] == TEAM)].tail(PRESENT_MATCHES)
    for _, match in team_matches.iterrows():
        home_points, away_points = match_points(match['home_score'], match['away_score'], match['neutral'])
    
        if match['home_team'] == TEAM:
            team_points += home_points
        elif match['away_team'] == TEAM:
            team_points += away_points
        
    handicaps[CONMEBOL_TEAMS[TEAM]] = math.sqrt(team_points / PRESENT_MATCHES_MAX_POINTS)

# Probabilities

class Results(enum.Enum):
    W = 1
    T = 0
    L = -1

conmebol_matches['name'] = conmebol_matches.apply(lambda match: match_name(match['home_team'], match['away_team']), axis=1)
conmebol_matches['team1_total'] = conmebol_matches.apply(lambda match: total_points(match, 'team1'), axis=1)
conmebol_matches['tie_total'] = conmebol_matches.apply(lambda match: total_points(match, 'tie'), axis=1)
conmebol_matches['team2_total'] = conmebol_matches.apply(lambda match: total_points(match, 'team2'), axis=1)

historic_simulation = conmebol_matches[['name', 'team1_total', 'tie_total', 'team2_total']]
historic_simulation = historic_simulation.groupby(['name'])

def final_prediction(team1_name, team2_name, historics, result):
    team1_total = sum(historics['team1_total']) * handicaps[team1_name]
    team2_total = sum(historics['team2_total']) * handicaps[team2_name]
    tie_total = sum(historics['tie_total'])
    
    full_total = team1_total + tie_total + team2_total
    result_total = team1_total if result == 'team1' else team2_total if result == 'team2' else tie_total
    
    return result_total / full_total

probabilities = {}
for match, historics in historic_simulation:
    team1_name, team2_name = teams(match)
    team1 = final_prediction(team1_name, team2_name, historics, 'team1')
    team2 = final_prediction(team1_name, team2_name, historics, 'team2')
    tie = final_prediction(team1_name, team2_name, historics, 'tie')
    
    if team1_name not in probabilities:
        probabilities[team1_name] = {}
        
    if team2_name not in probabilities:
        probabilities[team2_name] = {}
        
    probabilities[team1_name][team2_name] = { Results.W: team1, Results.T: tie, Results.L: team2 }
    probabilities[team2_name][team1_name] = { Results.W: team2, Results.T: tie, Results.L: team1 }

# Simulation

Points = 100
class Goals(enum.Enum):
    GF = 11
    GA = 9
    GD = 10

MATCHES_GROUP_A = ['BRA-VEN', 'COL-ECU', 'COL-VEN', 'BRA-PER', 'VEN-ECU', 'COL-PER', 'ECU-PER', 'BRA-COL', 'BRA-ECU', 'VEN-PER']
MATCHES_GROUP_B = ['ARG-CHI', 'PAR-BOL', 'CHI-BOL', 'ARG-URU', 'URU-CHI', 'ARG-PAR', 'BOL-URU', 'CHI-PAR', 'BOL-ARG', 'URU-PAR']
SIMULATION_RESULTS = {
    'ARG': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'BOL': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'BRA': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'CHI': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'COL': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'ECU': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'PAR': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'PER': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'URU': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
    'VEN': {'GROUPS': 0, 'QF': 0, '4TH': 0, '3RD': 0, '2ND': 0, '1ST': 0},
}


def matches_simulations(group_matches):
    results = []
    
    for match in group_matches:
        team1_name, team2_name = teams(match)
        
        team1 = probabilities[team1_name][team2_name][Results.W]
        team2 = probabilities[team1_name][team2_name][Results.L]
        tie = probabilities[team1_name][team2_name][Results.T]
        
        team1_win_max = team1
        team2_win_min = team1 + tie
        
        random_result = np.random.uniform(0, 1)
        if random_result < team1_win_max:
            # Team 1 won
            random_goals_team1 = math.ceil(np.random.exponential(0.5 / (1 - team1)))
            random_goals_team2 = math.floor(np.random.exponential(0.5 / (1 - team2)))
            while (random_goals_team2 >= random_goals_team1):
                random_goals_team2 = math.floor(np.random.exponential(1 / (1 - team2)))
            
        elif random_result > team2_win_min:
            # Team 2 won
            random_goals_team2 = math.ceil(np.random.exponential(0.5 / (1 - team2)))
            random_goals_team1 = math.floor(np.random.exponential(0.5 / (1 - team1)))
            while (random_goals_team1 >= random_goals_team2):
                random_goals_team1 = math.floor(np.random.exponential(1 / (1 - team1)))
                
        else:
            # Tie
            random_goals = int(np.random.exponential(0.5 / (1 - team1)) / 2)
            random_goals_team1 = random_goals
            random_goals_team2 = random_goals
            
        results += [(match, random_goals_team1, random_goals_team2)]
        
    return results

def group_stage(group_results):
    standings = {}
    
    for result in group_results:
        team1_name, team2_name = teams(result[0])
        team1_goals, team2_goals = result[1], result[2]
        
        if team1_name not in standings:
            standings[team1_name] = { Results.W: 0, Results.T: 0, Results.L: 0, Goals.GF: 0, Goals.GA: 0, Goals.GD: 0, Points: 0 }
        standings[team1_name][Goals.GF] += team1_goals
        standings[team1_name][Goals.GA] += team2_goals
        
        if team2_name not in standings:
            standings[team2_name] = { Results.W: 0, Results.T: 0, Results.L: 0, Goals.GF: 0, Goals.GA: 0, Goals.GD: 0, Points: 0 }
        standings[team2_name][Goals.GF] += team2_goals
        standings[team2_name][Goals.GA] += team1_goals

        if team1_goals > team2_goals:
            standings[team1_name][Results.W] += 1
            standings[team2_name][Results.L] += 1
        elif team2_goals > team1_goals:
            standings[team1_name][Results.L] += 1
            standings[team2_name][Results.W] += 1
        else:
            standings[team1_name][Results.T] += 1
            standings[team2_name][Results.T] += 1
            
    for country, final_results in standings.items():
        standings[country][Points] = final_results[Results.W] * 3 + final_results[Results.T]
        standings[country][Goals.GD] = final_results[Goals.GF] - final_results[Goals.GA]
        
    standings_list = list(standings.items())
    standings_list.sort(key=lambda team: (-team[1][Points], -team[1][Goals.GD], -team[1][Goals.GF]))
    
    # Special check for teams with same points, goal difference and goals for.
    for idx, team in enumerate(standings_list[1:]):
        previous_team = standings_list[idx]
        if previous_team[1][Points] == team[1][Points] and previous_team[1][Goals.GD] == team[1][Goals.GD] and previous_team[1][Goals.GF] == team[1][Goals.GF]:
            position_fixed = False
            for result in group_results:
                team1_goals = result[1]
                team2_goals = result[2]
                if result[0] == f'{previous_team[0]}-{team[0]}':
                    if team1_goals > team2_goals:
                        # Shouldn't change, previous team is properly sorted.
                        position_fixed = True
                    elif team2_goals > team1_goals:
                        # Should change positions
                        temp = standings_list[idx]
                        standings_list[idx] = standings_list[idx+1]
                        standings_list[idx+1] = temp
                        position_fixed = True
                    else:
                        # Tied match, should define randomly
                        coin = np.random.uniform(0, 1)
                        if coin > 0.5:
                            temp = standings_list[idx]
                            standings_list[idx] = standings_list[idx+1]
                            standings_list[idx+1] = temp
                            position_fixed = True
                elif result[0] == f'{team[0]}-{previous_team[0]}':
                    if team1_goals > team2_goals:
                        # Should change positions
                        temp = standings_list[idx]
                        standings_list[idx] = standings_list[idx+1]
                        standings_list[idx+1] = temp
                        position_fixed = True
                    elif team2_goals > team1_goals:
                        # Shouldn't change, previous team is properly sorted.
                        position_fixed = True
                    else:
                        # Tied match, should define randomly
                        coin = np.random.uniform(0, 1)
                        if coin > 0.5:
                            temp = standings_list[idx]
                            standings_list[idx] = standings_list[idx+1]
                            standings_list[idx+1] = temp
                            position_fixed = True
            
    return standings_list

def knockout_stage(matches):
    results = matches_simulations(matches)
    
    winners = []
    for result in results:
        team1_name, team2_name = teams(result[0])
        team1_goals, team2_goals = result[1], result[2]
        
        if team1_goals > team2_goals:
            winners += [team1_name]
        elif team2_goals > team1_goals:
            winners += [team2_name]
        else:
            penalties = np.random.uniform(0, 1)
            winners += [team1_name] if penalties > 0.5 else [team2_name]
    
    return winners

# Write data
output_folder = os.path.dirname(args.output)
pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True) 

def write_data():
    try:
        with open(args.output, "w") as outfile: 
            json.dump(SIMULATION_RESULTS, outfile)
    except IOError:
        logger.error('Error writing JSON file.')


SIMULATIONS_LOG_DELTA = math.ceil(args.simulations/113)
for iteration in range(0, args.simulations):
    if iteration % SIMULATIONS_LOG_DELTA == 0:
        write_data()
        ram = psutil.virtual_memory()
        logger.info(f'Iteration: #{iteration}. CPU: {psutil.cpu_percent(percpu=True)}. RAM: {round((ram.total - ram.available)/1024**3, 1)}G/{round(ram.total/1024**3, 1)}G ({ram.percent}%). Status: {int(iteration*100/args.simulations)}%.')

    groupA_results = matches_simulations(MATCHES_GROUP_A)
    groupB_results = matches_simulations(MATCHES_GROUP_B)
    
    groupA_standings = group_stage(groupA_results)
    groupB_standings = group_stage(groupB_results)
    
    groups_classified = [
        groupA_standings[0][0],
        groupA_standings[1][0],
        groupA_standings[2][0],
        groupA_standings[3][0],
        groupB_standings[0][0],
        groupB_standings[1][0],
        groupB_standings[2][0],
        groupB_standings[3][0]
    ]
    
    SIMULATION_RESULTS[groupA_standings[4][0]]['GROUPS'] += 1
    SIMULATION_RESULTS[groupB_standings[4][0]]['GROUPS'] += 1
    
    quarterfinals_matches = [
        f'{groupB_standings[0][0]}-{groupA_standings[3][0]}',
        f'{groupB_standings[1][0]}-{groupA_standings[2][0]}',
        f'{groupA_standings[0][0]}-{groupB_standings[3][0]}',
        f'{groupA_standings[1][0]}-{groupB_standings[2][0]}'
    ]
    
    quarterfinals_winners = knockout_stage(quarterfinals_matches)
    quarterfinals_losers = [team for team in groups_classified if team not in quarterfinals_winners]
    
    SIMULATION_RESULTS[quarterfinals_losers[0]]['QF'] += 1
    SIMULATION_RESULTS[quarterfinals_losers[1]]['QF'] += 1
    SIMULATION_RESULTS[quarterfinals_losers[2]]['QF'] += 1
    SIMULATION_RESULTS[quarterfinals_losers[3]]['QF'] += 1
    
    semifinals_matches = [
        f'{quarterfinals_winners[0]}-{quarterfinals_winners[1]}',
        f'{quarterfinals_winners[2]}-{quarterfinals_winners[3]}' 
    ]
    
    semifinals_winners = knockout_stage(semifinals_matches)
    semifinals_losers = [team for team in quarterfinals_winners if team not in semifinals_winners]
    
    third_place_match = [f'{semifinals_losers[0]}-{semifinals_losers[1]}']
    third_place_winner = knockout_stage(third_place_match)
    
    final_match = [f'{semifinals_winners[0]}-{semifinals_winners[1]}']
    final_winner = knockout_stage(final_match)
    
    fourth_place = [team for team in semifinals_losers if team not in third_place_winner][0]
    third_place = third_place_winner[0]
    second_place = [team for team in semifinals_winners if team not in final_winner][0]
    champion = final_winner[0]
    
    SIMULATION_RESULTS[fourth_place]['4TH'] += 1
    SIMULATION_RESULTS[third_place]['3RD'] += 1
    SIMULATION_RESULTS[second_place]['2ND'] += 1
    SIMULATION_RESULTS[champion]['1ST'] += 1

write_data()
