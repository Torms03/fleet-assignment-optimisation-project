import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum # library for solving optimisation problems

# subset of BA flights for a day - flitered BA flights for 1 Sept 2019
flights_data = pd.read_excel('BA_Flights_Data_Sept_01.xlsx')

flights_data = pd.DataFrame(flights_data)

# print(flights_data.head())

# print(flights_data.info())

# print(flights_data.isnull().sum())

# list of all UK airport codes
# source:https://www.caa.co.uk/commercial-industry/airports/aerodrome-licences/certificates/uk-certificated-aerodromes/
uk_aerodromes = ['EGPD','EGNM','EGAA','EGGP','EGAC','EGLC','EGPL','EGKK','EGBB','EGLL','EGHH','EGGW','EGGD','EGSS','EGEC','EGCC','EGSC','EGNT','EGFF','EGDG','EGAE','EGSH','EGPN','EGTK','EGNX','EGPK','EGPH','EGHI','EGTE','EGMC','EGPF','EGPO','EGNR','EGPB','EGNJ','EGNV','EGPE','EGPU','EGPI','EGPC','EGPA']

uk_flights_data = flights_data[(flights_data['ADEP'].isin(uk_aerodromes) | flights_data['ADES'].isin(uk_aerodromes))]

# print(uk_flights_data.head())

# print(uk_flights_data.info())

# print(uk_flights_data['AC Type'].unique())

# changing aicraft type to match BA fleet info for 2019 from 2019 annual reports for simplicity

uk_flights_data['AC Type'] = uk_flights_data['AC Type'].replace({'A20N': 'A320'})

uk_flights_data['AC Type'] = uk_flights_data['AC Type'].replace({'A21N': 'A321'})

uk_flights_data['AC Type'] = uk_flights_data['AC Type'].replace({'A332': 'A330'})

uk_flights_data['AC Type'] = uk_flights_data['AC Type'].replace({'A343': 'A330'})

uk_flights_data['AC Type'] = uk_flights_data['AC Type'].replace({'A388': 'A380'})

uk_flights_data['AC Type'] = uk_flights_data['AC Type'].replace({'B77W': 'B773'})

# print(uk_flights_data['AC Type'].unique())

#https://www.visitlondon.com/traveller-information/travel-to-london/airport/london-airport-map?_gl=1*1gbv5qz*_up*MQ..*_ga*MTY4MTY0MTYyNy4xNzM0NDQ4MTIz*_ga_BDFPSHTGM0*MTczNDQ0ODEyMS4xLjAuMTczNDQ0ODEyMS4wLjAuMA..
# Heathrow, London City, Gatwick, Luton, Stansted, Southend

london_airports = ['EGLC', 'EGKK', 'EGLL', 'EGGW', 'EGSS', 'EGMC']

london_flights_data = uk_flights_data[(uk_flights_data['ADEP'].isin(london_airports) | uk_flights_data['ADES'].isin(london_airports))]


# BA aircraft types to consider from annual report 2019
# source:https://www.iairgroup.com/investors-and-shareholders/financial-reporting/annual-reports/
aircraft_types = ["A318", "A319", "A320", "A321", "A330", "A350", "A380", "B744", "B772", "B773", "B788", "B789", "SB20", "E170", "E190"]

# Fleet availability for each aircraft type from annual report 2019
fleet_size = {"A318": 1, "A319": 39, "A320": 77, "A321": 27, "A330": 1, "A350": 3, "A380": 12, "B744": 32, "B772": 46, "B773": 12, "B788": 12, "B789": 18, "SB20": 1, "E170": 6, "E190": 18}

# Cost per aircraft type as per eurocontrol stats values
cost = {"A318": 4829, "A319": 4829, "A320": 4829, "A321": 4829, "A330": 7827, "A350": 0, "A380": 7827, "B744": 5357, "B772": 9507, "B773": 9507, "B788": 7184, "B789": 7184, "SB20": 0, "E170": 0, "E190": 4097}

# passenger seat capacity of each aircraft type (used for demand simulation)
capacity = {"A318": 107, "A319": 124, "A320": 150, "A321": 185, "A330": 277, "A350": 350, "A380": 555, "B744": 416, "B772": 317, "B773": 396, "B788": 242, "B789": 296,"SB20": 50, "E170": 76, "E190": 98}

# Filter flights to include only those with relevant aircraft types for BA fleet
flights_filtered = london_flights_data[london_flights_data['AC Type'].isin(aircraft_types)]

# Map the cost based on aircraft type
flights_filtered["cost"] = flights_filtered["AC Type"].map(cost)

# Simulate demand based on aircraft capacity
np.random.seed(42)  # For reproducibility, generally used in programming
demands = []
# simulates how many passengers want to take each flight using 75% - 85% (Statista report on Passenger Load) of the plane’s capacity, with random variation
# source:https://www.statista.com/statistics/1131673/passenger-load-facor-british-airways-plc/
# Ensures the value stays between 50 and the plane’s full capacity

for ac in flights_filtered['AC Type']:
    demand = np.random.uniform(capacity[ac] * 0.75, capacity[ac] * 0.85)
    demand = int(max(50, min(demand, capacity[ac])))
    demands.append(demand)

flights_filtered['demand'] = demands


# print("Filtered flights with costs and simulated demand:")
# print(flights_filtered)

# Using pulp library: https://realpython.com/linear-programming-python/
# Github project using pulp library for beer distribution project: https://github.com/coin-or/pulp/blob/master/examples/BeerDistributionProblemWarehouseExtension.py

# Create the optimisation problem. Goal is to minimize the total cost of assigning planes to flights
problem = LpProblem("Aircraft_Assignment_Cost_Minimization", LpMinimize)

# Decision variables: x[ECTRL ID, AC Type] = 1 if aircraft is assigned to flight
# Creates a variable for each flight-aircraft pair -> 1 if an aircraft type is assigned to a flight; 0 otherwise.

x = {
    (ectrl_id, ac_type): LpVariable(name=f"x_{ectrl_id}_{ac_type}", cat="Binary")
    for ectrl_id in flights_filtered["ECTRL ID"]
    for ac_type in aircraft_types
}

# Objective function: Minimise total assignment cost
problem += lpSum(
    flights_filtered.loc[flights_filtered['ECTRL ID'] == ectrl_id, 'cost'].values[0] * x[ectrl_id, ac_type]
    for ectrl_id in flights_filtered["ECTRL ID"]
    for ac_type in aircraft_types
), "Total_Cost"

# Constraint: Each flight is assigned exactly one aircraft
for ectrl_id in flights_filtered["ECTRL ID"]:
    problem += lpSum(x[ectrl_id, ac_type] for ac_type in aircraft_types) == 1, f"One_Aircraft_Per_Flight_{ectrl_id}"

# Constraint: Aircraft availability check
for ac_type in aircraft_types:
    problem += lpSum(x[ectrl_id, ac_type] for ectrl_id in flights_filtered["ECTRL ID"]) <= fleet_size[ac_type], f"Availability_{ac_type}"

# Constraint: Aircraft check if capacity meets simulated demand
for ectrl_id in flights_filtered["ECTRL ID"]:
    demand = flights_filtered.loc[flights_filtered['ECTRL ID'] == ectrl_id, 'demand'].values[0]
    problem += lpSum(capacity[ac_type] * x[ectrl_id, ac_type] for ac_type in aircraft_types) >= demand, f"Demand_Met_{ectrl_id}"


status = problem.solve()

# Output results
print(f"Status: {problem.status}")
results = []
for ectrl_id in flights_filtered["ECTRL ID"]:
    for ac_type in aircraft_types:
        if x[ectrl_id, ac_type].value() == 1:
            results.append({"ECTRL ID": ectrl_id, "assigned_aircraft": ac_type})
            print(f"Flight {ectrl_id} is assigned to {ac_type}")
