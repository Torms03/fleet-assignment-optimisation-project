import numpy as np
import random
import pandas as pd
from deap import base, creator, tools, algorithms
import math
import time


### Section 1: DATA LOADING + CLEANING FOR FLIGHTS DATA ###

# Load flight data from an Excel file
flights_data = pd.read_excel("D:\FYP\FINAL YEAR PROJECT PROGRESS\Datasets\Flights_Data.xlsx") 

# Clean column names by replacing spaces with underscores
flights_data.columns = flights_data.columns.str.replace(' ', '_')

# Filter flights that either depart or arrive in the UK (EG - ICAO code for UK)
uk_flights_data = flights_data[(flights_data['ADEP'].str[:2] == 'EG') | (flights_data['ADES'].str[:2] == 'EG')]

# Filter further to select only flights operated by certain operators (BAW, CFE, EFW)
BA_uk_flights = uk_flights_data[(uk_flights_data['AC_Operator'].isin(['BAW', 'CFE', 'EFW']))]

# Convert 'ACTUAL_OFF_BLOCK_TIME' column to datetime format
BA_uk_flights['ACTUAL_OFF_BLOCK_TIME'] = pd.to_datetime(BA_uk_flights['ACTUAL_OFF_BLOCK_TIME'], format='%d/%m/%Y %H:%M')

# Filter flights for a specific date (1st Sept 2019)
BA_uk_flights_01_Sept = BA_uk_flights[BA_uk_flights['ACTUAL_OFF_BLOCK_TIME'].dt.date == pd.to_datetime('2019-09-01').date()]

# Define list of London airport codes
london_airports = ['EGLL', 'EGKK', 'EGSS', 'EGGW', 'EGLC', 'EGMC']

# Filter further to include only flights departing or arriving at London airports
BA_london_flights_01_Sept = BA_uk_flights_01_Sept[(BA_uk_flights_01_Sept['ADEP'].isin(london_airports)) | (BA_uk_flights_01_Sept['ADES'].isin(london_airports))]

# Define fleet of aircraft types operated by British Airways
BA_fleet_AC_type = ['A318', 'A319', 'A320', 'A321', 'A339', 'A35K', 'A388', 'B744', 'B772', 'B773', 'B788', 'B789', 'SB20', 'E170', 'E190']

# Replace aircraft type codes for consistency with the fleet types
BA_london_flights_01_Sept['AC_Type'] = BA_london_flights_01_Sept['AC_Type'].replace({
    'A20N': 'A320', 'A21N': 'A321', 'B77W': 'B773', 'A343': 'A321', 'A332': 'A339'
})

# Fleet size definition for each aircraft type
fleet_size = {"A318": 1, "A319": 39, "A320": 77, "A321": 27, "A339": 1, "A35K": 3, "A388": 12, 
              "B744": 32, "B772": 46, "B773": 12, "B788": 12, "B789": 18, "SB20": 1, "E170": 6, "E190": 18}

# Cost for each aircraft type 
cost = {"A318": 35407.76, "A319": 35407.76, "A320": 35407.76, "A321": 35407.76, "A339": 82260.66, "A35K": 82260.66, "A388": 82260.66, 
        "B744": 82260.66, "B772": 110041.48, "B773": 110041.48, "B788": 84158.34, "B789": 84158.34, "SB20": 11262.86, "E170": 30018.42, "E190": 30018.42}

# Seating capacity for each aircraft type
capacity = {"A318": 132, "A319": 144, "A320": 180, "A321": 220, "A339": 465, "A35K": 331, "A388": 469, 
            "B744": 416, "B772": 336, "B773": 297, "B788": 214, "B789": 216, "SB20": 53, "E170": 76, "E190": 106}

# Filter flights for British Airways
flights_filtered = BA_london_flights_01_Sept[BA_london_flights_01_Sept['AC_Type'].isin(BA_fleet_AC_type)]

# Map costs to flights based on aircraft type
flights_filtered["cost"] = flights_filtered["AC_Type"].map(cost)


### Section 2: DATA LOADING + CLEANING FOR PASSENGER DEMAND DATA ###

# Load passenger traffic data and airport codes mapping
passenger_data = pd.read_csv("D:\FYP\FINAL YEAR PROJECT PROGRESS\Datasets\gb_pax_traffic.csv")
airport_codes = pd.read_excel("D:\FYP\FINAL YEAR PROJECT PROGRESS\Datasets\Airport codes IATA-ICAO.xlsx")

# Convert IATA codes to ICAO codes for both origin and destination airports
IATA_to_ICAO = airport_codes.set_index('IATA')['ICAO'].to_dict()
passenger_data['Origin_Airport_ICAO'] = passenger_data['Origin Airport'].map(IATA_to_ICAO)
passenger_data['Destination_Airport_ICAO'] = passenger_data['Destination Airport'].map(IATA_to_ICAO)

# Filter passenger data to include only flights involving London airports
passenger_data_london = passenger_data[
    (passenger_data['Origin_Airport_ICAO'].isin(london_airports)) | 
    (passenger_data['Destination_Airport_ICAO'].isin(london_airports))
]

# Further filter for non-stop flights (direct flights)
passenger_london_direct = passenger_data_london[passenger_data_london['Itinerary'] == 'NON-STOP']

# Load aircraft type counts per flight route
flights_ac_type_count = pd.read_excel("D:/FYP/FINAL YEAR PROJECT PROGRESS/Datasets/flights_data_ac_type.xlsx")

# Map aircraft types to each unique origin-destination route pair
ADEP_ADES_ac_type = flights_ac_type_count.set_index('ADEP-ADES')['AC Type'].to_dict()

# Create a combined route identifier (e.g. 'EGLL-EGKK') for direct flights
passenger_london_direct['Origin_Destination_ICAO'] = passenger_london_direct['Origin_Airport_ICAO'] + '-' + passenger_london_direct['Destination_Airport_ICAO']

# Map the aircraft type for each route based on the pre-defined mapping
passenger_london_direct['AC_Type'] = passenger_london_direct['Origin_Destination_ICAO'].map(ADEP_ADES_ac_type)

# Drop rows with missing aircraft type information
passenger_london_direct = passenger_london_direct.dropna(subset=['AC_Type'])

# Assign real demand values for each route
route_demand_mapping = passenger_london_direct.groupby(['Origin_Destination_ICAO'])['BA_PPDEW'].sum().to_dict()

# Create route identifiers for the flights
flights_filtered['Origin_Destination_ICAO'] = flights_filtered['ADEP'] + '-' + flights_filtered['ADES']

# Function to get the demand for a specific route (default to 50 if not found)
def get_route_demand(route):
    return route_demand_mapping.get(route, 50)

# Map the demand for each flight based on the route
flights_filtered['demand'] = flights_filtered['Origin_Destination_ICAO'].apply(get_route_demand)


### Section 3: DISTANCE CALCULATION ###

# Function to calculate the great-circle distance between two coordinates (haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)  # Convert latitudes to radians
    delta_phi, delta_lambda = math.radians(lat2-lat1), math.radians(lon2-lon1)  # Calculate differences
    a = math.sin(delta_phi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2.0)**2  # Haversine formula
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))  # Return distance in kilometers

# Calculate distance for each flight based on departure and destination coordinates
flights_filtered['distance_km'] = flights_filtered.apply(
    lambda row: haversine(row['ADEP_Latitude'], row['ADEP_Longitude'], row['ADES_Latitude'], row['ADES_Longitude']), axis=1
)

# Define aircraft range (in kilometers). Note that A318 and A339 were in nautical miles and converted to kilometers accordingly.
aircraft_range = {
    "A318": 4200 * 1.852, "A319": 6700, "A320": 6500, "A321": 5600, "A339": 7200 * 1.852, 
    "A35K": 16100, "A388": 15400, "B744": 12200, "B772": 6857, "B773": 14685, "B788": 15200, 
    "B789": 15400, "SB20": 1000 * 1.852, "E170": 3700, "E190": 3334
}

# Rename flights_data and aircraft_types variables
flights_data = flights_filtered
aircraft_types = BA_fleet_AC_type


### Section 4: OPTIMISATION WITH PENALTY VARIABLES ###

## Create the Fitness and Individual classes using the DEAP creator
# Define a 'FitnessMin' class that minimizes the cost (negative weights mean minimizing)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Define an 'Individual' class as a list of aircraft types
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize an individual (random aircraft assignments)
def init_individual():
    individual = []  # Create an empty list to represent an individual solution
    for i in range(len(flights_data)):  # Loop through each flight
        individual.append(random.choice(aircraft_types))  # Randomly select an aircraft for this flight
    return individual

# Cost Function (the goal is to minimize the total cost)
def eval_cost(individual):
    total_cost = 0  # Initialize the total cost to 0
    fleet_used = {}  # Create an empty dictionary to track how many of each aircraft type are used
    
    # Initialize the fleet_used dictionary with all aircraft types set to 0
    for ac in aircraft_types:
        fleet_used[ac] = 0

    # Evaluate the cost for the individual solution
    for i, ac in enumerate(individual):
        total_cost += cost[ac]  # Add the cost of the aircraft to the total cost
        fleet_used[ac] += 1  # Increase the count for the current aircraft type

        # Penalty if the fleet size is exceeded
        if fleet_used[ac] > fleet_size[ac]:
            total_cost += 10000  # Penalty if fleet size is exceeded
        
        # Penalty if the aircraft capacity is smaller than the flight demand
        if capacity[ac] < flights_data.iloc[i]['demand']:
            total_cost += 10000  # Penalty for not meeting demand

    # Check if aircraft can complete the flight (distance constraint)
    distance = flights_data.iloc[i]['distance_km']
    if aircraft_range.get(ac, 0) < distance:
        total_cost += 10000  # Penalty for range violation

    return (total_cost,)  # Return the total cost as a tuple

# Mutation Function (change an aircraft type for a flight randomly)
def mutate_individual(individual):
    # Randomly mutate an individual with a 10% chance
    if random.random() < 0.1:
        idx = random.randint(0, len(individual) - 1)  # Randomly select a flight
        individual[idx] = random.choice(aircraft_types)  # Assign a new random aircraft type
    return individual,

# Genetic Algorithm Components
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)  # Create individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Create population
toolbox.register("evaluate", eval_cost)  # Register evaluation function
toolbox.register("mate", tools.cxTwoPoint)  # Crossover function (mix two individuals)
toolbox.register("mutate", mutate_individual)  # Mutation function (change one individual)
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection function (pick the best)

# Run the Genetic Algorithm
population = toolbox.population(n=100)  # Create a population of 100 individuals

# Start the timer before running the GA
start_time = time.time()
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)  # Run the algorithm for 50 generations
# End the timer after the GA completes
end_time = time.time()

# Calculate the runtime
runtime_seconds = end_time - start_time
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds")

# Extract the Best Solution
best_ind = tools.selBest(population, k=1)[0]  # Get the best solution from the population



### Section 5: DIAGNOSTICS AND CONSTRAINT VIOLATION REPORT ###
print("\nGenetic Algorithm Diagnostics Report:")

# Track violations for the best solution
unassigned_flights = {}  
fleet_violations = {}
demand_violations = {}
range_violations = {}

# Calculate fleet usage from best individual
fleet_used = {ac_type: 0 for ac_type in aircraft_types}
for i, ac_type in enumerate(best_ind):
    fleet_used[ac_type] += 1

# Check for constraint violations
total_fleet_penalty = 0
total_demand_penalty = 0
total_range_penalty = 0

# Check fleet size violations
for ac_type in aircraft_types:
    if fleet_used[ac_type] > fleet_size[ac_type]:
        excess = fleet_used[ac_type] - fleet_size[ac_type]
        fleet_violations[ac_type] = excess
        total_fleet_penalty += excess * 10000

# Check for demand and range violations
for i, ac_type in enumerate(best_ind):
    ectrl_id = flights_data.iloc[i]['ECTRL_ID']
    demand = flights_data.iloc[i]['demand']
    distance = flights_data.iloc[i]['distance_km']
    
    # Check if demand is not met
    if capacity[ac_type] < demand:
        shortfall = demand - capacity[ac_type]
        demand_violations[ectrl_id] = shortfall
        total_demand_penalty += 10000  # Fixed penalty per violation 
    
    # Check if aircraft range is insufficient
    if aircraft_range.get(ac_type, 0) < distance:
        range_violations[ectrl_id] = distance - aircraft_range.get(ac_type, 0)
        total_range_penalty += 10000  # Fixed penalty per violation 

# Check for unassigned flights 
total_unassigned_penalty = 0
# In the genetic algorithm, every flight gets an aircraft assignment, but we could check for invalid ones
# For example, might consider cases where an aircraft can't meet the range requirements as effectively "unassigned"
for i, ac_type in enumerate(best_ind):
    ectrl_id = flights_data.iloc[i]['ECTRL_ID']
    # If we somehow have an invalid aircraft type (not in our list), count as unassigned
    if ac_type not in aircraft_types:
        unassigned_flights[ectrl_id] = 1
        total_unassigned_penalty += 10000

## CONSTRAINT VIOLATION REPORT
print("\nConstraint Violation Report:")

# Check flights missing aircraft assignment
print("\nFlights missing aircraft assignment:")
if unassigned_flights:
    for ectrl_id in unassigned_flights:
        print(f"  - Flight {ectrl_id}: Not assigned a valid aircraft")
else:
    print("  - All flights have aircraft assignments")

# Check fleet size violations
print("\nFleet size exceeded for the following aircraft types:")
if fleet_violations:
    for ac_type, excess in fleet_violations.items():
        print(f"  - Aircraft type {ac_type}: Exceeded by {excess} aircraft(s)")
else:
    print("  - No fleet size violations")

# Check flights not meeting demand
print("\nFlights not meeting passenger demand:")
if demand_violations:
    for ectrl_id, shortfall in demand_violations.items():
        print(f"  - Flight {ectrl_id}: Short by {round(shortfall, 2)} passengers")
else:
    print("  - All passenger demands are met")

# Check range violations
print("\nFlights with aircraft range violations:")
if range_violations:
    for ectrl_id, shortfall in range_violations.items():
        print(f"  - Flight {ectrl_id}: Range short by {round(shortfall, 2)} km")
else:
    print("  - No range violations")

print("\nConstraint Violation Check Complete.\n")

# PENALTY SUMMARY
print("\nPenalty Summary:")

print(f"Total Assignment Penalty (missing aircraft assignments): £{total_unassigned_penalty:,.2f}")
print(f"Total Fleet Penalty (exceeding fleet size): £{total_fleet_penalty:,.2f}")
print(f"Total Demand Penalty (not meeting passenger demand): £{total_demand_penalty:,.2f}")
print(f"Total Range Penalty (aircraft cannot cover distance): £{total_range_penalty:,.2f}")

total_penalty = total_unassigned_penalty + total_fleet_penalty + total_demand_penalty + total_range_penalty
print(f"Grand Total Penalty: £{total_penalty:,.2f}")

print("\nPenalty Summary Complete.\n")

# PENALTY AS % OF TOTAL COST
# Calculate the actual assignment cost from best individual
total_assignment_cost = sum(cost[ac_type] for ac_type in best_ind)

# Total cost including penalties
total_cost_with_penalties = total_assignment_cost + total_penalty

# Calculate penalty as a percentage of the total cost
if total_cost_with_penalties > 0:  # Avoid division by zero
    penalty_percentage = (total_penalty / total_cost_with_penalties) * 100
else:
    penalty_percentage = 0

print(f"\nTotal Assignment Cost (without penalties): £{total_assignment_cost:,.2f}")
print(f"Total Cost (with penalties): £{total_cost_with_penalties:,.2f}")
print(f"Penalty as % of Total Cost: {penalty_percentage:.2f}%")

print("\nPenalty as % of Total Cost calculation complete.\n")

# Export the final solution with complete details for better analysis
detailed_assignments = []
for i, ac_type in enumerate(best_ind):
    flight = flights_data.iloc[i]
    ectrl_id = flight['ECTRL_ID']
    demand = flight['demand']
    distance = flight['distance_km']
    
    # Calculate violations for this specific flight
    demand_violation = max(0, demand - capacity[ac_type])
    range_violation = 1 if aircraft_range.get(ac_type, 0) < distance else 0
    
    detailed_assignments.append({
        'ECTRL_ID': ectrl_id,
        'Assigned_Aircraft': ac_type,
        'Aircraft_Cost': cost[ac_type],
        'Aircraft_Capacity': capacity[ac_type],
        'Flight_Demand': demand,
        'Demand_Violation': demand_violation,
        'Flight_Distance': distance,
        'Aircraft_Range': aircraft_range.get(ac_type, 0),
        'Range_Violation': range_violation
    })

# Convert to DataFrame and save
detailed_results_df = pd.DataFrame(detailed_assignments)
detailed_results_df.to_csv("genetic_aircraft_assignments_detailed_diagnostics.csv", index=False)
print("\nDetailed diagnostics saved to CSV file.")


### Section 5: SAVING RESULTS OF ASSIGNMENT ###
# Assign aircraft to flights and save the result
assignments = []  

# Loop through the best individual solution and assign aircraft to flights
for i in range(len(best_ind)):
    assignments.append({
        'ECTRL ID': flights_data.iloc[i]['ECTRL_ID'],  # Get the ECTRL ID for this flight
        'Aircraft Assigned': best_ind[i]  # Assign the selected aircraft to this flight
    })

# Convert assignments to a DataFrame and save it as a CSV file
result_df = pd.DataFrame(assignments)
result_df.to_csv("genetic_aircraft_assignments.csv", index=False)



