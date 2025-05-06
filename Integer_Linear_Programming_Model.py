# Import necessary libraries after installing using pip install.
from pulp import LpProblem, LpMinimize, LpVariable, lpSum 
import pandas as pd  
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
aircraft_range_raw = {
    "A318": 4200 * 1.852, "A319": 6700, "A320": 6500, "A321": 5600, "A339": 7200 * 1.852, 
    "A35K": 16100, "A388": 15400, "B744": 12200, "B772": 6857, "B773": 14685, "B788": 15200, 
    "B789": 15400, "SB20": 1000 * 1.852, "E170": 3700, "E190": 3334
}



### Section 4: OPTIMISATION WITH PENALTY VARIABLES ###

# Define the linear programming problem as a minimization problem
problem = LpProblem("Aircraft_Assignment_With_Penalties", LpMinimize)

# Variables: Decision variables indicating which aircraft is assigned to each flight
x = {(ectrl_id, ac_type): LpVariable(f"x_{ectrl_id}_{ac_type}", cat="Binary") 
     for ectrl_id in flights_filtered["ECTRL_ID"]
     for ac_type in BA_fleet_AC_type}

# Slack variables for soft constraints
slack_one_aircraft = {ectrl_id: LpVariable(f"slack_one_aircraft_{ectrl_id}", lowBound=0, cat="Integer") 
                      for ectrl_id in flights_filtered["ECTRL_ID"]}

slack_fleet = {ac_type: LpVariable(f"slack_fleet_{ac_type}", lowBound=0, cat="Integer") 
               for ac_type in BA_fleet_AC_type}

slack_demand = {ectrl_id: LpVariable(f"slack_demand_{ectrl_id}", lowBound=0, cat="Integer") 
                for ectrl_id in flights_filtered["ECTRL_ID"]}

# Objective function: Minimize cost and penalties for constraint violations
problem += (
    lpSum(cost[ac_type] * x[ectrl_id, ac_type] for ectrl_id in flights_filtered["ECTRL_ID"] for ac_type in BA_fleet_AC_type)
    + 1e5 * lpSum(slack_one_aircraft.values())   # Penalty for missing aircraft assignment
    + 1e5 * lpSum(slack_fleet.values())           # Penalty for exceeding fleet size
    + 1e5 * lpSum(slack_demand.values())          # Penalty for demand not met
), "Total_Cost_With_Penalties"

## Constraints

# 1. Ensure that each flight gets exactly one aircraft (soft constraint)
for ectrl_id in flights_filtered["ECTRL_ID"]:
    problem += lpSum(x[ectrl_id, ac_type] for ac_type in BA_fleet_AC_type) + slack_one_aircraft[ectrl_id] == 1, f"One_Aircraft_{ectrl_id}"

# 2. Ensure that the number of aircraft of each type used does not exceed the fleet size (soft constraint)
for ac_type in BA_fleet_AC_type:
    problem += lpSum(x[ectrl_id, ac_type] for ectrl_id in flights_filtered["ECTRL_ID"]) <= fleet_size[ac_type] + slack_fleet[ac_type], f"Fleet_Availability_{ac_type}"

# 3. Ensure that passenger demand for each flight is met (soft constraint)
for ectrl_id in flights_filtered["ECTRL_ID"]:
    demand = flights_filtered.loc[flights_filtered['ECTRL_ID'] == ectrl_id, 'demand'].values[0]
    problem += lpSum(capacity[ac_type] * x[ectrl_id, ac_type] for ac_type in BA_fleet_AC_type) + slack_demand[ectrl_id] >= demand, f"Demand_Met_{ectrl_id}"

# 4. Ensure that aircraft range is sufficient for the flight (hard constraint)
for ectrl_id in flights_filtered["ECTRL_ID"]:
    distance = flights_filtered.loc[flights_filtered['ECTRL_ID'] == ectrl_id, 'distance_km'].values[0]
    for ac_type in BA_fleet_AC_type:
        if aircraft_range_raw.get(ac_type, 0) < distance:
            problem += x[ectrl_id, ac_type] == 0, f"Range_Limit_{ectrl_id}_{ac_type}"

# Solve the optimization problem
start_time = time.time()  # Start timing the optimization process
status = problem.solve()  # Solve the linear program
end_time = time.time()  # End timing
runtime_seconds = end_time - start_time  # Calculate the runtime

# Output solver status and execution time
print(f"\nSolver Status: {problem.status}")
print(f"Total Runtime: {runtime_seconds:.2f} seconds")

# Save the assignments (flight-to-aircraft assignments)
assignments = []
for ectrl_id in flights_filtered["ECTRL_ID"]:
    for ac_type in BA_fleet_AC_type:
        if x[ectrl_id, ac_type].value() == 1:
            assignments.append({"ECTRL_ID": ectrl_id, "Assigned Aircraft": ac_type})

assignments_df = pd.DataFrame(assignments)
assignments_df.to_csv("integer_programming_assignment.csv", index=False)

# Save the actual assignment
flights_filtered.to_csv("actual_aircraft_assignment.csv", index=False)

print("Assignments saved.")



### Section 5: CONSTRAINT VIOLATION REPORT ###

print("\nConstraint Violation Report:")

# Check for flights missing aircraft assignment
print("\nFlights missing aircraft assignment (due to slack_one_aircraft > 0):")
for ectrl_id, slack_var in slack_one_aircraft.items():
    if slack_var.value() > 0:
        print(f"  - Flight {ectrl_id}: Missing by {slack_var.value()} aircraft(s)")

# Check for fleet size violations
print("\nFleet size exceeded for the following aircraft types (due to slack_fleet > 0):")
for ac_type, slack_var in slack_fleet.items():
    if slack_var.value() > 0:
        print(f"  - Aircraft type {ac_type}: Exceeded by {slack_var.value()} aircraft(s)")

# Check for flights not meeting demand
print("\nFlights not meeting passenger demand (due to slack_demand > 0):")
for ectrl_id, slack_var in slack_demand.items():
    if slack_var.value() > 0:
        print(f"  - Flight {ectrl_id}: Short by {round(slack_var.value(), 2)} passengers")

print("\nConstraint Violation Check Complete.\n")



### Section 6: PENALTY SUMMARY ###
print("\nPenalty Summary:")

# Calculate total penalties for each type of violation
total_assignment_penalty = sum(slack_var.value() for slack_var in slack_one_aircraft.values()) * 1e6
total_fleet_penalty = sum(slack_var.value() for slack_var in slack_fleet.values()) * 1e5
total_demand_penalty = sum(slack_var.value() for slack_var in slack_demand.values()) * 1e5

# Print out the penalty summary
print(f"Total Assignment Penalty (missing aircraft assignments): £{total_assignment_penalty:,.2f}")
print(f"Total Fleet Penalty (exceeding fleet size): £{total_fleet_penalty:,.2f}")
print(f"Total Demand Penalty (not meeting passenger demand): £{total_demand_penalty:,.2f}")

# Calculate total penalty
total_penalty = total_assignment_penalty + total_fleet_penalty + total_demand_penalty
print(f"Grand Total Penalty: £{total_penalty:,.2f}")

print("\nPenalty Summary Complete.\n")

## PENALTY AS % OF TOTAL COST

# Calculate total assignment cost (without penalties)
total_assignment_cost = sum(cost[ac_type] * x[ectrl_id, ac_type].value() 
                            for ectrl_id in flights_filtered["ECTRL_ID"] 
                            for ac_type in BA_fleet_AC_type)

# Calculate total cost including penalties
total_cost_with_penalties = total_assignment_cost + total_penalty

# Calculate penalty as a percentage of the total cost
penalty_percentage = (total_penalty / total_cost_with_penalties) * 100

# Print out the penalty percentage
print(f"\nTotal Assignment Cost (without penalties): £{total_assignment_cost:,.2f}")
print(f"Total Cost (with penalties): £{total_cost_with_penalties:,.2f}")
print(f"\nPenalty as % of Total Cost: {penalty_percentage:.2f}%")

print("\nPenalty as % of Total Cost calculation complete.\n")
