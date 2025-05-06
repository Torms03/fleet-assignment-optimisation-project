import numpy as np  
import pandas as pd  
import random  
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
uk_flights_data = flights_filtered
aircraft_types = BA_fleet_AC_type



### Section 4: OPTIMISATION WITH PENALTY VARIABLES ###
# Initialize pheromone levels for Ant Colony Optimization
pheromone = {}
for ectrl_id in uk_flights_data['ECTRL_ID'].unique():
    pheromone[ectrl_id] = {ac_type: 1.0 for ac_type in aircraft_types}

# Hyperparameters for Ant Colony Optimization (ACO)
n_ants = 30  # Number of ants (solutions) to explore
n_iterations = 150  # Number of iterations (loops) for the ACO algorithm
alpha = 2  # Influence of pheromone level on decision-making
beta = 3  # Influence of cost on decision-making
evaporation_rate = 0.3  # Rate at which pheromone evaporates (gets weaker)

# Variables to store the best solution found during ACO
best_solution = None
best_cost = float('inf')  # Set the initial best cost to be a very large number

# Initiate runtime
start_time = time.time()  # Start timing

# Ant Colony Optimization Algorithm
for iteration in range(n_iterations):  # Loop through each iteration
    solutions = []  # List to store the solutions (assignments) found in this iteration
    costs = []  # List to store the total cost for each solution

    for ant in range(n_ants):  # Loop through each ant
        solution = {}  # Create an empty dictionary to store the solution for this ant
        total_cost = 0  # Initialize the total cost for this solution

        for ectrl_id in uk_flights_data['ECTRL_ID']:  # Loop through each flight
        # Get the current flight's distance
            flight_row = uk_flights_data[uk_flights_data['ECTRL_ID'] == ectrl_id].iloc[0]
            distance = flight_row['distance_km']

        # Filter aircraft that are available and have the required range
            available_aircraft = [
            ac_type for ac_type in aircraft_types
            if fleet_size[ac_type] > 0 and aircraft_range.get(ac_type, 0) >= distance
            ]

            # Fallback: no aircraft can meet the range -> apply penalty and assign a random one
            if not available_aircraft:
                selected_aircraft = random.choice(aircraft_types)
                total_cost += 1e5  # Heavy penalty for infeasible assignment
            else:
                # Compute selection probabilities
                probabilities = []
                total_pheromone = sum(
                    (pheromone[ectrl_id][ac] ** alpha) * ((1 / cost[ac]) ** beta)
                    for ac in available_aircraft
                )

                for ac in available_aircraft:
                    prob = (pheromone[ectrl_id][ac] ** alpha) * ((1 / cost[ac]) ** beta) / total_pheromone
                    probabilities.append(prob)

                selected_aircraft = random.choices(available_aircraft, probabilities)[0]
                total_cost += cost[selected_aircraft]

            # Store the selected aircraft for this flight
            solution[ectrl_id] = selected_aircraft


        solutions.append(solution)
        costs.append(total_cost)

        if total_cost < best_cost:
            best_solution = solution
            best_cost = total_cost

    for ectrl_id in pheromone:
        for ac_type in pheromone[ectrl_id]:
            pheromone[ectrl_id][ac_type] *= (1 - evaporation_rate)

    for solution, total_cost in zip(solutions, costs):
        if total_cost == best_cost:
            for ectrl_id, ac_type in solution.items():
                pheromone[ectrl_id][ac_type] += 1 / total_cost


# End runtime
end_time = time.time()  # End timing
print(f"\n=== ACO Runtime: {end_time - start_time:.2f} seconds ===\n")



# Output the best solution found and Save the best solution to a DataFrame to export as CSV
best_solution_df = pd.DataFrame(list(best_solution.items()), columns=['ECTRL_ID', 'Assigned_Aircraft'])
best_solution_df.to_csv('ant_colony_assignment.csv', index=False)


### Section 5: DIAGNOSTICS AND CONSTRAINT VIOLATION REPORT ###

print("########### Diagnostic 2 check ########################")

print("=== DIAGNOSTIC REPORT (ACO with Distance Constraint) ===")

# Merge best solution with flight data
diagnostic_df = uk_flights_data.copy()
diagnostic_df['Assigned_Aircraft'] = diagnostic_df['ECTRL_ID'].map(best_solution)

# Count how many flights were assigned to each aircraft type
assignment_counts = diagnostic_df['Assigned_Aircraft'].value_counts().to_dict()

# Fleet usage
print("\nAircraft Assignment Summary:")
for ac_type in aircraft_types:
    assigned = assignment_counts.get(ac_type, 0)
    total = fleet_size.get(ac_type, 0)
    print(f" - {ac_type}: Assigned {assigned} flights | Fleet Size: {total}")

# Overused aircraft types
print("\n Overused Aircraft Types:")
overused = False
for ac_type in aircraft_types:
    assigned = assignment_counts.get(ac_type, 0)
    if assigned > fleet_size.get(ac_type, 0):
        print(f" - {ac_type}: Overused by {assigned - fleet_size[ac_type]} flights")
        overused = True
if not overused:
    print("None")

# Demand capacity check
print("\nFlights Exceeding Assigned Aircraft Capacity:")
exceeds = diagnostic_df[diagnostic_df['demand'] > diagnostic_df['Assigned_Aircraft'].map(capacity)]
if exceeds.empty:
    print("None")
else:
    print(f"{len(exceeds)} flights exceed assigned aircraft capacity.")

# Distance feasibility check
print("\nFlights Exceeding Aircraft Range:")
exceeds_range = diagnostic_df[
    diagnostic_df['distance_km'] > diagnostic_df['Assigned_Aircraft'].map(aircraft_range)
]
if exceeds_range.empty:
    print("None")
else:
    print(f"{len(exceeds_range)} flights exceed assigned aircraft range.")

# Total Cost Summary
print(f"\nTotal Cost of Best Solution: £{best_cost:,.2f}")
print("=== END OF DIAGNOSTIC ===")




