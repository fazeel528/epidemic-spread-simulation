# Epidemic Spread Simulation

A Python-based simulation that models how an infectious disease spreads through a population over time. The project demonstrates the basic concepts of epidemic modeling, population states, and disease transmission using simulation and data visualization.

## Features

- Simulates the spread of an infectious disease
- Models different population states such as:
  - Susceptible
  - Infected
  - Recovered
- Tracks the number of people in each state over time
- Visualizes the spread using graphs
- Allows simulation parameters to be adjusted
- Helps understand how transmission affects an entire population

## Technologies Used

- Python
- NumPy
- Matplotlib
- Pandas

## How It Works

The population is divided into different categories based on their health status.

```text
Susceptible → Infected → Recovered
      ↑            |
      └────────────┘
       depending on
       transmission
