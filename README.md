# DistribuPrivacyCore

DistribuPrivacyCore is a Python-based project designed for [Brief Project Description]. This project was tested and run using **Python 3.10**. Please ensure that all required dependencies are installed (listed in `requirements.txt`).

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Hungarian Algorithm Integration**: The project implements a matching algorithm based on the Hungarian method for solving assignment problems, specifically matching drivers and passengers in a ride-sharing context.
  
- **Probabilistic Matching**: Incorporates a Las Vegas-style randomized algorithm for matching, which runs probabilistically until an optimal or acceptable match is found, ensuring randomness in the matching process.

- **Noise Incorporation**: The system adds artificial noise to passenger locations using polar coordinates, which ensures privacy and simulates real-world uncertainties in passenger location data.

- **Runtime Comparison**: Includes runtime comparisons between the Hungarian, Greedy, and Las Vegas algorithms, with results displayed graphically to illustrate performance across multiple trials.

- **Dynamic Visualization**: The project provides visualizations to represent matching decisions over time, showing how the system evolves and updates its driver-passenger assignments.

- **Scalable Framework**: Handles large-scale scenarios with hundreds or thousands of drivers and passengers, making the solution applicable to real-world urban transportation networks.

## Requirements

- **Python 3.10** or higher
- All required dependencies can be installed using the `requirements.txt` file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DistribuPrivacyCore.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd DistribuPrivacyCore
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. After installing the required dependencies, you can run the project with the following command:
   ```bash
   python main.py
   ```

## Code Overview

### main.py

- This script serves as the entry point for the project. It initializes the core modules and handles the overall process flow.

### auct.py

- Contains the core logic for the auction algorithm.

### requirements.txt

- List of all required Python packages. Use `pip` to install these dependencies.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

License
This project is licensed under the [choose license, e.g., MIT License].

## License

This project is licensed under the MIT License. 
## Contact

For questions or further information, please contact:
- **Jonathan** (jonathany@mail.tau.ac.il)
