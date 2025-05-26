# Project overview and setup instructions
# Elephant Corridor Prediction using Energy Landscapes and Circuit Theory

A comprehensive Python implementation for predicting elephant movement corridors using energy landscape modeling and circuit theory approaches, based on recent advances in movement ecology and conservation planning.

## Overview

This project implements a state-of-the-art pipeline for analyzing elephant movement patterns and predicting optimal corridors between habitat patches. The methodology combines:

- **Energy Landscape Modeling**: Based on Berti et al. (2025) energy landscapes approach using Pontzer's unified locomotion cost equations
- **Circuit Theory**: Using Circuitscape for connectivity analysis that considers multiple pathway options
- **Geospatial Analysis**: Comprehensive GIS workflow for processing GPS telemetry and environmental data

## Features

### 🔬 **Scientific Methods**
- Implementation of Pontzer's biomechanical energy cost equations
- Slope-based resistance surfaces accounting for elephant physiology
- Multi-scale corridor analysis (least-cost paths + circuit theory)
- Integration with NDVI for habitat quality assessment

### 🗺️ **Geospatial Processing**
- Automated GPS telemetry data ingestion and validation
- DEM processing and slope calculation
- Area of Interest (AOI) generation from tracking data
- CRS handling and projection management

### 📊 **Visualization & Analysis**
- Interactive corridor maps using Folium
- Static publication-quality figures
- Movement animation capabilities
- Comprehensive model validation metrics

### 🧪 **Robust Testing**
- Unit tests for all major components
- Synthetic data validation
- Cross-validation frameworks
- Performance benchmarking

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/elephant-corridors.git
cd elephant-corridors

# Create and activate conda environment
conda env create -f environment.yml
conda activate elephant-corridors

# Verify installation
python scripts/setup_environment.py
```

### Option 2: Pip Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### System Dependencies

For R integration (energyscape package):
```bash
# Ubuntu/Debian
sudo apt install r-base gdal-bin libgdal-dev libgeos-dev

# macOS
brew install r gdal geos proj

# Install R packages
R -e "install.packages('enerscape')"
```

## Quick Start

### 1. Prepare Your Data

Place your GPS collar data in `data/raw/`:
```
data/raw/
├── elephant_tracks_2024.csv
├── dem_study_area.tif
└── ndvi_composite.tif
```

### 2. Configure the Project

Copy and customize the environment file:
```bash
cp .env .env.local
# Edit .env.local with your study area parameters
```

### 3. Run the Analysis

```python
from elephant_corridors import CorridorAnalysis

# Initialize analysis
analysis = CorridorAnalysis(config_file='.env.local')

# Load and process data
analysis.load_gps_data('data/raw/elephant_tracks_2024.csv')
analysis.generate_energy_surface('data/raw/dem_study_area.tif')

# Predict corridors
corridors = analysis.predict_corridors()

# Generate visualizations
analysis.create_corridor_map(save_path='outputs/corridor_map.html')
```

## Project Structure

```
elephant-corridors/
├── src/                          # Source code
│   ├── data_ingestion.py         # GPS data processing
│   ├── processing/
│   │   ├── enerscape.py          # Energy landscape calculations
│   │   ├── corridors.py          # LCP and circuit theory
│   │   └── visualization.py      # Mapping and plotting
│   └── utils.py                  # Shared utilities
├── data/
│   ├── raw/                      # Original GPS and raster data
│   ├── processed/                # Intermediate results
│   └── outputs/                  # Final corridor predictions
├── notebooks/                    # Jupyter analysis notebooks
├── tests/                        # Unit and integration tests
├── scripts/                      # Utility scripts
├── config/                       # Configuration files
└── reports/                      # Generated reports and figures
```

## Methodology

### Energy Landscape Modeling

The energy cost surface is calculated using Pontzer's locomotion equation:
```
E = α × M^β × (1 + γ × |sin(θ)|)
```
Where:
- `E` = energy cost (kcal/km)
- `M` = body mass (kg) 
- `θ` = terrain slope (degrees)
- `α`, `β`, `γ` = species-specific coefficients

### Circuit Theory Analysis

Corridors are modeled as electrical circuits where:
- Resistance = Energy cost per cell
- Current flow = Movement probability
- Voltage = Accessibility from source habitats

This approach identifies both optimal pathways and broad connectivity zones.

### Default Parameters

- **Female elephant mass**: 2,744 kg
- **Male elephant mass**: 6,029 kg
- **DEM resolution**: 30m
- **Buffer distance**: 5km
- **Maximum slope**: 30°

## Validation

The pipeline includes multiple validation approaches:

1. **Cross-validation**: Using held-out GPS data
2. **Expert validation**: Comparison with known movement patterns
3. **Landscape validation**: Overlap with protected areas and barriers
4. **Biomechanical validation**: Energy cost plausibility checks

## Dependencies

### Core Scientific Libraries
- **GeoPandas**: Vector GIS operations
- **Rasterio**: Raster data I/O
- **NumPy/Pandas**: Numerical computing
- **SciPy**: Scientific algorithms
- **Scikit-image**: Image processing and pathfinding

### Specialized Packages
- **Circuitscape**: Circuit theory modeling
- **R/enerscape**: Energy landscape calculations (via rpy2)
- **Folium**: Interactive mapping
- **Matplotlib**: Static visualizations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/
isort src/

# Lint code
flake8 src/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{elephant_corridors_2025,
  title={Elephant Corridor Prediction using Energy Landscapes and Circuit Theory},
  author={Atencia, Guillaume},
  year={2025},
  url={https://github.com/yourusername/elephant-corridors},
  version={0.1.0}
}
```

And the underlying methodology:
```bibtex
@article{berti2025energy,
  title={Energy landscapes direct the movement preferences of elephants},
  author={Berti, Emilio and others},
  journal={Journal of Animal Ecology},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Berti et al. (2025)** for the energy landscapes methodology
- **Pontzer (2016)** for the biomechanical locomotion models
- **Circuitscape developers** for circuit theory implementation
- **GDAL/OGR community** for geospatial processing tools

## Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/elephant-corridors/issues)
- 📖 Documentation: [Read the Docs](https://elephant-corridors.readthedocs.io/)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/elephant-corridors/discussions)

## Roadmap

- [ ] **v0.2.0**: Multi-species support
- [ ] **v0.3.0**: Real-time corridor updates
- [ ] **v0.4.0**: Web-based dashboard
- [ ] **v1.0.0**: Production-ready release

---

**Status**: 🚧 Active Development | **Version**: 0.1.0 | **Python**: 3.9+