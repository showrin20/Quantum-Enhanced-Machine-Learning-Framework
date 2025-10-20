# Real-Time Water Quality Dataset for Aquaculture

## Dataset Overview

This directory contains a real-time water quality dataset comprising **40,280 records** collected from **5 ponds** in Bangladesh for aquaculture research.

## Dataset File

- **`realfishdataset.csv`** - Main dataset with fish species and environmental parameters

## Dataset Specifications

### Size & Scope
- **Total Records**: 40,280
- **Data Sources**: 5 fish ponds in Bangladesh
- **Collection Method**: Real-time water quality monitoring
- **Purpose**: Predicting fish species suitability for aquaculture environments

### Data Columns

#### Environmental Parameters (Features):
1. **`ph`**: Water pH level
   - Measures acidity/alkalinity of water
   - Critical for fish health and survival
   - Range: Typically 6.0-9.0 for aquaculture

2. **`temperature`**: Water temperature in °C
   - Affects fish metabolism and oxygen levels
   - Influences fish growth rates
   - Critical for species-specific suitability

3. **`turbidity`**: Water clarity measurement (NTU)
   - Impacts light penetration
   - Affects photosynthesis and feeding behavior
   - Indicator of water quality

#### Target Variable:
4. **`fish`**: Fish species name (categorical)
   - 11 different species
   - Common aquaculture species in Bangladesh

## Fish Species Included

The dataset includes 11 fish species commonly cultivated in Bangladesh:

1. **Sing** (Stinging Catfish) - *Heteropneustes fossilis*
2. **Silver Carp** - *Hypophthalmichthys molitrix*
3. **Katla** (Major Carp) - *Catla catla*
4. **Prawn** - Freshwater prawn species
5. **Shrimp** - Aquaculture shrimp species
6. **Rui** (Rohu) - *Labeo rohita*
7. **Pangas** (Pangasius) - *Pangasianodon hypophthalmus*
8. **Tilapia** - *Oreochromis niloticus*
9. **Koi** (Climbing Perch) - *Anabas testudineus*
10. Additional species (check dataset for complete list)

## Data Format

### CSV Structure
```csv
ph,temperature,turbidity,fish
7.2,28.5,12.3,Rui
6.8,26.1,8.7,Tilapia
7.5,29.0,10.5,Katla
8.1,27.5,15.2,Silver Carp
```

### Data Quality
- ✓ Quality-controlled measurements
- ✓ Balanced representation across environmental conditions
- ✓ Real-time monitoring data
- ✓ No missing values (preprocessed)

## Research Context

This dataset supports research on:
- **Aquaculture Optimization**: Determining optimal environmental conditions for fish species
- **Sustainable Farming**: Improving fish farming productivity in Bangladesh
- **Environmental Management**: Water quality monitoring and management
- **Predictive Analytics**: Machine learning for aquaculture decision support
- **Quantum ML Applications**: Testing quantum-enhanced prediction frameworks

## Data Usage

### Loading the Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/realfishdataset.csv')

# Display basic information
print(f"Total records: {len(df)}")
print(f"Fish species: {df['fish'].nunique()}")
print(f"\nDataset preview:")
print(df.head())
```

### Environmental Parameter Ranges

Check the distribution of environmental parameters:

```python
print(df[['ph', 'temperature', 'turbidity']].describe())
```

## Important Notes

- **Confidentiality**: This dataset is for research purposes
- **Citation Required**: Please cite the research paper when using this data
- **Data Privacy**: Collected from real aquaculture ponds in Bangladesh
- **Version Control**: Data files are not tracked in git for size management

## Data Collection Methodology

The data represents:
- Real-world aquaculture conditions
- Multiple pond environments
- Temporal variation in water quality
- Species-specific environmental preferences

## Applications

This dataset enables:
1. Fish species suitability prediction
2. Water quality optimization
3. Aquaculture management decisions
4. Environmental monitoring systems
5. Quantum-enhanced ML research

## Contact

For questions about the dataset:
- Refer to the main project README
- Contact the research team
- Check the research paper for methodology details
