#!/usr/bin/env python3
"""
Quick environment test to see what's working
"""

import sys
print(f"Python: {sys.version}")

# Test core libraries
try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError:
    print("❌ NumPy: MISSING")

try:
    import pandas as pd
    print("✅ Pandas:", pd.__version__)
except ImportError:
    print("❌ Pandas: MISSING")

try:
    import geopandas as gpd
    print("✅ GeoPandas:", gpd.__version__)
except ImportError as e:
    print("❌ GeoPandas: MISSING -", e)

try:
    import rasterio
    print("✅ Rasterio:", rasterio.__version__)
except ImportError:
    print("❌ Rasterio: MISSING")

try:
    import matplotlib
    print("✅ Matplotlib:", matplotlib.__version__)
except ImportError:
    print("❌ Matplotlib: MISSING")

try:
    import folium
    print("✅ Folium:", folium.__version__)
except ImportError:
    print("❌ Folium: MISSING")

try:
    from skimage import __version__ as skimage_version
    print("✅ Scikit-image:", skimage_version)
except ImportError:
    print("❌ Scikit-image: MISSING")

# Test R integration
try:
    import rpy2
    print("✅ rpy2:", rpy2.__version__)
    
    # Test R connection
    from rpy2.robjects import r
    r_version = r('R.version.string')[0]
    print("✅ R connection:", r_version)
    
    # Test terra package
    try:
        from rpy2.robjects.packages import importr
        terra = importr('terra')
        print("✅ R terra package: Available")
    except:
        print("❌ R terra package: Not available")
        
    # Test enerscape package  
    try:
        enerscape = importr('enerscape')
        print("✅ R enerscape package: Available")
    except:
        print("❌ R enerscape package: Not available")
        
except ImportError:
    print("❌ rpy2: MISSING")

# Test circuitscape alternatives
try:
    import networkx as nx
    print("✅ NetworkX:", nx.__version__, "(Circuitscape alternative)")
except ImportError:
    print("⚠️  NetworkX: Not installed (install with: pip install networkx)")

print("\n" + "="*50)
print("SUMMARY:")
print("- Core geospatial stack: Check above")
print("- R integration: Check rpy2 and R packages above") 
print("- Circuitscape: We'll implement alternatives")
print("="*50)