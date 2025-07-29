#!/usr/bin/env python3
"""
Drone Route Planning Data Preprocessor
Processes elevation data and prepares it for LLM-based route planning tasks
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import math

@dataclass
class Coordinate:
    longitude: float
    latitude: float
    elevation_m: float

@dataclass
class TerrainGrid:
    """Structured terrain representation for route planning"""
    lons: np.ndarray
    lats: np.ndarray
    elevations: np.ndarray
    grid_spacing_deg: float
    bounds: Dict[str, float]

class DroneTerrainProcessor:
    def __init__(self, elevation_file: str):
        """Initialize with elevation data file"""
        self.raw_data = self._load_elevation_data(elevation_file)
        self.terrain_grid = self._create_terrain_grid()
        
    def _load_elevation_data(self, filename: str) -> List[Coordinate]:
        """Load and parse elevation JSON data"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        for point in data:
            coord = Coordinate(
                longitude=point['longitude'],
                latitude=point['latitude'],
                elevation_m=point['elevation_m']
            )
            coordinates.append(coord)
        
        print(f"Loaded {len(coordinates)} elevation points")
        return coordinates
    
    def _create_terrain_grid(self) -> TerrainGrid:
        """Convert raw coordinates to structured grid"""
        # Extract coordinates
        lons = np.array([c.longitude for c in self.raw_data])
        lats = np.array([c.latitude for c in self.raw_data])
        elevs = np.array([c.elevation_m for c in self.raw_data])
        
        # Get unique sorted coordinates
        unique_lons = np.unique(lons)
        unique_lats = np.unique(lats)
        
        # Calculate grid spacing
        lon_spacing = np.mean(np.diff(unique_lons))
        lat_spacing = np.mean(np.diff(unique_lats))
        avg_spacing = (lon_spacing + lat_spacing) / 2
        
        # Create 2D elevation grid
        elevation_grid = np.zeros((len(unique_lats), len(unique_lons)))
        
        for coord in self.raw_data:
            lat_idx = np.argmin(np.abs(unique_lats - coord.latitude))
            lon_idx = np.argmin(np.abs(unique_lons - coord.longitude))
            elevation_grid[lat_idx, lon_idx] = coord.elevation_m
        
        bounds = {
            'min_lon': float(np.min(unique_lons)),
            'max_lon': float(np.max(unique_lons)),
            'min_lat': float(np.min(unique_lats)),
            'max_lat': float(np.max(unique_lats)),
            'min_elev': float(np.min(elevs)),
            'max_elev': float(np.max(elevs))
        }
        
        return TerrainGrid(
            lons=unique_lons,
            lats=unique_lats,
            elevations=elevation_grid,
            grid_spacing_deg=avg_spacing,
            bounds=bounds
        )
    
    def calculate_terrain_features(self) -> Dict:
        """Calculate terrain features relevant for drone operations"""
        
        # Calculate gradients (slopes)
        grad_lat, grad_lon = np.gradient(self.terrain_grid.elevations)
        
        # Convert to actual slopes (degrees)
        lat_spacing_m = self.terrain_grid.grid_spacing_deg * 111000  # rough meters per degree
        lon_spacing_m = lat_spacing_m * np.cos(np.radians(np.mean(self.terrain_grid.lats)))
        
        slope_lat = np.degrees(np.arctan(grad_lat / lat_spacing_m))
        slope_lon = np.degrees(np.arctan(grad_lon / lon_spacing_m))
        slope_magnitude = np.sqrt(slope_lat**2 + slope_lon**2)
        
        # Terrain roughness (standard deviation of elevation in local area)
        roughness = gaussian_filter(self.terrain_grid.elevations, sigma=2)
        roughness = np.abs(self.terrain_grid.elevations - roughness)
        
        # Ridge and valley detection
        ridges = self._detect_ridges()
        valleys = self._detect_valleys()
        
        return {
            'slope_magnitude': slope_magnitude,
            'slope_direction': np.degrees(np.arctan2(slope_lat, slope_lon)),
            'roughness': roughness,
            'ridges': ridges,
            'valleys': valleys,
            'elevation_stats': {
                'mean': float(np.mean(self.terrain_grid.elevations)),
                'std': float(np.std(self.terrain_grid.elevations)),
                'range': float(np.ptp(self.terrain_grid.elevations))
            }
        }
    
    def _detect_ridges(self) -> np.ndarray:
        """Detect ridge lines (local elevation maxima)"""
        from scipy.ndimage import maximum_filter
        
        # Local maxima detection
        local_maxima = maximum_filter(self.terrain_grid.elevations, size=5)
        ridges = (self.terrain_grid.elevations == local_maxima) & \
                 (self.terrain_grid.elevations > np.percentile(self.terrain_grid.elevations, 75))
        
        return ridges.astype(int)
    
    def _detect_valleys(self) -> np.ndarray:
        """Detect valley lines (local elevation minima)"""
        from scipy.ndimage import minimum_filter
        
        # Local minima detection
        local_minima = minimum_filter(self.terrain_grid.elevations, size=5)
        valleys = (self.terrain_grid.elevations == local_minima) & \
                  (self.terrain_grid.elevations < np.percentile(self.terrain_grid.elevations, 25))
        
        return valleys.astype(int)
    
    def calculate_line_of_sight(self, start_coord: Tuple[float, float], 
                              end_coord: Tuple[float, float], 
                              drone_height: float = 50) -> Dict:
        """Calculate line-of-sight between two points considering terrain"""
        
        # Convert coordinates to grid indices
        start_lat_idx = np.argmin(np.abs(self.terrain_grid.lats - start_coord[1]))
        start_lon_idx = np.argmin(np.abs(self.terrain_grid.lons - start_coord[0]))
        end_lat_idx = np.argmin(np.abs(self.terrain_grid.lats - end_coord[1]))
        end_lon_idx = np.argmin(np.abs(self.terrain_grid.lons - end_coord[0]))
        
        # Create path between points
        num_points = max(abs(end_lat_idx - start_lat_idx), abs(end_lon_idx - start_lon_idx)) + 1
        lat_indices = np.linspace(start_lat_idx, end_lat_idx, num_points, dtype=int)
        lon_indices = np.linspace(start_lon_idx, end_lon_idx, num_points, dtype=int)
        
        # Get elevations along path
        path_elevations = self.terrain_grid.elevations[lat_indices, lon_indices]
        
        # Calculate required altitude for clear line of sight
        start_elev = self.terrain_grid.elevations[start_lat_idx, start_lon_idx]
        end_elev = self.terrain_grid.elevations[end_lat_idx, end_lon_idx]
        
        # Linear interpolation of direct line
        direct_line_elevations = np.linspace(start_elev + drone_height, 
                                           end_elev + drone_height, 
                                           num_points)
        
        # Check for obstructions
        obstructed = np.any(path_elevations > direct_line_elevations - drone_height)
        
        # Minimum safe altitude
        min_safe_altitude = np.max(path_elevations) - min(start_elev, end_elev) + 20  # 20m safety margin
        
        return {
            'clear_line_of_sight': not obstructed,
            'path_elevations': path_elevations.tolist(),
            'min_safe_altitude_m': float(min_safe_altitude),
            'max_terrain_elevation_m': float(np.max(path_elevations)),
            'terrain_clearance_required_m': float(np.max(path_elevations) + 20)
        }
    
    def generate_safe_corridors(self, max_slope: float = 15, 
                              min_corridor_width: float = 0.01) -> List[Dict]:
        """Identify safe flight corridors based on terrain constraints"""
        
        features = self.calculate_terrain_features()
        
        # Find areas suitable for flight (low slope, low roughness)
        safe_mask = (features['slope_magnitude'] < max_slope) & \
                   (features['roughness'] < np.percentile(features['roughness'], 75))
        
        # Convert mask to corridor coordinates
        corridors = []
        safe_indices = np.where(safe_mask)
        
        if len(safe_indices[0]) > 0:
            # Group nearby safe areas
            for i in range(0, len(safe_indices[0]), 50):  # Sample every 50 points
                if i + 10 < len(safe_indices[0]):  # Ensure we have enough points
                    lat_idx = safe_indices[0][i:i+10]
                    lon_idx = safe_indices[1][i:i+10]
                    
                    corridor = {
                        'center_lat': float(np.mean(self.terrain_grid.lats[lat_idx])),
                        'center_lon': float(np.mean(self.terrain_grid.lons[lon_idx])),
                        'avg_elevation': float(np.mean(self.terrain_grid.elevations[lat_idx, lon_idx])),
                        'avg_slope': float(np.mean(features['slope_magnitude'][lat_idx, lon_idx])),
                        'safety_score': float(1.0 - np.mean(features['slope_magnitude'][lat_idx, lon_idx]) / max_slope)
                    }
                    corridors.append(corridor)
        
        return corridors[:20]  # Return top 20 corridors
    
    def prepare_prompt_data(self, mission_type: str = "reconnaissance") -> Dict:
        """Prepare structured data for LLM prompts"""
        
        features = self.calculate_terrain_features()
        safe_corridors = self.generate_safe_corridors()
        
        # Key terrain statistics
        terrain_summary = {
            'area_bounds': self.terrain_grid.bounds,
            'grid_resolution_deg': float(self.terrain_grid.grid_spacing_deg),
            'grid_resolution_m': float(self.terrain_grid.grid_spacing_deg * 111000),
            'elevation_stats': features['elevation_stats'],
            'terrain_complexity': {
                'avg_slope_deg': float(np.mean(features['slope_magnitude'])),
                'max_slope_deg': float(np.max(features['slope_magnitude'])),
                'roughness_index': float(np.mean(features['roughness'])),
                'ridge_density': float(np.sum(features['ridges']) / features['ridges'].size),
                'valley_density': float(np.sum(features['valleys']) / features['valleys'].size)
            }
        }
        
        # Flight recommendations
        flight_params = {
            'recommended_min_altitude_m': float(features['elevation_stats']['mean'] + 50),
            'recommended_max_altitude_m': float(self.terrain_grid.bounds['max_elev'] + 200),
            'safe_corridor_count': len(safe_corridors),
            'terrain_classification': self._classify_terrain(features),
            'weather_considerations': {
                'wind_exposure_risk': 'HIGH' if np.mean(features['slope_magnitude']) > 10 else 'MEDIUM',
                'turbulence_risk': 'HIGH' if features['elevation_stats']['range'] > 150 else 'MEDIUM'
            }
        }
        
        return {
            'mission_type': mission_type,
            'terrain_summary': terrain_summary,
            'flight_parameters': flight_params,
            'safe_corridors': safe_corridors[:10],  # Top 10 for prompt
            'tactical_considerations': self._get_tactical_considerations(features)
        }
    
    def _classify_terrain(self, features: Dict) -> str:
        """Classify terrain type for mission planning"""
        avg_slope = np.mean(features['slope_magnitude'])
        elevation_range = features['elevation_stats']['range']
        
        if avg_slope < 5 and elevation_range < 50:
            return "FLAT_OPEN"
        elif avg_slope < 10 and elevation_range < 100:
            return "ROLLING_HILLS"
        elif avg_slope < 15 and elevation_range < 150:
            return "MODERATE_TERRAIN"
        else:
            return "COMPLEX_TERRAIN"
    
    def _get_tactical_considerations(self, features: Dict) -> Dict:
        """Generate tactical considerations for mission planning"""
        
        return {
            'concealment_opportunities': {
                'ridge_masking': 'AVAILABLE' if np.sum(features['ridges']) > 100 else 'LIMITED',
                'valley_routes': 'AVAILABLE' if np.sum(features['valleys']) > 100 else 'LIMITED',
                'terrain_following': 'RECOMMENDED' if np.mean(features['slope_magnitude']) > 8 else 'OPTIONAL'
            },
            'navigation_challenges': {
                'gps_reliability': 'HIGH' if np.mean(features['slope_magnitude']) < 10 else 'MEDIUM',
                'visual_navigation': 'GOOD' if features['elevation_stats']['range'] > 100 else 'LIMITED',
                'landmark_availability': 'HIGH' if np.sum(features['ridges']) > 50 else 'MEDIUM'
            },
            'operational_risks': {
                'terrain_collision': 'LOW' if np.mean(features['slope_magnitude']) < 5 else 'MEDIUM',
                'forced_landing_sites': 'AVAILABLE' if np.sum(features['slope_magnitude'] < 5) > 1000 else 'LIMITED',
                'weather_exposure': 'HIGH' if self.terrain_grid.bounds['max_elev'] > 200 else 'MEDIUM'
            }
        }
    
    def export_for_gis(self, output_file: str):
        """Export data in GeoJSON format for GIS analysis"""
        
        features = []
        
        # Export elevation points
        for coord in self.raw_data:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [coord.longitude, coord.latitude]
                },
                "properties": {
                    "elevation_m": coord.elevation_m
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} points to {output_file}")

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DroneTerrainProcessor('sumy_elevations.json')
    
    # Generate prompt-ready data
    prompt_data = processor.prepare_prompt_data("multi_target_reconnaissance")
    
    # Save processed data
    with open('drone_mission_data.json', 'w') as f:
        json.dump(prompt_data, f, indent=2)
    
    print("Drone mission data prepared successfully!")
    print(f"Terrain Classification: {prompt_data['flight_parameters']['terrain_classification']}")
    print(f"Safe Corridors Found: {prompt_data['flight_parameters']['safe_corridor_count']}")
    
    # Example: Check line of sight between two points
    start_point = (34.0, 51.5)  # lon, lat
    end_point = (34.5, 52.0)
    
    los_result = processor.calculate_line_of_sight(start_point, end_point, drone_height=60)
    print(f"\nLine of sight analysis:")
    print(f"Clear path: {los_result['clear_line_of_sight']}")
    print(f"Min safe altitude: {los_result['min_safe_altitude_m']:.1f}m")
    
    # Export for GIS
    processor.export_for_gis('sumy_elevation_points.geojson')