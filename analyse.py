import json
import numpy as np
import math

def distance_km(lon1, lat1, lon2, lat2):
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def analyze_low_corridor_with_ew_avoidance(elevation_file):
    """
    Find the lowest elevation corridor from western start (33.6, 51.4) 
    to eastern target (34.65, 51.45) while avoiding EW Station Alpha
    """
    with open(elevation_file, 'r') as f:
        data = json.load(f)
    
    print("=== LOW CORRIDOR PATH ANALYSIS WITH EW AVOIDANCE ===\n")
    
    # Extract elevation points
    points = []
    for point in data:
        if 'elevation_m' in point:
            points.append({
                'lon': point['longitude'],
                'lat': point['latitude'], 
                'elev': point['elevation_m']
            })
    
    # Mission parameters
    start_point = (33.6, 51.4)
    target_point = (34.65, 51.45)
    ew_alpha = (34.300, 51.250)  # EW Station Alpha position
    ew_alpha_radius = 3.0  # 3km danger zone
    
    print(f"Start point: {start_point}")
    print(f"Target point: {target_point}")
    print(f"EW Alpha position: {ew_alpha} (danger radius: {ew_alpha_radius}km)\n")
    
    # Create corridor segments for analysis
    corridor_segments = [
        {"name": "Western Start Zone", "lon_range": (33.5, 33.8), "lat_range": (51.3, 51.5)},
        {"name": "Western Valley", "lon_range": (33.8, 34.1), "lat_range": (51.2, 51.4)},
        {"name": "Central Corridor", "lon_range": (34.1, 34.4), "lat_range": (51.0, 51.3)},
        {"name": "Eastern Approach", "lon_range": (34.4, 34.7), "lat_range": (51.2, 51.5)},
    ]
    
    optimal_waypoints = []
    
    for segment in corridor_segments:
        print(f"=== {segment['name']} ===")
        
        # Find all points in this corridor segment
        segment_points = []
        for point in points:
            if (segment["lon_range"][0] <= point['lon'] <= segment["lon_range"][1] and 
                segment["lat_range"][0] <= point['lat'] <= segment["lat_range"][1]):
                
                # Check if point is safe from EW Alpha
                dist_to_ew = distance_km(point['lon'], point['lat'], ew_alpha[0], ew_alpha[1])
                
                if dist_to_ew > ew_alpha_radius:  # Safe from EW station
                    segment_points.append({
                        'lon': point['lon'],
                        'lat': point['lat'],
                        'elev': point['elev'],
                        'ew_distance': dist_to_ew
                    })
        
        if segment_points:
            # Sort by elevation (lowest first)
            segment_points.sort(key=lambda x: x['elev'])
            
            print(f"  Safe points found: {len(segment_points)}")
            print(f"  Elevation range: {segment_points[0]['elev']:.1f}m - {segment_points[-1]['elev']:.1f}m")
            
            # Find the 5 lowest safe points
            lowest_points = segment_points[:5]
            
            print("  Top 5 lowest safe points:")
            for i, point in enumerate(lowest_points, 1):
                print(f"    {i}. Lon: {point['lon']:.3f}, Lat: {point['lat']:.3f}")
                print(f"       Elevation: {point['elev']:.1f}m, EW Distance: {point['ew_distance']:.2f}km")
            
            # Select the lowest point as optimal waypoint
            optimal = lowest_points[0]
            optimal_waypoints.append({
                'longitude': optimal['lon'],
                'latitude': optimal['lat'],
                'elevation': optimal['elev'],
                'ew_distance': optimal['ew_distance'],
                'segment': segment['name']
            })
            
        else:
            print(f"  WARNING: No safe points found in {segment['name']}!")
        
        print()
    
    # Generate optimized route waypoints
    print("=== OPTIMIZED ROUTE WAYPOINTS ===")
    print("(Safe from EW Alpha, following lowest terrain)")
    print()
    
    route_waypoints = [
        # Start
        {
            'longitude': start_point[0],
            'latitude': start_point[1], 
            'action': 'Takeoff - Western base',
            'note': 'Starting position'
        }
    ]
    
    # Add optimal corridor waypoints
    for waypoint in optimal_waypoints:
        altitude_agl = 50  # 50m above ground level
        altitude_msl = waypoint['elevation'] + altitude_agl
        
        route_waypoints.append({
            'longitude': waypoint['longitude'],
            'latitude': waypoint['latitude'],
            'altitude_msl': altitude_msl,
            'action': f"Low corridor - {waypoint['segment']} (EW safe: {waypoint['ew_distance']:.1f}km)",
            'note': f"Terrain: {waypoint['elevation']:.1f}m + 50m AGL"
        })
    
    # Add target
    route_waypoints.append({
        'longitude': target_point[0],
        'latitude': target_point[1],
        'action': 'Target - Factory reconnaissance',
        'note': 'Mission objective'
    })
    
    # Print optimized waypoints
    for i, wp in enumerate(route_waypoints, 1):
        print(f"Waypoint {i}:")
        print(f"  Lon: {wp['longitude']:.3f}, Lat: {wp['latitude']:.3f}")
        if 'altitude_msl' in wp:
            print(f"  Altitude: {wp['altitude_msl']:.0f}m MSL")
        print(f"  Action: {wp['action']}")
        print(f"  Note: {wp['note']}")
        print()
    
    return optimal_waypoints, route_waypoints

# Usage
if __name__ == "__main__":
    elevation_file = "sumy_elevations.json"
    optimal_waypoints, route_waypoints = analyze_low_corridor_with_ew_avoidance(elevation_file)
    
    # Save results to JSON for easy import
    with open('optimized_waypoints.json', 'w') as f:
        json.dump(route_waypoints, f, indent=2)
    
    print("Results saved to 'optimized_waypoints.json'")