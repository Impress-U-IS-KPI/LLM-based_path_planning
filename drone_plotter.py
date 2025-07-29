import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import argparse
from pathlib import Path

class DroneElevationPlotter:
    def __init__(self):
        self.elevation_data = None
        self.route_data = None
        
    def load_elevation_data(self, elevation_file):
        """
        Load elevation data from JSON file.
        Expected formats:
        1. Grid format: {"elevations": [[z11, z12, ...], [z21, z22, ...]], "x_coords": [...], "y_coords": [...]}
        2. Point format: {"points": [{"x": x1, "y": y1, "elevation": z1}, ...]}
        3. Simple grid: [[z11, z12, ...], [z21, z22, ...]]
        4. Lat/Lon format: [{"longitude": lon, "latitude": lat, "elevation_m": elev}, ...]
        """
        with open(elevation_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                # Simple 2D array format
                self.elevation_data = {
                    'grid': np.array(data),
                    'format': 'grid'
                }
            elif 'longitude' in data[0] and 'latitude' in data[0]:
                # Longitude/Latitude format
                points = [(p['longitude'], p['latitude'], p['elevation_m']) for p in data]
                self.elevation_data = {
                    'points': points,
                    'format': 'lat_lon_points',
                    'coordinate_system': 'geographic'
                }
            elif 'x' in data[0] and 'y' in data[0]:
                # List of waypoint objects with x, y coordinates
                points = [(p['x'], p['y'], p.get('elevation', p.get('elevation_m', 0))) for p in data]
                self.elevation_data = {
                    'points': points,
                    'format': 'points'
                }
        elif 'elevations' in data:
            # Grid format with coordinates
            self.elevation_data = {
                'grid': np.array(data['elevations']),
                'x_coords': np.array(data.get('x_coords', [])),
                'y_coords': np.array(data.get('y_coords', [])),
                'format': 'grid_with_coords'
            }
        elif 'points' in data:
            # Point format
            points = data['points']
            self.elevation_data = {
                'points': [(p['x'], p['y'], p['elevation']) for p in points],
                'format': 'points'
            }
        else:
            raise ValueError("Unsupported elevation data format")
    
    def load_route_data(self, route_file):
        """
        Load drone route data from JSON file.
        Expected formats:
        1. {"routes": [{"name": "route1", "waypoints": [{"longitude": x, "latitude": y, "altitude_msl": z, "time": t, "action": a}, ...]}, ...]}
        2. {"waypoints": [{"x": x1, "y": y1}, ...]}
        3. [{"x": x1, "y": y1}, ...]
        4. [[x1, y1], [x2, y2], ...]
        5. [{"longitude": lon, "latitude": lat}, ...]
        """
        with open(route_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                # Simple coordinate pairs
                self.route_data = [{'name': 'Route 1', 'waypoints': [{'x': p[0], 'y': p[1]} for p in data]}]
            elif 'longitude' in data[0] and 'latitude' in data[0]:
                # Longitude/Latitude format
                self.route_data = [{'name': 'Route 1', 'waypoints': [{'x': p['longitude'], 'y': p['latitude']} for p in data]}]
            elif 'x' in data[0]:
                # List of waypoint objects
                self.route_data = [{'name': 'Route 1', 'waypoints': data}]
        elif 'routes' in data:
            # Multiple routes format - handle mission route format with detailed waypoints
            processed_routes = []
            for route in data['routes']:
                waypoints = route['waypoints']
                if 'longitude' in waypoints[0]:
                    # Convert drone mission format to x/y for plotting
                    converted_waypoints = []
                    for wp in waypoints:
                        converted_wp = {
                            'x': wp['longitude'], 
                            'y': wp['latitude'],
                            'altitude': wp.get('altitude_msl', 0),
                            'time': wp.get('time', ''),
                            'action': wp.get('action', '')
                        }
                        converted_waypoints.append(converted_wp)
                    
                    route_info = {
                        'name': route.get('name', 'Route'),
                        'drone_id': route.get('drone_id', ''),
                        'mission': route.get('mission', ''),
                        'takeoff_time': route.get('takeoff_time', ''),
                        'landing_time': route.get('landing_time', ''),
                        'waypoints': converted_waypoints
                    }
                    processed_routes.append(route_info)
                else:
                    processed_routes.append(route)
            self.route_data = processed_routes
        elif 'waypoints' in data:
            # Single route format
            waypoints = data['waypoints']
            if 'longitude' in waypoints[0]:
                converted_waypoints = [{'x': wp['longitude'], 'y': wp['latitude']} for wp in waypoints]
                self.route_data = [{'name': 'Route 1', 'waypoints': converted_waypoints}]
            else:
                self.route_data = [{'name': 'Route 1', 'waypoints': waypoints}]
        else:
            raise ValueError("Unsupported route data format")
        with open(route_file, 'r') as f:
            full_data = json.load(f)
        self.map_objects = full_data.get('map_objects', [])    
    
    def prepare_elevation_grid(self, resolution=100):
        """Convert elevation data to a regular grid for heatmap display"""
        if self.elevation_data['format'] == 'grid':
            # Already a grid, use as-is
            return self.elevation_data['grid'], None, None
        
        elif self.elevation_data['format'] == 'grid_with_coords':
            # Grid with coordinate arrays
            grid = self.elevation_data['grid']
            x_coords = self.elevation_data.get('x_coords')
            y_coords = self.elevation_data.get('y_coords')
            return grid, x_coords, y_coords
        
        elif self.elevation_data['format'] in ['points', 'lat_lon_points']:
            # Interpolate points to regular grid
            points = np.array(self.elevation_data['points'])
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Create regular grid
            xi = np.linspace(x.min(), x.max(), resolution)
            yi = np.linspace(y.min(), y.max(), resolution)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate elevations
            zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method='cubic', fill_value=z.mean())
            
            return zi_grid, xi, yi
    
    def plot_elevation_and_routes(self, output_file=None, figsize=(12, 10), dpi=300):
        """Create the main visualization"""
        if not self.elevation_data or not self.route_data:
            raise ValueError("Both elevation and route data must be loaded first")
        
        # Prepare elevation grid
        elevation_grid, x_coords, y_coords = self.prepare_elevation_grid()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Create heatmap
        if x_coords is not None and y_coords is not None:
            extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
        else:
            extent = [0, elevation_grid.shape[1], 0, elevation_grid.shape[0]]
        
        # Create custom colormap for elevation
        colors = ['#2E8B57', '#90EE90', '#FFFF00', '#FFA500', '#8B4513', '#FFFFFF']  # Green to white
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('elevation', colors, N=n_bins)
        
        # Plot elevation heatmap
        im = ax.imshow(elevation_grid, extent=extent, origin='lower', 
                      cmap=cmap, alpha=0.7, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Elevation (m)', rotation=270, labelpad=20)
        
        # Plot drone routes
        route_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, route in enumerate(self.route_data):
            waypoints = route['waypoints']
            route_name = route.get('name', f'Route {i+1}')
            drone_id = route.get('drone_id', '')
            mission = route.get('mission', '')
            
            # Create full route label with additional info if available
            label = route_name
            if drone_id:
                label += f" ({drone_id})"
            
            # Extract coordinates
            x_coords_route = [wp['x'] for wp in waypoints]
            y_coords_route = [wp['y'] for wp in waypoints]
            
            # Plot route line
            color = route_colors[i % len(route_colors)]
            ax.plot(x_coords_route, y_coords_route, color=color, linewidth=2.5, 
                   label=label, marker='o', markersize=5, alpha=0.9, markeredgewidth=0.5,
                   markeredgecolor='white')
            
            # Mark start and end points
            ax.plot(x_coords_route[0], y_coords_route[0], color=color, 
                   marker='s', markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label='_nolegend_')
            ax.plot(x_coords_route[-1], y_coords_route[-1], color=color, 
                   marker='^', markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label='_nolegend_')
            
            # Add annotations for key waypoints (targets, takeoff/landing)
            for j, wp in enumerate(waypoints):
                action = wp.get('action', '').lower()
                if any(keyword in action for keyword in ['target', 'takeoff', 'landing', 'observation']):
                    # Only annotate important waypoints to avoid clutter
                    if j < len(x_coords_route):
                        ax.annotate(wp.get('action', ''), 
                                   (x_coords_route[j], y_coords_route[j]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8, color=color,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                           alpha=0.7, edgecolor=color))
        if hasattr(self, 'map_objects') and self.map_objects:
            for obj in self.map_objects:
                x, y = obj['longitude'], obj['latitude']
                obj_type = obj['type']
                name = obj['name']
                
                # Different colors and markers for different object types
                type_style = {
                    'infrastructure': {'color': 'red', 'marker': 's', 'size': 120},
                    'factory': {'color': 'orange', 'marker': 's', 'size': 100}, 
                    'ew_zone': {'color': 'yellow', 'marker': '^', 'size': 150},
                    'urban_area': {'color': 'purple', 'marker': 'H', 'size': 200},
                    'military': {'color': 'darkred', 'marker': 'D', 'size': 120}
                }
                
                style = type_style.get(obj_type, {'color': 'black', 'marker': 'o', 'size': 80})
                
                # Plot the object
                ax.scatter(x, y, c=style['color'], s=style['size'], 
                          marker=style['marker'], alpha=0.8, 
                          edgecolor='black', linewidth=1, zorder=10)
                
                # Add label
                ax.annotate(name, (x, y), 
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=7, alpha=0.9, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=style['color'], 
                                   alpha=0.7, edgecolor='black'))
        # Plot EW zones as circles if present in route data
        if hasattr(self, 'map_objects') and self.map_objects:
            for obj in self.map_objects:
                if obj['type'] == 'ew_zone' and 'radius_km' in obj:
                    # Convert radius from km to coordinate degrees (approximate)
                    # 1 degree ≈ 111 km at this latitude
                    radius_deg = obj['radius_km'] / 30#111.0 
                    
                    # Create circle
                    circle = plt.Circle((obj['longitude'], obj['latitude']), 
                                    radius_deg, 
                                    color='red', 
                                    fill=False, 
                                    linewidth=2, 
                                    alpha=0.8,
                                    linestyle='--')
                    ax.add_patch(circle)
                    
                    # Add danger zone label
                    # ax.annotate(f"EW DANGER\n{obj['radius_km']}km radius", 
                            # (obj['longitude'], obj['latitude']), 
                            # xytext=(0, 0), textcoords='offset points',
                            # fontsize=9, alpha=0.9, weight='bold', color='red',
                            # ha='center', va='center',
                            # bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                            #         alpha=0.7, edgecolor='red'))

        # Customize plot
        if self.elevation_data.get('coordinate_system') == 'geographic':
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_title('Drone Routes on Elevation Heatmap (Geographic Coordinates)', fontsize=16, fontweight='bold')
        else:
            ax.set_xlabel('X Coordinate (m)')
            ax.set_ylabel('Y Coordinate (m)')
            ax.set_title('Drone Routes on Elevation Heatmap', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3)
        
        # Add text annotations
        ax.text(0.02, 0.98, '■ Takeoff Point', transform=ax.transAxes, va='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.94, '▲ Landing Point', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.90, '● Waypoints', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add mission info if available
        if hasattr(self, 'route_data') and len(self.route_data) > 0:
            mission_info = []
            for route in self.route_data:
                if route.get('takeoff_time') and route.get('landing_time'):
                    mission_info.append(f"{route.get('drone_id', 'Drone')}: {route.get('takeoff_time')} - {route.get('landing_time')}")
            
            if mission_info:
                info_text = "Mission Timeline:\n" + "\n".join(mission_info)
                ax.text(0.98, 0.02, info_text, transform=ax.transAxes, va='bottom', ha='right',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=9)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
        else:
            plt.show()
        
        return fig, ax
    
    def generate_sample_data(self, elevation_file='sample_elevation.json', route_file='sample_routes.json'):
        """Generate sample data files for testing"""
        
        # Sample elevation data in your longitude/latitude format
        # Creating a realistic grid around Kyiv area coordinates
        elevations = []
        lon_start, lat_start = 33.501042, 50.999306
        
        # Create a grid of elevation points
        for i in range(10):  # longitude points
            for j in range(8):   # latitude points
                lon = lon_start + i * 0.02
                lat = lat_start + j * 0.015
                # Generate realistic elevation variation
                elev = 77 + 40 * np.sin(i * 0.8) * np.cos(j * 0.6) + 20 * np.random.random()
                elevations.append({
                    "longitude": round(lon, 6),
                    "latitude": round(lat, 6), 
                    "elevation_m": round(elev, 1)
                })
        
        with open(elevation_file, 'w') as f:
            json.dump(elevations, f, indent=2)
        
        # Sample route data matching the coordinate system
        route_data = {
            "routes": [
                {
                    "name": "Survey Route North",
                    "waypoints": [
                        {"longitude": 33.51, "latitude": 51.00},
                        {"longitude": 33.53, "latitude": 51.02},
                        {"longitude": 33.55, "latitude": 51.04},
                        {"longitude": 33.57, "latitude": 51.06},
                        {"longitude": 33.59, "latitude": 51.08}
                    ]
                },
                {
                    "name": "Survey Route South",
                    "waypoints": [
                        {"longitude": 33.52, "latitude": 51.09},
                        {"longitude": 33.54, "latitude": 51.07},
                        {"longitude": 33.56, "latitude": 51.05},
                        {"longitude": 33.58, "latitude": 51.03},
                        {"longitude": 33.60, "latitude": 51.01}
                    ]
                },
                {
                    "name": "Cross Route",
                    "waypoints": [
                        {"longitude": 33.505, "latitude": 51.02},
                        {"longitude": 33.525, "latitude": 51.03},
                        {"longitude": 33.545, "latitude": 51.04},
                        {"longitude": 33.565, "latitude": 51.05},
                        {"longitude": 33.585, "latitude": 51.06}
                    ]
                }
            ]
        }
        
        with open(route_file, 'w') as f:
            json.dump(route_data, f, indent=2)
        
        print(f"Sample data generated in longitude/latitude format:")
        print(f"  Elevation data: {elevation_file}")
        print(f"  Route data: {route_file}")
        print(f"  Coordinate range: {lon_start:.3f}-{lon_start + 0.18:.3f} longitude, {lat_start:.3f}-{lat_start + 0.105:.3f} latitude")

def main():
    parser = argparse.ArgumentParser(description='Plot drone routes over elevation heatmap')
    parser.add_argument('--elevation', '-e', type=str, help='Elevation JSON file')
    parser.add_argument('--routes', '-r', type=str, help='Routes JSON file') 
    parser.add_argument('--output', '-o', type=str, help='Output image file')
    parser.add_argument('--generate-sample', '-g', action='store_true', 
                       help='Generate sample data files')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 10],
                       help='Figure size (width height)')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    
    args = parser.parse_args()
    
    plotter = DroneElevationPlotter()
    
    if args.generate_sample:
        plotter.generate_sample_data()
        return
    
    if not args.elevation or not args.routes:
        print("Error: Both --elevation and --routes files are required")
        print("Use --generate-sample to create sample data files")
        return
    
    try:
        # Load data
        print(f"Loading elevation data from: {args.elevation}")
        plotter.load_elevation_data(args.elevation)
        
        print(f"Loading route data from: {args.routes}")
        plotter.load_route_data(args.routes)
        
        # Create plot
        print("Generating plot...")
        plotter.plot_elevation_and_routes(
            output_file=args.output,
            figsize=tuple(args.figsize),
            dpi=args.dpi
        )
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()