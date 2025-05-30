#!/usr/bin/env python3
"""
Comprehensive OpenTopography API Debugging Script
Diagnoses all aspects of API requests to identify authentication and parameter issues.
"""

import os
import sys
import requests
import json
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_opentopography.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OpenTopographyDebugger:
    """Comprehensive OpenTopography API debugger."""
    
    def __init__(self):
        """Initialize debugger."""
        self.api_key = os.getenv("OPENTOPOGRAPHY_API_KEY", "").strip()
        self.endpoints = {
            'current_public': 'https://portal.opentopography.org/API/globaldem',
            'old_swift': 'https://cloud.sdsc.edu/v1/AUTH_opentopography/API/globaldem',
            'alternative': 'https://portal.opentopography.org/API/global'
        }
        
        # Test coordinates (Central African Republic - should have SRTM data)
        self.test_bounds = {
            'name': 'Central_Africa_Test',
            'south': 4.0,    # Latitude
            'north': 4.5,    # Latitude 
            'west': 18.0,    # Longitude
            'east': 18.5     # Longitude
        }
        
        # Alternative test bounds (smaller area)
        self.small_test_bounds = {
            'name': 'Small_Test_Area',
            'south': 4.2,
            'north': 4.3,
            'west': 18.2,
            'east': 18.3
        }
        
        print("🔍 OpenTopography API Comprehensive Debugger")
        print("=" * 70)
        
    def debug_environment(self):
        """Debug environment variables and setup."""
        print("\n📋 ENVIRONMENT DEBUGGING")
        print("-" * 50)
        
        # Check API key
        logger.info(f"API Key from environment: {'SET' if self.api_key else 'NOT SET'}")
        if self.api_key:
            logger.info(f"API Key length: {len(self.api_key)} characters")
            logger.info(f"API Key starts with: {self.api_key[:8]}...")
            logger.info(f"API Key ends with: ...{self.api_key[-4:]}")
            logger.info(f"API Key contains only valid chars: {self.api_key.isalnum()}")
        else:
            logger.error("❌ OPENTOPOGRAPHY_API_KEY environment variable is not set!")
            logger.info("Set it with: export OPENTOPOGRAPHY_API_KEY=your_key_here")
            return False
        
        # Check environment details
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        
        # Check requests library
        logger.info(f"Requests library version: {requests.__version__}")
        
        return True
    
    def debug_coordinates(self, bounds):
        """Debug coordinate validity."""
        print(f"\n🌍 COORDINATE DEBUGGING - {bounds['name']}")
        print("-" * 50)
        
        # Check coordinate ranges
        south, north = bounds['south'], bounds['north']
        west, east = bounds['west'], bounds['east']
        
        logger.info(f"Bounds: South={south}, North={north}, West={west}, East={east}")
        
        # Validate latitude range
        if not (-90 <= south <= north <= 90):
            logger.error(f"❌ Invalid latitude range: {south} to {north}")
            return False
        else:
            logger.info("✅ Latitude range valid")
        
        # Validate longitude range  
        if not (-180 <= west <= east <= 180):
            logger.error(f"❌ Invalid longitude range: {west} to {east}")
            return False
        else:
            logger.info("✅ Longitude range valid")
        
        # Check area size
        width_deg = east - west
        height_deg = north - south
        area_deg_sq = width_deg * height_deg
        
        logger.info(f"Area dimensions: {width_deg:.3f}° × {height_deg:.3f}°")
        logger.info(f"Area size: {area_deg_sq:.6f} square degrees")
        
        # Estimate area in km²
        avg_lat = (south + north) / 2
        lat_to_km = 111.32  # km per degree latitude
        lon_to_km = 111.32 * abs(cos(radians(avg_lat)))  # Adjust for latitude
        
        area_km_sq = (width_deg * lon_to_km) * (height_deg * lat_to_km)
        logger.info(f"Estimated area: {area_km_sq:.1f} km²")
        
        # Check if area is too large (OpenTopography has limits)
        if area_deg_sq > 25:  # 5° × 5° is often the limit
            logger.warning(f"⚠️ Large area ({area_deg_sq:.2f} deg²) - may exceed API limits")
        elif area_deg_sq > 1:
            logger.info(f"📏 Medium area ({area_deg_sq:.2f} deg²) - should be acceptable")
        else:
            logger.info(f"📏 Small area ({area_deg_sq:.6f} deg²) - definitely acceptable")
        
        # Check if coordinates are in Africa (expected data region)
        if -20 <= west <= 55 and -35 <= south <= 35:
            logger.info("✅ Coordinates are in Africa (good SRTM coverage expected)")
        else:
            logger.warning("⚠️ Coordinates outside Africa - may have limited SRTM coverage")
        
        return True
    
    def debug_request_building(self, endpoint, bounds, dem_type='SRTMGL1'):
        """Debug request parameter building."""
        print(f"\n🔧 REQUEST BUILDING DEBUG - {endpoint}")
        print("-" * 50)
        
        # Build parameters
        params = {
            'demtype': dem_type,
            'south': str(bounds['south']),
            'north': str(bounds['north']),
            'west': str(bounds['west']),
            'east': str(bounds['east']),
            'outputFormat': 'GTiff'
        }
        
        # Test different API key parameter names
        api_key_variants = [
            ('api_key', self.api_key),           # Current standard
            ('API_Key', self.api_key),           # Old format
            ('apikey', self.api_key),            # No underscore
            ('key', self.api_key),               # Simple
        ]
        
        logger.info(f"Base parameters: {params}")
        
        results = []
        
        for key_param, key_value in api_key_variants:
            test_params = params.copy()
            test_params[key_param] = key_value
            
            # Build full URL
            query_string = urlencode(test_params)
            full_url = f"{endpoint}?{query_string}"
            
            logger.info(f"\nTesting API key parameter: '{key_param}'")
            logger.info(f"Full URL: {full_url}")
            logger.info(f"URL length: {len(full_url)} characters")
            
            # Parse URL to verify
            parsed = urlparse(full_url)
            parsed_params = parse_qs(parsed.query)
            logger.info(f"Parsed parameters: {parsed_params}")
            
            results.append({
                'key_param': key_param,
                'params': test_params,
                'url': full_url,
                'url_length': len(full_url)
            })
        
        return results
    
    def debug_network_request(self, url, timeout=30):
        """Debug network request with detailed logging."""
        print(f"\n🌐 NETWORK REQUEST DEBUG")
        print("-" * 50)
        
        logger.info(f"Making request to: {url}")
        logger.info(f"Timeout: {timeout} seconds")
        
        # Track timing
        start_time = time.time()
        
        try:
            # Make request with detailed debugging
            response = requests.get(
                url,
                timeout=timeout,
                stream=False,  # Don't stream for debugging
                allow_redirects=True,
                verify=True,  # Verify SSL
                headers={
                    'User-Agent': 'EnergyScape-Debugger/1.0 (Python/requests)',
                    'Accept': 'image/tiff, application/octet-stream, */*',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'close'
                }
            )
            
            request_time = time.time() - start_time
            
            # Log response details
            logger.info(f"✅ Request completed in {request_time:.2f} seconds")
            logger.info(f"Status code: {response.status_code}")
            logger.info(f"Reason: {response.reason}")
            
            # Log response headers
            logger.info("Response headers:")
            for header, value in response.headers.items():
                logger.info(f"  {header}: {value}")
            
            # Log response size
            content_length = len(response.content)
            logger.info(f"Response size: {content_length} bytes ({content_length/1024:.1f} KB)")
            
            # Analyze content type
            content_type = response.headers.get('content-type', 'unknown')
            logger.info(f"Content type: {content_type}")
            
            # Check if response looks like a TIFF
            if response.content.startswith(b'II*\x00') or response.content.startswith(b'MM\x00*'):
                logger.info("✅ Response appears to be a valid TIFF file")
                return True, response, "TIFF file received"
            elif response.content.startswith(b'<?xml'):
                logger.warning("⚠️ Response is XML (likely an error message)")
                xml_content = response.content.decode('utf-8', errors='ignore')
                logger.info(f"XML content: {xml_content}")
                return False, response, xml_content
            elif response.content.startswith(b'{'):
                logger.warning("⚠️ Response is JSON (likely an error message)")
                try:
                    json_content = json.loads(response.content.decode('utf-8'))
                    logger.info(f"JSON content: {json_content}")
                    return False, response, json_content
                except:
                    logger.error("Could not parse JSON response")
                    return False, response, "Invalid JSON"
            elif content_length < 1000:
                logger.warning("⚠️ Response is very small (likely an error)")
                text_content = response.content.decode('utf-8', errors='ignore')
                logger.info(f"Small response content: {text_content}")
                return False, response, text_content
            else:
                logger.info("📄 Response is larger content of unknown type")
                # Show first 200 bytes
                preview = response.content[:200]
                logger.info(f"Content preview: {preview}")
                return False, response, "Unknown large content"
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ Request timed out after {timeout} seconds")
            return False, None, "Timeout"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Connection error: {e}")
            return False, None, f"Connection error: {e}"
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request exception: {e}")
            return False, None, f"Request error: {e}"
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False, None, f"Unexpected error: {e}"
    
    def test_endpoint_combinations(self):
        """Test all combinations of endpoints, coordinates, and parameters."""
        print(f"\n🧪 COMPREHENSIVE ENDPOINT TESTING")
        print("=" * 70)
        
        if not self.debug_environment():
            return False
        
        # Test coordinates
        test_areas = [self.small_test_bounds, self.test_bounds]
        dem_types = ['SRTMGL1', 'SRTMGL3', 'SRTM_FF']
        
        results = []
        
        for area in test_areas:
            if not self.debug_coordinates(area):
                continue
                
            for endpoint_name, endpoint_url in self.endpoints.items():
                logger.info(f"\n🎯 Testing endpoint: {endpoint_name}")
                logger.info(f"URL: {endpoint_url}")
                
                # Build requests for this endpoint
                request_variants = self.debug_request_building(endpoint_url, area)
                
                for dem_type in dem_types:
                    logger.info(f"\n🏔️ Testing DEM type: {dem_type}")
                    
                    for variant in request_variants:
                        # Update DEM type
                        test_params = variant['params'].copy()
                        test_params['demtype'] = dem_type
                        
                        # Build URL
                        query_string = urlencode(test_params)
                        test_url = f"{endpoint_url}?{query_string}"
                        
                        logger.info(f"\n📡 Testing: {variant['key_param']} parameter with {dem_type}")
                        
                        # Make request
                        success, response, message = self.debug_network_request(test_url)
                        
                        # Store result
                        result = {
                            'area': area['name'],
                            'endpoint': endpoint_name,
                            'dem_type': dem_type,
                            'api_key_param': variant['key_param'],
                            'success': success,
                            'status_code': response.status_code if response else None,
                            'message': message,
                            'url': test_url
                        }
                        results.append(result)
                        
                        # Log result
                        if success:
                            logger.info("✅ SUCCESS!")
                        else:
                            logger.error(f"❌ FAILED: {message}")
                        
                        # Brief pause between requests
                        time.sleep(1)
        
        return results
    
    def analyze_results(self, results):
        """Analyze test results and provide recommendations."""
        print(f"\n📊 RESULTS ANALYSIS")
        print("=" * 70)
        
        # Count successes and failures
        total_tests = len(results)
        successes = [r for r in results if r['success']]
        failures = [r for r in results if not r['success']]
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successes: {len(successes)}")
        logger.info(f"Failures: {len(failures)}")
        logger.info(f"Success rate: {len(successes)/total_tests*100:.1f}%")
        
        if successes:
            print(f"\n✅ SUCCESSFUL CONFIGURATIONS:")
            for success in successes:
                logger.info(f"  ✓ {success['endpoint']} + {success['dem_type']} + {success['api_key_param']}")
                logger.info(f"    Area: {success['area']}")
                logger.info(f"    URL: {success['url']}")
        
        if failures:
            print(f"\n❌ FAILED CONFIGURATIONS:")
            
            # Group failures by error type
            error_groups = {}
            for failure in failures:
                error_key = f"{failure.get('status_code', 'N/A')} - {failure['message'][:50]}"
                if error_key not in error_groups:
                    error_groups[error_key] = []
                error_groups[error_key].append(failure)
            
            for error_type, error_list in error_groups.items():
                logger.error(f"  ❌ {error_type} ({len(error_list)} occurrences)")
                for error in error_list[:2]:  # Show first 2 examples
                    logger.error(f"     - {error['endpoint']} + {error['dem_type']} + {error['api_key_param']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if successes:
            # Find best working configuration
            best = successes[0]
            logger.info(f"✅ Use this working configuration:")
            logger.info(f"   Endpoint: {best['endpoint']}")
            logger.info(f"   DEM type: {best['dem_type']}")
            logger.info(f"   API key parameter: {best['api_key_param']}")
        else:
            logger.error("❌ No working configurations found!")
            logger.info("🔍 Check these potential issues:")
            logger.info("   1. API key may be invalid or expired")
            logger.info("   2. Account may not have proper permissions")
            logger.info("   3. Service may be temporarily unavailable")
            logger.info("   4. Network connectivity issues")
            
        return successes
    
    def save_debug_report(self, results, filename='opentopography_debug_report.json'):
        """Save detailed debug report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': {
                'api_key_set': bool(self.api_key),
                'api_key_length': len(self.api_key) if self.api_key else 0,
                'python_version': sys.version,
                'requests_version': requests.__version__,
                'working_directory': os.getcwd()
            },
            'test_configuration': {
                'endpoints': self.endpoints,
                'test_bounds': [self.small_test_bounds, self.test_bounds]
            },
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Debug report saved to: {filename}")

def run_comprehensive_debug():
    """Run comprehensive OpenTopography debugging."""
    debugger = OpenTopographyDebugger()
    
    # Run all tests
    results = debugger.test_endpoint_combinations()
    
    # Analyze results
    successes = debugger.analyze_results(results)
    
    # Save report
    debugger.save_debug_report(results)
    
    return len(successes) > 0

if __name__ == "__main__":
    # Import math functions for coordinate validation
    from math import cos, radians
    
    print("Starting comprehensive OpenTopography API debugging...")
    
    success = run_comprehensive_debug()
    
    if success:
        print("\n🎉 Found working API configuration!")
        print("Check the debug log and report for details.")
    else:
        print("\n❌ No working API configuration found.")
        print("Review the debug output to identify the issue.")
    
    print(f"\nDebug log saved to: debug_opentopography.log")
    print(f"JSON report saved to: opentopography_debug_report.json")