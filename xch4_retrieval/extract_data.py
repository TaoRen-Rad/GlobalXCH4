import h5py
import numpy as np
import pandas as pd
import os
from typing import Dict, List


def get_valid_points(l2_file_path: str, land_mask: np.ndarray, qa_threshold: float = 75.0) -> Dict:
    """Extract and filter valid points from L2 file."""
    invalid_value = 9.969209968386869e+36
    tolerance_value = 1e30
    
    with h5py.File(l2_file_path, 'r') as f:
        latitude = f["PRODUCT"]["latitude"][0, :, :]
        longitude = f["PRODUCT"]["longitude"][0, :, :]
        methane_mixing_ratio = f["PRODUCT"]["methane_mixing_ratio"][0, :, :]
        methane_profile_apriori = f["PRODUCT"]["SUPPORT_DATA"]["INPUT_DATA"]["methane_profile_apriori"][0, :, :, :]
        surface_pressure = f["PRODUCT"]["SUPPORT_DATA"]["INPUT_DATA"]["surface_pressure"][0, :, :]
        qa_value = f["PRODUCT"]["qa_value"][0, :, :]
        
        valid_methane = ~np.isclose(methane_mixing_ratio, invalid_value, rtol=0, atol=tolerance_value)
        valid_quality = (qa_value >= qa_threshold)
        
        valid_land = np.zeros_like(valid_methane, dtype=bool)
        lat_flat = latitude.flatten()
        lon_flat = longitude.flatten()
        valid_range = (lat_flat >= -60) & (lat_flat <= 90) & (lon_flat >= -180) & (lon_flat <= 180)
        lat_idx = ((lat_flat + 60) / 0.5).astype(int)
        lon_idx = ((lon_flat + 180) / 0.5).astype(int)
        valid_idx = (lat_idx >= 0) & (lat_idx < land_mask.shape[0]) & \
                   (lon_idx >= 0) & (lon_idx < land_mask.shape[1]) & valid_range
        valid_land_flat = np.zeros_like(lat_flat, dtype=bool)
        valid_land_flat[valid_idx] = land_mask[lat_idx[valid_idx], lon_idx[valid_idx]]
        valid_land = valid_land_flat.reshape(latitude.shape)
        
        valid_indices = np.where(valid_methane & valid_quality & valid_land)
        
        return {
            "latitude": latitude[valid_indices].tolist(),
            "longitude": longitude[valid_indices].tolist(),
            "xch4": methane_mixing_ratio[valid_indices].tolist(),
            "scanline": valid_indices[0].tolist(),
            "pixel": valid_indices[1].tolist(),
            "methane_profile_apriori": methane_profile_apriori[valid_indices[0], valid_indices[1], :].tolist(),
            "surface_pressure": surface_pressure[valid_indices].tolist(),
            "count": len(valid_indices[0])
        }


def extract_l1b_params(l1b_file_path: str, scanlines: np.ndarray, pixels: np.ndarray) -> List[Dict]:
    """Extract parameters from L1B file."""
    with h5py.File(l1b_file_path, 'r') as f:
        earth_sun_distance = float(f['BAND7_RADIANCE']['STANDARD_MODE']['GEODATA']['earth_sun_distance'][0])
        solar_zenith_angle_full = f['BAND7_RADIANCE']['STANDARD_MODE']['GEODATA']['solar_zenith_angle'][0, :, :]
        solar_azimuth_angle_full = f['BAND7_RADIANCE']['STANDARD_MODE']['GEODATA']['solar_azimuth_angle'][0, :, :]
        viewing_zenith_angle_full = f['BAND7_RADIANCE']['STANDARD_MODE']['GEODATA']['viewing_zenith_angle'][0, :, :]
        viewing_azimuth_angle_full = f['BAND7_RADIANCE']['STANDARD_MODE']['GEODATA']['viewing_azimuth_angle'][0, :, :]
        radiance_full = f['BAND7_RADIANCE']['STANDARD_MODE']['OBSERVATIONS']['radiance'][0, :, :, :]
        
        l1b_data = []
        for scanline, pixel in zip(scanlines, pixels):
            l1b_data.append({
                "earth_sun_distance": earth_sun_distance,
                "solar_zenith_angle": float(solar_zenith_angle_full[scanline, pixel]),
                "solar_azimuth_angle": float(solar_azimuth_angle_full[scanline, pixel]),
                "viewing_zenith_angle": float(viewing_zenith_angle_full[scanline, pixel]),
                "viewing_azimuth_angle": float(viewing_azimuth_angle_full[scanline, pixel]),
                "radiance": radiance_full[scanline, pixel, :].tolist()
            })
        return l1b_data


def process_l1b_l2_pair(
    l2_file_path: str,
    l1b_file_path: str,
    land_mask_path: str,
    output_file: str,
    qa_threshold: float = 75.0
) -> int:
    """Process L1B-L2 file pair and save to parquet."""
    land_mask = np.load(land_mask_path)
    result = get_valid_points(l2_file_path, land_mask, qa_threshold)
    
    if result['count'] == 0:
        return 0
    
    l2_basename = os.path.basename(l2_file_path)
    time_part = l2_basename[20:51]
    
    df = pd.DataFrame({
        "time_sequence": [time_part] * result['count'],
        "longitude": result['longitude'],
        "latitude": result['latitude'],
        "xch4": result['xch4'],
        "scanline": result['scanline'],
        "pixel": result['pixel'],
        "methane_profile_apriori": result['methane_profile_apriori'],
        "surface_pressure": result['surface_pressure']
    })
    
    l1b_data = extract_l1b_params(l1b_file_path, df['scanline'].values, df['pixel'].values)
    l1b_df = pd.DataFrame(l1b_data)
    df_combined = pd.concat([df, l1b_df], axis=1)
    
    df_combined.to_parquet(output_file, index=False)
    return len(df_combined)
