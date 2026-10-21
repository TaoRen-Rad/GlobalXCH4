from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xch4_retrieval.extract_data import process_l1b_l2_pair


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'
    l2_file = data_dir / 'l2' / 'S5P_OFFL_L2__CH4____20240101T074458_20240101T092629_32219_03_020600_20240102T234559.nc'
    l1b_file = data_dir / 'l1b' / 'S5P_OFFL_L1B_RA_BD7_20240101T074458_20240101T092629_32219_03_020100_20240101T111020.nc'
    output_file = script_dir / '20240101T074458_20240101T092629.parquet'
    land_mask = data_dir / 'global_land_mask.npy'
    qa_threshold = 75.0
    
    count = process_l1b_l2_pair(
        l2_file_path=l2_file,
        l1b_file_path=l1b_file,
        land_mask_path=land_mask,
        output_file=output_file,
        qa_threshold=qa_threshold
    )
    print(f"Processing completed. {count} records saved.")


if __name__ == "__main__":
    main()
