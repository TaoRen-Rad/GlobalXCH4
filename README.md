# Global XCH<sub style="font-size: 0.7em;">4</sub> Retrieval

## Overview
The Global XCH<sub style="font-size: 0.7em;">4</sub> Retrieval is a two-stage probabilistic multi-layer perceptron (MLP) model for fast and accurate retrievals of column-averaged methane dry air mole fraction (XCH<sub style="font-size: 0.7em;">4</sub>) from TROPOMI and GOSAT. This multi-satellite transfer learning framework enables efficient global XCH<sub style="font-size: 0.7em;">4</sub> products with high coverage, improved accuracy, and low computational cost, supporting near-real-time methane monitoring.

## Key Features
- **Fast and Accurate Retrievals**: Reduces computational time from minutes to milliseconds per retrieval.
- **Probabilistic Output**: Provides both mean predictions and uncertainty estimates for each retrieval.
- **Snapshot Ensemble**: Uses multiple model snapshots for robust and stable predictions.
- **Efficient Fine-Tuning**: Maintains accuracy with fine-tuning on GOSAT data using a small fraction of newly available data.
- **Multi-satellite Support**: Works with both TROPOMI and GOSAT satellite data.

## Data Preparation
Users need to prepare the data following the steps below:
1. Download TROPOMI Level 1B and Level 2 product data from the [Copernicus Open Access Hub](https://dataspace.copernicus.eu/).
2. Use the provided `extract_data.py` script to extract and combine data from L1B and L2 files into Parquet-formatted Pandas DataFrames.
3. Run `python run/process_data.py` to split 2019–2023 observations by overpass and fit standardization parameters on the training subset. Training, validation, and independent 2024–2025 test files are saved under `run/`.

Fine-tuning uses raw TROPOMI–GOSAT matched files with both satellites' coordinates and the original `gosat_xch4` in ppb. Pairs within 10 km Haversine distance are retained, split by overpass, and standardized using the baseline scalers. Model inputs follow the feature order saved in the baseline checkpoint.

## Structure
- **`xch4_retrieval` Package**: Core modules for model definition, training, fine-tuning, evaluation, and data processing.
- **Data Extraction Script** (`run/extract_data.py`): Extract and process TROPOMI L1B/L2 data.
- **Data Processing Script** (`run/process_data.py`): Generate standardization parameters and standardize data.
- **Training Script** (`run/baseline_model.py`): Model training and evaluation.
- **Fine-tuning Script** (`run/finetune_model.py`): Fine-tuning on GOSAT data.
- **Product Generation Script** (`run/generate_product_example.py`): Generate XCH<sub style="font-size: 0.7em;">4</sub> product files.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/GlobalXCH4.git
   cd GlobalXCH4
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the data as described above.
4. Train the model:
   ```bash
   python run/baseline_model.py
   ```
5. Fine-tune the model on GOSAT data:
   ```bash
   python run/finetune_model.py
   ```
6. Generate products from new data:
   ```bash
   python run/generate_product_example.py
   ```


## Data Availability

The fine-tuned global XCH4 product for 2024–2025 generated in this study is available at [https://doi.org/10.5281/zenodo.21618794](https://doi.org/10.5281/zenodo.21618794).

## Cite as

If you use this work in your research, please cite as follows,

```bibtex
@article{Luo2027GlobalXCH4,
  title = {Global {XCH4} retrievals from {TROPOMI} shortwave-infrared radiances through cross-satellite transfer learning},
  author = {Luo, Yuwen and Chen, Wei and Ren, Tao and Zhao, Changying and Mao, Huiqin and Zhou, Minqiang and Wang, Pucai},
  year = {2027},
  journal = {Journal of Quantitative Spectroscopy and Radiative Transfer},
  volume = {366},
  pages = {110158},
  issn = {0022-4073},
  doi = {10.1016/j.jqsrt.2026.110158}
}
```

Y. Luo, W. Chen, T. Ren\*, C. Zhao\*, H. Mao, M. Zhou and P. Wang. Global XCH4 retrievals from TROPOMI shortwave-infrared radiances through cross-satellite transfer learning. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 366: 110158, 2027. doi: [10.1016/j.jqsrt.2026.110158](https://doi.org/10.1016/j.jqsrt.2026.110158).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
