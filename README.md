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
3. Use the `process_data.py` script to generate standardization parameters and standardize the data for training.

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


## License

This project is licensed under the MIT License. See the LICENSE file for details.
