# RAGAS Evaluation Script

This Python script evaluates the performance of a language model using the RAGAS library. It takes an input Excel file containing questions, answers, ground truth, and context information, and calculates various metrics such as faithfulness, answer relevancy, context recall, and context precision.

## Prerequisites

Before running the script, make sure you have the following:

- Python 3.x installed
- Required Python packages installed:
  - pandas
  - dotenv
  - datasets
  - langchain_openai
  - ragas
- Azure OpenAI API credentials (API key and endpoint URL)
- Input Excel file with the required columns: "question", "answer", "ground_truth", "contexts", and "Buch"

## Setup

1. Install the required Python packages by running the following command:

   ```
   pip install pandas dotenv datasets langchain_openai ragas
   ```

2. Set up your Azure OpenAI API credentials:
   - Create a `.env` file in the same directory as the script.
   - Add the following lines to the `.env` file, replacing `<your_api_key>` and `<your_endpoint_url>` with your actual credentials:

     ```
     AZURE_OPENAI_API_KEY=<your_api_key>
     AZURE_OPENAI_BASE_URL=<your_endpoint_url>
     ```

## Usage

1. Prepare your input Excel file with the required columns: "question", "answer", "ground_truth", "contexts", and "Buch".

2. Run the script from the command line, providing the input file name as an argument:

   ```
   python script_name.py input_file.xlsx
   ```

   Replace `script_name.py` with the actual name of your Python script, and `input_file.xlsx` with the name of your input Excel file.

3. The script will evaluate the performance of the language model using the specified metrics and save the results in a new Excel file named `<input_file_name>_results_<timestamp>_eval_gpt4.xlsx`.

## Configuration

The script uses the following Azure OpenAI configurations:

- `openai_api_version`: The version of the OpenAI API to use (default: "2023-12-01-preview").
- `model_deployment`: The deployment name for the language model (default: "ragas").
- `model_name`: The name of the language model (default: "gpt-4").
- `embedding_deployment`: The deployment name for the embedding model (default: "embedding").
- `embedding_name`: The name of the embedding model (default: "text-embedding-ada-002").

You can modify these configurations in the `azure_configs` dictionary in the script if needed.

## Metrics

The script evaluates the following metrics using the RAGAS library:

- Faithfulness
- Answer Relevancy
- Context Recall
- Context Precision

These metrics assess different aspects of the language model's performance in generating relevant and accurate answers based on the provided context.

## Output

The script generates an output Excel file named `<input_file_name>_results_<timestamp>_eval_gpt4.xlsx`, where `<input_file_name>` is the name of the input file (without the extension) and `<timestamp>` is the current timestamp. The output file contains the original input data along with the calculated metric values for each question-answer pair.

## License

This script is provided under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute it as needed.
