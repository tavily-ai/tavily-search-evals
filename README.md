# **Evaluation Framework for Web Search APIs**

## **Overview**
This repository provides evaluation frameworks for benchmarking web search APIs, combining static benchmarks and dynamic datasets to measure accuracy, relevance, and retrieval performance across different providers.
### Benchmarks:
1. [SimpleQA](https://openai.com/index/introducing-simpleqa/) Benchmark
    -  Runs the full SimpleQA dataset against each provider.
    - Retrieved documents are reformatted for an LLM (we used `gpt-4.1`) to extract a predicted answer.
    - The predicted answer is graded using the official SimpleQA classifier.
    - For providers that return direct answers, it is possible to bypass the LLM step and compare the returned answer directly with the SimpleQA ground truth (though our evaluations used the classifier route).
2. Document Relevance Benchmark 
    - Uses [QuotientAI](https://docs.quotientai.co/data-collection/logs) to assess the relevance of retrieved documents against a given query.
    - Involves generating a dynamic dataset using the open-source [Dynamic Eval Datasets Generator](https://github.com/Eyalbenba/tavily-web-eval-generator).
    - You can use the provided dataset (`datasets/document_relevance_dynamic_test_set.json`) or easily create new datasets on topics of your choice with the above generator.
    - This flexibility allows evaluation on domain-specific or real-time topics, making the benchmark more reflective of production-like tasks than static datasets.

### **Features**
- Comparative evaluation of multiple search providers
- Out-of-the-box support for Tavily, Exa, Brave, Google (SERP via Serper), Perplexity Search, Perplexity, GPTR, and You.com
- Easy integration of additional providers (see [this section](#adding-a-new-search-provider-to-the-evaluation))
- Customizable configuration for each provider
- Parallelized, independent evaluation pipelines
- Automatic resume from the last checkpoint in case of errors

---

## **Evaluation Results**

The table below presents evaluation results across various search providers and LLMs on the SimpleQA benchmark. 

| Provider | Accuracy |
|----------|-------|
| Tavily   | 93.3%   |
| Perplexity Search | 85.92% |
| Google (SERP) using SERPER | 82.15% |
| Brave Search | 76.05% |
| Exa Search | 71.24%   |

The table below presents evaluation results across various search providers and LLMs on the Document Relevance benchmark. 

| Provider | Accuracy |
|----------|-------|
| Tavily   | 83.02%   |
| Perplexity Search | 71.2% |
| Google (SERP) using SERPER | 58.11% |
| Brave Search | 56.2% |
| Exa Search | 51.33%   |

NOTE: The `config.json` file contains the search parameters we used to evaluate each provider above. 

---

## **Running Locally**

1. **Clone the repository**:
    ```sh
    git clone https://github.com/tavily-ai/tavily-search-evals
    cd tavily-search-evals
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:  
    Create a `.env` file in the root directory and add the following:
    ```env
    TAVILY_API_KEY=XXX
    OPENAI_API_KEY=XXX
    EXA_API_KEY=XXX
    PERPLEXITY_API_KEY=XXX
    SERPER_API_KEY=XXX
    BRAVE_API_KEY=XXX
    YDC_API_KEY=XXX
    ```

4. **Run**:
```sh
python run_evaluation.py
```

### **Command Line Options**

- `--evaluation_type`: Type of evaluation to run (simpleqa or document_relevance, default: simpleqa)
- `--config`: Path to JSON config file with provider parameters (default: configs/config.json)
- `--start_index`: Starting index for examples (inclusive, default: 0)
- `--end_index`: Ending index for examples (exclusive, default: all examples)
- `--random_sample`: Number of random samples to select (overrides start/end index)
- `--post_process_model`: Model for post-processing for SimpleQA (default: gpt-4.1)
- `--output_dir`: Directory to save results (default: results)
- `--sequential`: Run providers sequentially instead of in parallel
- `--rerun`: Continue evaluation on existing results directory, output_dir must exist
- `--token_model`: Model for token consumption calculation (default: gpt-4.1)
- `--evaluator_model`: Model for correctness evaluation for SimpleQA (default: gpt-4.1)

### **Output**

Evaluation results are saved in the `results/` directory with the following structure:

```
results/
└── {evaluation_type}/                      # Evaluation type folder (simpleqa or document_relevance)
    └── YYYY-MM-DD_HH-MM-SS/               
        ├── summary.csv                     # Overall evaluation summary
        ├── config.json                     # Configuration used for this evaluation
        ├── {provider}_{evaluation_type}_results.csv   # Individual provider results
        └── ...                             # Additional provider result files
```

#### **Example Output:**
```bash
results/
├── simpleqa/
│   └── 2025-01-15_14-30-25/
│       ├── summary.csv
│       ├── config.json
│       ├── tavily_simpleqa_results.csv
│       ├── exa_simpleqa_results.csv
│       ├── serper_simpleqa_results.csv
│       ├── brave_simpleqa_results.csv
│       └── perplexity_search_simpleqa_results.csv
└── document_relevance/
    └── 2025-01-15_15-45-12/
        ├── summary.csv
        ├── config.json
        ├── tavily_document_relevance_results.csv
        ├── exa_document_relevance_results.csv
        └── ...
```

### **Config Example**

Configuration file `config.json` might look like:
```json
{
  "tavily": {
    "search_depth": "advanced",
    "include_raw_content": false,
    "max_results": 10,
  },
  "perplexity_search": {
    "max_results": 10,
    "max_tokens_per_page": 512
  }
}
```
### **Resume Evaluation**

If your evaluation is interrupted, you can continue from where it stopped using the `--rerun` flag (`output_dir` folder must exist with the previous run's partial results):

```sh
python run_evaluation.py --output_dir results/my_evaluation --rerun
```

This will:
1. Load existing results from the specified output directory
2. Skip questions that have already been evaluated
3. Continue with the remaining questions in the dataset
4. Update the summary statistics with all results when complete

---

## **Adding a New Search Provider to the Evaluation**
### Supported Search Providers
The current supported search providers are:
- `tavily`
- `perplexity`
- `perplexity_search`
- `gptr`
- `exa`
- `serper`
- `brave`
- `youcom`


You can extend the system to evaluate additional search providers by following these steps:

1. Create a new handler file in the `handlers` directory (e.g., `handlers/new_provider_handler.py`).

2. Add your provider to the handler registry:
- Update `handlers/__init__.py` to import and expose your new handler.
- Update the `get_search_handlers` function in `app.py` and `run_benchmark.py` to include your new provider.

3. Update environment variables, add your provider's API key to the `.env` file:
```
NEW_PROVIDER_API_KEY=your_api_key_here
```

4. Use your provider in evaluation config:
```json
{
  "new_provider": {
    "custom_param1": "value1",
    "custom_param2": "value2"
  }
}
```

Remember to implement appropriate error handling and respect any rate limits or API constraints for your new provider.

---

## **License**

This project is made available under the [MIT License](https://github.com/tavily-ai/tavily-mcp/blob/main/LICENCE).
