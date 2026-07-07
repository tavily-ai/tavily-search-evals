from datetime import datetime, timedelta, timezone
import logging
import os
import json
import asyncio
import argparse
import time
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from evaluators.correctness_evaluator import CorrectnessConfig


from handlers import TavilyHandler, ExaHandler, GPTRHandler, PerplexityHandler, SerperHandler, BraveHandler, PerplexitySearchHandler, YoucomHandler
from evaluators import CorrectnessEvaluator
from utils import PostProcessor, save_summary, load_csv_data, load_document_relevance_eval_data, prepare_examples, get_output_dir, save_result, get_quotient_ai_client, EvaluationType, copy_config_to_results

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


TAVILY_DEFAULT_CONFIG = {
    "search_depth": "advanced",
    "max_results": 10,
}

def get_dataset_path(evaluation_type: EvaluationType) -> str:
    """Get the dataset path based on evaluation type."""
    if evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
        return "datasets/document_relevance_dynamic_test_set.json"
    elif evaluation_type == EvaluationType.SIMPLEQA:
        return "datasets/simple_qa_test_set.csv"


def load_data(evaluation_type: EvaluationType, start_index: int = 0, end_index: Optional[int] = None, random_sample: Optional[int] = None):
    """Load data based on evaluation type."""
    dataset_path = get_dataset_path(evaluation_type)
    
    if evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
        return load_document_relevance_eval_data(dataset_path, start_index, end_index, random_sample)
    elif evaluation_type == EvaluationType.SIMPLEQA:
        return load_csv_data(dataset_path, start_index, end_index, random_sample)


async def get_search_handlers(search_provider_params: Dict[str, Dict[str, Any]], token_model: str = "gpt-4.1"):
    """Initialize search handlers based on provided parameters."""
    handler_map = {
        "tavily": TavilyHandler,
        "exa": ExaHandler,
        "gptr": GPTRHandler,
        "perplexity": PerplexityHandler,
        "perplexity_search": PerplexitySearchHandler,
        "serper": SerperHandler,
        "brave": BraveHandler,
        "youcom": YoucomHandler,
    }
    
    return [
        handler_class(params, token_model=token_model)
        for provider_name, params in search_provider_params.items()
        if (handler_class := handler_map.get(provider_name.lower()))
    ]


async def evaluate_provider_simple_qa(
    provider_name: str,
    search_handler,
    examples: List[Dict],
    post_processor: Optional[PostProcessor] = None,
    evaluator_model: str = "gpt-4.1",
    batch_size: int = 3,
):
    """Evaluate a single search provider on the dataset."""
    evaluator = CorrectnessEvaluator(CorrectnessConfig(model_name=evaluator_model))
    
    results = []
    correct_count = 0
    
    async def process_example(example):
        nonlocal correct_count
        
        query = example["question"]
        reference_answer = example["answer"]
        index = example["index"]
        
        try:
            search_result = await search_handler.search(query)
            
            original_answer = search_result.get("answer", "")
            
            is_llm_response = search_handler.is_llm_response
            if is_llm_response:
                search_ans = original_answer
            else:
                search_ans, token_count, token_avg = await search_handler.post_process(search_result)
            
            answer = post_processor.extract_answer(
                query=query, 
                is_llm_response=is_llm_response, 
                search_result=search_ans
            )
            # Evaluate the answer
            evaluation_result = await evaluator.evaluate(
                {"question": query},
                {"answer": answer},
                {"answer": reference_answer}
            )
            
            is_correct = evaluation_result['score'] == 1.0
            if is_correct:
                correct_count += 1

            grade = evaluation_result['value']
            result = {
                "index": index,
                "question": query,
                "reference_answer": reference_answer,
                "predicted_answer": answer,
                "is_correct": is_correct,
                "grade": grade,
                "token_count": token_count if not is_llm_response else 0,
                "token_avg": token_avg if not is_llm_response else 0
            }

            results.append(result)
            logger.info(f"[{provider_name}] Q{index}: Grade - {grade}, Query: '{query}'")
            save_result(result, provider_name, output_dir, evaluation_type)

            return result
        
        except Exception as e:
            logger.error(f"[{provider_name}] Error evaluating example {index}: {str(e)}")
            results.append({
                "index": index,
                "question": query,
                "reference_answer": reference_answer,
                "predicted_answer": "ERROR",
                "is_correct": False,
                "grade": "ERROR",
                "error": str(e),
            })
            return None
    
    # Process examples in batches
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        tasks = [process_example(example) for example in batch]
        await asyncio.gather(*tasks)
        time.sleep(3.0) # avoid rate limiting
    
    accuracy = correct_count / len(examples) if examples else 0
    accuracy = round(accuracy, 3)

    return {
        "provider": provider_name,
        "results": results,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(examples)
    }

async def evaluate_provider_document_relevance(
    provider_name: str,
    search_handler,
    examples: List[Dict],
    environment: str = "test",
):
    results = []
    quotient_client, app_name = get_quotient_ai_client(provider_name, environment)

    async def process_example(example):
        query = example["question"]
        
        try:
            search_result = await search_handler.search(query)

            documents, token_count, token_avg = await search_handler.post_process(search_result, evaluation_type=EvaluationType.DOCUMENT_RELEVANCE)
            quotient_client.log(
                user_query=query,
                documents=documents,
            )

            result = {
                "index": example["index"],
                "question": query,
                "token_count": token_count,
                "token_avg": token_avg,
                "grade": "completed",
            }

            results.append(result)
            save_result(result, provider_name, output_dir, evaluation_type)

            return result
        
        except Exception as e:
            logger.error(f"[{provider_name}] Error evaluating example {example['index']}: {str(e)}")
            error_result = {
                "index": example["index"],
                "question": query,
                "token_count": token_count,
                "token_avg": token_avg,
                "grade": "ERROR",
                "error": str(e),
            }
            results.append(error_result)
            return None
    
    # Process examples
    for i, example in enumerate(examples):
        if i % 100 == 0:
            await asyncio.sleep(5) # to avoid Quotient AI rate limiting
        await process_example(example)

    return {
        "provider": provider_name,
        "results": results,
        "quotient_client": quotient_client,
        "app_name": app_name
    }
async def run_evaluation(
    evaluation_type: EvaluationType,
    search_provider_params: Dict[str, Dict[str, Any]],
    config_path: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    random_sample: Optional[int] = None,
    post_process_model: str = "gpt-4.1",
    token_model: str = "gpt-4.1",
    evaluator_model: str = "gpt-4.1",
    parallel: bool = True,
    output_dir: str = "results",
    rerun: bool = False,
):
    """Run the benchmark evaluation using specified evaluation type.
    
    Args:
        evaluation_type: Type of evaluation (EvaluationType.DOCUMENT_RELEVANCE or EvaluationType.SIMPLEQA)
        search_provider_params: Dictionary mapping search provider names to their parameters
        start_index: Starting index for examples (inclusive)
        end_index: Ending index for examples (exclusive), defaults to the end of the dataset
        random_sample: Number of random samples to select (overrides start_index and end_index)
        post_process_model: Model to use for post-processing
        parallel: Whether to run evaluations for search providers in parallel
        output_dir: Directory to save results
        rerun: Whether to rerun evaluation on existing results directory, output_dir must exist
    """
    try:
        if "exa" in search_provider_params:
            if "contents" not in search_provider_params["exa"] or search_provider_params["exa"]["contents"] != {"highlights": True}:
                raise ValueError("'contents' field with 'highlights' is required for Exa. Please add it to the configuration.")
        
        # Load and prepare data based on evaluation type
        examples = load_data(evaluation_type, start_index, end_index, random_sample)
        examples = prepare_examples(examples, list(search_provider_params.keys()), rerun, output_dir, random_sample, evaluation_type)
    
        # Initialize search handlers
        search_handlers = await get_search_handlers(search_provider_params, token_model)
        provider_names = list(search_provider_params.keys())

        if len(search_handlers) == 0:
            raise Exception("No search handlers found")

        os.makedirs(output_dir, exist_ok=True)
        copy_config_to_results(config_path, output_dir=output_dir)
        
        provider_results = {}
        post_processor = PostProcessor(llm_model=post_process_model)

        if parallel:
            # Evaluate providers in parallel
            tasks = []
            for handler, provider_name in zip(search_handlers, provider_names):
                if evaluation_type == EvaluationType.SIMPLEQA:
                    task = evaluate_provider_simple_qa(
                        provider_name,
                        handler,
                        examples[provider_name],
                        post_processor,
                        evaluator_model,
                    ) 
                elif evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
                    task = evaluate_provider_document_relevance(
                        provider_name,
                        handler,
                        examples[provider_name],
                        environment="test",
                    )
                tasks.append(task)
            
            # Wait for all evaluations to complete
            results = await asyncio.gather(*tasks)
            for result in results:
                provider_name = result["provider"]
                provider_results[provider_name] = result
        else:
            # Evaluate providers sequentially
            for handler, provider_name in zip(search_handlers, provider_names):
                logger.info(f"Evaluating provider: {provider_name}")
                if evaluation_type == EvaluationType.SIMPLEQA:
                    result = await evaluate_provider_simple_qa(
                        provider_name,
                        handler,
                        examples[provider_name],
                        post_processor,
                        evaluator_model,
                    )
                elif evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
                    result = await evaluate_provider_document_relevance(
                        provider_name,
                        handler,
                        examples[provider_name],
                        environment="test",
                    )
                provider_results[provider_name] = result
        
        # For Document Relevance evaluation, wait for Quotient AI logs to be processed and calculate stats
        if evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
            QUOTIENT_AI_WAIT_TIME = 60
            logger.info(f"Waiting for {QUOTIENT_AI_WAIT_TIME} seconds for logs to be processed in Quotient AI")
            time.sleep(QUOTIENT_AI_WAIT_TIME)
            
            # Calculate relevance stats for each provider
            for provider_name, result in provider_results.items():
                quotient_client = result['quotient_client']
                app_name = result['app_name']
                if quotient_client:
                    n_relevant_docs = 0
                    n_docs = 0
                    
                    logs = quotient_client.logs.list(app_name=app_name)
                    for log in logs:
                        if hasattr(log, 'documents') and log.documents:
                            n_docs += len(log.documents)
                            for document in log.documents:
                                if document.get('is_relevant', False):
                                    n_relevant_docs += 1
                    
                    # Update the result with calculated stats
                    result['relevant_docs'] = n_relevant_docs
                    result['total_docs'] = n_docs
                    result['relevant_docs_percentage'] = (n_relevant_docs / n_docs * 100) if n_docs > 0 else 0
                    result['app_name'] = app_name
                    
                    logger.info(f"[{provider_name}] Relevance stats: {result['relevant_docs_percentage']:.1f}% ({result['relevant_docs']}/{result['total_docs']})")
                    

        save_summary(provider_results, output_dir, evaluation_type)

        print("\n===== EVALUATION RESULTS =====")
        print(f"Evaluation Type: {evaluation_type}")
        print(f"Dataset: {get_dataset_path(evaluation_type)}")
        print("-----------------------------")
        for provider_name, result in provider_results.items():
            if evaluation_type == EvaluationType.SIMPLEQA:
                print(f"{provider_name}: {result['accuracy']:.2%} ({result['correct_count']}/{result['total_count']})")
            elif evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
                print(f"{provider_name}: {result['relevant_docs_percentage']:.1f}% ({result['relevant_docs']}/{result['total_docs']})")
        print("=============================\n")
        
        return provider_results
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark evaluation using specified evaluation type")
    parser.add_argument("--evaluation_type", default=EvaluationType.SIMPLEQA.value, choices=[EvaluationType.SIMPLEQA.value, EvaluationType.DOCUMENT_RELEVANCE.value], help="Type of evaluation to run (simpleqa or document_relevance)")
    parser.add_argument("--config", default="configs/config.json", type=str, help="Path to JSON config file with provider parameters")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for examples (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="Ending index for examples (exclusive)")
    parser.add_argument("--random_sample", type=int, default=None, help="Number of random samples to select (overrides start/end index)")
    parser.add_argument("--post_process_model", default="gpt-4.1", help="Model for post-processing for SimpleQA")
    parser.add_argument("--token_model", default="gpt-4.1", help="Model for token consumption calculation")
    parser.add_argument("--evaluator_model", default="gpt-4.1", help="Model for correctness evaluation for SimpleQA")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--sequential", action="store_true", help="Run providers sequentially instead of in parallel")
    parser.add_argument("--rerun", action="store_true", help="Rerun evaluation on existing results directory, output_dir must exist")
    
    args = parser.parse_args()
    
    search_provider_params = {}
    
    try:
        with open(args.config, 'r') as f:
            search_provider_params = json.load(f)
        logger.info(f"Loaded provider configuration from file: {args.config}")
    except Exception as e:
        logger.info(f"Error loading provider configuration from file: {str(e)}")
        logger.info("Using default Tavily config")
        search_provider_params = {"tavily": TAVILY_DEFAULT_CONFIG}
    
    evaluation_type = EvaluationType(args.evaluation_type)
    output_dir = get_output_dir(evaluation_type=evaluation_type, output_dir=args.output_dir, rerun=args.rerun)

    
    asyncio.run(run_evaluation(
        evaluation_type=evaluation_type,
        search_provider_params=search_provider_params,
        config_path=args.config,
        start_index=args.start_index,
        end_index=args.end_index,
        random_sample=args.random_sample,
        post_process_model=args.post_process_model,
        token_model=args.token_model,
        evaluator_model=args.evaluator_model,
        parallel=not args.sequential,
        output_dir=output_dir,
        rerun=args.rerun,
    ))
