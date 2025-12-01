import os
import time
import re
from typing import Tuple, List, Optional, Dict

def treescore_metric(filename: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Calculates a treescore: the average quality score across the model's lineage tree.
    This metric traces the model's ancestry (base_model references) and averages scores.
    
    Args:
        filename (str): The full text content of the README file.
        verbosity (int): The verbosity level (0=silent, 1=INFO, 2=DEBUG).
        log_queue (multiprocessing.Queue): The queue for centralized logging.
    
    Returns:
        A tuple containing:
        - The treescore (average of lineage scores, 0.0 if unable to calculate).
        - The total time spent (float).
    """
    pid = os.getpid()
    start_time = time.perf_counter()
    score = 0.0
    
    try:
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [INFO] Treescore: Starting lineage analysis...")
        
        # Extract base_model information from README text
        readme_text = filename
        base_models = _extract_base_models(readme_text)
        
        if verbosity >= 2:
            log_queue.put(f"[{pid}] [DEBUG] Treescore: Found {len(base_models)} base model references")
            if base_models:
                log_queue.put(f"[{pid}] [DEBUG] Treescore: Base models = {base_models}")
        
        # Calculate lineage depth (current model + ancestors)
        lineage_depth = len(base_models) + 1  # +1 for current model
        
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [INFO] Treescore: Lineage depth = {lineage_depth}")
        
        # Score based on lineage characteristics
        # Since we don't have access to actual ancestor scores in this context,
        # we'll use a proxy metric based on lineage quality indicators
        lineage_score = _calculate_lineage_quality(readme_text, base_models, verbosity, log_queue, pid)
        
        score = lineage_score
        
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [INFO] Treescore: Final score = {score:.2f}")
        
    except Exception as e:
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] calculating treescore: {e}")
        score = 0.0
    
    time_taken = time.perf_counter() - start_time
    
    if verbosity >= 1:
        log_queue.put(f"[{pid}] [INFO] Finished treescore calculation. Score={score:.2f}, Time={time_taken:.3f}s")
    
    return score, time_taken


def _extract_base_models(readme_text: str) -> List[str]:
    """
    Extract base_model references from README text.
    Looks for common patterns like:
    - base_model: model/name
    - Fine-tuned from: model/name
    - Based on: model/name
    
    Returns:
        List of base model identifiers found
    """
    base_models = []
    
    # Pattern 1: YAML frontmatter style "base_model: owner/repo"
    yaml_pattern = r'base_model\s*:\s*([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)'
    matches = re.findall(yaml_pattern, readme_text, re.IGNORECASE)
    base_models.extend(matches)
    
    # Pattern 2: "fine-tuned from [model]" or "finetuned from [model]"
    finetune_pattern = r'fine-?tuned\s+(?:from|on)\s*:?\s*\[?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)\]?'
    matches = re.findall(finetune_pattern, readme_text, re.IGNORECASE)
    base_models.extend(matches)
    
    # Pattern 3: "based on [model]"
    based_pattern = r'based\s+on\s*:?\s*\[?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)\]?'
    matches = re.findall(based_pattern, readme_text, re.IGNORECASE)
    base_models.extend(matches)
    
    # Pattern 4: "parent model: [model]"
    parent_pattern = r'parent\s+model\s*:?\s*\[?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)\]?'
    matches = re.findall(parent_pattern, readme_text, re.IGNORECASE)
    base_models.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for model in base_models:
        if model not in seen:
            seen.add(model)
            unique_models.append(model)
    
    return unique_models


def _calculate_lineage_quality(readme_text: str, base_models: List[str], verbosity: int, log_queue, pid: int) -> float:
    """
    Calculate a quality score based on lineage characteristics.
    Since we don't have access to actual ancestor evaluation scores,
    we use proxy indicators:
    - Presence of well-known base models (higher quality)
    - Documentation of lineage (transparency)
    - Depth of lineage information
    
    Returns:
        Score between 0.0 and 1.0
    """
    score_components = []
    
    # Component 1: Lineage documentation (40% weight)
    # Does the model clearly document its lineage?
    if base_models:
        lineage_doc_score = 1.0
    else:
        # Check if this might be a base model itself
        base_model_indicators = [
            'base model', 'foundation model', 'pretrained from scratch',
            'trained from scratch', 'original model'
        ]
        is_likely_base = any(indicator in readme_text.lower() for indicator in base_model_indicators)
        lineage_doc_score = 0.8 if is_likely_base else 0.3
    
    score_components.append(('lineage_documentation', lineage_doc_score, 0.4))
    
    # Component 2: Well-known base models (30% weight)
    # Check if base models are from reputable sources
    well_known_orgs = [
        'google', 'meta', 'facebook', 'microsoft', 'openai', 'anthropic',
        'mistralai', 'stabilityai', 'bigscience', 'eleutherai', 'huggingface',
        'nvidia', 'ibm', 'cohere', 'allenai', 'cerebras'
    ]
    
    if base_models:
        reputable_count = sum(
            1 for model in base_models
            if any(org in model.lower() for org in well_known_orgs)
        )
        reputation_score = min(reputable_count / len(base_models), 1.0)
    else:
        # No base models mentioned - could be a base model itself
        reputation_score = 0.5
    
    score_components.append(('base_model_reputation', reputation_score, 0.3))
    
    # Component 3: Lineage depth and detail (30% weight)
    # More detailed lineage information suggests better engineering practices
    lineage_keywords = [
        'fine-tuned', 'finetuned', 'adapted', 'specialized', 'customized',
        'trained on', 'training data', 'base model', 'parent model',
        'checkpoint', 'weights', 'parameters'
    ]
    
    readme_lower = readme_text.lower()
    keyword_matches = sum(1 for kw in lineage_keywords if kw in readme_lower)
    detail_score = min(keyword_matches / 5.0, 1.0)  # Normalize to 0-1, cap at 5 keywords
    
    score_components.append(('lineage_detail', detail_score, 0.3))
    
    # Calculate weighted average
    total_score = sum(score * weight for _, score, weight in score_components)
    
    if verbosity >= 2:
        log_queue.put(f"[{pid}] [DEBUG] Treescore components:")
        for name, score, weight in score_components:
            log_queue.put(f"[{pid}] [DEBUG]   {name}: {score:.2f} (weight: {weight})")
        log_queue.put(f"[{pid}] [DEBUG] Treescore total: {total_score:.2f}")
    
    return round(total_score, 2)
