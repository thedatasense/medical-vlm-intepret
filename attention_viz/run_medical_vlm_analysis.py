#!/usr/bin/env python3
"""
Fixed Medical VLM Attention Analysis Pipeline for Google Colab

This script provides a full pipeline for:
1. Setting up the environment
2. Loading and comparing LLaVA-Rad and MedGemma models
3. Running robustness analysis on medical imaging questions
4. Generating comprehensive reports

Usage in Google Colab:
1. Upload this file to Colab
2. Run: !python run_medical_vlm_analysis.py
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from scipy.spatial.distance import jensenshannon


# ===========================
# Environment Setup Functions
# ===========================

def setup_colab_environment():
    """Setup the Google Colab environment with better error handling"""
    print("Setting up Google Colab environment...")
    
    # Clone repository if not exists
    if not os.path.exists('/content/medical-vlm-intepret'):
        print("Cloning repository...")
        subprocess.run(['git', 'clone', 'https://github.com/thedatasense/medical-vlm-intepret.git'], 
                      cwd='/content', check=True)
    
    # Change to repo directory
    os.chdir('/content/medical-vlm-intepret/attention_viz')
    
    # Install dependencies
    print("Installing dependencies...")
    packages = [
        'torch', 'torchvision', 'transformers>=4.36.0',
        'opencv-python', 'scipy', 'matplotlib', 'pillow',
        'bitsandbytes', 'accelerate', 'gradio'
    ]
    
    for package in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package])
    
    # Handle LLaVA installation more carefully
    if not os.path.exists('/content/LLaVA'):
        print("Installing LLaVA...")
        try:
            # Clone LLaVA
            subprocess.run(['git', 'clone', 'https://github.com/haotian-liu/LLaVA.git'], 
                          cwd='/content', check=True)
            
            # Try to install LLaVA dependencies without full package install
            llava_requirements = '/content/LLaVA/requirements.txt'
            if os.path.exists(llava_requirements):
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', llava_requirements], 
                             check=False)  # Don't fail if some deps are problematic
            
            # Install specific LLaVA dependencies we need
            llava_deps = [
                'einops', 'einops-exts', 'timm', 
                'gradio_client==0.6.1', 'deepspeed'
            ]
            for dep in llava_deps:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', dep], check=False)
                except:
                    pass  # Skip problematic dependencies
                    
        except subprocess.CalledProcessError as e:
            print(f"Warning: LLaVA installation had issues: {e}")
            print("Continuing without full LLaVA install - core functionality should still work")
    
    # Add to Python path
    sys.path.insert(0, '/content/LLaVA')
    sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')
    
    print("Environment setup complete!")
    return True


def mount_drive_and_verify_data():
    """Mount Google Drive and verify data paths"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    except ImportError:
        raise RuntimeError("This script must be run in Google Colab with Drive access")
    
    # Define data paths
    data_paths = {
        'data_root': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset',
        'image_root': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa',
        'csv_path': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz/medical-cxr-vqa-questions_sample.csv',
        'csv_variants_path': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz/medical-cxr-vqa-questions_sample_hardpositives.csv'
    }
    
    # Verify paths
    print("\nVerifying data paths:")
    missing_paths = []
    for name, path in data_paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {'✓' if exists else '✗'} {path}")
        if not exists:
            missing_paths.append((name, path))
    
    if missing_paths:
        print("\nERROR: Required data paths are missing:")
        for name, path in missing_paths:
            print(f"  - {name}: {path}")
        print("\nPlease ensure your Google Drive contains the medical dataset at the expected location.")
        raise FileNotFoundError("Required medical imaging data not found")
    
    return data_paths


# ===========================
# Model Loading Functions
# ===========================

def load_models(llava_8bit=True, medgemma_8bit=True):
    """Load both models with memory optimization"""
    print("\nLoading models...")
    
    # Import after environment setup
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig
    from medgemma_enhanced import load_model_enhanced, EnhancedAttentionExtractor, AttentionExtractionConfig
    
    # Load LLaVA-Rad
    print("Loading LLaVA-Rad...")
    llava_config = AttentionConfig(
        use_medical_colormap=True,
        multi_head_mode='mean',
        percentile_clip=(5, 95)
    )
    llava_vis = EnhancedLLaVARadVisualizer(config=llava_config)
    llava_vis.load_model(load_in_8bit=llava_8bit)
    
    # Load MedGemma
    print("Loading MedGemma...")
    medgemma_model, medgemma_processor = load_model_enhanced(
        model_id="google/medgemma-4b-it",
        load_in_8bit=medgemma_8bit
    )
    
    return llava_vis, medgemma_model, medgemma_processor


# ===========================
# Data Loading Functions
# ===========================

@dataclass
class StudySample:
    study_id: str
    image_path: str
    finding: str
    variant_id: int
    question: str
    answer_gt: str


def load_study_data(data_paths: Dict[str, str], n_studies: Optional[int] = None) -> List[StudySample]:
    """Load study data from CSV files"""
    samples = []
    
    # Try to load from CSV
    csv_path = data_paths.get('csv_path')
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Process base questions (variant 0)
        for _, row in df.iterrows():
            if n_studies and len(samples) >= n_studies * 6:
                break
                
            samples.append(StudySample(
                study_id=str(row.get('study_id', row.get('image_id', f"study_{len(samples)}"))),
                image_path=row.get('image_path', ''),
                finding=row.get('finding', 'unknown'),
                variant_id=0,
                question=row.get('question', 'Is there an abnormality?'),
                answer_gt=row.get('answer', 'unknown')
            ))
    
    # Load variants if available
    variants_path = data_paths.get('csv_variants_path')
    if variants_path and os.path.exists(variants_path):
        print(f"Loading variants from {variants_path}")
        df_variants = pd.read_csv(variants_path)
        # Add variant processing logic here
    
    # If no data loaded, this is a critical error
    if not samples:
        raise ValueError("No medical data loaded. Please check your CSV files and data paths.")
    
    print(f"Loaded {len(samples)} samples")
    return samples[:n_studies * 6] if n_studies else samples


# ===========================
# Inference Functions
# ===========================

def run_inference_on_sample(
    sample: StudySample,
    llava_vis,
    medgemma_model,
    medgemma_processor,
    output_dir: str
) -> Dict[str, Any]:
    """Run inference on a single sample with both models"""
    
    from medgemma_enhanced import EnhancedAttentionExtractor, AttentionExtractionConfig
    
    result = {
        'study_id': sample.study_id,
        'finding': sample.finding,
        'variant_id': sample.variant_id,
        'question': sample.question,
        'answer_gt': sample.answer_gt,
        'timestamp': datetime.now().isoformat()
    }
    
    # Verify image exists
    if not os.path.exists(sample.image_path):
        print(f"ERROR: Image not found: {sample.image_path}")
        raise FileNotFoundError(f"Medical image not found: {sample.image_path}")
    
    # Run LLaVA-Rad
    try:
        start_time = time.time()
        llava_result = llava_vis.generate_with_attention(
            sample.image_path,
            sample.question,
            max_new_tokens=50,
            use_cache=False
        )
        llava_time = time.time() - start_time
        
        result['llava_answer'] = llava_result.get('answer', '')
        result['llava_correct'] = normalize_answer(llava_result.get('answer', '')) == normalize_answer(sample.answer_gt)
        result['llava_latency_ms'] = int(llava_time * 1000)
        result['llava_attention_method'] = llava_result.get('attention_method', 'unknown')
        
    except Exception as e:
        print(f"LLaVA error on {sample.study_id}: {e}")
        result['llava_error'] = str(e)
    
    # Run MedGemma
    try:
        # Load image
        image = Image.open(sample.image_path).convert('RGB')
        
        # Create extractor
        extractor = EnhancedAttentionExtractor(
            AttentionExtractionConfig(
                attention_head_reduction='mean',
                fallback_chain=['cross_attention', 'gradcam', 'uniform']
            )
        )
        
        # Prepare prompt
        if hasattr(medgemma_processor, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample.question}
                ]
            }]
            prompt = medgemma_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"<image>{sample.question}"
        
        # Prepare inputs
        inputs = medgemma_processor(text=prompt, images=image, return_tensors="pt")
        device = next(medgemma_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = medgemma_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True
            )
        medgemma_time = time.time() - start_time
        
        # Decode answer
        raw_answer = medgemma_processor.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        
        # Clean answer
        if '<start_of_turn>model' in raw_answer:
            answer = raw_answer.split('<start_of_turn>model')[-1].split('<end_of_turn>')[0].strip()
        else:
            answer = raw_answer.split(sample.question)[-1].strip()
        
        result['medgemma_answer'] = answer
        result['medgemma_correct'] = normalize_answer(answer) == normalize_answer(sample.answer_gt)
        result['medgemma_latency_ms'] = int(medgemma_time * 1000)
        
        # Extract attention
        attention, _, method = extractor.extract_token_conditioned_attention_robust(
            medgemma_model, medgemma_processor, outputs,
            [sample.finding.split()[0]], image, prompt
        )
        result['medgemma_attention_method'] = method
        
    except Exception as e:
        print(f"MedGemma error on {sample.study_id}: {e}")
        result['medgemma_error'] = str(e)
    
    return result


def normalize_answer(answer: str) -> str:
    """Normalize answer to yes/no"""
    answer_lower = answer.lower().strip()
    if any(word in answer_lower for word in ['yes', 'positive', 'present', 'evidence', 'shows', 'visible']):
        return 'yes'
    elif any(word in answer_lower for word in ['no', 'negative', 'absent', 'normal', 'clear']):
        return 'no'
    else:
        return 'unknown'


# ===========================
# Analysis Functions
# ===========================

def analyze_results(results: List[Dict[str, Any]], output_dir: str):
    """Analyze results and generate reports"""
    print("\nAnalyzing results...")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate metrics
    metrics = {}
    
    # Overall accuracy
    for model in ['llava', 'medgemma']:
        correct_col = f'{model}_correct'
        if correct_col in df.columns:
            metrics[f'{model}_accuracy'] = df[correct_col].mean()
            print(f"{model.upper()} Accuracy: {metrics[f'{model}_accuracy']:.2%}")
    
    # Per-finding accuracy
    if 'finding' in df.columns:
        finding_accuracy = df.groupby('finding').agg({
            'llava_correct': 'mean',
            'medgemma_correct': 'mean'
        }).round(3)
        print("\nPer-finding Accuracy:")
        print(finding_accuracy)
    
    # Robustness analysis (consistency across variants)
    if 'study_id' in df.columns and 'variant_id' in df.columns:
        consistency = df.groupby('study_id').agg({
            'llava_correct': lambda x: x.std() == 0,
            'medgemma_correct': lambda x: x.std() == 0
        }).mean()
        print("\nConsistency across variants:")
        print(f"LLaVA: {consistency.get('llava_correct', 0):.2%}")
        print(f"MedGemma: {consistency.get('medgemma_correct', 0):.2%}")
    
    # Save results
    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'n_samples': len(df),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Save detailed results
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}")


# ===========================
# Main Pipeline
# ===========================

def main():
    parser = argparse.ArgumentParser(description='Medical VLM Robustness Analysis')
    parser.add_argument('--n_studies', type=int, default=10, 
                       help='Number of studies to process (default: 10)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup environment
    setup_colab_environment()
    
    # Mount drive and verify data
    data_paths = mount_drive_and_verify_data()
    if not data_paths:
        print("Warning: Could not mount drive, using local paths")
        data_paths = {
            'csv_path': 'medical-cxr-vqa-questions_sample.csv',
            'csv_variants_path': 'medical-cxr-vqa-questions_sample_hardpositives.csv'
        }
    
    # Load models
    llava_vis, medgemma_model, medgemma_processor = load_models()
    
    # Load data
    samples = load_study_data(data_paths, n_studies=args.n_studies)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    results = []
    results_file = os.path.join(args.output_dir, 'results.jsonl')
    
    print(f"\nProcessing {len(samples)} samples...")
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing {sample.study_id} variant {sample.variant_id}")
        
        result = run_inference_on_sample(
            sample, llava_vis, medgemma_model, medgemma_processor, args.output_dir
        )
        results.append(result)
        
        # Save incrementally
        with open(results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        # Print progress
        if 'llava_correct' in result and 'medgemma_correct' in result:
            print(f"  LLaVA: {result.get('llava_answer', 'N/A')} ({'✓' if result['llava_correct'] else '✗'})")
            print(f"  MedGemma: {result.get('medgemma_answer', 'N/A')} ({'✓' if result['medgemma_correct'] else '✗'})")
    
    # Analyze results
    analyze_results(results, args.output_dir)
    
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()