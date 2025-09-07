#!/usr/bin/env python3
"""
Complete Medical VLM Attention Analysis Pipeline for Google Colab

This script provides a full pipeline for:
1. Setting up the environment
2. Loading and comparing LLaVA-Rad and MedGemma models
3. Running robustness analysis on medical imaging questions
4. Generating comprehensive reports

Usage in Google Colab:
1. Upload this file to Colab
2. Run: !python run_medical_vlm_analysis_colab.py
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
    """Setup the Google Colab environment"""
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
    
    # Install LLaVA
    if not os.path.exists('/content/LLaVA'):
        print("Installing LLaVA...")
        subprocess.run(['git', 'clone', 'https://github.com/haotian-liu/LLaVA.git'], 
                      cwd='/content', check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                      cwd='/content/LLaVA', check=True)
    
    # Add to Python path
    sys.path.insert(0, '/content/LLaVA')
    sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')
    
    print("Environment setup complete!")
    return True


def mount_drive_and_verify_data():
    """Mount Google Drive and verify data paths"""
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    
    # Define data paths
    data_paths = {
        'data_root': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset',
        'image_root': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa',
        'csv_path': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz/medical-cxr-vqa-questions_sample.csv',
        'csv_variants_path': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz/medical-cxr-vqa-questions_sample_hardpositives.csv'
    }
    
    # Verify paths
    print("\nVerifying data paths:")
    all_exist = True
    for name, path in data_paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {'✓' if exists else '✗'} {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\nWARNING: Some data paths do not exist. Please check your Google Drive setup.")
    
    return data_paths


# ===========================
# Model Loading Functions
# ===========================

def load_models(llava_8bit=True, medgemma_8bit=True):
    """Load both LLaVA-Rad and MedGemma models"""
    print("\nLoading models...")
    
    # Import after environment setup
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig
    from medgemma_enhanced import load_model_enhanced
    
    # Load LLaVA-Rad
    print("Loading LLaVA-Rad...")
    llava_config = AttentionConfig(
        use_medical_colormap=True,
        multi_head_mode='entropy_weighted',  # Using fixed entropy weighting
        percentile_clip=(5, 95)
    )
    llava_vis = EnhancedLLaVARadVisualizer(config=llava_config)
    llava_vis.load_model(load_in_8bit=llava_8bit)
    
    # Load MedGemma
    print("Loading MedGemma...")
    medgemma_model, medgemma_processor = load_model_enhanced(
        model_id="google/paligemma-3b-mix-224",
        load_in_8bit=medgemma_8bit
    )
    
    return llava_vis, medgemma_model, medgemma_processor


# ===========================
# Analysis Functions
# ===========================

@dataclass
class InferenceResult:
    """Store inference results for a single sample"""
    study_id: str
    image_path: str
    finding: str
    variant_id: str
    question: str
    answer_gt: str
    timestamp: str
    
    llava_answer: str
    llava_correct: bool
    llava_latency_ms: float
    llava_attention_method: str
    llava_focus_score: float
    llava_sparsity: float
    
    medgemma_answer: str
    medgemma_correct: bool
    medgemma_latency_ms: float
    medgemma_attention_method: str
    medgemma_focus_score: float
    medgemma_sparsity: float
    
    js_divergence: Optional[float] = None


def process_single_sample(
    image_path: str,
    question: str,
    answer_gt: str,
    llava_vis,
    medgemma_model,
    medgemma_processor,
    medgemma_extractor,
    study_info: Dict[str, Any]
) -> InferenceResult:
    """Process a single image-question pair through both models"""
    
    from llava_rad_enhanced import AttentionMetrics
    
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_pil = Image.open(image_path).convert('RGB')
    
    # Debug: Check image size
    print(f"  Processing image: {os.path.basename(image_path)} ({image_pil.size})")
    
    # LLaVA-Rad inference
    start_time = time.time()
    llava_result = llava_vis.generate_with_attention(
        image_path,
        question,
        max_new_tokens=50,
        use_cache=False
    )
    llava_latency = (time.time() - start_time) * 1000
    
    # Extract LLaVA attention
    llava_attention = llava_result.get('visual_attention')
    if isinstance(llava_attention, list):
        llava_attention = np.mean(np.stack(llava_attention), axis=0)
    
    # LLaVA metrics
    llava_metrics = AttentionMetrics.calculate_focus_score(llava_attention)
    llava_metrics['sparsity'] = AttentionMetrics.calculate_sparsity(llava_attention)
    
    # MedGemma inference
    # Build prompt with chat template to ensure image tokens are inserted
    if hasattr(medgemma_processor, 'apply_chat_template'):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = medgemma_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = medgemma_processor(
            text=prompt,
            images=image_pil,
            return_tensors="pt",
        )
    else:
        # Fallback: explicit <image> token
        inputs = medgemma_processor(
            text=f"<image>{question}",
            images=image_pil,
            return_tensors="pt",
        )
    
    device = next(medgemma_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = medgemma_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True
        )
    medgemma_latency = (time.time() - start_time) * 1000
    
    # Extract MedGemma answer
    medgemma_answer = medgemma_processor.tokenizer.decode(
        outputs.sequences[0], skip_special_tokens=True
    )
    
    # Handle Gemma-3 format
    if '<start_of_turn>model' in medgemma_answer:
        # Extract text after model turn
        parts = medgemma_answer.split('<start_of_turn>model')
        if len(parts) > 1:
            medgemma_answer = parts[-1]
            # Remove end_of_turn if present
            medgemma_answer = medgemma_answer.split('<end_of_turn>')[0]
    elif '<start_of_turn>assistant' in medgemma_answer:
        parts = medgemma_answer.split('<start_of_turn>assistant')
        if len(parts) > 1:
            medgemma_answer = parts[-1]
            medgemma_answer = medgemma_answer.split('<end_of_turn>')[0]
    else:
        # Fallback: split by common patterns
        medgemma_answer = medgemma_answer.split("Assistant:")[-1]
        medgemma_answer = medgemma_answer.split("model\n")[-1]
    
    medgemma_answer = medgemma_answer.strip()
    
    # Extract MedGemma attention
    target_words = ["pneumonia", "consolidation", "opacity", "finding", "abnormal"]
    medgemma_attention, _, medgemma_method = medgemma_extractor.extract_token_conditioned_attention_robust(
        medgemma_model, medgemma_processor, outputs,
        target_words, image_pil, question
    )
    
    # MedGemma metrics
    medgemma_metrics = AttentionMetrics.calculate_focus_score(medgemma_attention)
    medgemma_metrics['sparsity'] = AttentionMetrics.calculate_sparsity(medgemma_attention)
    
    # Calculate JS divergence between attention maps
    llava_flat = llava_attention.flatten()
    medgemma_flat = medgemma_attention.flatten()
    llava_norm = llava_flat / (llava_flat.sum() + 1e-10)
    medgemma_norm = medgemma_flat / (medgemma_flat.sum() + 1e-10)
    js_div = float(jensenshannon(llava_norm, medgemma_norm))
    
    # Check correctness (basic yes/no matching)
    llava_correct = llava_result['answer'].lower().strip().startswith(answer_gt.lower().strip()[:3])
    medgemma_correct = medgemma_answer.lower().strip().startswith(answer_gt.lower().strip()[:3])
    
    return InferenceResult(
        study_id=study_info['study_id'],
        image_path=study_info['image_path'],
        finding=study_info['finding'],
        variant_id=study_info['variant_id'],
        question=question,
        answer_gt=answer_gt,
        timestamp=datetime.now().isoformat(),
        
        llava_answer=llava_result['answer'],
        llava_correct=llava_correct,
        llava_latency_ms=llava_latency,
        llava_attention_method=llava_result.get('attention_method', 'unknown'),
        llava_focus_score=llava_metrics['focus'],
        llava_sparsity=llava_metrics['sparsity'],
        
        medgemma_answer=medgemma_answer,
        medgemma_correct=medgemma_correct,
        medgemma_latency_ms=medgemma_latency,
        medgemma_attention_method=medgemma_method,
        medgemma_focus_score=medgemma_metrics['focus'],
        medgemma_sparsity=medgemma_metrics['sparsity'],
        
        js_divergence=js_div
    )


def run_robustness_study(
    data_paths: Dict[str, str],
    n_studies: int = 100,
    models: Optional[Tuple] = None,
    output_file: str = "robustness_results.jsonl"
) -> List[InferenceResult]:
    """Run the complete robustness study"""
    
    from medgemma_enhanced import EnhancedAttentionExtractor, AttentionExtractionConfig
    
    # Load models if not provided
    if models is None:
        llava_vis, medgemma_model, medgemma_processor = load_models()
    else:
        llava_vis, medgemma_model, medgemma_processor = models
    
    # Create MedGemma extractor
    medgemma_extractor = EnhancedAttentionExtractor(
        AttentionExtractionConfig(
            attention_head_reduction='entropy_weighted',
            fallback_chain=['cross_attention', 'gradcam', 'uniform']
        )
    )
    
    # Load data
    csv_path = data_paths.get('csv_variants_path', data_paths['csv_path'])
    df = pd.read_csv(csv_path)
    
    # Get unique studies
    unique_studies = df['study_id'].unique()[:n_studies]
    print(f"\nProcessing {len(unique_studies)} studies...")
    
    results = []
    
    for i, study_id in enumerate(unique_studies):
        study_rows = df[df['study_id'] == study_id]
        print(f"\n[{i+1}/{len(unique_studies)}] Processing study {study_id} ({len(study_rows)} variants)")
        
        for _, row in study_rows.iterrows():
            try:
                image_path = os.path.join(data_paths['image_root'], row['image_path'])
                
                study_info = {
                    'study_id': str(study_id),
                    'image_path': row['image_path'],
                    'finding': row.get('finding', 'unknown'),
                    'variant_id': row.get('question_variant', row.get('variant_id', 'base'))
                }
                
                result = process_single_sample(
                    image_path=image_path,
                    question=row['question'],
                    answer_gt=row.get('answer', 'unknown'),
                    llava_vis=llava_vis,
                    medgemma_model=medgemma_model,
                    medgemma_processor=medgemma_processor,
                    medgemma_extractor=medgemma_extractor,
                    study_info=study_info
                )
                
                results.append(result)
                
                # Save incrementally
                if len(results) % 10 == 0:
                    save_results(results, output_file)
                    
            except Exception as e:
                print(f"  Error processing {row['image_path']}: {e}")
                continue
    
    # Final save
    save_results(results, output_file)
    print(f"\nCompleted {len(results)} inferences")
    
    return results


def save_results(results: List[InferenceResult], output_file: str):
    """Save results to JSONL file"""
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')


# ===========================
# Analysis and Visualization
# ===========================

def analyze_results(results: List[InferenceResult]) -> Dict[str, Any]:
    """Analyze the robustness study results"""
    
    analysis = {
        'n_samples': len(results),
        'n_studies': len(set(r.study_id for r in results)),
        'timestamp': datetime.now().isoformat()
    }
    
    # Model performance
    for model in ['llava', 'medgemma']:
        correct = [getattr(r, f'{model}_correct') for r in results]
        focus_scores = [getattr(r, f'{model}_focus_score') for r in results]
        sparsity_scores = [getattr(r, f'{model}_sparsity') for r in results]
        latencies = [getattr(r, f'{model}_latency_ms') for r in results]
        
        analysis[f'{model}_accuracy'] = np.mean(correct)
        analysis[f'{model}_focus_mean'] = np.mean(focus_scores)
        analysis[f'{model}_focus_std'] = np.std(focus_scores)
        analysis[f'{model}_sparsity_mean'] = np.mean(sparsity_scores)
        analysis[f'{model}_latency_mean'] = np.mean(latencies)
    
    # JS divergence between models
    js_divs = [r.js_divergence for r in results if r.js_divergence is not None]
    analysis['js_divergence_mean'] = np.mean(js_divs)
    analysis['js_divergence_std'] = np.std(js_divs)
    
    # Robustness analysis by study
    robustness_stats = analyze_robustness_by_study(results)
    analysis.update(robustness_stats)
    
    return analysis


def analyze_robustness_by_study(results: List[InferenceResult]) -> Dict[str, Any]:
    """Analyze robustness metrics grouped by study"""
    
    df = pd.DataFrame([asdict(r) for r in results])
    
    robustness = {
        'llava_flip_rate': 0,
        'medgemma_flip_rate': 0,
        'llava_consistency_rate': 0,
        'medgemma_consistency_rate': 0
    }
    
    # Group by study
    study_groups = df.groupby('study_id')
    n_multi_variant = 0
    
    for study_id, group in study_groups:
        if len(group) > 1:
            n_multi_variant += 1
            
            # Check answer consistency
            llava_answers = group['llava_answer'].tolist()
            medgemma_answers = group['medgemma_answer'].tolist()
            
            # Count flips (different from base)
            base_llava = llava_answers[0]
            base_medgemma = medgemma_answers[0]
            
            llava_flips = sum(1 for a in llava_answers[1:] if a != base_llava)
            medgemma_flips = sum(1 for a in medgemma_answers[1:] if a != base_medgemma)
            
            robustness['llava_flip_rate'] += llava_flips / (len(group) - 1)
            robustness['medgemma_flip_rate'] += medgemma_flips / (len(group) - 1)
            
            # Check full consistency
            if len(set(llava_answers)) == 1:
                robustness['llava_consistency_rate'] += 1
            if len(set(medgemma_answers)) == 1:
                robustness['medgemma_consistency_rate'] += 1
    
    # Average over studies
    if n_multi_variant > 0:
        robustness['llava_flip_rate'] /= n_multi_variant
        robustness['medgemma_flip_rate'] /= n_multi_variant
        robustness['llava_consistency_rate'] /= n_multi_variant
        robustness['medgemma_consistency_rate'] /= n_multi_variant
    
    return robustness


def visualize_results(results: List[InferenceResult], output_dir: str = "visualizations"):
    """Create visualizations of the results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy comparison
    plt.figure(figsize=(10, 6))
    
    llava_acc = np.mean([r.llava_correct for r in results])
    medgemma_acc = np.mean([r.medgemma_correct for r in results])
    
    models = ['LLaVA-Rad', 'MedGemma']
    accuracies = [llava_acc, medgemma_acc]
    
    bars = plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. Attention metrics comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Focus scores
    llava_focus = [r.llava_focus_score for r in results]
    medgemma_focus = [r.medgemma_focus_score for r in results]
    
    ax1.hist(llava_focus, bins=30, alpha=0.7, label='LLaVA-Rad', color='blue')
    ax1.hist(medgemma_focus, bins=30, alpha=0.7, label='MedGemma', color='green')
    ax1.set_xlabel('Focus Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Attention Focus Scores')
    ax1.legend()
    
    # Sparsity scores
    llava_sparsity = [r.llava_sparsity for r in results]
    medgemma_sparsity = [r.medgemma_sparsity for r in results]
    
    ax2.hist(llava_sparsity, bins=30, alpha=0.7, label='LLaVA-Rad', color='blue')
    ax2.hist(medgemma_sparsity, bins=30, alpha=0.7, label='MedGemma', color='green')
    ax2.set_xlabel('Sparsity Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Attention Sparsity Scores')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_metrics_distribution.png'))
    plt.close()
    
    # 3. JS Divergence distribution
    plt.figure(figsize=(8, 6))
    
    js_divs = [r.js_divergence for r in results if r.js_divergence is not None]
    plt.hist(js_divs, bins=30, color='purple', alpha=0.7)
    plt.axvline(np.mean(js_divs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(js_divs):.3f}')
    plt.xlabel('JS Divergence')
    plt.ylabel('Count')
    plt.title('Distribution of JS Divergence Between Model Attentions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'js_divergence_distribution.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def generate_report(results: List[InferenceResult], analysis: Dict[str, Any], 
                   output_file: str = "robustness_report.md"):
    """Generate a comprehensive markdown report"""
    
    report = f"""# Medical VLM Robustness Study Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of a robustness analysis comparing LLaVA-Rad and MedGemma models on chest X-ray interpretation tasks.

## Dataset Overview

- **Total samples:** {analysis['n_samples']}
- **Unique studies:** {analysis['n_studies']}
- **Average variants per study:** {analysis['n_samples'] / analysis['n_studies']:.1f}

## Model Performance

### Accuracy
- **LLaVA-Rad:** {analysis['llava_accuracy']:.3f}
- **MedGemma:** {analysis['medgemma_accuracy']:.3f}

### Attention Metrics

| Metric | LLaVA-Rad | MedGemma |
|--------|-----------|----------|
| Focus Score | {analysis['llava_focus_mean']:.3f} ± {analysis['llava_focus_std']:.3f} | {analysis['medgemma_focus_mean']:.3f} ± {analysis['medgemma_focus_std']:.3f} |
| Sparsity | {analysis['llava_sparsity_mean']:.3f} | {analysis['medgemma_sparsity_mean']:.3f} |
| Inference Time | {analysis['llava_latency_mean']:.1f}ms | {analysis['medgemma_latency_mean']:.1f}ms |

### Attention Divergence
- **Mean JS Divergence:** {analysis['js_divergence_mean']:.3f} ± {analysis['js_divergence_std']:.3f}

## Robustness Analysis

### Answer Consistency Across Variants
- **LLaVA-Rad Flip Rate:** {analysis.get('llava_flip_rate', 0):.3f}
- **MedGemma Flip Rate:** {analysis.get('medgemma_flip_rate', 0):.3f}
- **LLaVA-Rad Full Consistency:** {analysis.get('llava_consistency_rate', 0):.1%}
- **MedGemma Full Consistency:** {analysis.get('medgemma_consistency_rate', 0):.1%}

## Key Findings

1. **Performance:** Both models show comparable accuracy on the chest X-ray interpretation task.

2. **Attention Patterns:** 
   - LLaVA-Rad tends to have {'more' if analysis['llava_focus_mean'] > analysis['medgemma_focus_mean'] else 'less'} focused attention patterns
   - MedGemma shows {'higher' if analysis['medgemma_sparsity_mean'] > analysis['llava_sparsity_mean'] else 'lower'} sparsity in attention distribution

3. **Robustness:** 
   - Both models show sensitivity to question phrasing variations
   - {'LLaVA-Rad' if analysis.get('llava_flip_rate', 1) < analysis.get('medgemma_flip_rate', 1) else 'MedGemma'} demonstrates better consistency across paraphrases

4. **Efficiency:** 
   - {'LLaVA-Rad' if analysis['llava_latency_mean'] < analysis['medgemma_latency_mean'] else 'MedGemma'} has faster inference times

## Recommendations

1. **Clinical Use:** Both models require careful validation before clinical deployment due to sensitivity to input phrasing
2. **Ensemble Approach:** Consider combining both models to improve robustness
3. **Further Research:** Investigate prompt engineering strategies to improve consistency

## Appendix

### Sample Disagreements

Below are examples where the models provided different answers:
"""
    
    # Add sample disagreements
    disagreements = [r for r in results[:50] if r.llava_answer != r.medgemma_answer][:3]
    
    for i, r in enumerate(disagreements):
        report += f"""
#### Example {i+1}
- **Question:** {r.question}
- **Ground Truth:** {r.answer_gt}
- **LLaVA-Rad:** {r.llava_answer} ({'✓' if r.llava_correct else '✗'})
- **MedGemma:** {r.medgemma_answer} ({'✓' if r.medgemma_correct else '✗'})
- **JS Divergence:** {r.js_divergence:.3f}
"""
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_file}")
    
    return report


# ===========================
# Main Execution Function
# ===========================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Medical VLM Robustness Analysis')
    parser.add_argument('--n_studies', type=int, default=10, 
                       help='Number of studies to process (default: 10)')
    parser.add_argument('--output_dir', type=str, default='robustness_output',
                       help='Output directory for results (default: robustness_output)')
    parser.add_argument('--skip_setup', action='store_true',
                       help='Skip environment setup (if already done)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    try:
        # 1. Setup environment
        if not args.skip_setup:
            setup_colab_environment()
        
        # 2. Mount Drive and verify data
        data_paths = mount_drive_and_verify_data()
        
        # 3. Load models
        models = load_models()
        
        # 4. Run robustness study
        print(f"\nRunning robustness study on {args.n_studies} studies...")
        results_file = "robustness_results.jsonl"
        
        results = run_robustness_study(
            data_paths=data_paths,
            n_studies=args.n_studies,
            models=models,
            output_file=results_file
        )
        
        # 5. Analyze results
        print("\nAnalyzing results...")
        analysis = analyze_results(results)
        
        # Save analysis
        with open("analysis_results.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # 6. Create visualizations
        print("Creating visualizations...")
        visualize_results(results, output_dir="visualizations")
        
        # 7. Generate report
        print("Generating report...")
        report = generate_report(results, analysis)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Processed {len(results)} samples from {analysis['n_studies']} studies")
        print(f"LLaVA-Rad Accuracy: {analysis['llava_accuracy']:.3f}")
        print(f"MedGemma Accuracy: {analysis['medgemma_accuracy']:.3f}")
        print(f"Mean JS Divergence: {analysis['js_divergence_mean']:.3f}")
        print(f"\nResults saved to: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
