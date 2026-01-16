#!/usr/bin/env python3
"""
Main experiment script for CHTC GPU Lab.
Runs medical diagnosis experiments on MIMIC data.
Faithful to original qwen_script.py

Supports batch processing via --start_idx and --end_idx arguments.
"""
import os
import sys
import json
import time
import argparse
import yaml
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    format_patient_info,
    MedicalAgent,
    run_experiment,
    run_experiment_no_dist
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--end_idx", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--job_id", type=int, default=1, help="Job ID for output file naming")
    args = parser.parse_args()

    print("=" * 60)
    print("CHTC Medical Diagnosis Experiment")
    print("=" * 60)
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Get settings from config
    max_try = config['experiment']['max_try']
    threshold = config['experiment']['threshold']
    disease_list = config['disease_list']
    
    # Environment info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)
    
    print(f"Model: {args.model_path}")
    print(f"Processing patients [{args.start_idx}, {args.end_idx})")
    print(f"Max tries: {max_try}")
    print(f"Threshold: {threshold}")
    print(f"Job ID: {args.job_id}")
    print(f"Disease list: {len(disease_list)} diseases")
    print("=" * 60)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load model
    print("Loading model...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True
    )
    
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")
    print("=" * 60)
    
    # Load data
    print("Loading data from mimic_data.csv...")
    df = pd.read_csv("mimic_data.csv")
    
    # Get subset based on indices
    start_idx = max(0, args.start_idx)
    end_idx = min(len(df), args.end_idx)
    subset_df = df.iloc[start_idx:end_idx]
    
    num_patients = len(subset_df)
    print(f"Processing {num_patients} patients (indices {start_idx} to {end_idx})")
    print("=" * 60)
    
    # Create agent with disease list from config
    agent = MedicalAgent(model, tokenizer, disease_list)
    
    # Results list
    results = []
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Output file
    output_file = f"results/result_{args.job_id}.json"
    
    # Run experiments
    experiment_start = time.time()
    
    for i, (idx, row) in enumerate(subset_df.iterrows()):
        print('=' * 60)
        print(f'EXPERIMENT {i + 1}/{num_patients}: Patient {row["subject_id_x"]} (global idx {idx})')
        print('=' * 60)
        
        true_diagnosis = row['target_label']
        
        # Format patient note (Q-LLM view - no diagnosis)
        note = format_patient_info(row, include_diagnosis=False)
        print(f"Patient Info:\n{note}\n")
        print(f"True Diagnosis: {true_diagnosis}\n")
        
        # Session result template
        session_result_init = {
            'patient_id': row['subject_id_x'],
            'global_idx': int(idx),
            'condition': 'EIG',
            'converged': False,
            'correct': False,
            'num_try': max_try,
        }
        
        # 1. INITIALIZE BELIEF
        current_probs = agent.get_initial_prior(note)
        print(f"Initial probs (top 3): {sorted(current_probs.items(), key=lambda x: x[1], reverse=True)[:3]}\n")

        # ----------------------------------------
        # Condition 1: NON-EIG with dist tracking
        # ----------------------------------------
        print('&' * 50)
        print('Condition: NON-EIG with dist tracking')
        session_result_control = run_experiment(
            control=True,
            agent=agent,
            true_diagnosis=true_diagnosis, 
            note=note,
            current_probs=current_probs.copy(),
            max_try=max_try, 
            threshold=threshold,
            session_result_init=session_result_init.copy())
        results.append(session_result_control)

        # ----------------------------------------
        # Condition 2: NON-EIG without dist tracking
        # ----------------------------------------
        print('&' * 50)
        print('Condition: NON-EIG without dist tracking')
        session_result_control_no_dist = run_experiment_no_dist(
            agent=agent,
            true_diagnosis=true_diagnosis, 
            note=note,
            max_try=max_try, 
            session_result_init=session_result_init.copy())
        results.append(session_result_control_no_dist)

        # ----------------------------------------
        # Condition 3: EIG
        # ----------------------------------------
        print('&' * 50)
        print('Condition: EIG')
        session_result = run_experiment(
            control=False,
            agent=agent,
            true_diagnosis=true_diagnosis, 
            note=note,
            threshold=threshold,
            current_probs=current_probs.copy(),
            max_try=max_try, 
            session_result_init=session_result_init.copy())
        results.append(session_result)
        
        # Save intermediate results after each patient (checkpoint in case of crash)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Checkpoint] Saved {len(results)} results to {output_file}")
    
    # Final summary
    total_time = time.time() - experiment_start
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total patients: {num_patients}")
    print(f"Total results: {len(results)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    if num_patients > 0:
        print(f"Avg time per patient: {total_time/num_patients:.2f}s")
    print(f"Results saved to: {output_file}")
    
    # Quick summary stats
    print("\n--- Quick Summary ---")
    for condition in ['NON-EIG', 'NON-EIG-NO-DIST', 'EIG']:
        subset = [r for r in results if r['condition'] == condition]
        if subset:
            accuracy = sum(1 for r in subset if r['correct']) / len(subset)
            avg_tries = sum(r['num_try'] for r in subset) / len(subset)
            print(f"{condition}: Accuracy={accuracy:.1%}, Avg Tries={avg_tries:.1f}")


if __name__ == "__main__":
    main()