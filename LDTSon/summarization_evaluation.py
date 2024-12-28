from rouge import Rouge
import os
from pathlib import Path
from bert_score import score
import numpy as np
import re

#BERTscore
from torchmetrics.text.bert import BERTScore


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extractive_fragments(A, S):
  """
  Hàm này trả về tập hợp các đoạn trích (extractive fragments) từ văn bản tóm tắt S 
  được trích xuất từ văn bản đầy đủ A.
  
  Args:
    A: Văn bản đầy đủ (string).
    S: Văn bản tóm tắt (string).

  Returns:
    Một tập hợp các đoạn trích (set of strings).
  """
  
  A = A.lower()
  S = S.lower()
  
  fragments = set()
  i = 0
  j = 0

  while i < len(S):
    f = ""
    while j < len(A):
      if S[i] == A[j]:
        i_prime = i
        j_prime = j
        while i_prime < len(S) and j_prime < len(A) and S[i_prime] == A[j_prime]:
          i_prime = i_prime + 1
          j_prime = j_prime + 1

        if len(f) < (i_prime - i - 1):
          f = S[i:i_prime]
        j = j_prime
      else:
        j = j + 1
    
    i = i + max(len(f), 1)
    j = 0
    fragments.add(f)
  
  return fragments

def extractive_fragment_coverage(A, S):
  """
  Hàm này tính toán Extractive Fragment Coverage giữa A và S.

  Args:
    A: Văn bản đầy đủ (string).
    S: Văn bản tóm tắt (string).

  Returns:
    Extractive Fragment Coverage (float).
  """
  
  fragments = extractive_fragments(A, S)
  total_fragment_length = sum([len(re.findall(r'\w+', f)) for f in fragments])
  total_summary_length = len(re.findall(r'\w+', S))
  
  return total_fragment_length / total_summary_length

def extractive_fragment_density(A, S):
  """
  Hàm này tính toán Extractive Fragment Density giữa A và S.

  Args:
    A: Văn bản đầy đủ (string).
    S: Văn bản tóm tắt (string).

  Returns:
    Extractive Fragment Density (float).
  """
  
  fragments = extractive_fragments(A, S)
  total_squared_fragment_length = sum([len(re.findall(r'\w+', f))**2 for f in fragments])
  total_summary_length = len(re.findall(r'\w+', S))
  
  return total_squared_fragment_length / total_summary_length

def compression_ratio(A, S):
  """
  Hàm này tính toán Compression Ratio giữa A và S.

  Args:
    A: Văn bản đầy đủ (string).
    S: Văn bản tóm tắt (string).

  Returns:
    Compression Ratio (float).
  """
  
  total_article_length = len(re.findall(r'\w+', A))
  total_summary_length = len(re.findall(r'\w+', S))
  
  return total_article_length / total_summary_length

# Ví dụ sử dụng
def evaluate_summaries_special(A, S):
    coverage = extractive_fragment_coverage(A, S)
    density = extractive_fragment_density(A, S)
    compression = compression_ratio(A, S)

    print(f"Extractive Fragment Coverage: {coverage}")
    print(f"Extractive Fragment Density: {density}")
    print(f"Compression Ratio: {compression}")

    return coverage, density, compression

def evaluate_summaries(ground_truth_dir, generated_summary_dir):
    rouge = Rouge()
    scores = []
    
    # Get sorted list of files from both directories
    ground_truth_files = sorted(Path(ground_truth_dir).glob('*'))
    summary_files = sorted(Path(generated_summary_dir).glob('*'))
    
    # Ensure we have matching files
    if len(ground_truth_files) != len(summary_files):
        raise ValueError("Number of files in both directories must match")
    
    # Compare each pair of files
    for gt_file, sum_file in zip(ground_truth_files, summary_files):
        reference = read_text_file(gt_file)
        hypothesis = read_text_file(sum_file)
        
        try:
            # Calculate ROUGE scores
            rouge_scores = rouge.get_scores(hypothesis, reference)[0]
            coverage, density, compression = evaluate_summaries_special(reference, hypothesis)
            # #BERTScore
            # bertscore = BERTScore()
            # bert_score = bertscore(hypothesis, reference)         
            scores.append({
                'file': gt_file.name,
                'rouge-1': rouge_scores['rouge-1']['f'],
                'rouge-2': rouge_scores['rouge-2']['f'],
                'rouge-l': rouge_scores['rouge-l']['f'],
                'coverage': coverage,
                'density': density,
                'compression': compression
            })
            
            print(f"Scores for {gt_file.name}:")
            print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
            print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
            print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
            print(f"Extractive Fragment Coverage: {coverage:.4f}")
            print(f"Extractive Fragment Density: {density:.4f}")
            print(f"Compression Ratio: {compression:.4f}")
            # print(f"BERTScore: {bert_score}")
            print("-" * 50)

            
        except Exception as e:
            print(f"Error processing {gt_file.name}: {str(e)}")
    
    return scores

def calculate_average_scores(scores):
    if not scores:
        return None
    
    avg_rouge1 = sum(score['rouge-1'] for score in scores) / len(scores)
    avg_rouge2 = sum(score['rouge-2'] for score in scores) / len(scores)
    avg_rougeL = sum(score['rouge-l'] for score in scores) / len(scores)
    avg_coverage = sum(score['coverage'] for score in scores) / len(scores)
    avg_density = sum(score['density'] for score in scores) / len(scores)
    avg_compression = sum(score['compression'] for score in scores) / len(scores)
    # avg_bert = sum(score['bert_score'] for score in scores) / len(scores)
    
    print("\nAverage ROUGE scores across all files:")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")
    print(f"Average Extractive Fragment Coverage: {avg_coverage:.4f}")
    print(f"Average Extractive Fragment Density: {avg_density:.4f}")
    print(f"Average Compression Ratio: {avg_compression:.4f}")
    # print(f"Average BERTScore: {avg_bert:.4f}")




if __name__ == "__main__":
    ground_truth_dir = r".\data_vietnews\abstract_ground_truth"
    summary_dir = r".\summary_out"
    
    scores = evaluate_summaries(ground_truth_dir, summary_dir)
    calculate_average_scores(scores)