"""
Utility functions and classes for medical diagnosis experiments.
"""
import json
import math
from typing import List, Dict
import torch
import pandas as pd


# ==========================================
# DISEASE LIST (loaded from config.yaml in main.py, passed to MedicalAgent)
# ==========================================
# Default list - will be overridden by config.yaml
DISEASE_LIST = [
    "Alcohol Abuse", "UTI", "Pneumonia", "Depression", "GI Bleed",
    "Cellulitis", "Heart Failure", "Acute Kidney Injury", "Epilepsy",
    "Asthma", "Ischemic Stroke", "Anemia", "Sepsis", "Atrial Fibrillation",
    "COPD", "Kidney Stone", "Anxiety", "Gastroenteritis", 
    "Intestinal Obstruction", "Viral Infection", "Hypertension",
    "Myocardial Infarction", "Type 2 Diabetes", "Pancreatitis", "Hematuria"
]


# ==========================================
# FORMATTING HELPERS
# ==========================================
def format_patient_info(row: pd.Series, include_diagnosis: bool = False) -> str:
    """
    Format patient information into a clinical note string.
    
    Args:
        row: A row from the MIMIC DataFrame
        include_diagnosis: If True, includes the target_label (for A-LLM)
    """
    note = f"""Patient Information:
- Chief Complaint: {row['chiefcomplaint']}
- Vitals: Temp {row['temperature']}°F, HR {row['heartrate']} bpm, RR {row['resprate']}, O2Sat {row['o2sat']}%, BP {row['sbp']}/{row['dbp']} mmHg
- Pain Level: {row['pain']}/10
- Acuity: {row['acuity']} (1=critical, 5=non-urgent)
- Demographics: {row['gender']}, {row['race']}
- Arrival: {row['arrival_transport']}"""
    
    if include_diagnosis:
        note += f"\n- TRUE DIAGNOSIS: {row['target_label']}"
    
    return note


# ==========================================
# MATH HELPERS
# ==========================================
def calculate_entropy(probs):
    """Calculate entropy of a probability distribution (takes list or dict values)."""
    if isinstance(probs, dict):
        probs = list(probs.values())
    return -sum(p * math.log2(p) for p in probs if p > 1e-9)


def bayesian_update(current_probs, likelihoods_yes, answer_is_yes):
    """Updates probabilities based on the answer."""
    new_probs = {}
    total_marginal = 0.0
    
    for d, p_prior in current_probs.items():
        p_yes_given_d = likelihoods_yes.get(d, 0.5)
        
        if answer_is_yes:
            likelihood = p_yes_given_d
        else:
            likelihood = 1.0 - p_yes_given_d
            
        unnormalized_posterior = likelihood * p_prior
        new_probs[d] = unnormalized_posterior
        total_marginal += unnormalized_posterior
        
    if total_marginal < 1e-9: 
        return current_probs 
    return {k: v/total_marginal for k, v in new_probs.items()}


def calculate_eig(current_probs, likelihoods_yes):
    """Calculates EIG for a question given its likelihood matrix."""
    p_vector = list(current_probs.values())
    p_yes_vector = [likelihoods_yes.get(d, 0.5) for d in current_probs.keys()]
    
    # 1. Marginal P(Yes)
    p_yes_marginal = sum(p * py for p, py in zip(p_vector, p_yes_vector))
    p_no_marginal = 1.0 - p_yes_marginal
    
    if p_yes_marginal < 1e-9 or p_no_marginal < 1e-9: 
        return 0.0

    # 2. Conditional Entropies
    post_yes = [(p * py) / p_yes_marginal for p, py in zip(p_vector, p_yes_vector)]
    h_yes = calculate_entropy(post_yes)
    
    post_no = [(p * (1-py)) / p_no_marginal for p, py in zip(p_vector, p_yes_vector)]
    h_no = calculate_entropy(post_no)
    
    # 3. EIG
    h_current = calculate_entropy(p_vector)
    expected_h_future = (p_yes_marginal * h_yes) + (p_no_marginal * h_no)
    
    return h_current - expected_h_future


# ==========================================
# MEDICAL AGENT CLASS
# ==========================================
class MedicalAgent:
    def __init__(self, model, tokenizer, disease_list):
        self.model = model
        self.tokenizer = tokenizer
        self.disease_list = disease_list
        # Handle device properly for device_map="auto"
        # For models with device_map="auto", model.device may not work
        # Get device from first parameter instead
        self.device = next(model.parameters()).device

    def _generate_response(self, prompt: str, system_prompt: str = "You are a helpful medical AI.", 
                          max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a response using the local model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _call_llm_json(self, prompt: str, system_prompt: str = "You are a helpful medical AI.", 
                       temperature: float = 0.0, max_retries: int = 3) -> dict:
        """Make the model return JSON format with retry logic."""
        
        for attempt in range(max_retries):
            response = ""
            try:
                # Use more tokens to avoid truncation
                response = self._generate_response(prompt, system_prompt, 
                                                   max_new_tokens=1024, 
                                                   temperature=temperature)
                
                # Try to extract JSON from the response
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                # Clean up common issues
                response = response.strip()
                
                # Try to fix truncated JSON by finding the last complete entry
                if response and not response.endswith('}'):
                    # Find last complete key-value pair
                    last_comma = response.rfind(',')
                    last_brace = response.rfind('{')
                    if last_comma > last_brace:
                        response = response[:last_comma] + '}'
                    elif last_brace >= 0:
                        response = response[:last_brace+1] + '}'
                
                return json.loads(response)
                
            except json.JSONDecodeError as e:
                print(f"[Attempt {attempt+1}/{max_retries}] JSON parse error: {e}")
                print(f"Raw response (first 300 chars): {response[:300] if response else 'empty'}...")
                
                # Try a more aggressive fix - extract any valid JSON object
                try:
                    # Find first { and try to parse from there
                    start = response.find('{')
                    if start >= 0:
                        # Try to find matching }
                        brace_count = 0
                        for i, char in enumerate(response[start:], start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    extracted = response[start:i+1]
                                    return json.loads(extracted)
                except:
                    pass
                    
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
                    
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] Error: {e}")
                if attempt < max_retries - 1:
                    continue
        
        print("All JSON parsing attempts failed, returning empty dict")
        return {}

    def _call_llm_text(self, prompt: str, system_prompt: str = "You are a helpful medical AI.",
                       temperature: float = 0.7) -> str:
        """Helper for standard text responses."""
        return self._generate_response(prompt, system_prompt, temperature=temperature)

    def get_initial_prior(self, note: str) -> Dict[str, float]:
        """Called ONLY ONCE at the start to get initial probability distribution."""
        prompt = f"""
Patient Information: {note}

Task: Estimate the initial probability (0-1) for EACH disease in the list below based on the patient information.
Disease List: {self.disease_list}

Output strictly JSON format: {{ "Disease1": 0.1, "Disease2": 0.2, ... }}
Ensure all probabilities sum roughly to 1.
"""
        # Use more tokens since we have 25 diseases
        data = self._call_llm_json(prompt, temperature=0.0, max_retries=3)
        
        # Normalize
        total = sum(data.values()) if data else 0
        if total == 0: 
            print("Warning: Empty or invalid prior, using uniform distribution")
            return {k: 1.0/len(self.disease_list) for k in self.disease_list}
        
        normalized = {}
        for d in self.disease_list:
            val = data.get(d, 0.0)
            normalized[d] = val / total
            
        return normalized

    def get_likelihood_matrix(self, question: str, diseases: List[str]) -> Dict[str, float]:
        """
        Estimate P(Answer=YES | Disease) for each disease.
        Uses simple yes/no generation and confidence estimation.
        """
        result = {}
        
        for disease in diseases:
            prompt = f"""Assume a patient has {disease}.

Question: "{question}"

Would this patient likely answer YES to this question?

Answer ONLY: YES or NO"""
            
            messages = [
                {"role": "system", "content": "Answer only YES or NO."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=None,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Get the generated text
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()
            
            # Try to get probability from logits
            if outputs.scores:
                first_token_logits = outputs.scores[0][0]
                probs = torch.softmax(first_token_logits, dim=-1)
                
                # Get token IDs for YES and NO
                yes_tokens = self.tokenizer.encode("YES", add_special_tokens=False)
                no_tokens = self.tokenizer.encode("NO", add_special_tokens=False)
                
                p_yes = probs[yes_tokens[0]].item() if yes_tokens else 0.5
                p_no = probs[no_tokens[0]].item() if no_tokens else 0.5
                
                # Normalize between YES and NO
                total = p_yes + p_no
                if total > 0:
                    p_yes = p_yes / total
                else:
                    p_yes = 0.5
            else:
                # Fallback: use binary answer
                p_yes = 0.9 if "YES" in answer else 0.1
            
            result[disease] = max(0.01, min(0.99, p_yes))
        
        return result

    def propose_binary_questions(self, note: str, history: List, current_probs: Dict) -> List[str]:
        """Proposes 3 questions based on the full probability distribution."""
        history_text = "\n".join([f"Q: {q} A: {a}" for q, a in history]) if history else "None"
        
        sorted_probs = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        dist_str = json.dumps(dict(sorted_probs), indent=2)
        
        prompt = f"""
Patient Information: {note}
All previous questions and answers: {history_text}

Current Disease Probabilities:
{dist_str}

Task: Propose 3 DISTINCT BINARY (Yes/No) questions to narrow down the diagnosis. 
Do not ask questions that have been answered above.
Strategy: Ask questions that differentiate the top likely diseases.

Return strictly a JSON object: {{ "questions": ["Question 1", "Question 2", "Question 3"] }}
"""
        data = self._call_llm_json(prompt, temperature=0.7, max_retries=3)
        questions = data.get("questions", [])
        
        # Fallback: if no questions, try to generate one simple question
        if not questions:
            print("Warning: Failed to get questions from JSON, using fallback")
            top_diseases = [d for d, _ in sorted_probs[:3]]
            questions = [f"Does the patient have symptoms consistent with {top_diseases[0]}?"]
        
        return questions
    
    def propose_one_binary_question_for_control(self, note: str, history: List, current_probs: Dict) -> str:
        """Proposes one question for NON-EIG condition with prob tracking."""
        history_text = "\n".join([f"Q: {q} A: {a}" for q, a in history]) if history else "None"
        
        sorted_probs = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        dist_str = json.dumps(dict(sorted_probs), indent=2)
        
        prompt = f"""
Patient Information: {note}
All previous questions and answers: {history_text}

Current Disease Probabilities:
{dist_str}

Task: Propose 1 DISTINCT BINARY (Yes/No) question to narrow down the diagnosis. 
Do not ask questions that have been answered above.
Strategy: Ask a question that differentiates the top likely diseases.
OUTPUT REQUIREMENT: Output ONLY the question text. Do not add any introductory text or explanations.
"""
        return self._call_llm_text(prompt)
    
    def propose_one_binary_question_for_control_without_dist(self, note: str, history: List) -> str:
        """Proposes one question for NON-EIG condition without prob tracking."""
        history_text = "\n".join([f"Q: {q} A: {a}" for q, a in history]) if history else "None"
        
        prompt = f"""
Patient Information: {note}
All previous questions and answers: {history_text}
Disease candidates: {self.disease_list}

Task: Propose 1 DISTINCT BINARY (Yes/No) question to narrow down the diagnosis. 
Do not ask questions that have been answered above.
OUTPUT REQUIREMENT: Output ONLY the question text. Do not add any introductory text or explanations.
"""
        return self._call_llm_text(prompt)

    def answer_simulation(self, question: str, true_diagnosis: str, note: str) -> str:
        """A-LLM answers the question based on true diagnosis and patient info."""
        prompt = f"""
Patient TRUE Diagnosis: {true_diagnosis}
Patient Information: {note}
Question: "{question}"

Task: Based on the patient's true diagnosis and information, answer strictly "YES" or "NO".
"""
        text = self._call_llm_text(prompt)
        return "YES" if "YES" in text.strip().upper() else "NO"
    
    def guess_answer_control_no_dist(self, note: str, history: List) -> str:
        """Guess the diagnosis without probability tracking."""
        history_text = "\n".join([f"Q: {q} A: {a}" for q, a in history]) if history else "None"
        
        prompt = f"""
Patient Information: {note}
All previous questions and answers: {history_text}
Disease candidates: {self.disease_list}

Task: Estimate the most likely disease (ONLY ONE) based on the above information.
OUTPUT REQUIREMENT: Output ONLY the disease name. Do not add any introductory text or explanations.
"""
        response = self._call_llm_text(prompt)
        # Try to match to a disease in our list
        response_lower = response.lower().strip()
        for d in self.disease_list:
            if d.lower() in response_lower or response_lower in d.lower():
                return d
        return response.strip()
    
    def update_probs_for_control(self, note: str, current_probs: Dict, history: List) -> Dict[str, float]:
        """Update probabilities for NON-EIG control condition."""
        history_text = "\n".join([f"Q: {q} A: {a}" for q, a in history])

        prompt = f"""
Patient Information: {note}
Current probabilities: {json.dumps(current_probs)}
All previous questions and answers: {history_text}

Task: Update the probability (0-1) for EACH disease in the list below.
Disease List: {self.disease_list}

Output strictly JSON format: {{ "Disease1": 0.1, ... }}
Ensure all probabilities sum to 1.
"""
        data = self._call_llm_json(prompt, temperature=0.0, max_retries=3)
        
        total = sum(data.values()) if data else 0
        if total == 0: 
            print("Warning: Empty update, keeping previous probs")
            return current_probs  # Return previous probs instead of uniform
        
        normalized = {}
        for d in self.disease_list:
            val = data.get(d, 0.0)
            normalized[d] = val / total
            
        return normalized


# ==========================================
# EXPERIMENT FUNCTIONS
# ==========================================
def run_experiment(
        agent: MedicalAgent, 
        threshold: float,
        true_diagnosis: str, 
        note: str,
        current_probs: Dict,
        max_try: int, 
        session_result_init: Dict, 
        control: bool = False):
    """
    Run experiment with probability tracking.
    control=True: NON-EIG (LLM picks questions)
    control=False: EIG (select best question by EIG)
    """
    print(f"--- TRUE DIAGNOSIS: {true_diagnosis} ---")

    history = []
    final_res = None

    if control:
        session_result_init['condition'] = 'NON-EIG'

    for step in range(1, max_try + 1):
        # Sort and print Top Belief
        sorted_beliefs = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_d, top_p = sorted_beliefs[0]
        
        print(f"Step {step} | Top: {top_d} ({top_p:.1%})")
        
        # Stop condition
        if top_p >= threshold:
            print(f">>> Confidence >= {threshold}. Diagnosis Reached.")
            final_res = (top_d, top_p)
            session_result_init['converged'] = True
            session_result_init['correct'] = top_d == true_diagnosis
            session_result_init['num_try'] = step 
            break

        ## If using EIG:
        if not control:
            # 2. Propose Questions
            questions = agent.propose_binary_questions(note, history, current_probs)
            if not questions:
                print("No questions proposed. Retrying...")
                continue

            # 3. Select Best Question (EIG)
            best_q = None
            best_eig = -1
            best_likelihoods = None 
            
            print("  Scanning questions for EIG...")
            for q in questions:
                # Skip duplicates
                if any(q in old_q for old_q, _ in history): 
                    continue
                
                # Get Matrix L
                l_matrix = agent.get_likelihood_matrix(q, agent.disease_list)
                
                # Calculate EIG
                eig = calculate_eig(current_probs, l_matrix)
                print(f"  Q: '{q}' -> EIG: {eig:.4f}") 
                
                if eig > best_eig:
                    best_eig = eig
                    best_q = q
                    best_likelihoods = l_matrix
            
            if not best_q: 
                print("No valid new questions found.")
                break
                
            # 4. Get Answer
            ans_str = agent.answer_simulation(best_q, true_diagnosis, note)
            print(f"  >>> ASKED: '{best_q}' -> {ans_str}\n")
            history.append((best_q, ans_str))
            
            # 5. BAYESIAN UPDATE
            is_yes = (ans_str == "YES")
            current_probs = bayesian_update(current_probs, best_likelihoods, is_yes)
        
        # If control (NON-EIG with prob tracking)
        else:
            # 2. Propose Question
            question = agent.propose_one_binary_question_for_control(note, history, current_probs)
            
            # 4. Get Answer
            ans_str = agent.answer_simulation(question, true_diagnosis, note)
            print(f"  >>> ASKED: '{question}' -> {ans_str}\n")
            history.append((question, ans_str))
            
            # 5. Prob updates (LLM-based, not Bayesian)
            current_probs = agent.update_probs_for_control(note, current_probs, history)

    if not final_res:
        print("Max steps reached without convergence")
        top_d, top_p = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)[0]
        session_result_init['correct'] = top_d == true_diagnosis
        final_res = (top_d, top_p)
        
    print(f"Final Result: {final_res}")
    return session_result_init


def run_experiment_no_dist(
        agent: MedicalAgent, 
        true_diagnosis: str, 
        note: str,
        max_try: int, 
        session_result_init: Dict):
    """
    PURE LLM WITHOUT TRACKING DISTRIBUTIONS
    """
    print(f"--- TRUE DIAGNOSIS: {true_diagnosis} ---")

    history = []
    # INITIAL ESTIMATE
    estimated_diagnosis = agent.guess_answer_control_no_dist(note, history)

    session_result_init['converged'] = 'N/A'
    final_res = None
    session_result_init['condition'] = 'NON-EIG-NO-DIST'

    for step in range(1, max_try + 1):
        print(f"Step {step} | Current guess: {estimated_diagnosis}")
        
        # Stop condition
        if estimated_diagnosis == true_diagnosis:
            session_result_init['num_try'] = step 
            session_result_init['correct'] = True 
            final_res = estimated_diagnosis
            print(f">>> Correct Diagnosis Reached.")              
            break

        # 2. Propose Question
        question = agent.propose_one_binary_question_for_control_without_dist(note, history)
        
        # 4. Get Answer
        ans_str = agent.answer_simulation(question, true_diagnosis, note)
        print(f"  >>> ASKED: '{question}' -> {ans_str}\n")
        
        history.append((question, ans_str))

        estimated_diagnosis = agent.guess_answer_control_no_dist(note, history)

    if not final_res:
        print("Max steps reached without convergence")
        session_result_init['correct'] = estimated_diagnosis == true_diagnosis
        
    print(f"Final Result: {final_res}")
    return session_result_init