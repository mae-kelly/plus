import openai
import json
from typing import List, Dict, Any
from snorkel.labeling import LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class LLMClassifier:
    def __init__(self, config: Dict):
        self.api_key = config.get('openai_api_key')
        self.model = config['model_config'].get('llm_model', 'gpt-4')
        self.batch_size = config['model_config'].get('llm_batch_size', 50)
        self.weak_supervision_ratio = config['model_config'].get('weak_supervision_ratio', 0.013)
        
        if self.api_key:
            openai.api_key = self.api_key
        
        self.label_model = LabelModel(cardinality=78, verbose=False)
        self.labeling_functions = []
    
    async def generate_labeling_functions(self, column_samples: Dict[str, List]) -> List[LabelingFunction]:
        prompts = self._create_prompts(column_samples)
        
        functions = []
        for prompt in prompts[:10]:
            try:
                response = await self._call_llm(prompt)
                function_code = self._parse_response(response)
                
                if function_code:
                    lf = self._create_labeling_function(function_code)
                    functions.append(lf)
                    
            except Exception as e:
                logger.debug(f"Failed to generate labeling function: {e}")
        
        self.labeling_functions.extend(functions)
        return functions
    
    async def classify_with_llm(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        prompt = self._create_classification_prompt(column_name, values)
        
        try:
            response = await self._call_llm(prompt)
            return self._parse_classification(response)
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {'type': 'unknown', 'confidence': 0.0}
    
    def _create_prompts(self, column_samples: Dict[str, List]) -> List[str]:
        prompts = []
        
        for column_name, values in list(column_samples.items())[:self.batch_size]:
            sample_values = values[:10]
            
            prompt = f"""Generate a Python labeling function to detect the semantic type of this column:
Column: {column_name}
Sample values: {sample_values}

Return a function that takes a pandas Series and returns:
- 0 for hostname/server
- 1 for IP address  
- 2 for email
- 3 for timestamp
- 4 for numeric ID
- -1 for unknown/other

def labeling_function(series):
    # Your code here
    pass
"""
            prompts.append(prompt)
        
        return prompts
    
    def _create_classification_prompt(self, column_name: str, values: List[Any]) -> str:
        sample_values = [str(v) for v in values[:20] if v]
        
        return f"""Classify this database column's semantic type:

Column name: {column_name}
Sample values:
{chr(10).join(sample_values)}

Possible types: hostname, ip_address, email, phone, address, person_name, organization, 
timestamp, date, numeric_id, text_id, amount, percentage, url, file_path, unknown

Return JSON: {{"type": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""
    
    async def _call_llm(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data type classification expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return ""
    
    def _parse_response(self, response: str) -> str:
        try:
            if 'def ' in response:
                start = response.find('def ')
                end = response.find('\n\n', start)
                if end == -1:
                    end = len(response)
                return response[start:end]
            return ""
            
        except Exception:
            return ""
    
    def _parse_classification(self, response: str) -> Dict[str, Any]:
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
                
        except Exception:
            pass
        
        return {'type': 'unknown', 'confidence': 0.0}
    
    def _create_labeling_function(self, function_code: str) -> LabelingFunction:
        local_scope = {}
        exec(function_code, {}, local_scope)
        
        func = local_scope.get('labeling_function')
        if func:
            return LabelingFunction(name=f"lf_{len(self.labeling_functions)}", f=func)
        return None
    
    def train_weak_supervision(self, df: pd.DataFrame):
        if not self.labeling_functions:
            return
        
        applier = PandasLFApplier(self.labeling_functions)
        L = applier.apply(df)
        
        self.label_model.fit(L)
        
        predictions = self.label_model.predict(L)
        confidences = self.label_model.predict_proba(L)
        
        return predictions, confidences