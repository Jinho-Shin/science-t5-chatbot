import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer

trained_model = T5ForConditionalGeneration.from_pretrained("jhs0640/science_t5", return_dict=True)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def generate_answer(question):
  source_encoding = tokenizer(
      question["question"],
      max_length=396,
      padding="max_length",
      truncation="only_second",
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt"
  )
  generated_ids = trained_model.generate(
      input_ids=source_encoding["input_ids"],
      attention_mask=source_encoding["attention_mask"],
      num_beams=1,
      max_length=200,
      repetition_penalty=2.5,
      length_penalty=1.0,
      early_stopping=True,
      use_cache=True
  )

  preds = [
      tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
      for generated_id in generated_ids
  ]

  return "".join(preds)
############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        query= text
        response = generate_answer({"question": text})
        output.append(response)
    
    return SimpleText(dict(text=output))
