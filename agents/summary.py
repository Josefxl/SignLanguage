import logging
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaryAgent:
    def __init__(self, model_name="google/flan-t5-base"):
        logger.info(f"Loading summary model: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )

            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                num_beams=4,
                device=device
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.error(f"Failed to load summary model: {e}")
            raise

        prompt_template = """
You are a language learning assistant. Based on the following lesson performance log,
generate a concise summary of how the student did. Highlight strengths, weaknesses,
patterns, and suggest one area for improvement. Be encouraging.

Log:
{log}
"""

        prompt = PromptTemplate.from_template(prompt_template)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def format_log(self, session_log):
        formatted = ""
        for entry in session_log:
            formatted += f"- Prompt: {entry['prompt']}\n"
            formatted += f"  Gloss: {entry['gloss']}\n"
            formatted += f"  Caption: {entry['caption']}\n"
            formatted += f"  Similarity: {entry['similarity']:.2f}\n"
            formatted += f"  Result: {entry['result']}\n\n"
        return formatted.strip()

    def summarize(self, session_log):
        try:
            formatted_log = self.format_log(session_log)
            summary = self.chain.run({"log": formatted_log})
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed."
