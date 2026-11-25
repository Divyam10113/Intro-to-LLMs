import pandas as pd
from openai import OpenAI
import os
import sys

# The 14 manually selected highly-relevant papers
SELECTED_TITLES = [
    "SD-MAD: Sign-Driven Few-shot Multi-Anomaly Detection in Medical Images",
    "MediCLIP: Adapting CLIP for Few-shot Medical Image Anomaly Detection",
    "Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images",
    "MadCLIP: Few-shot Medical Anomaly Detection with CLIP",
    "UltraAD: Fine-Grained Ultrasound Anomaly Classification via Few-Shot CLIP Adaptation",
    "Exploring Zero-Shot Anomaly Detection with CLIP in Medical Imaging: Are We There Yet?",
    "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection",
    "AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation",
    "Anomaly-Aware Vision-Language Adapters for Zero-Shot Anomaly Detection",
    "GlocalCLIP: Object-agnostic Global-Local Prompt Learning for Zero-shot Anomaly Detection",
    "AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection",
    "CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection",
    "Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models",
    "Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts"
]

def main():
    # 1. Setup API configuration
    api_key = os.environ.get("AGENTROUTER_API_KEY")
    if not api_key:
        print("Error: Please set the AGENTROUTER_API_KEY environment variable.")
        sys.exit(1)

    # We are returning to the OpenAI client targeting the /v1 endpoint,
    # as this is the most stable and standard way to communicate with AgentRouter.
    client = OpenAI(
        base_url="https://agentrouter.org/v1",
        api_key=api_key
    )

    # 2. Load the papers
    print("Loading ranked papers...")
    df = pd.read_csv('ranked_papers.csv')
    
    # Filter to only the 14 selected papers
    selected_df = df[df['Title'].isin(SELECTED_TITLES)].copy()
    print(f"Found {len(selected_df)} out of {len(SELECTED_TITLES)} selected papers.")
    
    # 3. Format the context for the LLM
    context_blocks = []
    # We use enumerate starting at 1 so the LLM cites them as [1], [2], etc.
    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        block = f"[{i}] Title: {row['Title']}\n"
        block += f"Year: {row.get('Year', 'N/A')}\n"
        block += f"Venue/Authors: {row.get('Venue', 'N/A')} / {row.get('Authors', 'N/A')}\n"
        block += f"Abstract: {row['Abstract']}\n"
        context_blocks.append(block)
    
    context_string = "\n\n".join(context_blocks)
    
    # 4. Construct the Prompts
    system_prompt = (
        "You are an expert AI researcher writing a comprehensive literature review. "
        "Your task is to write a literature review on 'Generalizable vision-language models "
        "for zero-shot/few-shot anomaly detection in medical images'.\n"
        "Constraints:\n"
        "- Synthesize the research by approaches, challenges, and themes, rather than simply listing them one-by-one.\n"
        "- You MUST use numbered citations (e.g., [1], [3], [4]) corresponding EXACTLY to the provided papers in the context.\n"
        "- Do not hallucinate or reference any papers outside of the provided context.\n"
        "- Make the review highly technical, analytical, and insightful."
    )

    user_prompt = f"Please generate the thematic literature review based on the following 14 papers:\n\n{context_string}"

    # You can change this to "claude-opus-5" or any other supported model!
    model_name = "gpt-5.6-sol"
    
    print(f"Calling AgentRouter API (model: {model_name})...")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        if hasattr(response, 'choices'):
            review_text = response.choices[0].message.content
        else:
            print("WARNING: API returned a non-standard response. Dumping raw string.")
            review_text = str(response)
        
        # 5. Save the output
        output_file = "generated_literature_review.md"
        with open(output_file, "w") as f:
            f.write(review_text)
            
        print(f"\nSuccess! Literature review saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred while calling the API: {e}")

if __name__ == "__main__":
    main()
