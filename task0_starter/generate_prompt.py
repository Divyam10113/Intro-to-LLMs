import pandas as pd

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
    print("Loading ranked papers...")
    df = pd.read_csv('ranked_papers.csv')
    
    # Filter to only the 14 selected papers
    selected_df = df[df['Title'].isin(SELECTED_TITLES)].copy()
    print(f"Found {len(selected_df)} out of {len(SELECTED_TITLES)} selected papers.")
    
    # Format the context for the LLM
    context_blocks = []
    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        block = f"[{i}] Title: {row['Title']}\n"
        block += f"Year: {row.get('Year', 'N/A')}\n"
        block += f"Venue/Authors: {row.get('Venue', 'N/A')} / {row.get('Authors', 'N/A')}\n"
        block += f"Abstract: {row['Abstract']}\n"
        context_blocks.append(block)
    
    context_string = "\n\n".join(context_blocks)
    
    # Construct the final prompt text
    final_prompt = (
        "You are an expert AI researcher writing a comprehensive literature review. "
        "Your task is to write a literature review on 'Generalizable vision-language models "
        "for zero-shot/few-shot anomaly detection in medical images'.\n\n"
        "Constraints:\n"
        "- Synthesize the research by approaches, challenges, and themes, rather than simply listing them one-by-one.\n"
        "- You MUST use numbered citations (e.g., [1], [3], [4]) corresponding EXACTLY to the provided papers in the context.\n"
        "- Do not hallucinate or reference any papers outside of the provided context.\n"
        "- Make the review highly technical, analytical, and insightful.\n\n"
        "--------------------------------------------------\n"
        "CONTEXT (THE 14 PAPERS):\n"
        "--------------------------------------------------\n\n"
        f"{context_string}\n\n"
        "--------------------------------------------------\n"
        "Please generate the thematic literature review based ONLY on the papers provided above."
    )

    # Save the prompt to a text file
    output_file = "llm_prompt.txt"
    with open(output_file, "w") as f:
        f.write(final_prompt)
        
    print(f"\nSuccess! The prompt has been generated and saved to {output_file}")
    print("You can now copy the contents of this file and paste it into ChatGPT, Claude, etc.")

if __name__ == "__main__":
    main()
