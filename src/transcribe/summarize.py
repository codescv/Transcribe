import os
from google import genai

def generate_summary(text: str) -> str:
    """
    Generate a summary of the given text using Gemini API.
    Splits text if it exceeds max_chars to avoid context/lost-in-middle issues.
    Requires GEMINI_API_KEY environment variable.
    """
    max_chars = 100000 # ~20,000 words limit for map-reduce chunks
    
    try:
        client = genai.Client()
        
        # 1. Split text into chunks if it exceeds max_chars
        chunks = []
        if len(text) <= max_chars:
            chunks = [text]
        else:
            current_chunk = []
            current_len = 0
            for line in text.split("\n"):
                if current_len + len(line) + 1 > max_chars and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(line)
                current_len += len(line) + 1
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                
        # 2. Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = (
                "You are an expert summarizer. Please provide a concise summary of the following transcription segment. "
                "Identify the main topics discussed and highlight any key points or actions. "
                "Use clear headings and bullet points where appropriate.\n\n"
                "**Important**: The output language must be the same as the transcription.\n"
                f"Transcription Segment {i+1}/{len(chunks)}:\n"
                f"{chunk}"
            )
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            summaries.append(response.text)
            
        # 3. If single chunk, return description
        if len(summaries) == 1:
            return summaries[0]
            
        # 4. Combine multiple summaries into a single cohesive digest
        combined_text = "\n\n=== Segment Summary ===\n\n".join(summaries)
        combine_prompt = (
            "You are an expert summarizer. Below are summaries of individual segments of a larger transcription. "
            "Please consolidate them into a single, cohesive overall summary. "
            "Organize by main topics or chronological order of topics discussed. "
            "Use clear headings and bullet points.\n\n"
            "**Important**: The output language must be the same as the summaries.\n"
            "Segment Summaries:\n"
            f"{combined_text}"
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=combine_prompt,
        )
        return response.text

    except Exception as e:
        return f"Error generating summary: {e}"
