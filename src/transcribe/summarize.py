import os
from google import genai

def generate_summary(text: str) -> str:
    """
    Generate a summary of the given text using Gemini API.
    Requires GEMINI_API_KEY environment variable.
    """
    try:
        # Initialize client. It automatically picks up GEMINI_API_KEY
        client = genai.Client()
        
        prompt = (
            "You are an expert summarizer. Please provide a concise summary of the following transcription. "
            "Identify the main topics discussed and highlight any key points or actions. "
            "Use clear headings and bullet points where appropriate.\n\n"
            "**Important**: The output language must be the same as the transcription.\n"
            "Transcription:\n"
            f"{text}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"
