#!/usr/bin/env python3
"""
Banner Prompt Generator

This script generates a prompt for creating a LinkedIn banner image using OpenAI's ChatGPT.
It reads the podcast script and uses a template to create an optimized image generation prompt.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Try to load .env from the script's directory or current working directory
    script_dir = Path(__file__).parent.absolute()
    env_file = script_dir / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Fall back to default behavior (searches from cwd up)
        load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv", file=sys.stderr)
    print("Continuing without .env file support...", file=sys.stderr)

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package is required. Install with: pip install openai")
    sys.exit(1)

class BannerPromptGenerator:
    def __init__(self, podcast_file: str, openai_api_key: str = None):
        self.podcast_file = podcast_file
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as argument.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Template for generating the prompt
        self.prompt_template = """Given is the following summary of biotechnology news: 



{PODCAST_SCRIPT}



Generate a precisely worded prompt that we can use to generate a LinkedIn banner image (or any landscape infographic-style image) that visually summarizes the content. 

CRITICAL REQUIREMENTS:
- The image must contain ABSOLUTELY NO text, writing, words, letters, numbers, or any form of written language
- The image must be purely visual - use symbols, icons, illustrations, colors, and visual metaphors only
- Do not include any labels, captions, titles, or text overlays
- The prompt should explicitly state "no text", "no writing", "no words", "no letters" to ensure the image generator understands this requirement
- Optimize the prompt for a LinkedIn banner with the aspect ratio of 1584×396

The generated prompt should be suitable for DALL-E, Midjourney, or similar AI image generators."""
    
    def read_podcast_script(self) -> str:
        """Read the podcast script from file."""
        if not os.path.exists(self.podcast_file):
            raise FileNotFoundError(f"Podcast file not found: {self.podcast_file}")
        
        try:
            with open(self.podcast_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Error reading podcast file: {e}")
    
    def generate_image_prompt(self) -> str:
        """Generate an image generation prompt using OpenAI ChatGPT."""
        # Read podcast script
        podcast_script = self.read_podcast_script()
        
        # Create the prompt using the template
        system_prompt = self.prompt_template.format(PODCAST_SCRIPT=podcast_script)
        
        try:
            # Call OpenAI API to generate the image prompt
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better prompt generation
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating image generation prompts for DALL-E and other AI image generators. You create concise, visual, and detailed prompts that capture the essence of content using ONLY visual elements. You NEVER include text, writing, words, letters, numbers, or any form of written language in your prompts. Your prompts explicitly state 'no text', 'no writing', 'no words', 'no letters' to ensure the generated images are purely visual."
                    },
                    {
                        "role": "user",
                        "content": system_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            generated_prompt = response.choices[0].message.content.strip()
            
            # Post-process: Ensure explicit no-text instructions are included
            generated_prompt = self.ensure_no_text_instructions(generated_prompt)
            
            return generated_prompt
            
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {e}")
    
    def ensure_no_text_instructions(self, prompt: str) -> str:
        """Ensure the prompt explicitly excludes all forms of text and writing."""
        # Check if prompt already contains no-text instructions
        no_text_keywords = ['no text', 'no writing', 'no words', 'no letters', 'no text overlay', 
                           'without text', 'text-free', 'no typography', 'no labels']
        
        prompt_lower = prompt.lower()
        has_no_text_instruction = any(keyword in prompt_lower for keyword in no_text_keywords)
        
        # If not explicitly stated, append clear instructions
        if not has_no_text_instruction:
            # Append negative instructions at the end
            negative_instructions = (
                "\n\nIMPORTANT: The image must contain absolutely no text, writing, words, "
                "letters, numbers, labels, captions, or any form of written language. "
                "The image should be purely visual with symbols, icons, and illustrations only."
            )
            prompt = prompt + negative_instructions
        
        return prompt
    
    def save_prompt(self, prompt: str, output_file: str):
        """Save the generated prompt to a file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            print(f"Banner image prompt saved to: {output_file}")
            
        except Exception as e:
            raise IOError(f"Error saving prompt file: {e}")

def main():
    """Main function to handle command line arguments and execute prompt generation."""
    parser = argparse.ArgumentParser(
        description="Generate a prompt for creating a LinkedIn banner image using OpenAI ChatGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python banner_prompt_generator.py --podcast podcast_script.txt
  python banner_prompt_generator.py --podcast podcast_script.txt --output banner_prompt.txt
  python banner_prompt_generator.py --podcast podcast_script.txt --api-key YOUR_API_KEY
        """
    )
    
    parser.add_argument('--podcast', '-p', required=True,
                       help='Podcast script file (required)')
    parser.add_argument('--output', '-o', default='banner_image_prompt.txt',
                       help='Output file for the generated prompt (default: banner_image_prompt.txt)')
    parser.add_argument('--api-key', '-k',
                       help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    try:
        # Check if API key is available before proceeding
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            error_msg = (
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable, add it to .env file, "
                "or pass it with --api-key argument."
            )
            print(f"Error: {error_msg}", file=sys.stderr)
            sys.exit(1)
        
        # Initialize generator
        generator = BannerPromptGenerator(args.podcast, args.api_key)
        
        print(f"Generating banner image prompt from {args.podcast}")
        print("Calling OpenAI ChatGPT API...")
        
        # Generate prompt
        prompt = generator.generate_image_prompt()
        
        # Save prompt
        generator.save_prompt(prompt, args.output)
        
        print("\nGenerated prompt:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print("\nBanner prompt generation complete!")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

