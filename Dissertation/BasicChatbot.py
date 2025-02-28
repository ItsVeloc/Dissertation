"""
DeepSeek-R1 Chatbot with Continuous Conversation Saving
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from config import MODEL_PATH, SAVE_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load model and tokenizer with error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16
        ).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f"\nERROR LOADING MODEL: {e}")
        input("Press Enter to exit...")
        raise SystemExit(1) from e

def initialize_pipeline(tokenizer, model):
    """Create text generation pipeline"""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        torch_dtype=torch.bfloat16
    )

def main():
    """Main chat loop execution"""
    print("\nInitializing DeepSeek-R1...")
    tokenizer, model = load_model()
    pipe = initialize_pipeline(tokenizer, model)
    
    # Create directory for SAVE_PATH if it doesn't exist
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    conversation_history = []
    print("\nChat ready! Type 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ('exit', 'quit'):
                break
                
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response
            prompt = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )
            
            outputs = pipe(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Extract and store response
            full_response = outputs[0]['generated_text']
            if len(full_response) <= len(prompt):
                print("Warning: Empty response from model")
                continue
                
            model_response = full_response[len(prompt):].strip()
            conversation_history.append({"role": "assistant", "content": model_response})
            
            # Save and display
            with open(SAVE_PATH, "a", encoding="utf-8") as f:
                f.write(f"You: {user_input}\nAssistant: {model_response}\n{'='*50}\n")
            
            print(f"\nAssistant: {model_response}\n")

        except KeyboardInterrupt:
            print("\nExiting conversation...")
            break
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("Resetting conversation...")
            conversation_history = []

    print(f"\nConversation history saved to: {SAVE_PATH}")
    input("Press Enter to close window...")

if __name__ == "__main__":
    main()