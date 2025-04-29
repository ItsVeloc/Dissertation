"""
DeepSeek-R1 Chatbot with Continuous Conversation Saving
Optimized for GPU memory usage and stability
"""

import os
# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gc
from config import MODEL_PATH, SAVE_PATH

# Check for CUDA, fallback to CPU if unavailable
DEVICE = "cuda" if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError("CUDA is not available"))

def load_model():
    """Load model and tokenizer with error handling and memory optimizations"""
    try:
        print(f"Loading model from {MODEL_PATH}...")
        print(f"Using device: {DEVICE}")
        
        # Log available GPU memory if using CUDA
        if DEVICE == "cuda":
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            print(f"Free GPU memory: {free_memory / (1024**3):.2f} GB")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,   # Use bfloat16 precision
            load_in_8bit=True,            # Use 8-bit quantization
            device_map="auto",            # Let accelerate handle device mapping
            low_cpu_mem_usage=True        # Reduce CPU memory usage during loading
        )
        
        print("Model loaded successfully")
        return tokenizer, model
    except Exception as e:
        print(f"\nERROR LOADING MODEL: {e}")
        print("\nTrying fallback options...")
        
        try:
            # Fallback option 1: CPU mode
            if DEVICE == "cuda":
                print("Attempting to load model on CPU instead...")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu"
                )
                return tokenizer, model
        except Exception as fallback_error:
            print(f"Fallback failed: {fallback_error}")
            input("Press Enter to exit...")
            raise SystemExit(1) from e

def initialize_pipeline(tokenizer, model):
    """Create text generation pipeline with optimized settings"""
    try:
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # Remove the device parameter since accelerate handles it
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"\nERROR INITIALIZING PIPELINE: {e}")
        input("Press Enter to exit...")
        raise SystemExit(1) from e

def cleanup_memory():
    """Clean up GPU memory"""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

def main():
    """Main chat loop execution with memory management"""
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
            
            # Generate with reduced token count to save memory
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
            
            # Clean up memory after each response
            cleanup_memory()

        except KeyboardInterrupt:
            print("\nExiting conversation...")
            break
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("Resetting conversation...")
            conversation_history = []
            cleanup_memory()

    print(f"\nConversation history saved to: {SAVE_PATH}")
    input("Press Enter to close window...")

if __name__ == "__main__":
    main()