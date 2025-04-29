"""
Script to test the teacher-tester conversation system.
"""
import argparse
import sys
from typing import Optional
from teacher_tester.agents.teacher import TeacherAgent
from teacher_tester.agents.tester import TesterAgent
from teacher_tester.data.schemas import Conversation
from teacher_tester.data.storage import get_storage
from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger
from teacher_tester.run_conversation import run_conversation

logger = setup_logger(__name__)

def display_conversation(conversation: Conversation) -> None:
    """
    Display a conversation in a readable format.
    
    Args:
        conversation: The conversation to display
    """
    print("\n" + "="*80)
    print(f"CONVERSATION: {conversation.id}")
    print(f"Subject: {conversation.subject}")
    print(f"Tester Rating: {conversation.true_rating}")
    print("="*80)
    
    for i, msg in enumerate(conversation.messages):
        # Clean up confidence lines before display
        content = msg.content
        if msg.role == "teacher" and "Confidence:" in content:
            # Extract confidence line separately
            lines = content.split('\n')
            confidence_line = None
            other_lines = []
            
            for line in lines:
                if line.startswith("Confidence:"):
                    confidence_line = line
                else:
                    other_lines.append(line)
            
            # Show confidence separately, then the actual message
            if confidence_line:
                print(f"\n[{msg.role.upper()} - {confidence_line}]")
            else:
                print(f"\n[{msg.role.upper()} {i+1}]")
            print("\n".join(other_lines).strip())
        else:
            print(f"\n[{msg.role.upper()} {i+1}]")
            print(content)
    
    print("\n" + "="*80)
    print(f"FINAL ASSESSMENT:")
    print(f"True Rating: {conversation.true_rating}")
    print(f"Predicted Rating: {conversation.predicted_rating}")
    print(f"Final Confidence: {conversation.final_confidence}")
    print(f"Error: {abs(conversation.true_rating - conversation.predicted_rating):.2f}")
    print(f"Terminated Reason: {conversation.terminated_reason}")
    print("="*80 + "\n")

def main():
    """Run a test conversation and display results."""
    parser = argparse.ArgumentParser(description="Run a test conversation between teacher and tester agents.")
    parser.add_argument("--subject", type=str, default="Python Programming", 
                        help="Subject for the conversation")
    parser.add_argument("--rating", type=float, default=7.5, 
                        help="True rating of the tester (0-10)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Display the full conversation")
    
    args = parser.parse_args()
    
    print(f"\nRunning conversation on {args.subject} with tester rating {args.rating}...\n")
    
    # Run the conversation
    conversation = run_conversation(args.subject, args.rating)
    
    # Display results
    if args.verbose:
        display_conversation(conversation)
    else:
        print(f"Conversation ID: {conversation.id}")
        print(f"Subject: {conversation.subject}")
        print(f"True Rating: {conversation.true_rating}")
        print(f"Predicted Rating: {conversation.predicted_rating}")
        print(f"Final Confidence: {conversation.final_confidence}")
        print(f"Error: {abs(conversation.true_rating - conversation.predicted_rating):.2f}")
        print(f"Number of exchanges: {len([m for m in conversation.messages if m.role == 'teacher'])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())