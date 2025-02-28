from teacher_tester.run_conversation import run_conversation

# Run a conversation with a tester of rating 7.5 on Python Programming
conversation = run_conversation("Python Programming", 7.5)

# Get results
print(f"Predicted rating: {conversation.predicted_rating}")
print(f"True rating: {conversation.true_rating}")
print(f"Final confidence: {conversation.final_confidence}")