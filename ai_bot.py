import cohere

class BrainTumorAIBot:
    def __init__(self, api_key: str = "Ff74gn0LI9T"):
        self.co = cohere.Client(api_key)
        self.chat_history = []

    def chat(self, message: str, tumor_context=None) -> str:
        try:
            # Format message for Cohere API
            formatted_message = {
                "role": "user",
                "content": message
            }
            
            # Add user's message to history
            self.chat_history.append(formatted_message)

            # Call Cohere's chat endpoint
            response = self.co.chat(
                message=message,
                chat_history=self.chat_history,
                model="command-r-plus",
                temperature=0.5,
                max_tokens=500
            )

            # Add assistant's reply to history
            assistant_message = {
                "role": "assistant",
                "content": response.text
            }
            self.chat_history.append(assistant_message)

            return response.text

        except Exception as e:
            return f"Error: {str(e)}"

    def generate_response(self, tumor_type=None, location=None, size=None):
        """Generate a response about the tumor"""
        prompt = f"Tell me about {tumor_type} tumors"
        if location:
            prompt += f" in the {location}"
        if size:
            prompt += f" of size {size}"
        
        return self.chat(prompt)

    def get_treatment_options(self, tumor_type):
        """Get treatment options for a tumor type"""
        return self.chat(f"What are the treatment options for {tumor_type} tumors?")

    def get_symptoms(self, tumor_type, location=None):
        """Get symptoms for a tumor type and location"""
        prompt = f"What are the symptoms of {tumor_type} tumors"
        if location:
            prompt += f" in the {location}"
        return self.chat(prompt)

    def generate_visualization_data(self, tumor_type):
        """Generate data for visualizations"""
        return self.chat(f"Provide visualization data for {tumor_type} tumors")

# Example usage
if __name__ == "__main__":
    # Create bot instance
    bot = BrainTumorAIBot()
    
    # Simple chat example
    response = bot.chat("Tell me about pituitary tumors")
    print(response)
