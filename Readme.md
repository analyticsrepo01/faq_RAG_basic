FAQ Chatbot (Streamlit + Vertex AI)

This Streamlit application provides a user-friendly chatbot interface that leverages Vertex AI's generative models and your SingPost knowledge base.

Key Features

CSV Knowledge Base: Upload a CSV file containing questions, answers, and optional image references.
Hybrid Retrieval: Combines BM25 and FAISS retrieval methods for enhanced accuracy in finding relevant information.
Vertex AI Integration: Uses Vertex AI's powerful Gemini model to generate informative and contextually relevant answers to user queries.
Image Display: If available, displays an image associated with the retrieved information.
Customizable: Easily adapt the PROJECT_ID and data_store_path variables to match your Vertex AI configuration.
How to Use

Install Requirements:

Bash
pip install streamlit pandas vertexai langchain langchain_google_vertexai langchain_community
Use code with caution.
content_copy
Prepare Your CSV:

Ensure your CSV has columns named question, answer, and (optionally) image_references.
Each row should contain a question, its corresponding answer, and the URL of a relevant image (if applicable).
Run the App:

Bash
streamlit run your_app_name.py
Use code with caution.
content_copy
Replace your_app_name.py with the filename of your script.

Interact:

Upload your CSV file through the app.
Type your questions in the input box.
Get informative responses, along with images where relevant.
Important Notes

Replace placeholders: Fill in your actual PROJECT_ID and data_store_path values.
Customize prompts: Modify the system_prompt to tailor the chatbot's behavior.
Example CSV Structure

question	answer	image_references
How do I track my package?	You can track your package on our website using the tracking number provided in your shipping confirmation email.	https://www.singpost.com/track-items
What are SingPost's rates?	Our rates vary depending on the destination, weight, and dimensions of your package. Please visit our website for a detailed pricing guide.	[invalid URL removed]
What is vPost?	vPost is our international shipping service that allows you to shop from overseas online stores and have your purchases delivered to your doorstep in Singapore.	https://www.vpost.com.sg/