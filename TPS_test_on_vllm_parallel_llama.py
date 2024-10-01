import time
from vllm import LLM, SamplingParams
import torch
import wandb
import os
import asyncio
import aiohttp

# For multi-GPU usage set visible gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"



torch.set_num_threads(24)

prompts = [ 
    "Tell me a joke.",
    "What is the capital of France?",
    "Explain quantum physics in simple terms.",
    "What is your favorite book?",
    "How do you make a perfect cup of coffee?",
    "Who was the first person to walk on the moon?",
    "Describe the process of photosynthesis.",
    "What is the theory of relativity?",
    "Write a short poem about nature.",
    "What are the benefits of meditation?",
    "How does blockchain technology work?",
    "What is the significance of the Great Wall of China?",
    "Explain the importance of renewable energy.",
    "What are the causes of climate change?",
    "Describe the basic principles of economics.",
    "How do vaccines work?",
    "What are the major components of a computer?",
    "Explain the concept of artificial intelligence.",
    "What is the difference between a virus and bacteria?",
    "What are the key factors in a successful business?",
    "How can we improve mental health awareness?",
    "What are the advantages of learning a second language?",
    "Describe the structure of an atom.",
    "What is the significance of the Renaissance period?",
    "How do social media platforms impact society?",
    "What is the role of government in the economy?",
    "Explain the concept of supply and demand.",
    "What are the main types of renewable energy?",
    "Describe the impact of technology on education.",
    "What are the benefits of a healthy diet?",
    "Explain how a bill becomes a law.",
    "What are the effects of pollution on wildlife?",
    "What is the history of the Internet?",
    "Describe the life cycle of a butterfly.",
    "What are the causes and effects of deforestation?",
    "Explain the concept of democracy.",
    "What is the role of DNA in heredity?",
    "Describe the impact of music on culture.",
    "What are the principles of design?",
    "Explain the importance of water conservation.",
    "What is the role of art in society?",
    "Describe the process of learning a new skill.",
    "What are the characteristics of effective leadership?",
    "How do different cultures approach conflict resolution?",
    "What are the psychological effects of color?",
    "Describe the impact of climate change on agriculture.",
    "What are the causes of poverty?",
    "How do natural disasters affect communities?",
    "Explain the significance of space exploration.",
    "What are the ethical implications of genetic engineering?",
    "Describe the concept of time management.",
    "What are the benefits of teamwork?",
    "Explain the role of the United Nations.",
    "What is the significance of cultural diversity?",
    "Describe the process of globalization.",
    "What are the effects of globalization on local economies?",
    "Explain the importance of critical thinking.",
    "What is the significance of the scientific method?",
    "Describe the relationship between art and politics.",
    "Write a short story about a time traveler who visits ancient Egypt.",
    "What are the key differences between classical and quantum physics?",
    "Explain how machine learning models can be used in healthcare.",
    "Describe the process of photosynthesis in plants.",
    "Create a dialogue between two aliens discussing Earth’s technology.",
    "What are the pros and cons of renewable energy sources?",
    "Generate a poem about the changing seasons.",
    "Write an email requesting a meeting with a CEO.",
    "What are the ethical implications of AI in military applications?",
    "Explain the concept of blockchain in simple terms.",
    "How does the brain process emotions during stressful situations?",
    "Describe a future where humans live on Mars.",
    "Write a product description for a smart home device.",
    "Summarize the plot of the novel '1984' by George Orwell.",
    "Explain how the internet works in layman's terms.",
    "Generate a motivational speech for a sports team before a big game.",
    "What are the main challenges of space exploration?",
    "Describe the impact of social media on modern communication.",
    "Write a recipe for a simple pasta dish.",
    "What is the theory of relativity, and why is it important?",
    "Explain the benefits of mindfulness and meditation.",
    "Create a fictional conversation between two historical figures.",
    "Describe the economic impacts of climate change.",
    "What are the main principles of democracy?",
    "Write a letter to a future version of yourself.",
    "Explain the basic structure of the human immune system.",
    "What are the key ingredients of a successful business plan?",
    "Write a news article about a recent scientific discovery.",
    "How can artificial intelligence be used in education?",
    "Describe the role of cybersecurity in protecting personal data.",
    "Write a movie review for a film you recently watched.",
    "What are the advantages and disadvantages of electric cars?",
    "Explain how vaccines work to protect the body from diseases.",
    "Describe a day in the life of a professional athlete.",
    "Write a dialogue between a human and a robot in the year 2100.",
    "What are the main causes and effects of deforestation?",
    "Create a story about a detective solving a mystery in a futuristic city.",
    "Explain the process of starting your own business.",
    "What are the key components of a balanced diet?",
    "Write an essay on the importance of environmental conservation.",
    "What is quantum computing, and how does it differ from classical computing?",
    "Describe the cultural significance of music in human history.",
    "Explain the major milestones in the development of the internet.",
    "Write a letter of advice to someone going through a difficult time.",
    "What are the psychological effects of working from home?",
    "Create a dialogue between two scientists debating climate change.",
    "Explain how renewable energy can reduce carbon emissions.",
    "Describe a utopian society and its core principles.",
    "What are the latest trends in artificial intelligence?",
    "Write a short story about a child discovering a hidden magical world."
]

#Creating vllm server with 2 gpu
#python3 -m vllm.entrypoints.api_server --model "meta-llama/Meta-Llama-3.1-8B-Instruct" --port 8000 --tensor-parallel-size 2

# Define the vLLM server URL - Add your IP from ssh here
VLLM_SERVER_URL = "http://10.205.205.27:8000/generate"





# Define an asynchronous function to send a prompt to vLLM
async def send_prompt(prompt, session):
    payload = {
        "prompt": prompt,
        "temperature": 0.1,  # Adjust generation settings if needed
        "max_tokens": 100,
        "top_p":0.1
    }
    async with session.post(VLLM_SERVER_URL, json=payload) as response:
        result = await response.json()
        return result




async def measure_tps(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()  # Start time for TPS measurement
        for prompt in prompts:
            tasks.append(send_prompt(prompt, session))

        # Gather results (send all prompts concurrently)
        responses = await asyncio.gather(*tasks)
        
        # End time for TPS measurement
        end_time = time.time()
        total_time = end_time - start_time  # Total time taken to process 50 prompts


        # Calculate TPS (Throughput Per Second)
        tps = len(responses) / total_time
        print(f"Throughput Per Second (TPS): {tps:.2f}")

        # Calculate TokensPS (Tokens Per Second)
        tps = len(responses) / total_time *100 *0.75
        print(f"Tokens Per Second (TPS): {tps:.2f}")
        


# Main function to run the parallel inference
async def main():
    await measure_tps(prompts)


# Run the asynchronous code
if __name__ == "__main__":
    asyncio.run(main())








