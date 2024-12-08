import os
import re
import streamlit as st
from langchain.llms import HuggingFaceEndpoint
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.agents import load_tools, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import openai
from transformers import AutoTokenizer, pipeline
from google.cloud import translate_v2 as translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ''
os.environ['OPENAI_API_KEY'] = ''
HUGGINGFACEHUB_API_TOKEN = ""
os.environ["SERPER_API_KEY"] = ''
openai.api_key = ''
chromadb = "./db/chroma14"
upload_folder = "upload"
os.makedirs(upload_folder, exist_ok=True)

# Find which class prompt belongs to
tokenizer = AutoTokenizer.from_pretrained("bayartsogt/mongolian-gpt2")
tokenizer.pad_token = tokenizer.eos_token

generator = pipeline("text-generation", model="Ikhee10/khasbank_three_classifier_v13", tokenizer = tokenizer, device=0, num_beams=5)

translate_client = translate.Client()

def process_generated_text(text):
    matches = re.findall(r'<(.*?)>', text)
    ans = matches[1] if len(matches) > 0 else ''
    if ' ' in ans:
        ans = ans.split(' ')[0].strip()
    return ans
    
def generate_and_process(prompt):
    result = generator(prompt, max_length=32, pad_token_id=tokenizer.eos_token_id)
    generated_text = result[0]['generated_text']

    return process_generated_text(generated_text)


# Set up the embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Set up the vector store (Chroma)
try:
    vectorstore = Chroma(collection_name='v_db', persist_directory=chromadb, embedding_function=embeddings)
except Exception as e:
    print(f"Error Loading ChromaDB: {e}")

def chroma_retrieval_tool(query): 
    # Retrieve documents relevant to the query 
    query_mn = translate_client.translate(query, target_language="mn")["translatedText"]
    clas = generate_and_process('<s> bb: ' + query_mn)
    # TODO: noa
    if clas == "noa": 
        return "Асуулт нь хамааралгүй байна. Хариулах боломжгүй. Өөр дахиж хайх, хэрэггүй зогсоо."

    docs = vectorstore.similarity_search(query_mn, k=5, filter={"type": clas}) 
    context = "\n".join([doc.page_content for doc in docs]) 
    return context

# Load model function
repo_id = "meta-llama/Meta-Llama-3-70B-Instruct"

def load_model():
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.1,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    return llm

def get_agent_response(user_input):
    res = agent.run(user_input)
    res = translate_client.translate(res, target_language="mn")["translatedText"]
    return res

# Streamlit UI
# Apply custom CSS to lower the image
st.markdown("""
    <style>
        .lower-image {
            margin-top: 30px; /* Adjust this value to control the distance */
        }
    </style>
""", unsafe_allow_html=True)

# Create columns for image and text in one line
col1, col2 = st.columns([1, 8])  # Adjust the ratio (col1 for the image and col2 for the text)

with col1:
    # Apply the class to the image for custom margin
    st.markdown('<div class="lower-image">', unsafe_allow_html=True)
    st.image('./xacbank_logo-removebg-preview.png', width=80)  # Display image
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.header("Xacqa chatbot! 👋")  # Display header
    st.markdown("*IHH багийн бүтээсэн чатботод тавтай морил.*")

# Add a separator line between the text and input field
st.markdown("---")  # Horizontal line separator

# Title and input field for the user (input a bit lower)
query = st.text_input("Хурууны үзүүрээр банкаа удирд!")  # Adjusted input field text

# Check if the "Send" button is clicked
if st.button("Send"):
    llm = load_model()

    template = """Доорх асуултанд нөхөрсгөөр хариулаарай. Хариулахад дараах мэдээллийг ашиглана уу. Хэрэв та хариултаа мэдэхгүй бол мэдэхгүй гэж хэлээрэй, хариулт зохиох гэж бүү оролдоорой. Хамгийн ихдээ гурван өгүүлбэр ашигла. Хариултаа аль болох товчхон бичээрэй. Хариултын төгсгөлд "Асуусанд баярлалаа!" гэж үргэлж хэлээрэй.
        {context}
        Асуулт: {query}	
        Хэрвээ дараахи асуултуудийг асуувал хариулт гэсэн зааврийн дагуу хариул:
        Пин код мартсан    Хариулт:Дижитал банкны карт цэсээр болон өөрт ойрхон салбарт шинэ пин код үүсгэх боломжтой  Дижитал банкаар картын пин код солих заавар:https://www youtube com/watch?v=kCJS3hY3Wng
        Интернэт пин код авах  Хариулт: Та Дижитал банк, ХасБанкны АТМ-ээс юмуу 18001888 лавлах дугаарт холбогдон интернэт пин код авна уу  Картын онлайн гүйлгээний И-пин код авах заавар:https://www youtube com/watch?v=FxkzNptP-Sw
        Карт хаалгах  Хариулт:Та Дижитал банкаар картаа түр хаах боломжтой  Картаа түр хаах заавар:https://www youtube com/watch?v=ZX9_JkjAhEM
        Гүйлгээний лимит  Хариулт:Картын төрлөөс хамааран өдрийн гүйлгээний лимит өөр байдаг 	
        Карт нээлгэх  Хариулт:Та гээгдүүлсэн картаа нээлгэх бол өөрт ойр байрлах салбарт хандаж нээлгэнэ  Шинээр захиалсан картаа идэвхжүүлэх бол 18001888 лавлах дугаарт холбогдоно уу 
        Карт захиалах  Хариулт:Та XacBank Digital аппликэйшн татаж аван эсвэл интернэт банкаар карт захиална уу 
        QPay-Цахим түрийвч бүртгүүлэх  Хариулт:Та гар утсандаа QPay Wallet АПП татаж, бүртгэл үүсгэн ашиглах боломжтой 	
        QPay-Цахим түрийвч Карт бүртгүүлэх  Хариулт:Та АПП-аар нэвтэрч орсны дараа баруун буланд байрлах + товчийг сонгон картын мэдээллээ бүртгүүлнэ үү 	
        """

    # Define the REACT_DOCSTORE tool 
    react_docstore_tool = Tool( 
        name="react-docstore", 
        description="Мэдээлэл авахдаа эхлээд боддог дараа нь үйлдэл хийдэг агент. Энэ агент нь асуултад хариулахын тулд холбогдох мэдээллийг хайх боломжтой мэдээллийн санг ашиглах боломжтой.",
        func=chroma_retrieval_tool 
    )
    tools = load_tools(["google-serper", "llm-math"], llm=llm)
    tools.append(react_docstore_tool)

    # Initialize conversation memory
    memory = ConversationBufferMemory()

    agent = initialize_agent(
        tools, 
        llm, 
        agent="zero-shot-react-description", 
        memory=memory, 
        verbose=True, 
        prompt=template
    )
    # template_init = """Доорх асуултанд хариулаарай. Мэдээлэл авахдаа эхлээд боддог дараа нь үйлдэл хийдэг агент. Энэ агент нь асуултад хариулахын тулд холбогдох мэдээллийг хайх боломжтой мэдээллийн санг ашиглах боломжтой.
    #     Асуулт: {query}
    #     Хариулт: Хариулахад алхам алхмаар бодож үзье.   
    # """
    # prompt = PromptTemplate.from_template(template_init)
    # formatted_prompt = prompt.format(query=query)

    # # Create the RetrievalQA chain with the formatted prompt and the model
    # qa = RetrievalQA.from_llm(llm, retriever=vectorstore.as_retriever())

    # If the query is provided, run the QA chain
    if query:
        # answer = qa.run(formatted_prompt)
        answer = agent.run(query)
        answer = translate_client.translate(answer, target_language="mn")["translatedText"]
        # Display the first part of the answer
        st.write(answer[:500])  # Display the first 500 characters

        # Check if the answer exceeds the displayed limit
        if len(answer) > 500:
            # Display "Continue" button
            if st.button("Continue"):
                # Display the rest of the answer
                st.write(answer[500:])