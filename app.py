import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd
from datetime import datetime
import os

# Define file path for server-side CSV storage
csv_file_path = 'poem_history_server.csv'

# Function to append a new entry to the CSV file
def save_to_csv_server_side(prompt, poem):
    new_entry = pd.DataFrame({
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'prompt': [prompt],
        'poem': [poem]
    })
    
    if os.path.exists(csv_file_path):
        # Append without writing header
        new_entry.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        # Write with header
        new_entry.to_csv(csv_file_path, mode='w', header=True, index=False)

# Set page title and layout
st.set_page_config(page_title="Thai Poem Generator", layout="wide")

# Custom CSS to improve readability
st.markdown("""
<style>
    .poem-output {
        font-family: 'Sarabun', sans-serif;
        font-size: 18px;
        line-height: 1.6;
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 5px;
        white-space: pre-wrap;
    }
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌾 PhoneticKlon-LM : AI แต่งกลอน")
st.markdown("""
โปรเจกต์นี้เป็นส่วนหนึ่งของวิชา 2110572 NLP SYSTEM
เป็นการพัฒนาโมเดลที่สามารถสร้างกลอนแปดได้ให้ถูกต้องตามฉันทลักษณ์ ด้วยการเสนอเทคนิคในการ Fine-Tune แบบใหม่ เพื่อให้โมเดลสามารถจับความสัมพันธ์ของคำที่คล้องจองกัน แม้มี Dataset ที่น้อย    
""")

# Define available models
available_model_id = {
    "nmt-mixed": "Pongsaky/llama3.2-typhoon2-1b-instruct-tagged_nmt-mixed",
    "nmt_syllable_mixed": "Pongsaky/llama3.2-typhoon2-1b-instruct-tagged_nmt_syllable_mixed",
    "non-nmt": "Pongsaky/llama3.2-typhoon2-1b-instruct-tagged_non-nmt",
}

# Model tags list (same as dictionary keys)
model_tags = list(available_model_id.keys())

# Sidebar for model settings
st.sidebar.title("Model Settings")

# Model selection dropdown
model_choice = st.sidebar.selectbox(
    "Select Model", 
    options=range(len(model_tags)),
    format_func=lambda x: model_tags[x],
)
model_id = available_model_id[model_tags[model_choice]]

# Generation parameters
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.9, step=0.1)
top_p = st.sidebar.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=50, value=15, step=1)
max_new_tokens = st.sidebar.slider("Maximum Length", min_value=50, max_value=1000, value=512, step=50)
tab_format = st.sidebar.slider("tab_format", min_value=1, max_value=5, value=4, step=1)


if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Text input area
user_input = st.text_area(
    "Enter your prompt:", 
    value=st.session_state.user_input,
    placeholder="ช่วยแต่งกลอนที่...", 
    height=100,
    key="user_input"  # This connects the widget to session state
)

# Helper functions
upper_vowel = ["่", "้", "๊", "๋", "็", "ิ", "ี", "ึ", "ื", "ุ", "ู", "์", "ั"]

@st.cache_resource
def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model

def print_poem_formatted(text):
    # lines = text.split('\n')
    
    # max_width = -1
    # for line in lines:
    #     if '\t' in line:
    #         first_tab = line.split('\t')[0]
    #         tab_length = len(first_tab) - sum(1 for char in first_tab if char in upper_vowel)
    #         max_width = max(max_width, tab_length)
    
    # max_width += 4
    # formatted_poem = []
    
    # for line in lines:
    #     if '\t' in line:
    #         tabs = line.split('\t')
    #         formatted_line = ""
    #         for i, tab in enumerate(tabs):
    #             vowels_count = sum(1 for char in tab if char in upper_vowel)
    #             char_length = len(tab) - vowels_count
                
    #             formatted_line += tab
    #             if i < len(tabs) - 1:  # If not the last tab
    #                 formatted_line += "<span style=\"display:inline-block; width:0.6em;\"></span>" * (max_width - char_length)
    #         formatted_poem.append(formatted_line)
    #     else:
    #         formatted_poem.append(line)
    
    # return "\n".join(formatted_poem)
    return text.replace("\t", "\t"*tab_format).replace("\n", "<br>")  # Replace tabs with spaces and newlines with <br>

def remove_non_thai_characters(text):
    thai_char_pattern = r"[ก-๙\n\t]"
    cleaned_text = "".join(re.findall(thai_char_pattern, text))
    return cleaned_text
st.caption("📝 Your prompt and the generated poem will be saved to help improve the model.")

# Generate button
if st.button("Generate Poem", type="primary"):
    if user_input:
        with st.spinner("กำลังแต่งกลอน โปรดรอสักครู่..."):
            try:
                # Load model and tokenizer
                tokenizer, model = load_model(model_id)
                
                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert Thai poet specializing in `กลอนแปด` (Eight-syllable verse) poetry. When a user provides a prompt, respond ONLY with a Thai poem (2-4 stanzas) that addresses their request, without any explanations or commentary. Your poem must strictly follow traditional กลอนแปด structure and rhyming patterns. Include phonetic rhyming tags for all rhyming words using the format: `<r>[vowel][ending consonant]word</r>` (examples: `<r>[a][w]เขา</r>`, `<r>[o][k]นก</r>`, `<r>[a]ผา</r>`, `<r>[i]ศรี</r>`). These tags should mark all external and internal rhymes according to proper กลอนแปด structure. Create vivid, culturally appropriate poetry that demonstrates mastery of Thai prosody while faithfully addressing the user's requested theme or scenario."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    },
                ]
                
                # Generate poem
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                
                response = outputs[0][input_ids.shape[-1]:]
                text_response = tokenizer.decode(response, skip_special_tokens=True)
                thai_response = remove_non_thai_characters(text_response)
                save_to_csv_server_side(user_input, thai_response)

                # # Display results in tabs for better organization
                # tab1, tab2 = st.tabs(["Untagged Output", "Tagged Output (with rhyming tags)"])
                
                # with tab1:
                st.markdown(f'<div class="poem-output">{print_poem_formatted(thai_response)}</div>', unsafe_allow_html=True)
                
                # with tab2:
                #     st.markdown(f'<div class="poem-output">{text_response}</div>', unsafe_allow_html=True)
                
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a prompt to generate a poem.")

# Example prompts section
st.markdown("## Example Prompts")
col1, col2 = st.columns(2)

examples = [ "ช่วยแต่งกลอนที่พูดถึงทะเลที่สวยงามและมีคลื่นลมแรง",
            "อยากได้กลอนที่พรรณนาถึงความสวยงามของดวงดาราที่ระยิบระยับอยู่บนท้องฟ้า",
            "ช่วยแต่งกลอนที่เกี่ยวกับความรักของหนุ่มสาวสองคนกำลังเกี้ยวพาราสีกันให้หน่อย",
            "ขอกลอนที่เล่าถึงความสวยงามจากวิวเมืองเชียงใหม่ที่มองลงมาจากดอยสุเทพหน่อยสิ",
            "เพียงสบตาเท่านั้น หัวใจฉันก็อบอุ่นใจ เพียงสบตาเท่านั้น ฉันก็รู้ทันใด ว่าเธอคือใครคนนั้นที่ฉันรอ",
            "เยลลี่ยุทธเมืองสุรินทร์ ใครได้เห็นก็ต้องอยากกิน โคตรอร่อยเคี้ยวเพลินนุ่มลิ้น จะไปบอกแม่ว่าหนูอยากกิน ",
            "แต่งกลอนแปดเกี่ยวกับความปวดหัวของการเป็นผู้เฝ้ายามอยู่ประจำจนเช้าทุกวันเพื่อเฝ้าบ้านให้เธอให้หน่อย",
]

# Function to set example text
def set_example(example_text):
    st.session_state.user_input = example_text

st.button(f"🌊 \"ช่วยแต่งกลอนที่พูดถึงทะเลที่สวยงามและมีคลื่นลมแรง\"", key="example_0", on_click=set_example, args=(examples[0],))
st.button(f"🌌 \"อยากได้กลอนที่พรรณนาถึงความสวยงามของดวงดาราที่ระยิบระยับอยู่บนท้องฟ้า\"", key="example_1", on_click=set_example, args=(examples[1],))
st.button(f"🌹 \"ช่วยแต่งกลอนที่เกี่ยวกับความรักของหนุ่มสาวสองคนกำลังเกี้ยวพาราสีกันให้หน่อย\"", key="example_2", on_click=set_example, args=(examples[2],))
st.button(f"⛰️ \"ขอกลอนที่เล่าถึงความสวยงามจากวิวเมืองเชียงใหม่ที่มองลงมาจากดอยสุเทพหน่อยสิ\"", key="example_3", on_click=set_example, args=(examples[3],))
st.button(f"🥰 \"เพียงสบตาเท่านั้น หัวใจฉันก็อบอุ่นใจ เพียงสบตาเท่านั้น ฉันก็รู้ทันใด ว่าเธอคือใครคนนั้นที่ฉันรอ\"", key="example_4", on_click=set_example, args=(examples[4],))
st.button(f"🍭 \"เยลลี่ยุทธเมืองสุรินทร์ ใครได้เห็นก็ต้องอยากกิน โคตรอร่อยเคี้ยวเพลินนุ่มลิ้น จะไปบอกแม่ว่าหนูอยากกิน \"", key="example_5", on_click=set_example, args=(examples[5],))
st.button(f"👮🏻 \"แต่งกลอนแปดเกี่ยวกับความปวดหัวของการเป็นผู้เฝ้ายามอยู่ประจำจนเช้าทุกวันเพื่อเฝ้าบ้านให้เธอให้หน่อย\"", key="example_6", on_click=set_example, args=(examples[6],))


# Footer
st.markdown("---")
st.markdown("โปรเจกต์นี้เป็นส่วนหนึ่งของวิชา 2110572 NLP SYSTEM")
st.markdown("**สมาชิกกลุ่ม:**")
st.markdown("- กัมปนาท ยิ่งเสรี")
st.markdown("- พงศกร แก้วใจดี")
st.markdown("- ธนธรณ์ ปยะชาติ")
st.markdown("ผลงานต่อยอดจาก <a href='https://medium.com/@kampanatyingseree4704/klonsuphap-lm-%E0%B9%81%E0%B8%95%E0%B9%88%E0%B8%87%E0%B8%81%E0%B8%A5%E0%B8%AD%E0%B8%99%E0%B9%81%E0%B8%9B%E0%B8%94-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2-gpt-2-d2baffc80907'>Klonsuphap-LM</a>", unsafe_allow_html=True)

st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <p style="font-size: 0.8rem; color: gray; text-align: center;">
    ข้อจำกัดความรับผิดชอบ: ข้อความที่สร้างขึ้นโดยโมเดลปัญญาประดิษฐ์นี้เป็นเป็นผลลัพธ์จากการประมวลผลอัตโนมัติ ผู้จัดทำไม่รับรองความถูกต้อง ครบถ้วน หรือความเหมาะสมของเนื้อหาดังกล่าว และไม่ขอรับผิดชอบต่อผลลัพธ์ใด ๆ ที่อาจเกิดจากการนำข้อความไปใช้
    </p>
    """,
    unsafe_allow_html=True
)
