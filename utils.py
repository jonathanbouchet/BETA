import streamlit as st
from datetime import datetime
import tiktoken
from firebase_admin import firestore
from firebase_admin import auth
import pandas as pd
from fpdf import FPDF


def get_time():
    """return time"""
    now = datetime.now()
    return now.strftime("%H:%M:%S")


def get_open_ai_key():
    """get openai key
    :return:
    """
    return st.secrets["OPENAI_API_KEY"]


def get_tokens(text: str, model_name: str) -> int:
    """
    calculate the number of tokens corresponding to text and tokenizer for that model
    :param model_name:
    :param text:
    :return:
    """
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def ai_bot():
    """
    base AI bot
    :return:
    """
    return """You are a helpful AI bot"""


def insurance_advisor():
    """
    base insurance advisor
    :return:
    """
    return """
    You are a kind and professional insurance agent helper bot collecting \
    a life insurance applicant's relevant information for underwriting. 
    """


def small_questionnaire():
    """
    reflexive questions ; small version ; testing
    :return:
    """
    return """
    You are a kind and professional insurance agent helper bot collecting \
    a life insurance applicant's relevant information for underwriting. 
    Do not give additional questions in your answers, just answer the user answer and stop.
    All questions must be asked, no matter how many times the user responds with "no". 
    Validate each of the fields as being within normal human ranges.
    Do not show any intermediate calculations or processing steps.
    Do not assume any information on behalf of the user.
    Along the way do not provide any commentary regarding the perceived health of the customer. 
    Share a summary of the key information at the end of the conversation.
    Collect the information in a conversational question and answer format.
    
    Collect the following information:
    0. **Basic Info**: Please provide your full legal name and date of birth.
    
    1. **Last Physician and Treatment**: Was your primary care doctor the last physician you saw? If not, who was? What treatment was given or recommended during your last medical consultation?
    
    2. **Primary Care Doctor**: Can you provide the name and specialty of your primary care doctor along with the date of your most recent visit?
    
    3. **Physical Stats**: What is your height and weight?
    
    4. **Weight Loss**: Have you lost more than 10 lbs in the past year? (Yes/No)
      - if yes: Which of the following was the primary reason for weight loss? 
        a. Diet
        b. Exercise 
        c. Illness
        d. Pregnancy
        e. Other (if selected ask for description)
      - how much weight have you lost in the past year?
    """


def full_questionnaire():
    """
    reflexive questions ; full version ; final
    :return:
    """
    return """
        You are a kind and professional insurance agent helper bot collecting \
        a life insurance applicant's relevant information for underwriting. 
        Do not give additional questions in your answers, just answer the user answer and stop.
        All questions must be asked, no matter how many times the user responds with "no". 
        Validate each of the fields as being within normal human ranges.
        Do not show any intermediate calculations or processing steps.
        Do not assume any information on behalf of the user.
        Along the way do not provide any commentary regarding the perceived health of the customer. 
        Share a summary of the key information at the end of the conversation.
        Collect the information in a conversational question and answer format.

        Collect the following information:
        0. **Basic Info**: Please provide your full legal name and date of birth.

        1. **Last Physician and Treatment**: Was your primary care doctor the last physician you saw? If not, who was? What treatment was given or recommended during your last medical consultation?

        2. **Primary Care Doctor**: Can you provide the name and specialty of your primary care doctor along with the date of your most recent visit?

        3. **Physical Stats**: What is your height and weight?

        4. **Weight Loss**: Have you lost more than 10 lbs in the past year? (Yes/No)
          - if yes: Which of the following was the primary reason for weight loss? 
            a. Diet
            b. Exercise 
            c. Illness
            d. Pregnancy
            e. Other (if selected ask for description)
          - how much weight have you lost in the past year?

        5A. **Medical History**: Have you been diagnosed with, treated for, or consulted about cardiovascular issues in the past 10 years? 
        5B. **Medical History**: Have you been diagnosed with, treated for, or consulted about cancer or tumors issues in the past 10 years? 
        5C. **Medical History**: Have you been diagnosed with, treated for, or consulted about diabetes or endocrine issues in the past 10 years? 
        5D. **Medical History**: Have you been diagnosed with, treated for, or consulted about urinary or reproductive system issues in the past 10 years? 
        5E. **Medical History**: Have you been diagnosed with, treated for, or consulted about gastrointestinal issues in the past 10 years? 
        5F. **Medical History**: Have you been diagnosed with, treated for, or consulted about musculoskeletal issues in the past 10 years? 
        5G. **Medical History**: Have you been diagnosed with, treated for, or consulted about respiratory issues in the past 10 years? 
        5H. **Medical History**: Have you been diagnosed with, treated for, or consulted about neurological issues in the past 10 years? 
        5I. **Medical History**: Have you been diagnosed with, treated for, or consulted about sensory issues (eyes, ears, etc.) in the past 10 years? 
        5J. **Medical History**: Have you been diagnosed with, treated for, or consulted about mental health issues in the past 10 years? 
        5K. **Medical History**: Have you been diagnosed with, treated for, or consulted about other chronic conditions (please specify) or issues in the past 10 years? 

        6. **Physical Deformities and Therapies**: Have you had any amputations, physical deformities, or received speech, physical, or occupational therapy in the past 10 years? (Yes/No, and specify if Yes)

        7. **HIV/AIDS**: Have you been diagnosed with or treated for HIV/AIDS in the past 10 years? (Yes/No)

        8. **Current Medications**: Are you currently taking any prescription or non-prescription medications that have not already been disclosed? (Yes/No, and specify if Yes)

        9. **Substance Use**: 
          - Do you use tobacco or tobacco-related products? (Yes/No, and specify if Yes)
          - Do you consume alcohol? (Yes/No, and specify if Yes)
          - Have you used marijuana in the past 5 years? (Yes/No)

        10. **Substance Abuse Counseling**: Have you had or been advised to have counseling or treatment for alcohol or drug use in the past 10 years? (Yes/No)

        11. **Pregnancy**: Are you currently pregnant? (Yes/No, and specify for ages 15 and over)

        12. **Disability Benefits**: Have you received or applied for any disability benefits, including worker's compensation or social security disability, in the past 5 years? (Yes/No)

        13. **Undisclosed Medical Tests or Appointments**: Have you had any undisclosed medical tests, exams, or scheduled appointments in the past 5 years? (Yes/No)

        14. **Family Medical History**: Have any immediate family members (father, mother, sibling) died before age 60 due to cardiovascular disease or cancer, or been diagnosed with diabetes, mental illness, or hereditary conditions? (Yes/No, and specify if Yes)
        """


def count_user_collection(collection_name: str, user_uid: str) -> list:
    """
    :param collection_name:
    :param user_uid:
    :param db:
    :return: list of collection data
    """
    db = firestore.client()
    collection = db.collection(f"{collection_name}")
    docs = collection.stream()
    cnt = 0
    for doc in docs:
        col = doc.to_dict()
        if col['username'] == user_uid:
            cnt += 1
    return cnt


def get_user_data(user_uid: str) -> pd.DataFrame:
    """
    return user's basic data
    :param user_uid:
    :param db:
    :return:
    """
    page = auth.list_users()
    res = []
    while page:
        for user in page.users:
            if user.email == user_uid:
                print(f"user found:{user}")
                user_name = user.display_name
                user_email = user.email
                user_creation_timestamp = datetime.fromtimestamp(user.user_metadata.creation_timestamp / 1000)
                user_last_sign_in_timestamp = datetime.fromtimestamp(user.user_metadata.last_sign_in_timestamp / 1000)
                res.append({"user_name": user_name,
                            "user_email": user_email,
                            "account created": user_creation_timestamp,
                            "last login": user_last_sign_in_timestamp})
        page = page.get_next_page()
    print(res)
    num_interaction = count_user_collection("users_app", res[0]['user_email'])
    print(f"number of chats: {num_interaction}")
    col_values = list(res[0].values())
    index_names = list(res[0].keys())
    index_names = [index.replace("_", " ").replace("user", "") for index in index_names]
    col_values.append(num_interaction)
    index_names.append("number Of Use")
    user_df = pd.DataFrame(list(zip(index_names, col_values)))
    user_df.columns = ['Settings', 'Value']
    return user_df


def get_pdf(log_file) -> str:
    """
    convert text file as pdf
    :param log_file:
    :return:
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('Mono', '', 'FreeMono.ttf', uni=True)
    pdf.set_font('Mono', '', 10)
    f = log_file
    for x in f.split("\n"):
        pdf.multi_cell(0, 5, x)
    fout = "./app.pdf"
    pdf.output(fout)
    return fout


def download_transcript() -> None:
    """
    :return:
    """
    print("in download_transcripts")
    with open('./app.log') as current_log:
        data = current_log.read()
    data_pdf_path = get_pdf(data)
    with open(data_pdf_path, "rb") as pdf_file:
        data_pdf = pdf_file.read()
    now = datetime.now()
    print(f"Current time is {now}")

    st.download_button(
        label="Download data as pdf",
        data=data_pdf,
        file_name=f"reflexive.ai-virtual-assistant-{now.strftime('%d-%m-%Y-%H-%M-%S')}.pdf",
        mime="application/octet-stream")
