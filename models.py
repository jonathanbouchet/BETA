from pydantic import BaseModel, Field, ValidationError
from typing import Union
from langchain.chains import create_tagging_chain_pydantic, create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from pydantic import Extra
from datetime import datetime


class Tags0(BaseModel):
    """
    pydantic model for user's data
    """
    full_name: Union[str, None] = Field(
        description="this is the full name of the user",
        default=None)
    first_name: Union[str, None] = Field(
        description="this is the first name of the user",
        default=None)
    last_name: Union[str, None]  = Field(
        description="this is the last name of the user",
        default=None)
    age: Union[int, None] = Field(
        description="this is the age of the user",
        default=None)
    date_of_birth: Union[str, None] = Field(
        description="this is the date of birth of the user",
        default=None)
    weight: Union[float, None] = Field(
        description="this is the weight of the user",
        default=None)
    height: Union[float, None] = Field(
        description="this is the height of the user",
        default=None)
    weight_unit: Union[str, None] = Field(
        description="this is the unit of the weight of the user",
        default=None)
    height_unit: Union[str, None]  = Field(
        description="this is the unit of the weight of the user",
        default=None)
    BMI: Union[float, None] = Field(
        description="this is the BMI of the user",
        default=None)
    class Config:
        extra = Extra.allow


def add_bmi(user_data: dict) -> dict:
    """calculate BMI as w / h*h where w = weight in kilograms and h = height in meter
    :params user_data: dict
    "returns: same user data with BMI added
    """
    height = user_data["height"]
    weight = user_data["weight"]
    if user_data["height_unit"] == "cm":
        height /= 100
    if user_data["height_unit"] == '"' or user_data["height_unit"] == "'":
        tmp = str(height).split(".")
        feet = int(tmp[0])
        inches = int(tmp[1])
        height = 2.54 * (feet*12 + inches) / 100
    if user_data["weight_unit"] == "lbs":
        weight /= 2.205
    bmi = weight / (height * height)
    # user_data['bmi'] = weight / (height * height)
    return bmi


def add_age(user_data: dict) -> dict:
    dob = user_data['date_of_birth']
    age = None
    if dob:
        dob_datetime = datetime.strptime(dob, "%m/%d/%Y")
        age = int(datetime.now().year - dob_datetime.year)
        # user_data["age"] = int(age)
    return age


def extract_data(assistant_summary: str, openai_key) -> dict:
    """
    run langchain tagging chain to extract data from summary according a pydantic model
    :param assistant_summary:
    :return:
    """

    llm_extraction = ChatOpenAI(temperature=0,
                                model="gpt-4",
                                openai_api_key=openai_key)

    # first we extract the summary data
    user_summary_extraction = create_extraction_chain_pydantic(Tags0, llm_extraction)
    extracted_data = user_summary_extraction.run(assistant_summary)
    print(f"Pydantic data extracted:{extracted_data}")
    single_extracted_data = extracted_data[0]
    print(f"single Pydantic data extracted:{extracted_data}")
    single_extracted_data_dict = dict(single_extracted_data)
    print(f"data dict extracted:{single_extracted_data_dict}")
    current_bmi = add_bmi(single_extracted_data_dict)
    current_age = add_age(single_extracted_data_dict)
    single_extracted_data.age = current_age
    single_extracted_data.BMI = current_bmi
    print(f"after adding BMI and AGe: {single_extracted_data}")

    return single_extracted_data


def check_extracted_data(extracted_data, query) -> None:
    """
    check the missing fields of the current response with respect a pydantic model
    :params extracted data:
    :returns: None
    """
    try:
        if len(extracted_data) == 0:
            print(f"no extracted data for the query: '{query}'")
        else:
            print(f"only partial extraction for the query: {extracted_data}")
    except ValidationError as e:
        print(f"validation missing errors: '{query}'")
        error_msg = e.errors()
        print(error_msg)


def add_non_empty_details(current_details, new_details):
    """
    add current fields extracted from user's query to existing pydantic model
    :params current_details:
    :params new_details:
    """
    new_details_dict = dict(new_details)
    non_empty_details = {k: v for k, v in new_details_dict.items() if v not in [None, "", 0]}
    updated_details = current_details.copy(update=non_empty_details)  # v1
    return updated_details


def filter_response(text_input: str, person, llm,):
    """
    search for fields of the current response with respect a pydantic model
    :params text_input: user's query
    :params person: pydantic model
    :params text_input: llm model for extraction
    """
    pydantic_model = Tags0

    chain = create_tagging_chain_pydantic(pydantic_model, llm)
    res = chain.run(text_input)
    print(f"current result :{res}, type:{type(res)}")

    person = add_non_empty_details(person, res)
    return person
