from pydantic import BaseModel, Field, ValidationError
from typing import Union
from langchain.chains import create_tagging_chain_pydantic


class Tags0(BaseModel):
    """
    pydantic model for user's data
    """
    first_name: Union[str, None]
    last_name: Union[str, None]
    age: Union[int, None]
    weight: Union[float, None]
    height: Union[float, None]
    weight_unit: Union[str, None]
    height_unit: Union[str, None]
    BMI: Union[float, None]


class Tags1(BaseModel):
    """
    pydantic model for user's data
    """
    first_name: str | None = Field(
        description="this is the first name of the user",
        default=None)
    last_name:  str | None = Field(
        description="this is the last name of the user",
        default=None)
    age:  float | None = Field(
        description="this is the age of the user",
        default=None)
    weight:  float | None = Field(
        description="this is the weight of the user",
        default=None)
    height:  float | None = Field(
        description="this is the height of the user",
        default=None)
    weight_unit:  str | None = Field(
        description="this is the unit of the weight of the user",
        default=None)
    height_unit:  str | None = Field(
        description="this is the unit of the weight of the user",
        default=None)
    BMI:  float | None = Field(
        description="this is the BMI of the user",
        default=None)

    # @validator("age")
    # def ensure_age_over_18(cls, v):
    #     if v < 18:
    #         raise ValueError("age must be 18 or older")
    #     elif v > 150:
    #         raise ValueError("age must be < 150")
    #     return v
    #
    # @validator("age", "income", "total_household_debt", "total_household_savings")
    # def ensure_positive_value(cls, v):
    #     if v < 0:
    #         raise ValueError("must be positive number")
    #     return v


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


def filter_response(text_input: str, person, llm, conversation_type: str):
    """
    search for fields of the current response with respect a pydantic model
    :params text_input: user's query
    :params person: pydantic model
    :params text_input: llm model for extraction
    :params conversation_type:
    """
    if conversation_type == "life_insurance":
        pydantic_model = Tags1
    else:
        return person

    chain = create_tagging_chain_pydantic(pydantic_model, llm)
    res = chain.run(text_input)
    print(f"current result :{res}, type:{type(res)}")

    person = add_non_empty_details(person, res)
    return person
