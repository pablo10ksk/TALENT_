# https://docs.streamlit.io/
# streamlit run main.py - python -m streamlit run main.py
import streamlit as st
from elasticsearch import Elasticsearch
import requests
import json
import os
import webbrowser
import re
from dotenv import load_dotenv

load_dotenv()

PRIVATE_KEY_MIST = os.getenv("PRIVATE_MISTRAL_TOKEN")

if "messages" not in st.session_state:
    st.session_state.messages = []


def qry_data(keywords):
    obj_es = Elasticsearch(
        cloud_id=os.getenv("ELASTIC_LINK"),
        basic_auth=(os.getenv("ELASTIC_USR"), os.getenv("ELASTIC_PSS")),
    )
    query = create_es_query(keywords)
    res = obj_es.search(index="talent_ml", body=query)
    return res


def get_result_data(object_result, prompt) -> str:
    return_str = ""
    for hit_ in object_result["hits"]["hits"]:
        return_str += f"Title: {hit_['_source'].get('titulo', '')}, Content: {hit_['_source'].get('contenido_seccion', '')}, Notes: {hit_['_source'].get('texto_notas', '')}\n"
    return return_str, object_result["hits"]["hits"]


def ask_mistral(prompt):
    url_ = "https://api.mistral.ai/v1/chat/completions"
    payload_ = json.dumps(
        {
            "model": "open-mixtral-8x7b",
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 500,
            "stream": False,
            "safe_prompt": False,
        }
    )
    headers = {
        "Authorization": f"Bearer {PRIVATE_KEY_MIST}",
        "Content-Type": "application/json",
    }
    print(headers)
    response_ = requests.post(url_, data=payload_, headers=headers)
    print(response_)
    return response_.json()


def extract_important_info(prompt):
    preguntaMistralExtractInfo = (
        f"Extract the important keywords from this question: {prompt}"
    )
    mistral_response = ask_mistral(preguntaMistralExtractInfo)
    print(mistral_response)
    # Asumiendo que la respuesta tiene una estructura similar a la original
    return mistral_response["choices"][0]["message"]["content"]


def super_result_algorithm(prompt):
    important_info = extract_important_info(prompt)
    keywords = important_info.split()  # Separar las palabras importantes
    results_ = qry_data(keywords)
    string_results_, array_results_ = get_result_data(results_, prompt)
    prompt_full = f"The user has asked for the following question '{prompt}' and we have found that it can be located in the following sections: {string_results_}. List the locations to the user and tell them that they can navigate through the web.Asked in Spanish"
    llm_response = ask_mistral(prompt_full)
    return llm_response["choices"][0]["message"]["content"], array_results_


def openurl(textdata):
    webbrowser.open_new_tab(url)


def create_es_query(strings):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"query_string": {"query": " ".join(strings), "default_field": "*"}}
                ]
            }
        },
        "knn": {
            "field": "ml.inference.outmsgvector_.predicted_value",
            "k": 10,
            "num_candidates": 100,
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": ".multilingual-e5-small_linux-x86_64",
                    "model_text": " ".join(strings),
                }
            },
        },
        "rank": {"rrf": {}},
        "size": 5,
    }

    return query


# Obtain the Mistral generated description/context for each result
def get_description_for_result(result):
    content = result["_source"].get("contenido_seccion", "")
    prompt = f"Provide a very brief, one-sentence description for the following content: {content}. Always answer in spanish!"
    mistral_response = ask_mistral(prompt)
    return mistral_response["choices"][0]["message"]["content"]


# Interfaz de usuario con Streamlit
if prompt := st.chat_input("Pregunta algo del gobierno de Salamanca"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Ya mismo te contesto..."):
            str_response, response_ = super_result_algorithm(prompt)
            print(response_)
            st.session_state.messages.append({"name": "assistant", "text": response_})
            st.write(str_response)

            breadcrumbs = response_[0]["_source"]["title_migaspan"].upper()
            st.write(breadcrumbs)

            for idx, response in enumerate(response_):

                # Obtain and show the Mistral generated description/context for each result
                description = get_description_for_result(response)
                st.markdown(
                    "<p style='font-size: 18px;'><b>Descripción:</b></p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 14px;'>{description}</p>",
                    unsafe_allow_html=True,
                )
                if not "titulo" in response["_source"]:
                    continue
                with st.expander(response["_source"]["titulo"]):
                    st.header(response["_source"]["titulo"])

                    if "breve_descripcion_tramite" in response["_source"]:
                        st.subheader("Descripción:")
                        st.write(response["_source"]["breve_descripcion_tramite"])

                    st.subheader("Contenido de Sección:")
                    max_len = max(
                        len(response["_source"]["secciones"]),
                        len(response["_source"]["contenido_seccion"]),
                    )
                    for i, (seccion, contenido) in enumerate(
                        zip(
                            response["_source"]["secciones"],
                            response["_source"]["contenido_seccion"],
                        )
                    ):
                        st.markdown(f"**{seccion}:**")
                        st.write(contenido)

                    st.markdown("<hr>", unsafe_allow_html=True)

                    st.success("¿Necesitas más información?")
                    st.write("¡Haz clic en el botón a continuación!")

                    st.button(
                        "Más información",
                        on_click=openurl,
                        args=(
                            "https://www.aytosalamanca.gob.es/es/tramitesgestiones/presentacion/",
                        ),
                        key=f"mas_info_{idx}",
                    )