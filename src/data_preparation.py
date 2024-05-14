# src/data_preparation.py

import xml.etree.ElementTree as ET
import pandas as pd
from src.text_cleaning import limpiar_texto

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    data = []

    for question in root.findall('NLM-QUESTION'):
        question_id = question.get('questionid')
        subject_elem = question.find('SUBJECT')
        subject = limpiar_texto(subject_elem.text) if subject_elem is not None and subject_elem.text else ''
        message_elem = question.find('MESSAGE')
        message = limpiar_texto(message_elem.text) if message_elem is not None and message_elem.text else ''
        
        for sub_question in question.findall('SUB-QUESTIONS/SUB-QUESTION'):
            sub_question_id = sub_question.get('subqid')
            focus_elem = sub_question.find('ANNOTATIONS/FOCUS')
            focus = limpiar_texto(focus_elem.text) if focus_elem is not None and focus_elem.text else ''
            q_type_elem = sub_question.find('ANNOTATIONS/TYPE')
            q_type = limpiar_texto(q_type_elem.text) if q_type_elem is not None and q_type_elem.text else ''
            
            for answer in sub_question.findall('ANSWERS/ANSWER'):
                answer_id = answer.get('answerid')
                answer_text = limpiar_texto(answer.text) if answer.text else ''
                data.append({
                    'question_id': question_id,
                    'subject': subject,
                    'message': message,
                    'sub_question_id': sub_question_id,
                    'focus': focus,
                    'type': q_type,
                    'answer_id': answer_id,
                    'answer': answer_text
                })
    
    return pd.DataFrame(data)

def load_data(file_path):
    return parse_xml(file_path)
