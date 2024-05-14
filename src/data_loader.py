# src/data_loader.py

import pandas as pd
import xml.etree.ElementTree as ET

def load_data(filepath):
    # Parse XML file
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Extract data into DataFrame
    data = []
    for question in root.findall('.//NLM-QUESTION'):
        question_id = question.get('questionid')
        subject_elem = question.find('SUBJECT')
        message_elem = question.find('MESSAGE')

        # Handle cases where SUBJECT or MESSAGE tags are missing or empty
        subject = subject_elem.text.strip() if subject_elem is not None and subject_elem.text is not None else ''
        message = message_elem.text.strip() if message_elem is not None and message_elem.text is not None else ''

        # Extract subquestions
        for subquestion in question.findall('.//SUB-QUESTION'):
            subquestion_id = subquestion.get('subqid')
            focus_elem = subquestion.find('.//FOCUS')
            type_elem = subquestion.find('.//TYPE')

            focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text is not None else ''
            type_ = type_elem.text.strip() if type_elem is not None and type_elem.text is not None else ''

            # Extract answers
            for answer in subquestion.findall('.//ANSWER'):
                answer_id = answer.get('answerid')
                answer_text = answer.text.strip() if answer.text is not None else ''

                data.append({
                    'QuestionID': question_id,
                    'Subject': subject,
                    'Message': message,
                    'SubQuestionID': subquestion_id,
                    'Focus': focus,
                    'Type': type_,
                    'AnswerID': answer_id,
                    'Answer': answer_text
                })

    df = pd.DataFrame(data)
    return df
