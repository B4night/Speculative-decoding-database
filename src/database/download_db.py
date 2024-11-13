from db import data, LC_QuAD_query

download_location = '/home/feic/pjs/Speculative-decoding-database/data/qa_reduced.csv'

def Store_question_answer():
    questions = []
    answers = []
    with open(download_location, mode='w') as f:
        # f.write("Question§Answer\n")
        for i in range(len(data)):
            question = data[i]['corrected_question']
            query = data[i]['sparql_query']
            # if question != "Name some comic characters created by Bruce Timm?":
            #     continue
            answer = LC_QuAD_query(query)
            questions.append(question)
            # convert '\n' to ' '
            answer = answer.replace('\n', ' ')
            answers.append(answer)
            # flush the data to csv file
            f.write(f"{question}§{answer}\n")
            f.flush()
            
Store_question_answer()